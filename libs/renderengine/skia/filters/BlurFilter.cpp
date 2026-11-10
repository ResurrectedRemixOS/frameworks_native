/*
 * Copyright 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS
#include "BlurFilter.h"
#include <SkBlendMode.h>
#include <SkCanvas.h>
#include <SkM44.h>
#include <SkPaint.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <common/ThreadStateCrashLogger.h>
#include <common/trace.h>
#include <log/log.h>

#include "RuntimeEffectManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>

namespace android {
namespace renderengine {
namespace skia {

// Catalyxx glass: a blur region drawn as a slab of glass with a rounded rim.
// The rim bends in what lies just past the edge (the glass's own shadow too),
// splits it into colors and catches the light; the flat middle shows the
// content behind it, clear or frosted. The glass has its own color, which
// turns from one tint to another where the content behind is light, so what
// sits on the glass stays readable.
//
// The shader comes in parts, as glass that melts together (kGlassMeltPart) is drawn just like
// glass on its own (kGlassOnePiecePart), only from a different outline.
static const char kGlassCommonPart[] = R"(
    uniform shader blurredInput;
    uniform shader originalInput;

    // How far past the edge the rim reaches, in rim widths.
    uniform float bend;
    // Strength of the light on the rim, 0 to 1.
    uniform float rimLight;
    // Color split of the bent light, 0 to 1.
    uniform float dispersion;
    // Darkness of the glass's shadow, 0 to 1.
    uniform float shadow;
    // Unit vector toward the light, y pointing down, and how far it has turned from its usual
    // place at the top left, in radians.
    uniform float2 lightDir;
    uniform float lightTurn;
    // The brightness of what's behind (0 to 1) around which the glass turns from its color over
    // dark content to its color over light content; 0 keeps the first everywhere.
    uniform float adaptPoint;
    // RR's lens edge, 0 to 1: how clear the rim of frosted glass stays. Above 0 the rim bends
    // what's under the glass out to its edge, not what's past the edge in.
    uniform float lensEdge;
    // 1 for glass that shows what's behind it a little richer in color, as RR's does; else 0.
    uniform float rich;

    float3 richer(float3 color) {
        float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
        float3 vivid = (float3(luma) + (color - float3(luma)) * 1.2) * 1.03;
        return mix(color, clamp(vivid, float3(0.0), float3(1.0)), rich);
    }

    // Signed distance to a rounded rect's outline (negative inside) and the outward normal.
    // Corner radii: top-left, top-right, bottom-right, bottom-left.
    float outlineOf(float2 p, float4 bounds, float4 radii, out float2 normal) {
        float2 center = (bounds.xy + bounds.zw) * 0.5;
        float2 halfSize = (bounds.zw - bounds.xy) * 0.5;
        float2 q = p - center;
        float2 side = float2(q.x < 0.0 ? -1.0 : 1.0, q.y < 0.0 ? -1.0 : 1.0);
        float radius = q.y < 0.0 ? (q.x < 0.0 ? radii.x : radii.y)
                                 : (q.x < 0.0 ? radii.w : radii.z);
        float2 d = abs(q) - halfSize + radius;
        if (d.x > 0.0 && d.y > 0.0) {
            normal = side * normalize(d);
        } else if (d.x > d.y) {
            normal = float2(side.x, 0.0);
        } else {
            normal = float2(0.0, side.y);
        }
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
    }
)";

// Glass on its own, one rounded rect, which Skia draws with its edge smoothed.
static const char kGlassOnePiecePart[] = R"(
    // Glass outline in local coordinates; joined sides are pushed far out.
    uniform float4 shapeBounds;
    uniform float4 cornerRadii;
    // Width of the curved rim.
    uniform float rimWidth;
    // Blurred share of the flat middle, 0 to 1.
    uniform float frost;
    // The falloff and offset of the glass's shadow.
    uniform float shadowSoftness;
    uniform float2 shadowDrop;
    // The glass's own color over dark and over light content (straight alpha = how much it
    // covers).
    uniform float4 tintOverDark;
    uniform float4 tintOverLight;

    // Darkness of the glass's shadow on the content at p. It fades in across the
    // edge, so the rim's bent view of it has no hard line.
    float shadowAt(float2 p, float rim) {
        if (shadow <= 0.0) {
            return 0.0;
        }
        float2 unused;
        float fadeIn = smoothstep(-0.4 * rim, 0.4 * rim,
                                  outlineOf(p, shapeBounds, cornerRadii, unused));
        float x = clamp((outlineOf(p - shadowDrop, shapeBounds, cornerRadii, unused) +
                         0.5 * shadowSoftness) / (1.5 * shadowSoftness), 0.0, 1.0);
        return shadow * fadeIn * (1.0 - x) * (1.0 - x);
    }

    half4 main(float2 p) {
        float2 normal;
        float depth = max(-outlineOf(p, shapeBounds, cornerRadii, normal), 0.0);
        const half cover = 1.0;
)";

// Up to 4 pieces of glass melted into one surface, which this draws all of, edge included: where
// two come within meltSize of each other their outlines reach for each other, and from a third of
// it a neck joins them. Each keeps its own rim width, alpha, frost and colors, which blend where
// they meet.
static const char kGlassMeltPart[] = R"(
    uniform float4 shapeBounds[4];
    uniform float4 cornerRadii[4];
    uniform float4 shapeRims;
    uniform float4 shapeAlphas;
    uniform float4 shapeFrosts;
    uniform float4 shapeTintsOverDark[4];
    uniform float4 shapeTintsOverLight[4];
    uniform float shapeCount;
    uniform float meltSize;
    // The falloff and offset of the glass's shadow, in rim widths.
    uniform float shadowSoftnessInRims;
    uniform float2 shadowDropInRims;

    // Signed distance to the melted outline, its slope, which is shorter than 1 where the pieces
    // blend, and what the glass is like there: its rim width, alpha and frost, and colors.
    float surfaceAt(float2 p, out float2 normal, out float3 look, out float4 overDark,
                    out float4 overLight) {
        float d = outlineOf(p, shapeBounds[0], cornerRadii[0], normal);
        look = float3(shapeRims[0], shapeAlphas[0], shapeFrosts[0]);
        overDark = shapeTintsOverDark[0];
        overLight = shapeTintsOverLight[0];
        for (int i = 1; i < 4; i++) {
            if (float(i) < shapeCount) {
                float2 n;
                float other = outlineOf(p, shapeBounds[i], cornerRadii[i], n);
                // A cubic smooth minimum: it only reaches where the two are within meltSize of
                // each other, and its slope and curve are smooth, so what the rim bends has no
                // kinks. The farther piece counts for up to half of the slope.
                float h = max(meltSize - abs(d - other), 0.0) / meltSize;
                float far = 0.5 * h * h;
                float share = d < other ? far : 1.0 - far;
                normal = mix(normal, n, share);
                look = mix(look, float3(shapeRims[i], shapeAlphas[i], shapeFrosts[i]), share);
                overDark = mix(overDark, shapeTintsOverDark[i], share);
                overLight = mix(overLight, shapeTintsOverLight[i], share);
                d = min(d, other) - h * h * h * meltSize / 6.0;
            }
        }
        return d;
    }

    // Signed distance to the melted outline alone.
    float meltedOutline(float2 p) {
        float2 unused;
        float d = outlineOf(p, shapeBounds[0], cornerRadii[0], unused);
        for (int i = 1; i < 4; i++) {
            if (float(i) < shapeCount) {
                float other = outlineOf(p, shapeBounds[i], cornerRadii[i], unused);
                float h = max(meltSize - abs(d - other), 0.0) / meltSize;
                d = min(d, other) - h * h * h * meltSize / 6.0;
            }
        }
        return d;
    }

    float shadowAt(float2 p, float rim) {
        if (shadow <= 0.0) {
            return 0.0;
        }
        float fadeIn = smoothstep(-0.4 * rim, 0.4 * rim, meltedOutline(p));
        float softness = shadowSoftnessInRims * rim;
        float x = clamp((meltedOutline(p - shadowDropInRims * rim) + 0.5 * softness) /
                        (1.5 * softness), 0.0, 1.0);
        return shadow * fadeIn * (1.0 - x) * (1.0 - x);
    }

    half4 main(float2 p) {
        float2 normal;
        float3 look;
        float4 tintOverDark;
        float4 tintOverLight;
        float rawDistance = surfaceAt(p, normal, look, tintOverDark, tintOverLight);
        // The distance as near as the slope tells, so the rim and the line along it keep their
        // width along a neck. The slope itself stays as it is, shorter than 1: it fades to nothing
        // where the rim turns inside a neck, so what the rim bends has no seam there.
        float signedDistance = rawDistance / max(length(normal), 0.3);
        normal.x += 0.0001;
        half cover = half(clamp(0.5 - signedDistance, 0.0, 1.0) * look.y);
        if (cover <= 0.0) {
            return half4(0.0);
        }
        float rimWidth = look.x;
        float frost = look.z;
        float depth = max(-signedDistance, 0.0);
)";

// The rest of the shader's main(), from how deep in the glass p is, what the glass is like there
// and how much of the pixel it covers.
static const char kGlassBodyPart[] = R"(
        // How bright the content behind this spot is, averaged over about a rim width so
        // small details such as text don't make the glass flicker between its two colors.
        float overLight = 0.0;
        if (adaptPoint > 0.0) {
            float reach = 0.75 * rimWidth;
            float3 around = blurredInput.eval(p + float2(reach, 0.0)).rgb +
                            blurredInput.eval(p - float2(reach, 0.0)).rgb +
                            blurredInput.eval(p + float2(0.0, reach)).rgb +
                            blurredInput.eval(p - float2(0.0, reach)).rgb;
            float backdrop = dot(0.25 * around, float3(0.2126, 0.7152, 0.0722));
            overLight = smoothstep(adaptPoint - 0.12, adaptPoint + 0.12, backdrop);
        }
        float4 tint = mix(tintOverDark, tintOverLight, overLight);

        // Past the rim nothing bends, and no shadow, line or glint reaches: all the rim's work
        // below comes to just this there, for a fraction of it.
        float lineWidth = max(1.5, rimWidth * 0.06);
        if (depth >= max(rimWidth, lineWidth)) {
            float3 middle = frost >= 1.0 ? blurredInput.eval(p).rgb
                          : frost <= 0.0 ? originalInput.eval(p).rgb
                          : mix(originalInput.eval(p).rgb, blurredInput.eval(p).rgb, frost);
            return half4(half3(mix(richer(middle), tint.rgb, tint.a)) * cover, cover);
        }

        // 1 on the very edge, 0 where the rim meets the flat middle.
        float edge = 1.0 - clamp(depth / rimWidth, 0.0, 1.0);

        // The rim pulls in the content past the edge, its outer part the most. RR's lens edge
        // pulls what's under the glass out to the edge instead.
        float2 offset = normal * (bend * rimWidth * edge * edge * sqrt(edge)) *
                        (lensEdge > 0.0 ? -1.0 : 1.0);
        // Frost goes right up to the edge, where the bend only moves the blur around. RR's lens
        // edge keeps the outer part of the rim clear, so its bend shows around the frost.
        float3 color = blurredInput.eval(p + offset).rgb;
        float clearness = max(1.0 - frost, lensEdge * 0.8 * smoothstep(0.35, 1.0, edge));
        if (clearness > 0.0) {
            // Blue bends further than red.
            float3 sharp = float3(originalInput.eval(p + offset * (1.0 - 0.5 * dispersion)).r,
                                  originalInput.eval(p + offset).g,
                                  originalInput.eval(p + offset * (1.0 + 0.5 * dispersion)).b);
            color = mix(color, sharp, clearness);
        }
        color = richer(color);
        color *= 1.0 - shadowAt(p + offset, rimWidth);
        color = mix(color, tint.rgb, tint.a);

        // Light: a thin line all along the edge, brightest facing the light and
        // brighter on the far side than in between, tinted with a rainbow that
        // turns around the outline, over a soft glow inside the lit part of the rim.
        float facing = dot(normal, lightDir);
        float catchLight = 0.3 + 0.7 * pow(max(facing, 0.0), 2.0) +
                           0.4 * pow(max(-facing, 0.0), 2.0);
        float line = 1.0 - smoothstep(0.0, lineWidth, depth);
        float angle = atan(normal.y, normal.x);
        // Whole turns only, so the colors meet up again all the way around. They turn with
        // the light.
        float3 rainbow = 0.5 + 0.5 * cos(5.76 + lightTurn - angle + float3(0.0, 2.094, 4.189));
        float3 lineColor = mix(float3(1.0), rainbow, 0.8 * dispersion);
        // A wet glint where the rim turns straight toward the light.
        float glint = pow(max(facing, 0.0), 12.0) * smoothstep(0.35, 1.0, edge);
        // Over light content a bright line can't be seen; it gives way to a thin dark edge.
        float3 light = rimLight * (1.0 - 0.8 * overLight) *
                       (line * catchLight * lineColor +
                        0.25 * edge * edge * edge * max(facing, 0.0) + 0.6 * glint);
        // Added like light on a screen: bright parts don't clip.
        color = 1.0 - (1.0 - color) * (1.0 - min(light, 1.0));
        color *= 1.0 - 0.35 * rimLight * overLight * line;
        return half4(half3(color) * cover, cover);
    }
)";

static SkString makeSource(std::initializer_list<const char*> parts) {
    SkString source;
    for (const char* part : parts) {
        source.append(part);
    }
    return source;
}

// Most glass pieces one shadow can go around; matches the arrays below.
static constexpr size_t kGlassShadowMaxShapes = 8;

// Catalyxx glass shadow: one soft shadow around all the glass of a layer,
// falling a little lower than it. Glass pieces that touch or overlap share
// one shadow, and nothing is drawn under the glass itself.
static const char kGlassShadowCommonPart[] = R"(
    uniform float4 shapeBounds[8];
    // Corner radii: top-left, top-right, bottom-right, bottom-left.
    uniform float4 cornerRadii[8];
    uniform float shapeCount;
    // Size of the shadow's soft falloff.
    uniform float softness;
    // How far the shadow falls.
    uniform float2 drop;
    // Darkness where the shadow starts, 0 to 1.
    uniform float strength;

    float outline(float2 p, float4 bounds, float4 radii) {
        float2 center = (bounds.xy + bounds.zw) * 0.5;
        float2 halfSize = (bounds.zw - bounds.xy) * 0.5;
        float2 q = p - center;
        float radius = q.y < 0.0 ? (q.x < 0.0 ? radii.x : radii.y)
                                 : (q.x < 0.0 ? radii.w : radii.z);
        float2 d = abs(q) - halfSize + radius;
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
    }
)";

// Pieces of glass on their own: the shadow goes around all of them.
static const char kGlassShadowOnePiecePart[] = R"(
    float united(float d, float other, int i) {
        return min(d, other);
    }
)";

// Where glass melts together, its first meltCount pieces, the shadow follows the melted outline.
static const char kGlassShadowMeltPart[] = R"(
    uniform float meltCount;
    uniform float meltSize;

    float united(float d, float other, int i) {
        if (float(i) < meltCount) {
            float h = max(meltSize - abs(d - other), 0.0) / meltSize;
            return min(d, other) - h * h * h * meltSize / 6.0;
        }
        return min(d, other);
    }
)";

static const char kGlassShadowBodyPart[] = R"(
    half4 main(float2 p) {
        float here = 1.0e9;
        for (int i = 0; i < 8; i++) {
            if (float(i) < shapeCount) {
                here = united(here, outline(p, shapeBounds[i], cornerRadii[i]), i);
            }
        }
        // Nothing is drawn under the glass itself, so no need to look further there
        if (here < 0.0) {
            return half4(0.0);
        }
        float dropped = 1.0e9;
        for (int i = 0; i < 8; i++) {
            if (float(i) < shapeCount) {
                dropped = united(dropped, outline(p - drop, shapeBounds[i], cornerRadii[i]), i);
            }
        }
        float x = clamp((dropped + 0.5 * softness) / (1.5 * softness), 0.0, 1.0);
        return half4(0.0, 0.0, 0.0, half(strength * (1.0 - x) * (1.0 - x)));
    }
)";

// Joined glass sides pushed this far out show no rim.
static constexpr float kGlassFarAway = 10000.0f;
// The glass's shadow: its falloff and how far it falls, in rim widths.
static constexpr float kGlassShadowSoftness = 0.9f;
static constexpr float kGlassShadowDrop = 0.25f;
// Unit vector toward the usual light (up and a little to the left), y pointing down.
static constexpr SkV2 kGlassLightDir = {-0.447214f, -0.894427f};
// The sides in BlurRegion::glassJoinedSides, which also holds BlurRegion::kGlassFixedLight.
static constexpr uint32_t kGlassSides = BlurRegion::kGlassSideLeft | BlurRegion::kGlassSideTop |
        BlurRegion::kGlassSideRight | BlurRegion::kGlassSideBottom;
// Where two pieces of glass that are to melt line up within this, they're one body of glass.
static constexpr float kGlassJoinTolerance = 1.5f;

namespace {

// Where the light on the glass comes from (DisplaySettings::glassLight), as a unit vector,
// and how far it has turned from the usual light. With the light, the glass's shadow and the
// colors on its rim turn too.
struct GlassLight {
    SkV2 direction;
    // Radians
    float turn;

    // The shadow falls this far straight down when the light is the usual one.
    SkV2 shadowDrop(float drop) const {
        return {-drop * std::sin(turn), drop * std::cos(turn)};
    }
};

GlassLight getGlassLight(const vec2& light) {
    const float length = std::hypot(light.x, light.y);
    if (!std::isfinite(length) || length < 0.01f) {
        return {kGlassLightDir, 0.0f};
    }
    const SkV2 direction = {light.x / length, light.y / length};
    return {direction,
            std::atan2(direction.y, direction.x) - std::atan2(kGlassLightDir.y, kGlassLightDir.x)};
}

// A glass outline as its shaders take it.
struct GlassShape {
    // Left, top, right, bottom.
    SkV4 bounds;
    // Top-left, top-right, bottom-right, bottom-left, as circles.
    SkV4 radii;
};

GlassShape getGlassShape(const SkRRect& outline, uint32_t joinedSides, bool pushJoinedSides) {
    const SkRect& rect = outline.rect();
    const auto radius = [&outline](SkRRect::Corner corner) {
        const SkVector radii = outline.radii(corner);
        return std::min(radii.x(), radii.y());
    };
    GlassShape shape{{rect.left(), rect.top(), rect.right(), rect.bottom()},
                     {radius(SkRRect::kUpperLeft_Corner), radius(SkRRect::kUpperRight_Corner),
                      radius(SkRRect::kLowerRight_Corner), radius(SkRRect::kLowerLeft_Corner)}};
    if (pushJoinedSides) {
        if (joinedSides & BlurRegion::kGlassSideLeft) shape.bounds.x -= kGlassFarAway;
        if (joinedSides & BlurRegion::kGlassSideTop) shape.bounds.y -= kGlassFarAway;
        if (joinedSides & BlurRegion::kGlassSideRight) shape.bounds.z += kGlassFarAway;
        if (joinedSides & BlurRegion::kGlassSideBottom) shape.bounds.w += kGlassFarAway;
    }
    return shape;
}

// The rim is no wider than half the glass, across each direction it ends in.
float getGlassRimWidth(const BlurRegion& region, const SkRect& rect) {
    float rimWidth = region.glassRimWidth;
    const uint32_t joined = region.glassJoinedSides;
    if (!(joined & (BlurRegion::kGlassSideLeft | BlurRegion::kGlassSideRight))) {
        rimWidth = std::min(rimWidth, rect.width() / 2);
    }
    if (!(joined & (BlurRegion::kGlassSideTop | BlurRegion::kGlassSideBottom))) {
        rimWidth = std::min(rimWidth, rect.height() / 2);
    }
    return std::max(rimWidth, 1.0f);
}

} // namespace

const SkString kEffectSource_BlurFilter_MixEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float mixFactor;

    half4 main(float2 xy) {
        return half4(mix(originalInput.eval(xy), blurredInput.eval(xy), mixFactor)).rgb1;
    }
)");

static SkMatrix getShaderTransform(const SkCanvas* canvas, const SkRect& blurRect,
                                   const float scale, const float zoomScale) {
    // 1. Apply the blur shader matrix, which scales up the blurred surface to its real size
    auto matrix = SkMatrix::Scale(scale, scale);
    // 2. Since the blurred surface has the size of the layer, we align it with the
    // top left corner of the layer position.
    matrix.postConcat(SkMatrix::Translate(blurRect.fLeft, blurRect.fTop));
    // 3. Apply the "zoom" effect as an extra scale + translate around the center of the blur.
    if (zoomScale != 1.0f) {
        matrix.postScale(zoomScale, zoomScale);
        matrix.postTranslate(
                blurRect.width() * (1 - zoomScale) / 2.0f,
                blurRect.height() * (1 - zoomScale) / 2.0f);
    }
    // 4. Finally, apply the inverse canvas matrix. The snapshot made in the BlurFilter is in the
    // original surface orientation. The inverse matrix has to be applied to align the blur
    // surface with the current orientation/position of the canvas.
    SkMatrix drawInverse;
    if (canvas != nullptr && canvas->getTotalMatrix().invert(&drawInverse)) {
        matrix.postConcat(drawInverse);
    }
    return matrix;
}

BlurFilter::BlurFilter(RuntimeEffectManager& effectManager, const float maxCrossFadeRadius)
      : mMaxCrossFadeRadius(maxCrossFadeRadius),
        mMixEffect(effectManager.mKnownEffects[kBlurFilter_MixEffect]) {
    SkRuntimeEffect::Options options;
    options.fName = "RE_CatalyxxGlass";
    auto [glassEffect, glassError] = SkRuntimeEffect::MakeForShader(
            makeSource({kGlassCommonPart, kGlassOnePiecePart, kGlassBodyPart}), options);
    options.fName = "RE_CatalyxxGlassShadow";
    auto [shadowEffect, shadowError] = SkRuntimeEffect::MakeForShader(
            makeSource({kGlassShadowCommonPart, kGlassShadowOnePiecePart, kGlassShadowBodyPart}),
            options);
    if (!glassEffect || !shadowEffect) {
        ALOGE("Catalyxx glass is drawn as plain blur, its effects failed to build: %s %s",
              glassError.c_str(), shadowError.c_str());
        return;
    }
    mGlassEffect = std::move(glassEffect);
    mGlassShadowEffect = std::move(shadowEffect);
}

float BlurFilter::getMaxCrossFadeRadius() const {
    return mMaxCrossFadeRadius;
}

void BlurFilter::drawBlurRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                const uint32_t blurRadius, const float zoomScale,
                                const float blurAlpha,
                                const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                                sk_sp<SkImage> input, SkIPoint inputOrigin) {
    SFTRACE_CALL();

    SkPaint paint;
    paint.setAlphaf(blurAlpha);

    auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, zoomScale);

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    if (blurRadius < mMaxCrossFadeRadius) {
        LOG_THREAD_STATE_AND_CRASH_IF(!input);

        // For sampling Skia's API expects the inverse of what logically seems appropriate. In this
        // case you might expect the matrix to simply be the canvas matrix.
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        if (zoomScale != 1.0f) {
            inputMatrix.preTranslate(
                    blurRect.width() * (1 - zoomScale) / 2.0f,
                    blurRect.height() * (1 - zoomScale) / 2.0f);
            inputMatrix.preScale(zoomScale, zoomScale);
        }
        inputMatrix.preTranslate(inputOrigin.fX, inputOrigin.fY);

        SkRuntimeShaderBuilder blurBuilder(mMixEffect);
        blurBuilder.child("blurredInput") = blurShader;
        blurBuilder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);

        blurBuilder.uniform("mixFactor") = blurRadius / mMaxCrossFadeRadius;

        paint.setShader(blurBuilder.makeShader());
    } else {
        paint.setShader(blurShader);
    }

    if (effectRegion.isRect()) {
        if (blurAlpha == 1.0f) {
            paint.setBlendMode(SkBlendMode::kSrc);
        }
        canvas->drawRect(effectRegion.rect(), paint);
    } else {
        paint.setAntiAlias(true);
        canvas->drawRRect(effectRegion, paint);
    }
}

SkRRect BlurFilter::getGlassRRect(const BlurRegion& region) {
    const uint32_t joined = region.glassJoinedSides;
    const bool left = joined & BlurRegion::kGlassSideLeft;
    const bool top = joined & BlurRegion::kGlassSideTop;
    const bool right = joined & BlurRegion::kGlassSideRight;
    const bool bottom = joined & BlurRegion::kGlassSideBottom;
    const auto corner = [](bool square, float x, float y) {
        return square ? SkVector::Make(0, 0) : SkVector::Make(x, y);
    };
    const SkVector radii[4] = {
            corner(left || top, region.cornerRadiusTLX, region.cornerRadiusTLY),
            corner(right || top, region.cornerRadiusTRX, region.cornerRadiusTRY),
            corner(right || bottom, region.cornerRadiusBRX, region.cornerRadiusBRY),
            corner(left || bottom, region.cornerRadiusBLX, region.cornerRadiusBLY)};
    SkRRect outline;
    outline.setRectRadii(SkRect::MakeLTRB(region.left, region.top, region.right, region.bottom),
                         radii);
    return outline;
}

bool BlurFilter::buildGlassMeltEffects() {
    if (mGlassMeltEffect && mGlassMeltShadowEffect) {
        return true;
    }
    if (mGlassMeltFailed) {
        return false;
    }
    SkRuntimeEffect::Options options;
    options.fName = "RE_CatalyxxGlassMelt";
    auto [glassEffect, glassError] = SkRuntimeEffect::MakeForShader(
            makeSource({kGlassCommonPart, kGlassMeltPart, kGlassBodyPart}), options);
    options.fName = "RE_CatalyxxGlassMeltShadow";
    auto [shadowEffect, shadowError] = SkRuntimeEffect::MakeForShader(
            makeSource({kGlassShadowCommonPart, kGlassShadowMeltPart, kGlassShadowBodyPart}),
            options);
    if (!glassEffect || !shadowEffect) {
        ALOGE("Catalyxx glass doesn't melt, its effects failed to build: %s %s",
              glassError.c_str(), shadowError.c_str());
        mGlassMeltFailed = true;
        return false;
    }
    mGlassMeltEffect = std::move(glassEffect);
    mGlassMeltShadowEffect = std::move(shadowEffect);
    return true;
}

float BlurFilter::getGlassFrost(const BlurRegion& region) const {
    // Like plain blur, small radii fade toward the sharp content.
    float frost = region.glassFrost;
    if (region.blurRadius < mMaxCrossFadeRadius) {
        frost *= region.blurRadius / mMaxCrossFadeRadius;
    }
    return frost;
}

float BlurFilter::getGlassLensEdge(bool lensEdge, uint32_t layerBlurRadius) const {
    if (!lensEdge) {
        return 0.0f;
    }
    if (layerBlurRadius == 0) {
        return 1.0f;
    }
    if (mMaxCrossFadeRadius <= 0.0f) {
        return 0.0f;
    }
    return 1.0f - std::min(layerBlurRadius / mMaxCrossFadeRadius, 1.0f);
}

bool BlurFilter::glassReadsSharp(const BlurRegion& region, uint32_t layerBlurRadius) const {
    return getGlassFrost(region) < 1.0f ||
            getGlassLensEdge(region.glassJoinedSides & BlurRegion::kGlassLensEdge,
                             layerBlurRadius) > 0.0f;
}

BlurFilter::GlassMelts BlurFilter::findGlassMelts(const std::vector<BlurRegion>& regions) {
    GlassMelts melts;
    // Nearly always, nothing melts
    if (!canDrawGlass() ||
        std::none_of(regions.begin(), regions.end(),
                     [](const BlurRegion& region) {
                         return region.isGlass() && region.blurRadius > 0 && region.glassMelt > 0.0f;
                     }) ||
        !buildGlassMeltEffects()) {
        return melts;
    }

    // A body of glass: a region, or pieces joined to each other that make one rounded rect.
    struct Body {
        SkRect rect;
        SkV4 radii;
        uint32_t joinedSides;
        float melt;
        bool movingLight;
        uint64_t regions;
        const BlurRegion* region;
    };
    std::vector<Body> bodies;
    for (size_t i = 0; i < regions.size() && i < sizeof(uint64_t) * 8; i++) {
        const BlurRegion& region = regions[i];
        if (!region.isGlass() || region.blurRadius == 0) {
            continue;
        }
        const SkRRect outline = getGlassRRect(region);
        bodies.push_back({outline.rect(),
                          getGlassShape(outline, 0, false).radii,
                          region.glassJoinedSides & kGlassSides,
                          region.glassMelt,
                          region.hasMovingLight(),
                          uint64_t{1} << i,
                          &region});
    }

    // Pieces carrying on into each other, such as the volume panel's ringer drawer and its slider:
    // the piece above or to the left comes first.
    const auto carriesOn = [](const Body& first, const Body& second) {
        const SkRect& a = first.rect;
        const SkRect& b = second.rect;
        if ((first.joinedSides & BlurRegion::kGlassSideBottom) &&
            (second.joinedSides & BlurRegion::kGlassSideTop)) {
            return std::abs(a.left() - b.left()) <= kGlassJoinTolerance &&
                    std::abs(a.right() - b.right()) <= kGlassJoinTolerance &&
                    a.top() <= b.top() && b.top() <= a.bottom() + kGlassJoinTolerance;
        }
        if ((first.joinedSides & BlurRegion::kGlassSideRight) &&
            (second.joinedSides & BlurRegion::kGlassSideLeft)) {
            return std::abs(a.top() - b.top()) <= kGlassJoinTolerance &&
                    std::abs(a.bottom() - b.bottom()) <= kGlassJoinTolerance &&
                    a.left() <= b.left() && b.left() <= a.right() + kGlassJoinTolerance;
        }
        return false;
    };
    for (size_t i = 0; i < bodies.size(); i++) {
        for (size_t j = 0; j < bodies.size(); j++) {
            if (i == j || !carriesOn(bodies[i], bodies[j])) {
                continue;
            }
            Body& first = bodies[i];
            const Body& second = bodies[j];
            const bool vertical = first.joinedSides & BlurRegion::kGlassSideBottom;
            // The outer corners are each piece's own
            first.radii = vertical ? SkV4{first.radii.x, first.radii.y, second.radii.z, second.radii.w}
                                   : SkV4{first.radii.x, second.radii.y, second.radii.z, first.radii.w};
            first.joinedSides = vertical
                    ? (first.joinedSides & ~BlurRegion::kGlassSideBottom) |
                            (second.joinedSides & ~BlurRegion::kGlassSideTop)
                    : (first.joinedSides & ~BlurRegion::kGlassSideRight) |
                            (second.joinedSides & ~BlurRegion::kGlassSideLeft);
            first.rect.join(second.rect);
            first.melt = std::max(first.melt, second.melt);
            first.movingLight = first.movingLight || second.movingLight;
            first.regions |= second.regions;
            bodies.erase(bodies.begin() + j);
            // Whatever comes next starts over, with this piece as it is now
            i = static_cast<size_t>(-1);
            break;
        }
    }

    // Which bodies reach each other, in groups. A piece still carrying on into something else
    // keeps its own way of drawing that side.
    std::vector<size_t> groupOf(bodies.size());
    for (size_t i = 0; i < bodies.size(); i++) {
        groupOf[i] = i;
    }
    for (size_t i = 0; i < bodies.size(); i++) {
        for (size_t j = i + 1; j < bodies.size(); j++) {
            const Body& a = bodies[i];
            const Body& b = bodies[j];
            const float size = std::max(a.melt, b.melt);
            if (size <= 0.0f || a.joinedSides != 0 || b.joinedSides != 0) {
                continue;
            }
            const float gapX = std::max({a.rect.left() - b.rect.right(),
                                         b.rect.left() - a.rect.right(), 0.0f});
            const float gapY = std::max({a.rect.top() - b.rect.bottom(),
                                         b.rect.top() - a.rect.bottom(), 0.0f});
            if (std::hypot(gapX, gapY) >= size) {
                continue;
            }
            const size_t from = groupOf[j];
            const size_t to = groupOf[i];
            for (size_t& group : groupOf) {
                if (group == from) {
                    group = to;
                }
            }
        }
    }

    for (size_t group = 0; group < bodies.size(); group++) {
        if (std::count(groupOf.begin(), groupOf.end(), group) < 2) {
            continue;
        }
        GlassMelt melt;
        melt.area = SkRect::MakeEmpty();
        for (size_t i = 0; i < bodies.size() && melt.count < GlassMelt::kMaxShapes; i++) {
            if (groupOf[i] != group) {
                continue;
            }
            const Body& body = bodies[i];
            const BlurRegion& region = *body.region;
            const size_t shape = melt.count++;
            melt.bounds[shape] = {body.rect.left(), body.rect.top(), body.rect.right(),
                                  body.rect.bottom()};
            melt.radii[shape] = body.radii;
            // The rim is no wider than half the glass.
            melt.rims[shape] = std::max(std::min({region.glassRimWidth, body.rect.width() / 2,
                                                  body.rect.height() / 2}),
                                        1.0f);
            melt.alphas[shape] = region.alpha;
            melt.frosts[shape] = getGlassFrost(region);
            const auto& overDark = region.glassTintOverDark;
            const auto& overLight = region.glassTintOverLight;
            melt.tintsOverDark[shape] = {overDark[0], overDark[1], overDark[2], overDark[3]};
            melt.tintsOverLight[shape] = {overLight[0], overLight[1], overLight[2], overLight[3]};
            // What isn't each piece's own is the first's
            if (shape == 0) {
                melt.bend = region.glassBend;
                melt.rimLight = region.glassRimLight;
                melt.dispersion = region.glassDispersion;
                melt.adaptPoint = region.glassAdaptPoint;
                melt.lensEdge = region.glassJoinedSides & BlurRegion::kGlassLensEdge;
            }
            melt.shadow = std::max(melt.shadow, region.glassShadow);
            melt.size = std::max(melt.size, body.melt);
            // One light for all of it. What keeps a fixed light, such as the dock, stays where it
            // is while things melt into it, and its light mustn't jump for it.
            melt.movingLight = (shape == 0 || melt.movingLight) && body.movingLight;
            melt.regions |= body.regions;
            melt.area.join(body.rect);
        }
        const float bulge = melt.size * kGlassMeltBulge + 2.0f;
        melt.area.outset(bulge, bulge);
        melts.melted |= melt.regions;
        melts.melts.push_back(melt);
    }
    return melts;
}

void BlurFilter::drawGlassShadows(SkCanvas* canvas, const std::vector<BlurRegion>& regions,
                                  const GlassMelts& melts, const vec2& light) {
    if (!canDrawGlass()) {
        return;
    }
    SFTRACE_CALL();

    std::array<SkV4, kGlassShadowMaxShapes> bounds{};
    std::array<SkV4, kGlassShadowMaxShapes> radii{};
    size_t count = 0;
    float strength = 0.0f;
    float rimWidth = 0.0f;
    // The layer's one shadow falls in the usual light while all its glass has a fixed light.
    bool movingLight = false;
    SkRect area = SkRect::MakeEmpty();

    // What melts comes first, for the shadow to follow its melted outline.
    float meltSize = 0.0f;
    for (const GlassMelt& melt : melts.melts) {
        if (melt.shadow <= 0.0f) {
            continue;
        }
        for (size_t shape = 0; shape < melt.count && count < kGlassShadowMaxShapes; shape++) {
            bounds[count] = melt.bounds[shape];
            radii[count] = melt.radii[shape];
            count++;
            strength = std::max(strength, melt.shadow * melt.alphas[shape]);
            rimWidth = std::max(rimWidth, melt.rims[shape]);
        }
        movingLight = movingLight || melt.movingLight;
        meltSize = std::max(meltSize, melt.size);
        area.join(melt.area);
    }
    const size_t meltCount = count;

    for (size_t i = 0; i < regions.size(); i++) {
        const BlurRegion& region = regions[i];
        // Regions without a blur radius aren't drawn, so they cast no shadow either.
        if (!region.isGlass() || region.blurRadius == 0 || region.glassShadow <= 0.0f ||
            melts.contains(i)) {
            continue;
        }
        if (count == kGlassShadowMaxShapes) {
            break;
        }
        const SkRRect outline = getGlassRRect(region);
        const GlassShape shape = getGlassShape(outline, region.glassJoinedSides, false);
        bounds[count] = shape.bounds;
        radii[count] = shape.radii;
        count++;
        movingLight = movingLight || region.hasMovingLight();
        strength = std::max(strength, region.glassShadow * region.alpha);
        rimWidth = std::max(rimWidth, getGlassRimWidth(region, outline.rect()));
        area.join(outline.rect());
    }
    if (count == 0 || strength <= 0.0f) {
        return;
    }
    const float softness = rimWidth * kGlassShadowSoftness;
    const float drop = rimWidth * kGlassShadowDrop;
    area.outset(1.5f * softness + drop, 1.5f * softness + drop);

    SkRuntimeShaderBuilder builder(meltCount > 0 ? mGlassMeltShadowEffect : mGlassShadowEffect);
    builder.uniform("shapeBounds").set(bounds.data(), kGlassShadowMaxShapes);
    builder.uniform("cornerRadii").set(radii.data(), kGlassShadowMaxShapes);
    builder.uniform("shapeCount") = static_cast<float>(count);
    builder.uniform("softness") = softness;
    builder.uniform("drop") = getGlassLight(movingLight ? light : vec2(0.f, 0.f)).shadowDrop(drop);
    builder.uniform("strength") = strength;
    if (meltCount > 0) {
        builder.uniform("meltCount") = static_cast<float>(meltCount);
        builder.uniform("meltSize") = meltSize;
    }
    // Without its shader the paint would fill the whole area black.
    sk_sp<SkShader> shader = builder.makeShader();
    if (!shader) {
        ALOGE("Catalyxx glass shadow could not be made");
        return;
    }

    SkPaint paint;
    paint.setShader(std::move(shader));
    canvas->drawRect(area, paint);
}

void BlurFilter::drawGlassRegion(SkCanvas* canvas, const BlurRegion& region, const vec2& light,
                                 const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                                 sk_sp<SkImage> input, SkIPoint inputOrigin,
                                 uint32_t layerBlurRadius) {
    if (!canDrawGlass() || (!input && glassReadsSharp(region, layerBlurRadius))) {
        drawBlurRegion(canvas, getGlassRRect(region), region.blurRadius, 1.0f, region.alpha,
                       blurRect, blurredImage, input, inputOrigin);
        return;
    }
    SFTRACE_CALL();

    const SkRRect outline = getGlassRRect(region);
    const GlassShape shape = getGlassShape(outline, region.glassJoinedSides, true);
    const float rimWidth = getGlassRimWidth(region, outline.rect());

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, 1.0f);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    SkRuntimeShaderBuilder builder(mGlassEffect);
    builder.child("blurredInput") = blurShader;
    if (input) {
        // For sampling Skia's API expects the inverse of what logically seems appropriate, as in
        // drawBlurRegion().
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        inputMatrix.preTranslate(inputOrigin.fX, inputOrigin.fY);
        builder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);
    } else {
        // Glass that only shows its blur never reads the sharp content
        builder.child("originalInput") = blurShader;
    }
    builder.uniform("shapeBounds") = shape.bounds;
    builder.uniform("cornerRadii") = shape.radii;
    builder.uniform("rimWidth") = rimWidth;
    builder.uniform("bend") = region.glassBend;
    builder.uniform("frost") = getGlassFrost(region);
    builder.uniform("rimLight") = region.glassRimLight;
    builder.uniform("dispersion") = region.glassDispersion;
    builder.uniform("shadow") = region.glassShadow;
    // Glass with a fixed light keeps the usual one, (0, 0)
    const GlassLight glassLight = getGlassLight(region.hasMovingLight() ? light : vec2(0.f, 0.f));
    builder.uniform("shadowSoftness") = rimWidth * kGlassShadowSoftness;
    builder.uniform("shadowDrop") = glassLight.shadowDrop(rimWidth * kGlassShadowDrop);
    builder.uniform("lightDir") = glassLight.direction;
    builder.uniform("lightTurn") = glassLight.turn;
    const auto& overDark = region.glassTintOverDark;
    const auto& overLight = region.glassTintOverLight;
    builder.uniform("tintOverDark") = SkV4{overDark[0], overDark[1], overDark[2], overDark[3]};
    builder.uniform("tintOverLight") = SkV4{overLight[0], overLight[1], overLight[2], overLight[3]};
    builder.uniform("adaptPoint") = region.glassAdaptPoint;
    const bool lensEdge = region.glassJoinedSides & BlurRegion::kGlassLensEdge;
    builder.uniform("lensEdge") = getGlassLensEdge(lensEdge, layerBlurRadius);
    builder.uniform("rich") = lensEdge ? 1.0f : 0.0f;
    // Without its shader the paint would fill the glass black, so draw plain blur instead.
    sk_sp<SkShader> shader = builder.makeShader();
    if (!shader) {
        ALOGE("Catalyxx glass could not be made, drawing plain blur");
        drawBlurRegion(canvas, outline, region.blurRadius, 1.0f, region.alpha, blurRect,
                       blurredImage, input, inputOrigin);
        return;
    }

    SkPaint paint;
    paint.setAlphaf(region.alpha);
    paint.setAntiAlias(true);
    paint.setShader(std::move(shader));
    canvas->drawRRect(outline, paint);
}

void BlurFilter::drawGlassMelt(SkCanvas* canvas, const GlassMelt& melt, const vec2& light,
                               const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                               sk_sp<SkImage> input, SkIPoint inputOrigin,
                               uint32_t layerBlurRadius) {
    // Glass melts only once its effects are built
    if (!mGlassMeltEffect) {
        return;
    }
    const float lensEdge = getGlassLensEdge(melt.lensEdge, layerBlurRadius);
    if (!input) {
        // Without the sharp input only glass that shows nothing but its blur can be drawn
        bool readsSharp = lensEdge > 0.0f;
        for (size_t shape = 0; shape < melt.count; shape++) {
            readsSharp = readsSharp || melt.frosts[shape] < 1.0f;
        }
        if (readsSharp) {
            return;
        }
    }
    SFTRACE_CALL();

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, 1.0f);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    SkRuntimeShaderBuilder builder(mGlassMeltEffect);
    builder.child("blurredInput") = blurShader;
    if (input) {
        // For sampling Skia's API expects the inverse of what logically seems appropriate, as in
        // drawBlurRegion().
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        inputMatrix.preTranslate(inputOrigin.fX, inputOrigin.fY);
        builder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);
    } else {
        builder.child("originalInput") = blurShader;
    }
    builder.uniform("shapeBounds").set(melt.bounds.data(), GlassMelt::kMaxShapes);
    builder.uniform("cornerRadii").set(melt.radii.data(), GlassMelt::kMaxShapes);
    builder.uniform("shapeRims").set(melt.rims.data(), GlassMelt::kMaxShapes);
    builder.uniform("shapeAlphas").set(melt.alphas.data(), GlassMelt::kMaxShapes);
    builder.uniform("shapeFrosts").set(melt.frosts.data(), GlassMelt::kMaxShapes);
    builder.uniform("shapeTintsOverDark").set(melt.tintsOverDark.data(), GlassMelt::kMaxShapes);
    builder.uniform("shapeTintsOverLight").set(melt.tintsOverLight.data(), GlassMelt::kMaxShapes);
    builder.uniform("shapeCount") = static_cast<float>(melt.count);
    builder.uniform("meltSize") = melt.size;
    builder.uniform("bend") = melt.bend;
    builder.uniform("rimLight") = melt.rimLight;
    builder.uniform("dispersion") = melt.dispersion;
    builder.uniform("shadow") = melt.shadow;
    // Glass with a fixed light keeps the usual one, (0, 0)
    const GlassLight glassLight = getGlassLight(melt.movingLight ? light : vec2(0.f, 0.f));
    builder.uniform("shadowSoftnessInRims") = kGlassShadowSoftness;
    builder.uniform("shadowDropInRims") = glassLight.shadowDrop(kGlassShadowDrop);
    builder.uniform("lightDir") = glassLight.direction;
    builder.uniform("lightTurn") = glassLight.turn;
    builder.uniform("adaptPoint") = melt.adaptPoint;
    builder.uniform("lensEdge") = lensEdge;
    builder.uniform("rich") = melt.lensEdge ? 1.0f : 0.0f;
    sk_sp<SkShader> shader = builder.makeShader();
    if (!shader) {
        ALOGE("Catalyxx melted glass could not be made");
        return;
    }

    // It draws its own edge, with what it covers of each pixel
    SkPaint paint;
    paint.setShader(std::move(shader));
    canvas->drawRect(melt.area, paint);
}

} // namespace skia
} // namespace renderengine
} // namespace android
