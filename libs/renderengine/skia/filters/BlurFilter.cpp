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

namespace android {
namespace renderengine {
namespace skia {

// Catalyxx glass: a blur region drawn as a slab of glass with a rounded rim.
// The rim bends in what lies just past the edge (the glass's own shadow too),
// splits it into colors and catches the light; the flat middle shows the
// content behind it, clear or frosted. The glass has its own color, which
// turns from one tint to another where the content behind is light, so what
// sits on the glass stays readable.
static const SkString kGlassEffectSource(R"(
    uniform shader blurredInput;
    uniform shader originalInput;

    // Glass outline in local coordinates; joined sides are pushed far out.
    uniform float4 shapeBounds;
    // Corner radii: top-left, top-right, bottom-right, bottom-left.
    uniform float4 cornerRadii;
    // Width of the curved rim.
    uniform float rimWidth;
    // How far past the edge the rim reaches, in rim widths.
    uniform float bend;
    // Blurred share of the flat middle, 0 to 1.
    uniform float frost;
    // Strength of the light on the rim, 0 to 1.
    uniform float rimLight;
    // Color split of the bent light, 0 to 1.
    uniform float dispersion;
    // Darkness of the glass's shadow, 0 to 1, and its falloff and offset.
    uniform float shadow;
    uniform float shadowSoftness;
    uniform float2 shadowDrop;
    // Unit vector toward the light, y pointing down, and how far it has turned from its usual
    // place at the top left, in radians.
    uniform float2 lightDir;
    uniform float lightTurn;
    // The glass's own color over dark and over light content (straight alpha = how much it
    // covers), and the brightness of what's behind (0 to 1) around which it turns from one to
    // the other; 0 keeps tintOverDark everywhere.
    uniform float4 tintOverDark;
    uniform float4 tintOverLight;
    uniform float adaptPoint;

    // Signed distance to the outline (negative inside) and the outward normal.
    float outline(float2 p, out float2 normal) {
        float2 center = (shapeBounds.xy + shapeBounds.zw) * 0.5;
        float2 halfSize = (shapeBounds.zw - shapeBounds.xy) * 0.5;
        float2 q = p - center;
        float2 side = float2(q.x < 0.0 ? -1.0 : 1.0, q.y < 0.0 ? -1.0 : 1.0);
        float radius = q.y < 0.0 ? (q.x < 0.0 ? cornerRadii.x : cornerRadii.y)
                                 : (q.x < 0.0 ? cornerRadii.w : cornerRadii.z);
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

    // Darkness of the glass's shadow on the content at p. It fades in across the
    // edge, so the rim's bent view of it has no hard line.
    float shadowAt(float2 p) {
        if (shadow <= 0.0) {
            return 0.0;
        }
        float2 unused;
        float fadeIn = smoothstep(-0.4 * rimWidth, 0.4 * rimWidth, outline(p, unused));
        float x = clamp((outline(p - shadowDrop, unused) + 0.5 * shadowSoftness) /
                        (1.5 * shadowSoftness), 0.0, 1.0);
        return shadow * fadeIn * (1.0 - x) * (1.0 - x);
    }

    half4 main(float2 p) {
        float2 normal;
        float depth = max(-outline(p, normal), 0.0);

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
            return half4(half3(mix(middle, tint.rgb, tint.a)), 1.0);
        }

        // 1 on the very edge, 0 where the rim meets the flat middle.
        float edge = 1.0 - clamp(depth / rimWidth, 0.0, 1.0);

        // The rim pulls in the content past the edge, its outer part the most.
        float2 offset = normal * (bend * rimWidth * edge * edge * sqrt(edge));
        // Blue bends further than red.
        float3 sharp = float3(originalInput.eval(p + offset * (1.0 - 0.5 * dispersion)).r,
                              originalInput.eval(p + offset).g,
                              originalInput.eval(p + offset * (1.0 + 0.5 * dispersion)).b);
        float3 blurred = blurredInput.eval(p + offset).rgb;
        // The middle is as frosted as asked; the rim stays clear so its bend shows.
        float3 color = mix(sharp, blurred, frost * smoothstep(0.0, 1.0, 1.0 - edge));
        color *= 1.0 - shadowAt(p + offset);
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
        return half4(half3(color), 1.0);
    }
)");

// Most glass pieces one shadow can go around; matches the arrays below.
static constexpr size_t kGlassShadowMaxShapes = 8;

// Catalyxx glass shadow: one soft shadow around all the glass of a layer,
// falling a little lower than it. Glass pieces that touch or overlap share
// one shadow, and nothing is drawn under the glass itself.
static const SkString kGlassShadowEffectSource(R"(
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

    half4 main(float2 p) {
        float here = 1.0e9;
        for (int i = 0; i < 8; i++) {
            if (float(i) < shapeCount) {
                here = min(here, outline(p, shapeBounds[i], cornerRadii[i]));
            }
        }
        // Nothing is drawn under the glass itself, so no need to look further there
        if (here < 0.0) {
            return half4(0.0);
        }
        float dropped = 1.0e9;
        for (int i = 0; i < 8; i++) {
            if (float(i) < shapeCount) {
                dropped = min(dropped, outline(p - drop, shapeBounds[i], cornerRadii[i]));
            }
        }
        float x = clamp((dropped + 0.5 * softness) / (1.5 * softness), 0.0, 1.0);
        return half4(0.0, 0.0, 0.0, half(strength * (1.0 - x) * (1.0 - x)));
    }
)");

// Joined glass sides pushed this far out show no rim.
static constexpr float kGlassFarAway = 10000.0f;
// The glass's shadow: its falloff and how far it falls, in rim widths.
static constexpr float kGlassShadowSoftness = 0.9f;
static constexpr float kGlassShadowDrop = 0.25f;
// Unit vector toward the usual light (up and a little to the left), y pointing down.
static constexpr SkV2 kGlassLightDir = {-0.447214f, -0.894427f};

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
    auto [glassEffect, glassError] = SkRuntimeEffect::MakeForShader(kGlassEffectSource, options);
    options.fName = "RE_CatalyxxGlassShadow";
    auto [shadowEffect, shadowError] =
            SkRuntimeEffect::MakeForShader(kGlassShadowEffectSource, options);
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
                                sk_sp<SkImage> input) {
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

void BlurFilter::drawGlassShadows(SkCanvas* canvas, const std::vector<BlurRegion>& regions,
                                  const vec2& light) {
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
    for (const BlurRegion& region : regions) {
        // Regions without a blur radius aren't drawn, so they cast no shadow either.
        if (!region.isGlass() || region.blurRadius == 0 || region.glassShadow <= 0.0f) {
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

    SkRuntimeShaderBuilder builder(mGlassShadowEffect);
    builder.uniform("shapeBounds").set(bounds.data(), kGlassShadowMaxShapes);
    builder.uniform("cornerRadii").set(radii.data(), kGlassShadowMaxShapes);
    builder.uniform("shapeCount") = static_cast<float>(count);
    builder.uniform("softness") = softness;
    builder.uniform("drop") = getGlassLight(movingLight ? light : vec2(0.f, 0.f)).shadowDrop(drop);
    builder.uniform("strength") = strength;
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
                                 sk_sp<SkImage> input) {
    if (!canDrawGlass() || !input) {
        drawBlurRegion(canvas, getGlassRRect(region), region.blurRadius, 1.0f, region.alpha,
                       blurRect, blurredImage, input);
        return;
    }
    SFTRACE_CALL();

    const SkRRect outline = getGlassRRect(region);
    const GlassShape shape = getGlassShape(outline, region.glassJoinedSides, true);
    const float rimWidth = getGlassRimWidth(region, outline.rect());

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, 1.0f);
    // For sampling Skia's API expects the inverse of what logically seems appropriate, as in
    // drawBlurRegion().
    SkMatrix inputMatrix;
    if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
        ALOGE("matrix was unable to be inverted");
    }

    // Like plain blur, small radii fade toward the sharp content.
    float frost = region.glassFrost;
    if (region.blurRadius < mMaxCrossFadeRadius) {
        frost *= region.blurRadius / mMaxCrossFadeRadius;
    }

    SkRuntimeShaderBuilder builder(mGlassEffect);
    builder.child("blurredInput") =
            blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                     &blurMatrix);
    builder.child("originalInput") =
            input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                              inputMatrix);
    builder.uniform("shapeBounds") = shape.bounds;
    builder.uniform("cornerRadii") = shape.radii;
    builder.uniform("rimWidth") = rimWidth;
    builder.uniform("bend") = region.glassBend;
    builder.uniform("frost") = frost;
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
    // Without its shader the paint would fill the glass black, so draw plain blur instead.
    sk_sp<SkShader> shader = builder.makeShader();
    if (!shader) {
        ALOGE("Catalyxx glass could not be made, drawing plain blur");
        drawBlurRegion(canvas, outline, region.blurRadius, 1.0f, region.alpha, blurRect,
                       blurredImage, input);
        return;
    }

    SkPaint paint;
    paint.setAlphaf(region.alpha);
    paint.setAntiAlias(true);
    paint.setShader(std::move(shader));
    canvas->drawRRect(outline, paint);
}

} // namespace skia
} // namespace renderengine
} // namespace android
