/*
 * Copyright 2020 The Android Open Source Project
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

#pragma once

#include <SkCanvas.h>
#include <SkImage.h>
#include <SkM44.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkSurface.h>
#include <math/vec2.h>
#include <ui/BlurRegion.h>

#include <array>
#include <vector>

#include "../compat/SkiaGpuContext.h"
#include "RuntimeEffectManager.h"

using namespace std;

namespace android {
namespace renderengine {
namespace skia {

class BlurFilter {
public:
    // Downsample FBO to improve performance
    static constexpr float kInputScale = 0.25f;
    // Downsample scale factor used to improve performance
    static constexpr float kInverseInputScale = 1.0f / kInputScale;

    explicit BlurFilter(RuntimeEffectManager& effectManager, float maxCrossFadeRadius = 10.0f);
    virtual ~BlurFilter(){}

    // Execute blur, saving it to a TEMPORARY texture. WARNING: the returned SkImage is only
    // guaranteed to be valid until the next call to generateTemporaryImage on this BlurFilter. This
    // limitation is to avoid unnecessary allocations.
    virtual sk_sp<SkImage> generateTemporaryImage(SkiaGpuContext* context,
                                                  const DisplaySettings& display,
                                                  const uint32_t radius,
                                                  const sk_sp<SkImage> blurInput,
                                                  const SkRect& blurRect) const = 0;

    /**
     * Draw the blurred content (from the generate method) into the canvas.
     * @param canvas is the destination/output for the blur
     * @param effectRegion the RoundRect in canvas coordinates that determines the blur coverage
     * @param blurRadius radius of the blur used to determine the intensity of the crossfade effect
     * @param blurAlpha alpha value applied to the effectRegion when the blur is drawn
     * @param blurRect bounds of the blurredImage translated into canvas coordinates
     * @param blurredImage down-sampled blurred content that was produced by the generate() method
     * @param input original unblurred input that is used to crossfade with the blurredImage
     * @param inputOrigin where input's top left corner is in device coordinates, when it holds
     *                    only part of the display
     */
    void drawBlurRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                const uint32_t blurRadius, const float zoomScale,
                                const float blurAlpha, const SkRect& blurRect,
                                sk_sp<SkImage> blurredImage, sk_sp<SkImage> input,
                                SkIPoint inputOrigin = SkIPoint::Make(0, 0));

    float getMaxCrossFadeRadius() const;

    // Catalyxx glass (BlurRegion::isGlass()): false if its effects failed to build, in which
    // case glass regions are drawn as plain blur.
    bool canDrawGlass() const { return mGlassEffect && mGlassShadowEffect; }

    // How far melted glass can bulge past the pieces' own outlines, as a share of the size it melts
    // at: as the cubic smooth minimum in its shader goes.
    static constexpr float kGlassMeltBulge = 1.0f / 6.0f;

    /**
     * Catalyxx glass of a layer that melts together (BlurRegion::glassMelt), drawn as one surface
     * from its pieces. Whatever isn't each piece's own is the first's.
     */
    struct GlassMelt {
        static constexpr size_t kMaxShapes = 4;

        // Left, top, right, bottom.
        std::array<SkV4, kMaxShapes> bounds{};
        // Top-left, top-right, bottom-right, bottom-left, as circles.
        std::array<SkV4, kMaxShapes> radii{};
        std::array<float, kMaxShapes> rims{};
        std::array<float, kMaxShapes> alphas{};
        std::array<float, kMaxShapes> frosts{};
        std::array<SkV4, kMaxShapes> tintsOverDark{};
        std::array<SkV4, kMaxShapes> tintsOverLight{};
        size_t count = 0;
        // How close the pieces start to reach for each other
        float size = 0.f;
        float bend = 0.f;
        float rimLight = 0.f;
        float dispersion = 0.f;
        float shadow = 0.f;
        float adaptPoint = 0.f;
        bool movingLight = false;
        // All it can cover
        SkRect area = SkRect::MakeEmpty();
        // Bit i: it draws region i of the layer
        uint64_t regions = 0;
    };

    /** All the glass of a layer that melts together, from findGlassMelts(). */
    struct GlassMelts {
        std::vector<GlassMelt> melts;
        // Bit i: region i of the layer is drawn by one of the melts
        uint64_t melted = 0;

        bool contains(size_t region) const {
            return region < sizeof(melted) * 8 && (melted >> region) & 1;
        }
    };

    /**
     * Which of a layer's glass regions melt together now: those where one has a glassMelt and
     * others are within it. The rest are drawn each on their own.
     */
    GlassMelts findGlassMelts(const std::vector<BlurRegion>& regions);

    // The outline a glass region is drawn in: the region's rounded rect with square corners
    // next to its joined sides, so neighbouring pieces meet without a gap.
    static SkRRect getGlassRRect(const BlurRegion& region);

    /**
     * Draw one soft shadow around all the glass regions in the list, before any of them is
     * drawn. Pieces that touch or overlap share one shadow, which falls away from the light.
     * @param melts what melts together among them
     * @param light where the light comes from, as DisplaySettings::glassLight
     */
    void drawGlassShadows(SkCanvas* canvas, const std::vector<BlurRegion>& regions,
                          const GlassMelts& melts, const vec2& light);

    /**
     * Draw a glass region: its rim bends in the content just past the edge and catches the
     * light, its middle shows the content behind it, clear or frosted.
     * @param light where the light comes from, as DisplaySettings::glassLight
     * @param blurRect bounds of the blurredImage translated into canvas coordinates
     * @param blurredImage down-sampled blurred content that was produced by the generate() method
     * @param input original unblurred input, which the rim bends
     * @param inputOrigin where input's top left corner is in device coordinates, as for
     *                    drawBlurRegion()
     */
    void drawGlassRegion(SkCanvas* canvas, const BlurRegion& region, const vec2& light,
                         const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                         sk_sp<SkImage> input, SkIPoint inputOrigin = SkIPoint::Make(0, 0));

    /**
     * Draw glass that melts together as one surface, as drawGlassRegion() draws one piece.
     */
    void drawGlassMelt(SkCanvas* canvas, const GlassMelt& melt, const vec2& light,
                       const SkRect& blurRect, sk_sp<SkImage> blurredImage, sk_sp<SkImage> input,
                       SkIPoint inputOrigin = SkIPoint::Make(0, 0));

    virtual void preallocateBuffers(SkiaGpuContext* context, ui::Size size) {}
    virtual bool areBuffersPreallocated(const SkiaGpuContext* context, ui::Size displaySize) const {
        return true;
    }

private:
    // How much of a glass region's flat middle is blurred, as far as its blur radius goes.
    float getGlassFrost(const BlurRegion& region) const;

    // Builds the effects for glass that melts together, the first time any does. False if they
    // failed to build: the glass then stays apart.
    bool buildGlassMeltEffects();

    // To avoid downscaling artifacts, we interpolate the blurred fbo with the full composited
    // image, up to this radius.
    const float mMaxCrossFadeRadius;

    // Optional blend used for crossfade only if mMaxCrossFadeRadius > 0
    const sk_sp<SkRuntimeEffect> mMixEffect;

    // Catalyxx glass. Built here rather than with the known effects, whose build failures
    // are fatal: if these fail, glass falls back to plain blur.
    sk_sp<SkRuntimeEffect> mGlassEffect;
    sk_sp<SkRuntimeEffect> mGlassShadowEffect;
    sk_sp<SkRuntimeEffect> mGlassMeltEffect;
    sk_sp<SkRuntimeEffect> mGlassMeltShadowEffect;
    bool mGlassMeltFailed = false;
};

} // namespace skia
} // namespace renderengine
} // namespace android
