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

#include <inttypes.h>
#include <array>
#include <iosfwd>
#include <iostream>

#include <math/HashCombine.h>

namespace android {

struct BlurRegion {
    // Sides of a region, for glassJoinedSides.
    static constexpr uint32_t kGlassSideLeft = 1 << 0;
    static constexpr uint32_t kGlassSideTop = 1 << 1;
    static constexpr uint32_t kGlassSideRight = 1 << 2;
    static constexpr uint32_t kGlassSideBottom = 1 << 3;
    // Also in glassJoinedSides: glass that keeps the usual light whatever the phone's tilt, such
    // as glass that shows nearly all the time, which mustn't keep the tilt sensor running and
    // the screen being drawn again as the light moves.
    static constexpr uint32_t kGlassFixedLight = 1 << 4;

    uint32_t blurRadius;
    float cornerRadiusTLX;
    float cornerRadiusTLY;
    float cornerRadiusTRX;
    float cornerRadiusTRY;
    float cornerRadiusBLX;
    float cornerRadiusBLY;
    float cornerRadiusBRX;
    float cornerRadiusBRY;
    float alpha;
    int left;
    int top;
    int right;
    int bottom;

    // Catalyxx glass: while glassRimWidth is above 0, the region is drawn as a slab of
    // glass with a curved, lit rim instead of plain blur.
    // Width of the curved rim, in the region's coordinates.
    float glassRimWidth = 0.f;
    // How far past the edge the rim reaches, in rim widths.
    float glassBend = 0.f;
    // Blurred share of the flat middle, 0 to 1.
    float glassFrost = 0.f;
    // Strength of the light on the rim, 0 to 1.
    float glassRimLight = 0.f;
    // Color split of the bent light, 0 to 1.
    float glassDispersion = 0.f;
    // Darkness of the glass's shadow, 0 to 1.
    float glassShadow = 0.f;
    // kGlassSide* bits: sides where the glass carries on into another region of the
    // same layer, so they get no rim and no shadow; and kGlassFixedLight.
    uint32_t glassJoinedSides = 0;
    // The glass's own color over dark content and over light content: straight-alpha RGBA,
    // where alpha is how much of the glass it covers.
    std::array<float, 4> glassTintOverDark{};
    std::array<float, 4> glassTintOverLight{};
    // Brightness of what's behind (0 to 1) around which the glass turns from its dark-content
    // color to its light-content color; 0 keeps the dark-content color everywhere.
    float glassAdaptPoint = 0.f;

    inline bool isGlass() const { return glassRimWidth > 0.f; }

    // Glass whose light follows the phone's tilt: all glass but kGlassFixedLight's.
    inline bool hasMovingLight() const {
        return isGlass() && !(glassJoinedSides & kGlassFixedLight);
    }

    inline bool operator==(const BlurRegion& other) const {
        return blurRadius == other.blurRadius && cornerRadiusTLX == other.cornerRadiusTLX &&
                cornerRadiusTLY == other.cornerRadiusTLY &&
                cornerRadiusTRX == other.cornerRadiusTRX &&
                cornerRadiusTRY == other.cornerRadiusTRY &&
                cornerRadiusBLX == other.cornerRadiusBLX &&
                cornerRadiusBLY == other.cornerRadiusBLY &&
                cornerRadiusBRX == other.cornerRadiusBRX &&
                cornerRadiusBRY == other.cornerRadiusBRY && alpha == other.alpha &&
                left == other.left && top == other.top && right == other.right &&
                bottom == other.bottom && glassRimWidth == other.glassRimWidth &&
                glassBend == other.glassBend && glassFrost == other.glassFrost &&
                glassRimLight == other.glassRimLight &&
                glassDispersion == other.glassDispersion && glassShadow == other.glassShadow &&
                glassJoinedSides == other.glassJoinedSides &&
                glassTintOverDark == other.glassTintOverDark &&
                glassTintOverLight == other.glassTintOverLight &&
                glassAdaptPoint == other.glassAdaptPoint;
    }

    inline bool operator!=(const BlurRegion& other) const { return !(*this == other); }
};

namespace {
// A newline character followed by N*4 spaces.
static inline constexpr std::string IndentedNewline(uint8_t indent) {
    return "\n" + std::string(static_cast<size_t>(indent * 4), ' ');
}
} // namespace

static inline void PrintTo(const BlurRegion& blurRegion, ::std::ostream* os,
                           const uint8_t currentIndent = 0) {
    const std::string newline = IndentedNewline(currentIndent + 1);
    *os << "BlurRegion {";
    *os << newline << ".blurRadius = " << blurRegion.blurRadius;
    *os << newline << ".cornerRadiusTLX = " << blurRegion.cornerRadiusTLX;
    *os << newline << ".cornerRadiusTLY = " << blurRegion.cornerRadiusTLY;
    *os << newline << ".cornerRadiusTRX = " << blurRegion.cornerRadiusTRX;
    *os << newline << ".cornerRadiusTRY = " << blurRegion.cornerRadiusTRY;
    *os << newline << ".cornerRadiusBLX = " << blurRegion.cornerRadiusBLX;
    *os << newline << ".cornerRadiusBLY = " << blurRegion.cornerRadiusBLY;
    *os << newline << ".cornerRadiusBRX = " << blurRegion.cornerRadiusBRX;
    *os << newline << ".cornerRadiusBRY = " << blurRegion.cornerRadiusBRY;
    *os << newline << ".alpha = " << blurRegion.alpha;
    *os << newline << ".left = " << blurRegion.left;
    *os << newline << ".top = " << blurRegion.top;
    *os << newline << ".right = " << blurRegion.right;
    *os << newline << ".bottom = " << blurRegion.bottom;
    if (blurRegion.isGlass()) {
        *os << newline << ".glassRimWidth = " << blurRegion.glassRimWidth;
        *os << newline << ".glassBend = " << blurRegion.glassBend;
        *os << newline << ".glassFrost = " << blurRegion.glassFrost;
        *os << newline << ".glassRimLight = " << blurRegion.glassRimLight;
        *os << newline << ".glassDispersion = " << blurRegion.glassDispersion;
        *os << newline << ".glassShadow = " << blurRegion.glassShadow;
        *os << newline << ".glassJoinedSides = " << blurRegion.glassJoinedSides;
        const auto& dark = blurRegion.glassTintOverDark;
        const auto& light = blurRegion.glassTintOverLight;
        *os << newline << ".glassTintOverDark = (" << dark[0] << ", " << dark[1] << ", "
            << dark[2] << ", " << dark[3] << ")";
        *os << newline << ".glassTintOverLight = (" << light[0] << ", " << light[1] << ", "
            << light[2] << ", " << light[3] << ")";
        *os << newline << ".glassAdaptPoint = " << blurRegion.glassAdaptPoint;
    }
    *os << IndentedNewline(currentIndent) << "}";
}

// copied from skia/src/core/SkBlurMask.cpp
inline float convertBlurUserRadiusToSigma(float radius) {
    return radius > 0 ? 0.57735f * radius + 0.5f : 0.0f;
}
// copied from skia/src/core/SkBlurEngine.h
inline float convertBlurSigmaToKernelRadius(float sigma) {
    return sigma <= 0.03f ? 0 : ceilf(3.f * sigma);
}

} // namespace android

namespace std {
template <>
struct hash<android::BlurRegion> {
    size_t operator()(const android::BlurRegion& region) const {
        return android::hashCombine(region.blurRadius, region.cornerRadiusTLX,
                                    region.cornerRadiusTLY, region.cornerRadiusTRX,
                                    region.cornerRadiusTRY, region.cornerRadiusBLX,
                                    region.cornerRadiusBLY, region.cornerRadiusBRX,
                                    region.cornerRadiusBRY, region.alpha, region.left, region.top,
                                    region.right, region.bottom, region.glassRimWidth,
                                    region.glassBend, region.glassFrost, region.glassRimLight,
                                    region.glassDispersion, region.glassShadow,
                                    region.glassJoinedSides, region.glassTintOverDark[0],
                                    region.glassTintOverDark[1], region.glassTintOverDark[2],
                                    region.glassTintOverDark[3], region.glassTintOverLight[0],
                                    region.glassTintOverLight[1], region.glassTintOverLight[2],
                                    region.glassTintOverLight[3], region.glassAdaptPoint);
    }
};
} // namespace std