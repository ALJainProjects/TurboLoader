#pragma once

#include "turboloader/decoders/image_decoder.hpp"

namespace turboloader {

/**
 * WebP decoder using libwebp
 *
 * Features:
 * - Decodes WebP to RGB
 * - Handles both lossy and lossless
 * - Fast decompression (SIMD optimized)
 */
class WebPDecoder : public ImageDecoder {
public:
    WebPDecoder() = default;
    ~WebPDecoder() override = default;

    DecodedImage decode(std::span<const uint8_t> data) override;
    bool can_decode(std::span<const uint8_t> data) const override;
    std::string name() const override { return "webp"; }

    /**
     * Check if data looks like WebP (magic bytes)
     */
    static bool is_webp(std::span<const uint8_t> data);
};

}  // namespace turboloader
