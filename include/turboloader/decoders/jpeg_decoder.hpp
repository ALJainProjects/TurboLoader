#pragma once

#include "turboloader/decoders/image_decoder.hpp"
#include <memory>

namespace turboloader {

/**
 * High-performance JPEG decoder using libjpeg-turbo
 *
 * Features:
 * - Fast decompression (SIMD optimized)
 * - RGB output format
 * - Error handling
 * - Batch decoding support
 */
class JpegDecoder : public ImageDecoder {
public:
    JpegDecoder();
    ~JpegDecoder() override;

    // Non-copyable, movable
    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;
    JpegDecoder(JpegDecoder&&) noexcept;
    JpegDecoder& operator=(JpegDecoder&&) noexcept;

    /**
     * Decode JPEG data to RGB
     * @param jpeg_data JPEG compressed data
     * @return Decoded image with RGB pixels
     */
    DecodedImage decode(std::span<const uint8_t> jpeg_data) override;

    /**
     * Check if data can be decoded (is JPEG)
     */
    bool can_decode(std::span<const uint8_t> data) const override;

    /**
     * Get decoder name
     */
    std::string name() const override { return "jpeg"; }

    /**
     * Check if data looks like JPEG (magic bytes)
     */
    static bool is_jpeg(std::span<const uint8_t> data);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace turboloader
