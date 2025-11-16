#pragma once

#include "turboloader/decoders/image_decoder.hpp"
#include <memory>

namespace turboloader {

/**
 * PNG decoder using libpng
 *
 * Features:
 * - Decodes PNG to RGB
 * - Handles transparency (converts RGBA -> RGB)
 * - Supports all PNG color types
 * - Fast decompression
 */
class PngDecoder : public ImageDecoder {
public:
    PngDecoder();
    ~PngDecoder() override;

    // Non-copyable, movable
    PngDecoder(const PngDecoder&) = delete;
    PngDecoder& operator=(const PngDecoder&) = delete;
    PngDecoder(PngDecoder&&) noexcept;
    PngDecoder& operator=(PngDecoder&&) noexcept;

    DecodedImage decode(std::span<const uint8_t> data) override;
    bool can_decode(std::span<const uint8_t> data) const override;
    std::string name() const override { return "png"; }

    /**
     * Check if data looks like PNG (magic bytes)
     */
    static bool is_png(std::span<const uint8_t> data);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace turboloader
