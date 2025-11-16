#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <memory>
#include <string>

namespace turboloader {

/**
 * Decoded image data (common format for all decoders)
 */
struct DecodedImage {
    std::vector<uint8_t> data;  // RGB pixels (HWC format)
    int width;
    int height;
    int channels;  // Always 3 (RGB)

    size_t size() const {
        return width * height * channels;
    }
};

/**
 * Image decoder interface
 *
 * All image decoders (JPEG, PNG, WebP, etc.) implement this interface
 */
class ImageDecoder {
public:
    virtual ~ImageDecoder() = default;

    /**
     * Decode image data to RGB
     * @param data Compressed image data
     * @return Decoded RGB image
     */
    virtual DecodedImage decode(std::span<const uint8_t> data) = 0;

    /**
     * Check if data is valid for this decoder
     * @param data Image data
     * @return true if this decoder can handle the data
     */
    virtual bool can_decode(std::span<const uint8_t> data) const = 0;

    /**
     * Get decoder name
     */
    virtual std::string name() const = 0;
};

/**
 * Auto-detecting image decoder
 *
 * Tries multiple decoders and uses the first one that can handle the data
 */
class AutoImageDecoder : public ImageDecoder {
public:
    AutoImageDecoder();
    ~AutoImageDecoder() override = default;

    DecodedImage decode(std::span<const uint8_t> data) override;
    bool can_decode(std::span<const uint8_t> data) const override;
    std::string name() const override { return "auto"; }

    /**
     * Add a decoder to try
     */
    void add_decoder(std::unique_ptr<ImageDecoder> decoder);

private:
    std::vector<std::unique_ptr<ImageDecoder>> decoders_;
};

/**
 * Detect image format from magic bytes
 */
enum class ImageFormat {
    Unknown,
    JPEG,
    PNG,
    WebP,
    BMP,
    GIF
};

ImageFormat detect_image_format(std::span<const uint8_t> data);
const char* image_format_name(ImageFormat format);

}  // namespace turboloader
