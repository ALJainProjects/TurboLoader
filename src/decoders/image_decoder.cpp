#include "turboloader/decoders/image_decoder.hpp"
#include "turboloader/decoders/jpeg_decoder.hpp"
#include "turboloader/decoders/png_decoder.hpp"
#include "turboloader/decoders/webp_decoder.hpp"
#include <stdexcept>

namespace turboloader {

// ============================================================================
// AutoImageDecoder
// ============================================================================

AutoImageDecoder::AutoImageDecoder() {
    // Add decoders in priority order
    // JPEG is most common, check it first
    add_decoder(std::make_unique<JpegDecoder>());
    add_decoder(std::make_unique<PngDecoder>());
    add_decoder(std::make_unique<WebPDecoder>());
}

void AutoImageDecoder::add_decoder(std::unique_ptr<ImageDecoder> decoder) {
    decoders_.push_back(std::move(decoder));
}

DecodedImage AutoImageDecoder::decode(std::span<const uint8_t> data) {
    // Try each decoder
    for (const auto& decoder : decoders_) {
        if (decoder->can_decode(data)) {
            return decoder->decode(data);
        }
    }

    // No decoder could handle this data
    auto format = detect_image_format(data);
    throw std::runtime_error(
        std::string("No decoder available for image format: ") +
        image_format_name(format)
    );
}

bool AutoImageDecoder::can_decode(std::span<const uint8_t> data) const {
    for (const auto& decoder : decoders_) {
        if (decoder->can_decode(data)) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Format Detection
// ============================================================================

ImageFormat detect_image_format(std::span<const uint8_t> data) {
    if (data.size() < 12) {
        return ImageFormat::Unknown;
    }

    // JPEG: FF D8 FF
    if (data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
        return ImageFormat::JPEG;
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) {
        return ImageFormat::PNG;
    }

    // WebP: RIFF ... WEBP
    if (data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
        data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P') {
        return ImageFormat::WebP;
    }

    // BMP: 42 4D
    if (data[0] == 0x42 && data[1] == 0x4D) {
        return ImageFormat::BMP;
    }

    // GIF: 47 49 46
    if (data[0] == 0x47 && data[1] == 0x49 && data[2] == 0x46) {
        return ImageFormat::GIF;
    }

    return ImageFormat::Unknown;
}

const char* image_format_name(ImageFormat format) {
    switch (format) {
        case ImageFormat::JPEG: return "JPEG";
        case ImageFormat::PNG: return "PNG";
        case ImageFormat::WebP: return "WebP";
        case ImageFormat::BMP: return "BMP";
        case ImageFormat::GIF: return "GIF";
        case ImageFormat::Unknown: return "Unknown";
    }
    return "Unknown";
}

}  // namespace turboloader
