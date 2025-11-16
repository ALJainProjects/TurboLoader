#include "turboloader/decoders/webp_decoder.hpp"
#include <stdexcept>
#include <cstring>

#ifdef HAVE_WEBP
#include <webp/decode.h>
#endif

namespace turboloader {

DecodedImage WebPDecoder::decode(std::span<const uint8_t> data) {
    if (!is_webp(data)) {
        throw std::invalid_argument("Not a valid WebP file");
    }

#ifdef HAVE_WEBP
    // Get image dimensions
    int width, height;
    if (!WebPGetInfo(data.data(), data.size(), &width, &height)) {
        throw std::runtime_error("Failed to get WebP info");
    }

    // Allocate output buffer
    DecodedImage result;
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.data.resize(width * height * 3);

    // Decode to RGB
    uint8_t* output = WebPDecodeRGBInto(
        data.data(), data.size(),
        result.data.data(), result.data.size(),
        width * 3  // stride
    );

    if (!output) {
        throw std::runtime_error("Failed to decode WebP");
    }

    return result;
#else
    throw std::runtime_error("WebP decoder not available - install libwebp and rebuild");
#endif
}

bool WebPDecoder::can_decode(std::span<const uint8_t> data) const {
    return is_webp(data);
}

bool WebPDecoder::is_webp(std::span<const uint8_t> data) {
    // WebP magic bytes: "RIFF" ... "WEBP"
    if (data.size() < 12) {
        return false;
    }

    return data[0] == 'R' &&
           data[1] == 'I' &&
           data[2] == 'F' &&
           data[3] == 'F' &&
           data[8] == 'W' &&
           data[9] == 'E' &&
           data[10] == 'B' &&
           data[11] == 'P';
}

}  // namespace turboloader
