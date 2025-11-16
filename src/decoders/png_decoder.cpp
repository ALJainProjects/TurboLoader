#include "turboloader/decoders/png_decoder.hpp"
#include <png.h>
#include <stdexcept>
#include <cstring>
#include <csetjmp>

namespace turboloader {

// Memory read state for PNG
struct PngReadState {
    const uint8_t* data;
    size_t size;
    size_t offset;
};

// Custom PNG read function
static void png_read_data(png_structp png_ptr, png_bytep out_data, png_size_t length) {
    PngReadState* state = static_cast<PngReadState*>(png_get_io_ptr(png_ptr));

    if (state->offset + length > state->size) {
        png_error(png_ptr, "PNG read error: not enough data");
        return;
    }

    std::memcpy(out_data, state->data + state->offset, length);
    state->offset += length;
}

struct PngDecoder::Impl {
    png_structp png_ptr = nullptr;
    png_infop info_ptr = nullptr;
};

PngDecoder::PngDecoder()
    : pimpl_(std::make_unique<Impl>()) {
}

PngDecoder::~PngDecoder() = default;
PngDecoder::PngDecoder(PngDecoder&&) noexcept = default;
PngDecoder& PngDecoder::operator=(PngDecoder&&) noexcept = default;

DecodedImage PngDecoder::decode(std::span<const uint8_t> data) {
    if (!is_png(data)) {
        throw std::invalid_argument("Not a valid PNG file");
    }

    // Create PNG read struct
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        throw std::runtime_error("Failed to create PNG read struct");
    }

    // Create PNG info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        throw std::runtime_error("Failed to create PNG info struct");
    }

    // Error handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        throw std::runtime_error("PNG decode error");
    }

    // Set up custom read function
    PngReadState state{data.data(), data.size(), 0};
    png_set_read_fn(png_ptr, &state, png_read_data);

    // Read PNG info
    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Convert to RGB
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
    }
    if (bit_depth == 16) {
        png_set_strip_16(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_RGBA) {
        png_set_strip_alpha(png_ptr);
    }

    png_read_update_info(png_ptr, info_ptr);

    // Allocate output buffer
    DecodedImage result;
    result.width = width;
    result.height = height;
    result.channels = 3;  // Always RGB
    result.data.resize(width * height * 3);

    // Allocate row pointers
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = result.data.data() + y * width * 3;
    }

    // Read image data
    png_read_image(png_ptr, row_pointers.data());

    // Cleanup
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

    return result;
}

bool PngDecoder::can_decode(std::span<const uint8_t> data) const {
    return is_png(data);
}

bool PngDecoder::is_png(std::span<const uint8_t> data) {
    // PNG magic bytes: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
    if (data.size() < 8) {
        return false;
    }

    return data[0] == 0x89 &&
           data[1] == 0x50 &&  // 'P'
           data[2] == 0x4E &&  // 'N'
           data[3] == 0x47 &&  // 'G'
           data[4] == 0x0D &&
           data[5] == 0x0A &&
           data[6] == 0x1A &&
           data[7] == 0x0A;
}

}  // namespace turboloader
