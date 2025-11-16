/**
 * @file png_decoder.hpp
 * @brief PNG decoder using libpng
 *
 * Features:
 * - Lossless compression support
 * - All color types (RGB, RGBA, grayscale, palette, alpha)
 * - Automatic conversion to RGB output
 * - 16-bit to 8-bit conversion
 * - Zero-copy memory-mapped input
 * - Pooled output buffers
 * - Thread-safe (one instance per thread)
 */

#pragma once

#include "../core/sample.hpp"
#include "../core/object_pool.hpp"
#include <png.h>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace turboloader {

/**
 * @brief High-performance PNG decoder
 *
 * Supports all PNG color types and automatically converts to RGB:
 * - Grayscale -> RGB conversion
 * - Grayscale + Alpha -> RGB (strip alpha)
 * - RGB -> RGB (no conversion)
 * - RGBA -> RGB (strip alpha)
 * - Palette -> RGB expansion
 * - 16-bit -> 8-bit reduction
 *
 * Thread-safe when each thread has its own instance.
 */
class PNGDecoder {
public:
    /**
     * @brief Construct decoder with optional buffer pool
     */
    explicit PNGDecoder(BufferPool* pool = nullptr)
        : buffer_pool_(pool) {}

    /**
     * @brief Decode PNG data to RGB
     *
     * @param png_data Span of compressed PNG bytes
     * @param output Output buffer for RGB data (resized automatically)
     * @param width Output: image width
     * @param height Output: image height
     * @param channels Output: number of channels (always 3 for RGB)
     *
     * @throws std::runtime_error on decode failure
     */
    void decode(
        std::span<const uint8_t> png_data,
        std::vector<uint8_t>& output,
        int& width,
        int& height,
        int& channels
    ) {
        if (png_data.empty()) {
            throw std::runtime_error("Empty PNG data");
        }

        // Create PNG read struct
        png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png_ptr) {
            throw std::runtime_error("Failed to create PNG read struct");
        }

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

        // Set up memory reader
        struct MemoryReader {
            const uint8_t* data;
            size_t size;
            size_t offset;
        };

        MemoryReader reader{png_data.data(), png_data.size(), 0};

        png_set_read_fn(png_ptr, &reader, [](png_structp png_ptr, png_bytep data, size_t length) {
            MemoryReader* reader = static_cast<MemoryReader*>(png_get_io_ptr(png_ptr));
            if (reader->offset + length > reader->size) {
                png_error(png_ptr, "Read past end of PNG data");
            }
            std::memcpy(data, reader->data + reader->offset, length);
            reader->offset += length;
        });

        // Read PNG info
        png_read_info(png_ptr, info_ptr);

        width = png_get_image_width(png_ptr, info_ptr);
        height = png_get_image_height(png_ptr, info_ptr);
        int color_type = png_get_color_type(png_ptr, info_ptr);
        int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

        // Transform to RGB
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
        if (color_type == PNG_COLOR_TYPE_RGB_ALPHA || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
            png_set_strip_alpha(png_ptr);
        }

        png_read_update_info(png_ptr, info_ptr);

        channels = 3;  // Always RGB

        // Allocate output buffer
        size_t row_stride = width * channels;
        size_t total_size = height * row_stride;
        output.resize(total_size);

        // Read image rows
        std::vector<png_bytep> row_pointers(height);
        for (int y = 0; y < height; ++y) {
            row_pointers[y] = output.data() + (y * row_stride);
        }

        png_read_image(png_ptr, row_pointers.data());
        png_read_end(png_ptr, nullptr);

        // Cleanup
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    }

    /**
     * @brief Decode PNG into Sample with pooled buffer
     */
    void decode_sample(Sample& sample) {
        if (sample.jpeg_data.empty()) {
            throw std::runtime_error("Sample has no image data");
        }

        // Get pooled buffer if available
        std::vector<uint8_t> buffer;
        if (buffer_pool_) {
            auto pooled = buffer_pool_->acquire();
            buffer = std::move(*pooled);
        }

        // Decode PNG
        decode(sample.jpeg_data, buffer, sample.width, sample.height, sample.channels);

        // Move buffer into sample
        sample.decoded_rgb = std::move(buffer);
    }

    /**
     * @brief Get decoder info
     */
    static std::string version_info() {
        return "libpng " PNG_LIBPNG_VER_STRING;
    }

private:
    BufferPool* buffer_pool_;
};

} // namespace turboloader
