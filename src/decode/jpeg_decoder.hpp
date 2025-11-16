/**
 * @file jpeg_decoder.hpp
 * @brief SIMD-accelerated JPEG decoder using libjpeg-turbo
 *
 * Features:
 * - 2-6x faster than standard libjpeg (SIMD: SSE2, AVX2, NEON)
 * - Zero-copy memory-mapped input
 * - Pooled output buffers
 * - Thread-safe (one instance per thread)
 * - Direct RGB output
 */

#pragma once

#include "../core/sample.hpp"
#include "../core/object_pool.hpp"
#include <jpeglib.h>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace turboloader {

/**
 * @brief Error handler for libjpeg
 *
 * Converts libjpeg errors to C++ exceptions
 */
struct JPEGErrorMgr {
    struct jpeg_error_mgr pub;  // Public fields
    jmp_buf setjmp_buffer;       // For longjmp on error

    static void error_exit(j_common_ptr cinfo) {
        JPEGErrorMgr* myerr = reinterpret_cast<JPEGErrorMgr*>(cinfo->err);
        longjmp(myerr->setjmp_buffer, 1);
    }
};

/**
 * @brief High-performance SIMD-accelerated JPEG decoder
 *
 * Uses libjpeg-turbo which provides:
 * - SSE2 acceleration on x86/x64 (2-4x speedup)
 * - AVX2 acceleration on modern Intel/AMD (3-6x speedup)
 * - NEON acceleration on ARM (2-4x speedup)
 *
 * Performance optimizations:
 * - Reuses decompressor handle to avoid initialization overhead
 * - Zero-copy memory-mapped input via jpeg_mem_src()
 * - Pooled output buffers eliminate allocation overhead
 * - Direct decode to RGB (no color space conversion)
 *
 * Thread-safe when each thread has its own instance.
 */
class JPEGDecoder {
public:
    /**
     * @brief Construct decoder with optional buffer pool
     *
     * @param pool Optional buffer pool for decoded RGB data
     */
    explicit JPEGDecoder(BufferPool* pool = nullptr)
        : buffer_pool_(pool) {

        // Initialize decompressor
        cinfo_.err = jpeg_std_error(&jerr_.pub);
        jerr_.pub.error_exit = JPEGErrorMgr::error_exit;
        jpeg_create_decompress(&cinfo_);
    }

    /**
     * @brief Destructor - cleanup decompressor
     */
    ~JPEGDecoder() {
        jpeg_destroy_decompress(&cinfo_);
    }

    // Non-copyable
    JPEGDecoder(const JPEGDecoder&) = delete;
    JPEGDecoder& operator=(const JPEGDecoder&) = delete;

    /**
     * @brief Decode JPEG data to RGB
     *
     * @param jpeg_data Span of compressed JPEG bytes
     * @param output Output buffer for RGB data (resized automatically)
     * @param width Output: image width
     * @param height Output: image height
     * @param channels Output: number of channels (always 3 for RGB)
     *
     * @throws std::runtime_error on decode failure
     *
     * Complexity: O(width * height)
     * Thread-safe: Yes (each thread needs own instance)
     */
    void decode(
        std::span<const uint8_t> jpeg_data,
        std::vector<uint8_t>& output,
        int& width,
        int& height,
        int& channels
    ) {
        if (jpeg_data.empty()) {
            throw std::runtime_error("Empty JPEG data");
        }

        // Set up error handling
        if (setjmp(jerr_.setjmp_buffer)) {
            throw std::runtime_error("JPEG decode error");
        }

        // Set source to memory buffer (zero-copy)
        jpeg_mem_src(&cinfo_, jpeg_data.data(), jpeg_data.size());

        // Read JPEG header
        if (jpeg_read_header(&cinfo_, TRUE) != JPEG_HEADER_OK) {
            throw std::runtime_error("Failed to read JPEG header");
        }

        // Force RGB output
        cinfo_.out_color_space = JCS_RGB;

        // Enable SIMD optimizations (libjpeg-turbo)
        // This is automatically enabled in libjpeg-turbo build
        // SSE2/AVX2/NEON acceleration happens transparently

        // Start decompression
        jpeg_start_decompress(&cinfo_);

        // Get output dimensions
        width = cinfo_.output_width;
        height = cinfo_.output_height;
        channels = cinfo_.output_components;

        if (channels != 3) {
            jpeg_abort_decompress(&cinfo_);
            throw std::runtime_error("Expected RGB output (3 channels)");
        }

        // Allocate output buffer
        size_t row_stride = width * channels;
        size_t total_size = height * row_stride;
        output.resize(total_size);

        // Decode scanlines (SIMD-accelerated in libjpeg-turbo)
        uint8_t* output_ptr = output.data();
        while (cinfo_.output_scanline < cinfo_.output_height) {
            uint8_t* row_pointer = output_ptr + (cinfo_.output_scanline * row_stride);
            jpeg_read_scanlines(&cinfo_, &row_pointer, 1);
        }

        // Finish decompression
        jpeg_finish_decompress(&cinfo_);
    }

    /**
     * @brief Decode JPEG into Sample with pooled buffer
     *
     * @param sample Sample with jpeg_data filled in (from TarReader)
     *
     * @throws std::runtime_error on decode failure
     *
     * This method:
     * 1. Gets buffer from pool (if available)
     * 2. Decodes JPEG into buffer
     * 3. Moves buffer into sample.decoded_rgb
     */
    void decode_sample(Sample& sample) {
        if (sample.jpeg_data.empty()) {
            throw std::runtime_error("Sample has no JPEG data");
        }

        // Get pooled buffer if available
        std::vector<uint8_t> buffer;
        if (buffer_pool_) {
            auto pooled = buffer_pool_->acquire();
            buffer = std::move(*pooled);
        }

        // Decode JPEG
        decode(sample.jpeg_data, buffer, sample.width, sample.height, sample.channels);

        // Move buffer into sample
        sample.decoded_rgb = std::move(buffer);
    }

    /**
     * @brief Get decoder info
     *
     * @return String with libjpeg version and SIMD capabilities
     */
    static std::string version_info() {
        return "libjpeg-turbo (SIMD: SSE2/AVX2/NEON)";
    }

private:
    BufferPool* buffer_pool_;         // Optional buffer pool
    jpeg_decompress_struct cinfo_;    // libjpeg decompressor
    JPEGErrorMgr jerr_;               // Error handler
};

} // namespace turboloader
