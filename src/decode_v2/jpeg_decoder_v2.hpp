/**
 * @file jpeg_decoder_v2.hpp
 * @brief High-performance JPEG decoder using libjpeg-turbo
 *
 * Uses libjpeg-turbo for SIMD-accelerated decoding (2-6x faster than libjpeg).
 * Integrates with object pool for zero-allocation buffer reuse.
 *
 * Design:
 * - libjpeg-turbo provides SIMD optimizations (SSE2, AVX2, NEON)
 * - Reuses decompressor handles to avoid initialization overhead
 * - Uses pooled buffers for decoded RGB data
 * - Direct decode to RGB format (no color space conversion needed)
 *
 * Performance:
 * - libjpeg-turbo: 2-6x faster than standard libjpeg
 * - Buffer pooling: Eliminates malloc/free overhead
 * - Handle reuse: Reduces decompressor initialization cost
 */

#pragma once

#include "../core_v2/sample_v2.hpp"
#include "../core_v2/object_pool.hpp"
#include <jpeglib.h>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace turboloader {
namespace v2 {

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
 * @brief High-performance JPEG decoder
 *
 * Thread-safe when each thread has its own instance.
 * Reuses decompressor handle and pooled buffers for maximum performance.
 */
class JPEGDecoderV2 {
public:
    /**
     * @brief Construct decoder with optional buffer pool
     *
     * @param pool Optional buffer pool for decoded RGB data
     */
    explicit JPEGDecoderV2(BufferPool* pool = nullptr)
        : buffer_pool_(pool) {

        // Initialize decompressor
        cinfo_.err = jpeg_std_error(&jerr_.pub);
        jerr_.pub.error_exit = JPEGErrorMgr::error_exit;
        jpeg_create_decompress(&cinfo_);
    }

    /**
     * @brief Destructor - cleanup decompressor
     */
    ~JPEGDecoderV2() {
        jpeg_destroy_decompress(&cinfo_);
    }

    // Non-copyable
    JPEGDecoderV2(const JPEGDecoderV2&) = delete;
    JPEGDecoderV2& operator=(const JPEGDecoderV2&) = delete;

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

        // Set source to memory buffer
        jpeg_mem_src(&cinfo_, jpeg_data.data(), jpeg_data.size());

        // Read JPEG header
        if (jpeg_read_header(&cinfo_, TRUE) != JPEG_HEADER_OK) {
            throw std::runtime_error("Failed to read JPEG header");
        }

        // Force RGB output
        cinfo_.out_color_space = JCS_RGB;

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

        // Decode scanlines
        uint8_t* output_ptr = output.data();
        while (cinfo_.output_scanline < cinfo_.output_height) {
            uint8_t* row_pointer = output_ptr + (cinfo_.output_scanline * row_stride);
            jpeg_read_scanlines(&cinfo_, &row_pointer, 1);
        }

        // Finish decompression
        jpeg_finish_decompress(&cinfo_);
    }

    /**
     * @brief Decode JPEG into SampleV2 with pooled buffer
     *
     * @param sample Sample with jpeg_data filled in (from TarReaderV2)
     *
     * @throws std::runtime_error on decode failure
     *
     * This method:
     * 1. Gets buffer from pool (if available)
     * 2. Decodes JPEG into buffer
     * 3. Moves buffer into sample.decoded_rgb
     */
    void decode_sample(SampleV2& sample) {
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
     * @return String with libjpeg version
     */
    static std::string version_info() {
        return "libjpeg-turbo (SIMD-accelerated)";
    }

private:
    BufferPool* buffer_pool_;         // Optional buffer pool
    jpeg_decompress_struct cinfo_;    // libjpeg decompressor
    JPEGErrorMgr jerr_;               // Error handler
};

/**
 * @brief Batch JPEG decoder with parallel decoding
 *
 * Decodes multiple JPEGs in parallel using worker pool.
 * Each worker has its own decoder instance to avoid contention.
 */
class BatchJPEGDecoder {
public:
    /**
     * @brief Construct batch decoder
     *
     * @param num_workers Number of decoder threads
     * @param pool Buffer pool for decoded RGB data
     */
    explicit BatchJPEGDecoder(size_t num_workers = 4, BufferPool* pool = nullptr)
        : buffer_pool_(pool) {

        // Create decoder for each worker
        for (size_t i = 0; i < num_workers; ++i) {
            decoders_.push_back(std::make_unique<JPEGDecoderV2>(pool));
        }
    }

    /**
     * @brief Decode batch of samples
     *
     * @param batch Batch with JPEG data filled in
     *
     * Decodes samples in parallel using worker pool.
     * Each sample is assigned to a worker based on index.
     */
    void decode_batch(BatchV2& batch) {
        if (batch.empty()) {
            return;
        }

        // For now, use simple sequential decoding
        // TODO: Implement parallel decoding with thread pool
        size_t worker_id = 0;
        for (auto& sample : batch) {
            decoders_[worker_id]->decode_sample(sample);
            worker_id = (worker_id + 1) % decoders_.size();
        }
    }

    /**
     * @brief Get number of workers
     *
     * @return Number of decoder workers
     */
    size_t num_workers() const {
        return decoders_.size();
    }

private:
    BufferPool* buffer_pool_;
    std::vector<std::unique_ptr<JPEGDecoderV2>> decoders_;
};

} // namespace v2
} // namespace turboloader
