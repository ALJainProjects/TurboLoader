/**
 * @file image_decoder.hpp
 * @brief Multi-format image decoder orchestrator with auto-detection
 *
 * This is the main entry point for image decoding. It auto-detects the format
 * from magic bytes and delegates to the appropriate specialized decoder.
 *
 * Supported formats:
 * - JPEG: libjpeg-turbo (SIMD: SSE2/AVX2/NEON)
 * - PNG: libpng (lossless)
 * - WebP: libwebp (SIMD: SSE2/AVX2/NEON) [optional]
 * - BMP: native C++ (24/32-bit uncompressed)
 *
 * All decoders output RGB format.
 */

#pragma once

#include "./jpeg_decoder.hpp"
#include "./png_decoder.hpp"
#include "./webp_decoder.hpp"
#include "./bmp_decoder.hpp"
#include "./tiff_decoder.hpp"
#include "../core/sample.hpp"
#include "../core/object_pool.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace turboloader {

/**
 * @brief Image format enum
 */
enum class ImageFormat {
    JPEG,
    PNG,
    WEBP,
    BMP,
    TIFF,
    UNKNOWN
};

/**
 * @brief Detect image format from magic bytes
 *
 * @param data Span of image data (at least 12 bytes)
 * @return Detected format
 *
 * Complexity: O(1) - simple byte comparisons
 */
inline ImageFormat detect_format(std::span<const uint8_t> data) {
    if (data.size() < 12) {
        return ImageFormat::UNKNOWN;
    }

    // JPEG: FF D8 FF
    if (data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
        return ImageFormat::JPEG;
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if (data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 &&
        data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A) {
        return ImageFormat::PNG;
    }

    // WebP: RIFF ???? WEBP
    if (data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
        data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P') {
        return ImageFormat::WEBP;
    }

    // BMP: 42 4D (BM)
    if (data[0] == 0x42 && data[1] == 0x4D) {
        return ImageFormat::BMP;
    }

    return ImageFormat::UNKNOWN;
}

/**
 * @brief Get format name as string
 */
inline std::string format_name(ImageFormat format) {
    switch (format) {
        case ImageFormat::JPEG: return "JPEG";
        case ImageFormat::PNG: return "PNG";
        case ImageFormat::WEBP: return "WebP";
        case ImageFormat::BMP: return "BMP";
        default: return "Unknown";
    }
}

/**
 * @brief Unified multi-format image decoder with auto-detection
 *
 * This is the main image decoder class. It automatically detects the format
 * from magic bytes and delegates to the appropriate specialized decoder:
 *
 * - JPEG → JPEGDecoder (libjpeg-turbo, SIMD-accelerated)
 * - PNG → PNGDecoder (libpng)
 * - WebP → WebPDecoder (libwebp, SIMD-accelerated) [if available]
 * - BMP → BMPDecoder (native C++)
 *
 * All decoders are thread-safe when each thread has its own ImageDecoder instance.
 *
 * Performance:
 * - Format detection: O(1) magic byte check
 * - Decoder reuse: No initialization overhead
 * - Buffer pooling: Eliminates allocation overhead
 */
class ImageDecoder {
public:
    /**
     * @brief Construct decoder with optional buffer pool
     *
     * @param pool Optional buffer pool for decoded RGB data
     */
    explicit ImageDecoder(BufferPool* pool = nullptr)
        : buffer_pool_(pool),
          jpeg_decoder_(pool),
          png_decoder_(pool),
#ifdef HAVE_WEBP
          webp_decoder_(pool),
#endif
          bmp_decoder_(pool)
    {}

    /**
     * @brief Decode image to RGB (auto-detects format)
     *
     * @param image_data Span of compressed image bytes
     * @param output Output buffer for RGB data (resized automatically)
     * @param width Output: image width
     * @param height Output: image height
     * @param channels Output: number of channels (always 3 for RGB)
     *
     * @throws std::runtime_error on decode failure or unsupported format
     */
    void decode(
        std::span<const uint8_t> image_data,
        std::vector<uint8_t>& output,
        int& width,
        int& height,
        int& channels
    ) {
        // Auto-detect format
        ImageFormat format = detect_format(image_data);

        // Delegate to appropriate decoder
        switch (format) {
            case ImageFormat::JPEG:
                jpeg_decoder_.decode(image_data, output, width, height, channels);
                break;

            case ImageFormat::PNG:
                png_decoder_.decode(image_data, output, width, height, channels);
                break;

#ifdef HAVE_WEBP
            case ImageFormat::WEBP:
                webp_decoder_.decode(image_data, output, width, height, channels);
                break;
#endif

            case ImageFormat::BMP:
                bmp_decoder_.decode(image_data, output, width, height, channels);
                break;

            default:
                throw std::runtime_error("Unsupported image format: " + format_name(format));
        }
    }

    /**
     * @brief Decode image into Sample with pooled buffer
     *
     * @param sample Sample with jpeg_data filled in (from TarReader)
     *
     * @throws std::runtime_error on decode failure
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

        // Auto-detect and decode
        decode(sample.jpeg_data, buffer, sample.width, sample.height, sample.channels);

        // Move buffer into sample
        sample.decoded_rgb = std::move(buffer);
    }

    /**
     * @brief Get format of image data
     *
     * @param image_data Span of image bytes
     * @return Detected format
     */
    ImageFormat get_format(std::span<const uint8_t> image_data) const {
        return detect_format(image_data);
    }

    /**
     * @brief Get version info for all decoders
     */
    std::string version_info() const {
        std::string info = "ImageDecoder (multi-format):\n";
        info += "  - JPEG: " + JPEGDecoder::version_info() + "\n";
        info += "  - PNG: " + PNGDecoder::version_info() + "\n";
#ifdef HAVE_WEBP
        info += "  - WebP: " + WebPDecoder::version_info() + "\n";
#endif
        info += "  - BMP: " + BMPDecoder::version_info();
        return info;
    }

private:
    BufferPool* buffer_pool_;
    JPEGDecoder jpeg_decoder_;
    PNGDecoder png_decoder_;
#ifdef HAVE_WEBP
    WebPDecoder webp_decoder_;
#endif
    BMPDecoder bmp_decoder_;
};

/**
 * @brief Batch image decoder with parallel decoding
 *
 * Decodes multiple images in parallel using worker pool.
 * Each worker has its own decoder instance to avoid contention.
 *
 * Supports all formats via auto-detection.
 */
class BatchImageDecoder {
public:
    /**
     * @brief Construct batch decoder
     *
     * @param num_workers Number of decoder threads
     * @param pool Buffer pool for decoded RGB data
     */
    explicit BatchImageDecoder(size_t num_workers = 4, BufferPool* pool = nullptr)
        : buffer_pool_(pool) {

        // Create decoder for each worker
        for (size_t i = 0; i < num_workers; ++i) {
            decoders_.push_back(std::make_unique<ImageDecoder>(pool));
        }
    }

    /**
     * @brief Decode batch of samples
     *
     * @param batch Batch with image data filled in
     *
     * Decodes samples in parallel using worker pool.
     * Each sample is assigned to a worker based on index.
     */
    void decode_batch(Batch& batch) {
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
    std::vector<std::unique_ptr<ImageDecoder>> decoders_;
};

} // namespace turboloader
