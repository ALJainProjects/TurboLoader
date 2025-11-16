#pragma once

#include "turboloader/decoders/image_decoder.hpp"
#include <memory>
#include <vector>
#include <span>

namespace turboloader {

/**
 * GPU-accelerated JPEG decoder using NVIDIA nvJPEG
 *
 * Features:
 * - Hardware-accelerated JPEG decoding on NVIDIA GPUs
 * - Batch decoding for maximum throughput
 * - Zero-copy GPU memory output
 * - Async decode pipeline with CUDA streams
 * - 5-10x faster than CPU decoding for large batches
 *
 * Requirements:
 * - NVIDIA GPU with CUDA capability >= 6.0
 * - CUDA Toolkit >= 11.0
 * - nvJPEG library
 */
class GpuJpegDecoder : public ImageDecoder {
public:
    struct Config {
        int device_id{0};              // CUDA device ID
        size_t max_batch_size{32};     // Maximum batch size
        bool use_cuda_stream{true};    // Use CUDA streams for async decode
        bool pinned_memory{true};      // Use pinned host memory for faster transfers
        size_t max_cpu_threads{4};     // CPU threads for nvJPEG preprocessing
    };

    /**
     * Construct GPU decoder with configuration
     * @param config GPU decoder configuration
     */
    explicit GpuJpegDecoder(const Config& config = Config{});
    ~GpuJpegDecoder() override;

    // Non-copyable, movable
    GpuJpegDecoder(const GpuJpegDecoder&) = delete;
    GpuJpegDecoder& operator=(const GpuJpegDecoder&) = delete;
    GpuJpegDecoder(GpuJpegDecoder&&) noexcept;
    GpuJpegDecoder& operator=(GpuJpegDecoder&&) noexcept;

    /**
     * Decode single JPEG image to GPU memory
     * @param jpeg_data JPEG compressed data
     * @return Decoded image (data in GPU memory)
     */
    DecodedImage decode(std::span<const uint8_t> jpeg_data) override;

    /**
     * Batch decode multiple JPEG images to GPU memory (optimized)
     * @param jpeg_batches Vector of JPEG compressed data
     * @return Vector of decoded images (data in GPU memory)
     */
    std::vector<DecodedImage> decode_batch(
        const std::vector<std::span<const uint8_t>>& jpeg_batches);

    /**
     * Check if data can be decoded (is JPEG)
     */
    bool can_decode(std::span<const uint8_t> data) const override;

    /**
     * Get decoder name
     */
    std::string name() const override { return "gpu_jpeg"; }

    /**
     * Get CUDA device pointer for decoded image
     * @param image Decoded image from decode()
     * @return GPU memory pointer
     */
    void* get_device_ptr(const DecodedImage& image) const;

    /**
     * Synchronize CUDA stream (wait for decode to complete)
     */
    void synchronize();

    /**
     * Check if GPU decoder is available on this system
     */
    static bool is_available();

    /**
     * Get number of available CUDA devices
     */
    static int get_device_count();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * GPU memory buffer for decoded images
 *
 * Manages GPU memory lifecycle with RAII
 */
class GpuImageBuffer {
public:
    GpuImageBuffer(int device_id, size_t width, size_t height, size_t channels);
    ~GpuImageBuffer();

    // Non-copyable, movable
    GpuImageBuffer(const GpuImageBuffer&) = delete;
    GpuImageBuffer& operator=(const GpuImageBuffer&) = delete;
    GpuImageBuffer(GpuImageBuffer&&) noexcept;
    GpuImageBuffer& operator=(GpuImageBuffer&&) noexcept;

    void* device_ptr() const { return device_ptr_; }
    size_t size() const { return size_; }
    int device_id() const { return device_id_; }

    /**
     * Copy data from GPU to CPU
     */
    std::vector<uint8_t> to_cpu() const;

    /**
     * Copy data from CPU to GPU
     */
    void from_cpu(std::span<const uint8_t> data);

private:
    void* device_ptr_{nullptr};
    size_t size_{0};
    int device_id_{0};
};

}  // namespace turboloader
