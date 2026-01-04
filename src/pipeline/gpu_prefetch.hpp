/**
 * @file gpu_prefetch.hpp
 * @brief GPU memory prefetcher with double-buffering (v2.23.0)
 *
 * Provides asynchronous GPU prefetching to hide data transfer latency.
 * Uses double-buffering: while GPU processes one batch, the next is
 * being transferred in the background.
 *
 * Features:
 * - Asynchronous host→GPU transfers using CUDA streams
 * - Double-buffering for overlap with computation
 * - Automatic pinned memory management
 * - Works with or without CUDA (CPU fallback)
 *
 * Usage:
 * ```cpp
 * GPUPrefetcher prefetcher(batch_size, channels, height, width, device_id);
 *
 * for (auto& batch : data_loader) {
 *     // Start prefetching next batch while current is processed
 *     prefetcher.prefetch_async(next_batch);
 *
 *     // Get current batch (blocks until ready)
 *     auto* gpu_data = prefetcher.get_current();
 *
 *     // Process on GPU
 *     model(gpu_data);
 *
 *     // Swap buffers for next iteration
 *     prefetcher.swap_buffers();
 * }
 * ```
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef TURBOLOADER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace turboloader {

/**
 * @brief GPU buffer for prefetched data
 */
struct GPUPrefetchBuffer {
    float* device_ptr = nullptr;
    float* pinned_ptr = nullptr;
    size_t size_bytes = 0;
    bool ready = false;
    bool in_use = false;

#ifdef TURBOLOADER_HAS_CUDA
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
#endif
};

/**
 * @brief GPU prefetcher with double-buffering
 *
 * Manages two GPU buffers for overlapping data transfer with computation.
 * While one buffer is being used for processing, the other is filled
 * with the next batch asynchronously.
 */
class GPUPrefetcher {
public:
    /**
     * @brief Create GPU prefetcher
     * @param batch_size Maximum batch size
     * @param channels Number of image channels (3 for RGB)
     * @param height Image height
     * @param width Image width
     * @param device_id CUDA device ID (-1 for CPU fallback)
     * @param num_buffers Number of buffers for prefetching (default 2)
     */
    GPUPrefetcher(size_t batch_size, size_t channels, size_t height, size_t width,
                   int device_id = 0, size_t num_buffers = 2)
        : batch_size_(batch_size),
          channels_(channels),
          height_(height),
          width_(width),
          device_id_(device_id),
          current_buffer_(0),
          prefetch_buffer_(1),
          use_gpu_(device_id >= 0) {

        if (num_buffers < 2) {
            throw std::invalid_argument("num_buffers must be >= 2");
        }

        // Calculate buffer size
        elements_per_sample_ = channels * height * width;
        size_t total_elements = batch_size * elements_per_sample_;
        size_t size_bytes = total_elements * sizeof(float);

        // Allocate buffers
        buffers_.resize(num_buffers);

#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to set CUDA device: " +
                    std::string(cudaGetErrorString(err)));
            }

            for (auto& buf : buffers_) {
                buf.size_bytes = size_bytes;

                // Allocate device memory
                err = cudaMalloc(&buf.device_ptr, size_bytes);
                if (err != cudaSuccess) {
                    cleanup();
                    throw std::runtime_error("Failed to allocate GPU memory: " +
                        std::string(cudaGetErrorString(err)));
                }

                // Allocate pinned host memory for fast transfers
                err = cudaMallocHost(&buf.pinned_ptr, size_bytes);
                if (err != cudaSuccess) {
                    cleanup();
                    throw std::runtime_error("Failed to allocate pinned memory: " +
                        std::string(cudaGetErrorString(err)));
                }

                // Create stream and event for this buffer
                err = cudaStreamCreate(&buf.stream);
                if (err != cudaSuccess) {
                    cleanup();
                    throw std::runtime_error("Failed to create CUDA stream: " +
                        std::string(cudaGetErrorString(err)));
                }

                err = cudaEventCreate(&buf.event);
                if (err != cudaSuccess) {
                    cleanup();
                    throw std::runtime_error("Failed to create CUDA event: " +
                        std::string(cudaGetErrorString(err)));
                }
            }
        } else
#endif
        {
            // CPU fallback - just allocate aligned memory
            for (auto& buf : buffers_) {
                buf.size_bytes = size_bytes;
                // Round up to multiple of 64 for aligned_alloc
                size_t aligned_size = ((size_bytes + 63) / 64) * 64;
                if (aligned_size == 0) aligned_size = 64;  // Minimum allocation
                buf.pinned_ptr = static_cast<float*>(std::aligned_alloc(64, aligned_size));
                if (!buf.pinned_ptr) {
                    cleanup();
                    throw std::runtime_error("Failed to allocate aligned memory");
                }
                buf.device_ptr = buf.pinned_ptr;  // Same pointer for CPU
            }
        }

        initialized_ = true;
    }

    ~GPUPrefetcher() {
        cleanup();
    }

    // Non-copyable
    GPUPrefetcher(const GPUPrefetcher&) = delete;
    GPUPrefetcher& operator=(const GPUPrefetcher&) = delete;

    /**
     * @brief Start asynchronous prefetch of data to GPU
     * @param data Source data pointer (CPU memory)
     * @param num_samples Number of samples in batch
     */
    void prefetch_async(const float* data, size_t num_samples) {
        if (!initialized_) return;
        if (num_samples > batch_size_) {
            throw std::invalid_argument("num_samples exceeds batch_size");
        }

        auto& buf = buffers_[prefetch_buffer_];
        size_t copy_bytes = num_samples * elements_per_sample_ * sizeof(float);

#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            // Wait for any previous transfer on this buffer to complete
            cudaStreamSynchronize(buf.stream);

            // Copy to pinned memory
            std::memcpy(buf.pinned_ptr, data, copy_bytes);

            // Async transfer to GPU
            cudaMemcpyAsync(buf.device_ptr, buf.pinned_ptr, copy_bytes,
                           cudaMemcpyHostToDevice, buf.stream);

            // Record event for synchronization
            cudaEventRecord(buf.event, buf.stream);
        } else
#endif
        {
            // CPU: just copy to buffer
            std::memcpy(buf.pinned_ptr, data, copy_bytes);
        }

        buf.ready = true;
        current_samples_ = num_samples;
    }

    /**
     * @brief Prefetch from vector of uint8_t data (with normalization)
     * @param data Source data (HWC uint8_t format)
     * @param num_samples Number of samples
     * @param mean Per-channel mean for normalization
     * @param std Per-channel std for normalization
     */
    void prefetch_async(const std::vector<uint8_t>& data, size_t num_samples,
                        const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
                        const std::vector<float>& std = {0.229f, 0.224f, 0.225f}) {
        if (!initialized_) return;

        auto& buf = buffers_[prefetch_buffer_];
        size_t pixels_per_sample = height_ * width_;

#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            cudaStreamSynchronize(buf.stream);
        }
#endif

        // Convert uint8_t HWC to float CHW with normalization
        for (size_t n = 0; n < num_samples; ++n) {
            for (size_t c = 0; c < channels_; ++c) {
                for (size_t h = 0; h < height_; ++h) {
                    for (size_t w = 0; w < width_; ++w) {
                        size_t src_idx = n * channels_ * height_ * width_ +
                                         h * width_ * channels_ + w * channels_ + c;
                        size_t dst_idx = n * elements_per_sample_ +
                                         c * pixels_per_sample + h * width_ + w;

                        float val = data[src_idx] / 255.0f;
                        val = (val - mean[c]) / std[c];
                        buf.pinned_ptr[dst_idx] = val;
                    }
                }
            }
        }

        size_t copy_bytes = num_samples * elements_per_sample_ * sizeof(float);

#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            cudaMemcpyAsync(buf.device_ptr, buf.pinned_ptr, copy_bytes,
                           cudaMemcpyHostToDevice, buf.stream);
            cudaEventRecord(buf.event, buf.stream);
        }
#endif

        buf.ready = true;
        current_samples_ = num_samples;
    }

    /**
     * @brief Get current buffer's device pointer (blocks until ready)
     */
    float* get_current() {
        if (!initialized_) return nullptr;

        auto& buf = buffers_[current_buffer_];

#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            // Wait for transfer to complete
            cudaEventSynchronize(buf.event);
        }
#endif

        buf.in_use = true;
        return buf.device_ptr;
    }

    /**
     * @brief Get current buffer's pinned host pointer
     */
    float* get_current_host() {
        if (!initialized_) return nullptr;
        return buffers_[current_buffer_].pinned_ptr;
    }

    /**
     * @brief Swap current and prefetch buffers
     */
    void swap_buffers() {
        if (!initialized_) return;

        buffers_[current_buffer_].in_use = false;
        std::swap(current_buffer_, prefetch_buffer_);
    }

    /**
     * @brief Wait for all pending transfers to complete
     */
    void synchronize() {
#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            for (auto& buf : buffers_) {
                cudaStreamSynchronize(buf.stream);
            }
        }
#endif
    }

    /**
     * @brief Check if prefetch buffer is ready
     */
    bool prefetch_ready() const {
        return buffers_[prefetch_buffer_].ready;
    }

    // Getters
    size_t batch_size() const { return batch_size_; }
    size_t channels() const { return channels_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }
    int device_id() const { return device_id_; }
    bool using_gpu() const { return use_gpu_; }
    size_t current_samples() const { return current_samples_; }
    size_t num_buffers() const { return buffers_.size(); }

    /**
     * @brief Get total GPU memory allocated (bytes)
     */
    size_t gpu_memory_usage() const {
        if (!use_gpu_) return 0;
        size_t total = 0;
        for (const auto& buf : buffers_) {
            total += buf.size_bytes;
        }
        return total;
    }

    /**
     * @brief Get total pinned memory allocated (bytes)
     */
    size_t pinned_memory_usage() const {
        size_t total = 0;
        for (const auto& buf : buffers_) {
            total += buf.size_bytes;
        }
        return total;
    }

private:
    void cleanup() {
#ifdef TURBOLOADER_HAS_CUDA
        if (use_gpu_) {
            for (auto& buf : buffers_) {
                if (buf.event) cudaEventDestroy(buf.event);
                if (buf.stream) cudaStreamDestroy(buf.stream);
                if (buf.pinned_ptr) cudaFreeHost(buf.pinned_ptr);
                if (buf.device_ptr) cudaFree(buf.device_ptr);
            }
        } else
#endif
        {
            for (auto& buf : buffers_) {
                if (buf.pinned_ptr) {
                    std::free(buf.pinned_ptr);
                }
            }
        }
        buffers_.clear();
        initialized_ = false;
    }

    size_t batch_size_;
    size_t channels_;
    size_t height_;
    size_t width_;
    int device_id_;
    bool use_gpu_;
    bool initialized_ = false;

    size_t elements_per_sample_;
    size_t current_samples_ = 0;

    std::vector<GPUPrefetchBuffer> buffers_;
    size_t current_buffer_;
    size_t prefetch_buffer_;
};

/**
 * @brief Triple-buffered prefetcher for even higher throughput
 *
 * Uses three buffers: one being processed, one being transferred,
 * and one being filled with CPU data.
 */
class TripleBufferPrefetcher : public GPUPrefetcher {
public:
    TripleBufferPrefetcher(size_t batch_size, size_t channels, size_t height, size_t width,
                           int device_id = 0)
        : GPUPrefetcher(batch_size, channels, height, width, device_id, 3) {}
};

}  // namespace turboloader
