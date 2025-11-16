/**
 * @file multi_gpu_pipeline.hpp
 * @brief Multi-GPU data loading pipeline for TurboLoader
 *
 * Enables efficient data loading across multiple GPUs with CUDA streams
 * and GPU-direct memory transfers for maximum throughput.
 */

#pragma once

#include "../pipeline/pipeline.hpp"
#include <vector>
#include <memory>
#include <string>

namespace turboloader {
namespace gpu {

/**
 * @brief Configuration for multi-GPU pipeline
 */
struct MultiGPUConfig {
    // Data source
    std::string data_path;

    // Pipeline configuration
    size_t batch_size = 32;
    size_t num_workers = 4;
    size_t queue_size = 256;
    bool shuffle = false;

    // GPU configuration
    std::vector<int> gpu_ids;  // GPU device IDs to use (e.g., {0, 1, 2, 3})
    bool pin_memory = true;     // Pin CPU memory for faster GPU transfers
    bool use_cuda_streams = true;  // Use CUDA streams for async transfers
    int prefetch_batches = 2;   // Number of batches to prefetch per GPU
};

/**
 * @brief Multi-GPU data loading pipeline
 *
 * Distributes data loading across multiple GPUs with:
 * - Per-GPU pinned memory buffers
 * - Async CUDA streams for overlapped data transfer
 * - Round-robin or custom data distribution strategies
 */
class MultiGPUPipeline {
public:
    /**
     * @brief Initialize multi-GPU pipeline
     */
    explicit MultiGPUPipeline(const MultiGPUConfig& config);

    /**
     * @brief Clean up GPU resources
     */
    ~MultiGPUPipeline();

    /**
     * @brief Start the pipeline
     */
    void start();

    /**
     * @brief Stop the pipeline
     */
    void stop();

    /**
     * @brief Get next batch for a specific GPU
     *
     * @param gpu_id GPU device ID
     * @return Batch of samples on the specified GPU
     */
    Batch next_batch(int gpu_id);

    /**
     * @brief Get next batches for all GPUs
     *
     * @return Vector of batches, one per GPU (in order of gpu_ids)
     */
    std::vector<Batch> next_batch_all();

    /**
     * @brief Check if pipeline is finished
     */
    bool is_finished() const;

    /**
     * @brief Get number of GPUs
     */
    size_t num_gpus() const { return config_.gpu_ids.size(); }

    /**
     * @brief Get GPU device IDs
     */
    const std::vector<int>& gpu_ids() const { return config_.gpu_ids; }

private:
    MultiGPUConfig config_;

    // Per-GPU pipelines
    std::vector<std::unique_ptr<UnifiedPipeline>> pipelines_;

    // CUDA resources (opaque pointers)
    std::vector<void*> cuda_streams_;  // cudaStream_t per GPU
    std::vector<void*> pinned_buffers_;  // Pinned memory buffers

    bool initialized_ = false;

    // Initialize CUDA resources
    void init_cuda_resources();

    // Cleanup CUDA resources
    void cleanup_cuda_resources();

    // Transfer batch to GPU
    void transfer_to_gpu(const Batch& cpu_batch, Batch& gpu_batch, int gpu_id);
};

} // namespace gpu
} // namespace turboloader
