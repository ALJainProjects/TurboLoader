#pragma once

#include "turboloader/pipeline/pipeline.hpp"
#include <memory>
#include <vector>
#include <string>

namespace turboloader {
namespace distributed {

/**
 * Distributed data loading for multi-GPU training
 *
 * Features:
 * - Automatic data sharding across GPUs
 * - NCCL/Gloo backend support for distributed communication
 * - GPU Direct RDMA for fast inter-GPU transfers
 * - Synchronized epoch boundaries across ranks
 * - Drop-in replacement for PyTorch DistributedDataParallel
 *
 * Example usage with PyTorch DDP:
 * ```cpp
 * // Initialize distributed backend
 * DistributedPipeline::init("nccl", rank, world_size);
 *
 * // Create distributed pipeline (automatically shards data)
 * DistributedPipeline pipeline({"/data/train.tar"}, config);
 * pipeline.start();
 *
 * // Each rank gets different samples
 * auto batch = pipeline.next_batch(32);  // Rank 0 gets samples 0-31
 *                                         // Rank 1 gets samples 32-63, etc.
 * ```
 */
class DistributedPipeline {
public:
    enum class Backend {
        NCCL,      // NVIDIA NCCL (GPU-only, fastest)
        GLOO,      // Facebook Gloo (CPU/GPU, more portable)
        MPI        // MPI (fallback)
    };

    struct Config {
        // Pipeline config
        size_t num_workers{4};
        size_t queue_size{256};
        size_t prefetch_factor{2};
        bool decode_jpeg{true};
        bool gpu_decode{false};        // Use GPU JPEG decoder

        // Distributed config
        Backend backend{Backend::NCCL};
        int rank{0};                   // Process rank (0 to world_size-1)
        int world_size{1};             // Total number of processes
        int local_rank{0};             // GPU device ID
        bool shuffle{true};            // Shuffle samples
        bool drop_last{true};          // Drop incomplete final batch

        // Performance config
        bool use_gpu_direct{true};     // Use GPU Direct RDMA (NCCL only)
        bool pin_memory{true};         // Pin host memory for faster GPU transfers
    };

    /**
     * Initialize distributed backend (call once at startup)
     * @param backend Backend type (nccl/gloo/mpi)
     * @param rank Process rank
     * @param world_size Total number of processes
     * @param master_addr Master process address (default: localhost)
     * @param master_port Master process port (default: 29500)
     */
    static void init(
        Backend backend,
        int rank,
        int world_size,
        const std::string& master_addr = "localhost",
        int master_port = 29500
    );

    /**
     * Initialize from environment variables (PyTorch-compatible)
     * Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from env
     */
    static void init_from_env(Backend backend = Backend::NCCL);

    /**
     * Finalize distributed backend (call at shutdown)
     */
    static void finalize();

    /**
     * Construct distributed pipeline
     * @param tar_paths TAR file paths (same across all ranks)
     * @param config Distributed configuration
     */
    DistributedPipeline(const std::vector<std::string>& tar_paths, const Config& config);
    ~DistributedPipeline();

    // Non-copyable, movable
    DistributedPipeline(const DistributedPipeline&) = delete;
    DistributedPipeline& operator=(const DistributedPipeline&) = delete;
    DistributedPipeline(DistributedPipeline&&) noexcept;
    DistributedPipeline& operator=(DistributedPipeline&&) noexcept;

    /**
     * Start the pipeline
     */
    void start();

    /**
     * Stop the pipeline
     */
    void stop();

    /**
     * Get next batch (automatically sharded by rank)
     * @param batch_size Batch size per rank
     * @return Batch of samples for this rank
     */
    std::vector<Sample> next_batch(size_t batch_size);

    /**
     * Get total number of samples visible to this rank
     */
    size_t total_samples() const;

    /**
     * Get global total samples (across all ranks)
     */
    size_t global_total_samples() const;

    /**
     * Synchronize all ranks (barrier)
     */
    void barrier();

    /**
     * Get current rank
     */
    int rank() const;

    /**
     * Get world size
     */
    int world_size() const;

    /**
     * Check if distributed is initialized
     */
    static bool is_initialized();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * Distributed sampler for data sharding
 *
 * Splits dataset indices across multiple ranks
 */
class DistributedSampler {
public:
    DistributedSampler(size_t total_samples, int rank, int world_size, bool shuffle = true);

    /**
     * Get sample indices for this rank
     */
    std::vector<size_t> get_indices() const;

    /**
     * Get number of samples per rank
     */
    size_t samples_per_rank() const { return num_samples_; }

    /**
     * Set epoch (for shuffling)
     */
    void set_epoch(int epoch);

private:
    size_t total_samples_;
    int rank_;
    int world_size_;
    bool shuffle_;
    int epoch_{0};
    size_t num_samples_;  // Samples for this rank
};

/**
 * Multi-GPU data transfer helper
 *
 * Efficiently transfers data between GPUs using:
 * - GPU Direct RDMA (NCCL)
 * - Peer-to-peer GPU transfers
 * - Pinned memory staging
 */
class MultiGpuTransfer {
public:
    struct Config {
        bool use_gpu_direct{true};
        bool use_p2p{true};          // Use peer-to-peer GPU transfers
        bool pin_memory{true};
        size_t staging_buffer_size{256 * 1024 * 1024};  // 256 MB
    };

    MultiGpuTransfer(const Config& config = Config{});
    ~MultiGpuTransfer();

    /**
     * Transfer batch from CPU to GPU
     * @param batch CPU batch
     * @param device_id Target GPU device
     * @return GPU pointers
     */
    std::vector<void*> cpu_to_gpu(const std::vector<Sample>& batch, int device_id);

    /**
     * Transfer batch from one GPU to another
     * @param gpu_ptrs Source GPU pointers
     * @param src_device Source GPU device
     * @param dst_device Destination GPU device
     * @return Destination GPU pointers
     */
    std::vector<void*> gpu_to_gpu(
        const std::vector<void*>& gpu_ptrs,
        int src_device,
        int dst_device
    );

    /**
     * Broadcast batch from rank 0 to all ranks (NCCL)
     */
    void broadcast(std::vector<void*>& gpu_ptrs, size_t size);

    /**
     * Check if peer-to-peer access is available between devices
     */
    static bool can_access_peer(int src_device, int dst_device);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace distributed
}  // namespace turboloader
