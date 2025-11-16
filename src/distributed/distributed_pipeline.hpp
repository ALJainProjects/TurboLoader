/**
 * @file distributed_pipeline.hpp
 * @brief Multi-node distributed data loading for TurboLoader
 *
 * Enables data parallel training across multiple nodes using MPI or TCP.
 * Each node loads a shard of the data with automatic work distribution.
 */

#pragma once

#include "../pipeline/pipeline.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace turboloader {
namespace distributed {

/**
 * @brief Communication backend for distributed coordination
 */
enum class CommBackend {
    MPI,    // MPI-based communication (recommended for HPC clusters)
    TCP,    // TCP-based communication (recommended for cloud environments)
    NCCL    // NVIDIA NCCL (for GPU-heavy workloads)
};

/**
 * @brief Configuration for distributed data loading
 */
struct DistributedConfig {
    // Data source
    std::string data_path;

    // Pipeline configuration
    size_t batch_size = 32;
    size_t num_workers = 4;
    size_t queue_size = 256;
    bool shuffle = false;

    // Distributed settings
    CommBackend backend = CommBackend::TCP;
    int world_rank = 0;      // Rank of this process (0 to world_size-1)
    int world_size = 1;      // Total number of processes
    std::string master_addr = "localhost";  // Address of rank 0
    int master_port = 29500; // Port for coordination

    // Sharding strategy
    bool drop_last = false;  // Drop incomplete batches at end
    int seed = 42;           // Seed for shuffling (same across all ranks)
};

/**
 * @brief Distributed data loading pipeline
 *
 * Coordinates data loading across multiple nodes/processes.
 * Each rank loads a disjoint subset of the data.
 */
class DistributedPipeline {
public:
    /**
     * @brief Initialize distributed pipeline
     */
    explicit DistributedPipeline(const DistributedConfig& config);

    /**
     * @brief Clean up resources
     */
    ~DistributedPipeline();

    /**
     * @brief Start the pipeline
     */
    void start();

    /**
     * @brief Stop the pipeline
     */
    void stop();

    /**
     * @brief Get next batch (rank-specific shard)
     */
    Batch next_batch();

    /**
     * @brief Check if pipeline is finished
     */
    bool is_finished() const;

    /**
     * @brief Get this rank's ID
     */
    int get_rank() const { return config_.world_rank; }

    /**
     * @brief Get total number of ranks
     */
    int get_world_size() const { return config_.world_size; }

    /**
     * @brief Barrier synchronization across all ranks
     */
    void barrier();

    /**
     * @brief Broadcast metadata from rank 0 to all ranks
     */
    void broadcast_metadata();

private:
    DistributedConfig config_;
    std::unique_ptr<UnifiedPipeline> pipeline_;

    // Sharding state
    size_t total_samples_ = 0;
    size_t samples_per_rank_ = 0;
    size_t start_index_ = 0;
    size_t end_index_ = 0;

    // Communication state
    void* comm_context_ = nullptr;  // MPI communicator or TCP socket
    bool initialized_ = false;

    // Initialize communication backend
    void init_backend();

    // Calculate this rank's data shard
    void calculate_shard();

    // Cleanup communication
    void cleanup_backend();
};

/**
 * @brief Helper to initialize distributed environment from env vars
 *
 * Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from environment.
 * Compatible with PyTorch's distributed.launch.
 */
DistributedConfig init_distributed_from_env();

} // namespace distributed
} // namespace turboloader
