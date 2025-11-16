/**
 * @file distributed_pipeline.cpp
 * @brief Multi-node distributed data loading implementation
 */

#include "distributed_pipeline.hpp"
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef TURBOLOADER_ENABLE_MPI
#include <mpi.h>
#endif

namespace turboloader {
namespace distributed {

DistributedPipeline::DistributedPipeline(const DistributedConfig& config)
    : config_(config), comm_context_(nullptr), initialized_(false) {

#ifndef TURBOLOADER_ENABLE_MPI
    if (config_.backend == CommBackend::MPI) {
        throw std::runtime_error("TurboLoader was not built with MPI support. "
                               "Rebuild with -DENABLE_MPI=ON");
    }
#endif

    init_backend();
    calculate_shard();

    // Create underlying pipeline for this rank's shard
    pipeline_ = std::make_unique<UnifiedPipeline>(
        config_.data_path,
        config_.batch_size,
        config_.num_workers,
        config_.queue_size,
        config_.shuffle
    );

    initialized_ = true;
}

DistributedPipeline::~DistributedPipeline() {
    cleanup_backend();
}

void DistributedPipeline::init_backend() {
#ifdef TURBOLOADER_ENABLE_MPI
    if (config_.backend == CommBackend::MPI) {
        // MPI should already be initialized by the user's application
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("MPI not initialized. Call MPI_Init before creating DistributedPipeline");
        }

        // Store MPI communicator
        MPI_Comm* comm = new MPI_Comm(MPI_COMM_WORLD);
        comm_context_ = static_cast<void*>(comm);

        // Get rank and size from MPI
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Update config with MPI values
        config_.world_rank = rank;
        config_.world_size = size;
    }
#endif

    // For TCP and NCCL backends, we rely on config values being set correctly
    if (config_.world_rank < 0 || config_.world_rank >= config_.world_size) {
        throw std::runtime_error("Invalid rank/size configuration");
    }
}

void DistributedPipeline::cleanup_backend() {
#ifdef TURBOLOADER_ENABLE_MPI
    if (config_.backend == CommBackend::MPI && comm_context_ != nullptr) {
        delete static_cast<MPI_Comm*>(comm_context_);
        comm_context_ = nullptr;
    }
#endif
}

void DistributedPipeline::calculate_shard() {
    // For now, we'll use a simple round-robin sharding strategy
    // In production, this would need to determine total_samples from the data source

    // Estimate total samples - this is simplified
    // In reality, we'd need to scan the dataset or get metadata
    total_samples_ = 10000;  // Placeholder

    samples_per_rank_ = total_samples_ / config_.world_size;
    if (!config_.drop_last && config_.world_rank == config_.world_size - 1) {
        // Last rank gets remainder
        samples_per_rank_ += total_samples_ % config_.world_size;
    }

    start_index_ = config_.world_rank * samples_per_rank_;
    end_index_ = start_index_ + samples_per_rank_;
}

void DistributedPipeline::start() {
    if (!initialized_) {
        throw std::runtime_error("DistributedPipeline not initialized");
    }
    pipeline_->start();
}

void DistributedPipeline::stop() {
    if (pipeline_) {
        pipeline_->stop();
    }
}

Batch DistributedPipeline::next_batch() {
    if (!pipeline_) {
        return Batch();
    }

    // Get next batch from this rank's pipeline
    return pipeline_->next_batch();
}

bool DistributedPipeline::is_finished() const {
    if (!pipeline_) {
        return true;
    }
    return pipeline_->is_finished();
}

void DistributedPipeline::barrier() {
#ifdef TURBOLOADER_ENABLE_MPI
    if (config_.backend == CommBackend::MPI) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
#endif

    // For other backends, this would need custom implementation
    // For now, no-op for TCP/NCCL
}

void DistributedPipeline::broadcast_metadata() {
#ifdef TURBOLOADER_ENABLE_MPI
    if (config_.backend == CommBackend::MPI) {
        // Broadcast total_samples from rank 0
        MPI_Bcast(&total_samples_, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        return;
    }
#endif

    // For other backends, this would need custom implementation
}

DistributedConfig init_distributed_from_env() {
    DistributedConfig config;

    // Read environment variables
    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");
    const char* master_addr_env = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");

    if (rank_env) {
        config.world_rank = std::atoi(rank_env);
    }

    if (world_size_env) {
        config.world_size = std::atoi(world_size_env);
    }

    if (master_addr_env) {
        config.master_addr = std::string(master_addr_env);
    }

    if (master_port_env) {
        config.master_port = std::atoi(master_port_env);
    }

    return config;
}

} // namespace distributed
} // namespace turboloader
