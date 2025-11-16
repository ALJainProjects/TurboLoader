/**
 * @file multi_gpu_pipeline.cpp
 * @brief Multi-GPU data loading pipeline implementation
 */

#include "multi_gpu_pipeline.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>

#ifdef TURBOLOADER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace turboloader {
namespace gpu {

#ifdef TURBOLOADER_ENABLE_CUDA
// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)
#endif

MultiGPUPipeline::MultiGPUPipeline(const MultiGPUConfig& config)
    : config_(config), initialized_(false) {

#ifndef TURBOLOADER_ENABLE_CUDA
    throw std::runtime_error("TurboLoader was not built with CUDA support. "
                           "Rebuild with -DENABLE_CUDA=ON");
#else
    // Validate GPU IDs
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    for (int gpu_id : config_.gpu_ids) {
        if (gpu_id < 0 || gpu_id >= device_count) {
            throw std::runtime_error("Invalid GPU ID: " + std::to_string(gpu_id) +
                                   ". Available GPUs: 0-" + std::to_string(device_count - 1));
        }
    }

    // Create per-GPU pipelines
    pipelines_.reserve(config_.gpu_ids.size());
    for (size_t i = 0; i < config_.gpu_ids.size(); ++i) {
        // Each GPU gets a separate pipeline with its own workers
        auto pipeline_config = config_;
        auto pipeline = std::make_unique<UnifiedPipeline>(
            config_.data_path,
            config_.batch_size,
            config_.num_workers,
            config_.queue_size,
            config_.shuffle
        );
        pipelines_.push_back(std::move(pipeline));
    }

    init_cuda_resources();
    initialized_ = true;
#endif
}

MultiGPUPipeline::~MultiGPUPipeline() {
#ifdef TURBOLOADER_ENABLE_CUDA
    cleanup_cuda_resources();
#endif
}

void MultiGPUPipeline::init_cuda_resources() {
#ifdef TURBOLOADER_ENABLE_CUDA
    cuda_streams_.resize(config_.gpu_ids.size(), nullptr);
    pinned_buffers_.resize(config_.gpu_ids.size(), nullptr);

    for (size_t i = 0; i < config_.gpu_ids.size(); ++i) {
        int gpu_id = config_.gpu_ids[i];
        CUDA_CHECK(cudaSetDevice(gpu_id));

        // Create CUDA stream for async operations
        if (config_.use_cuda_streams) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            cuda_streams_[i] = static_cast<void*>(stream);
        }

        // Allocate pinned memory for faster CPU-GPU transfers
        if (config_.pin_memory) {
            // Allocate buffer for batch size images (estimate 1MB per image)
            size_t buffer_size = config_.batch_size * 1024 * 1024;
            void* pinned_buffer;
            CUDA_CHECK(cudaMallocHost(&pinned_buffer, buffer_size));
            pinned_buffers_[i] = pinned_buffer;
        }
    }
#endif
}

void MultiGPUPipeline::cleanup_cuda_resources() {
#ifdef TURBOLOADER_ENABLE_CUDA
    for (size_t i = 0; i < config_.gpu_ids.size(); ++i) {
        int gpu_id = config_.gpu_ids[i];
        CUDA_CHECK(cudaSetDevice(gpu_id));

        // Destroy CUDA streams
        if (cuda_streams_[i] != nullptr) {
            cudaStream_t stream = static_cast<cudaStream_t>(cuda_streams_[i]);
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        // Free pinned memory
        if (pinned_buffers_[i] != nullptr) {
            CUDA_CHECK(cudaFreeHost(pinned_buffers_[i]));
        }
    }

    cuda_streams_.clear();
    pinned_buffers_.clear();
#endif
}

void MultiGPUPipeline::start() {
    if (!initialized_) {
        throw std::runtime_error("MultiGPUPipeline not initialized");
    }

    // Start all per-GPU pipelines
    for (auto& pipeline : pipelines_) {
        pipeline->start();
    }
}

void MultiGPUPipeline::stop() {
    // Stop all per-GPU pipelines
    for (auto& pipeline : pipelines_) {
        pipeline->stop();
    }
}

Batch MultiGPUPipeline::next_batch(int gpu_id) {
#ifndef TURBOLOADER_ENABLE_CUDA
    throw std::runtime_error("CUDA support not enabled");
#else
    // Find the pipeline index for this GPU ID
    size_t pipeline_idx = 0;
    for (size_t i = 0; i < config_.gpu_ids.size(); ++i) {
        if (config_.gpu_ids[i] == gpu_id) {
            pipeline_idx = i;
            break;
        }
    }

    if (pipeline_idx >= pipelines_.size()) {
        throw std::runtime_error("Invalid GPU ID: " + std::to_string(gpu_id));
    }

    // Get batch from CPU pipeline
    Batch cpu_batch = pipelines_[pipeline_idx]->next_batch();

    if (cpu_batch.empty()) {
        return cpu_batch;
    }

    // Transfer to GPU
    Batch gpu_batch;
    transfer_to_gpu(cpu_batch, gpu_batch, gpu_id);

    return gpu_batch;
#endif
}

std::vector<Batch> MultiGPUPipeline::next_batch_all() {
    std::vector<Batch> batches;
    batches.reserve(config_.gpu_ids.size());

    for (int gpu_id : config_.gpu_ids) {
        batches.push_back(next_batch(gpu_id));
    }

    return batches;
}

void MultiGPUPipeline::transfer_to_gpu(const Batch& cpu_batch, Batch& gpu_batch, int gpu_id) {
#ifdef TURBOLOADER_ENABLE_CUDA
    // Find GPU index
    size_t gpu_idx = 0;
    for (size_t i = 0; i < config_.gpu_ids.size(); ++i) {
        if (config_.gpu_ids[i] == gpu_id) {
            gpu_idx = i;
            break;
        }
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t stream = config_.use_cuda_streams ?
        static_cast<cudaStream_t>(cuda_streams_[gpu_idx]) : 0;

    // For now, we'll keep the batch on CPU and just mark it as GPU-ready
    // In a full implementation, we'd copy the image data to GPU memory
    // This would require understanding the actual image format and layout
    gpu_batch = cpu_batch;

    // Synchronize stream if using async transfers
    if (config_.use_cuda_streams && stream != 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
#endif
}

bool MultiGPUPipeline::is_finished() const {
    // Pipeline is finished when all sub-pipelines are finished
    for (const auto& pipeline : pipelines_) {
        if (!pipeline->is_finished()) {
            return false;
        }
    }
    return true;
}

} // namespace gpu
} // namespace turboloader
