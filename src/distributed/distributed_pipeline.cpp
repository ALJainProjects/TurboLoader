#include "turboloader/distributed/distributed_pipeline.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cstdlib>

#ifdef TURBOLOADER_WITH_NCCL
#include <nccl.h>
#include <cuda_runtime.h>
#endif

#ifdef TURBOLOADER_WITH_GLOO
#include <gloo/context.h>
#include <gloo/rendezvous/context.h>
#include <gloo/transport/tcp/device.h>
#endif

namespace turboloader {
namespace distributed {

// Global distributed state
static struct {
    bool initialized{false};
    DistributedPipeline::Backend backend;
    int rank{0};
    int world_size{1};
#ifdef TURBOLOADER_WITH_NCCL
    ncclComm_t nccl_comm;
#endif
#ifdef TURBOLOADER_WITH_GLOO
    std::shared_ptr<gloo::Context> gloo_context;
#endif
} g_dist_state;

// DistributedPipeline implementation
struct DistributedPipeline::Impl {
    std::unique_ptr<Pipeline> pipeline_;
    DistributedSampler sampler_;
    Config config_;
    std::vector<size_t> shard_indices_;
    size_t current_index_{0};

    Impl(const std::vector<std::string>& tar_paths, const Config& config)
        : config_(config),
          sampler_(0, config.rank, config.world_size, config.shuffle) {

        if (!g_dist_state.initialized) {
            throw std::runtime_error("Distributed not initialized. Call DistributedPipeline::init() first");
        }

        // Create underlying pipeline
        Pipeline::Config pipeline_config{
            .num_workers = config.num_workers,
            .queue_size = config.queue_size,
            .prefetch_factor = config.prefetch_factor,
            .shuffle = false,  // We handle shuffling via sampler
            .decode_jpeg = config.decode_jpeg
        };

        pipeline_ = std::make_unique<Pipeline>(tar_paths, pipeline_config);

        // Get total samples and create sampler
        size_t total = pipeline_->total_samples();
        sampler_ = DistributedSampler(total, config.rank, config.world_size, config.shuffle);
        shard_indices_ = sampler_.get_indices();
    }

    void start() {
        pipeline_->start();
        current_index_ = 0;
    }

    void stop() {
        pipeline_->stop();
    }

    std::vector<Sample> next_batch(size_t batch_size) {
        std::vector<Sample> batch;
        batch.reserve(batch_size);

        // Get samples from our shard
        while (batch.size() < batch_size && current_index_ < shard_indices_.size()) {
            // Read samples until we find the one at our shard index
            auto sample_batch = pipeline_->next_batch(1);
            if (sample_batch.empty()) break;

            // Check if this sample belongs to our shard
            if (sample_batch[0].index == shard_indices_[current_index_]) {
                batch.push_back(std::move(sample_batch[0]));
                current_index_++;
            }
        }

        return batch;
    }
};

void DistributedPipeline::init(
    Backend backend,
    int rank,
    int world_size,
    const std::string& master_addr,
    int master_port) {

    if (g_dist_state.initialized) {
        throw std::runtime_error("Distributed already initialized");
    }

    g_dist_state.backend = backend;
    g_dist_state.rank = rank;
    g_dist_state.world_size = world_size;

#ifdef TURBOLOADER_WITH_NCCL
    if (backend == Backend::NCCL) {
        // Initialize NCCL
        ncclUniqueId nccl_id;
        if (rank == 0) {
            ncclGetUniqueId(&nccl_id);
            // TODO: Broadcast nccl_id to other ranks via TCP/MPI
        }
        // TODO: Receive nccl_id if rank != 0

        ncclCommInitRank(&g_dist_state.nccl_comm, world_size, nccl_id, rank);
    }
#endif

#ifdef TURBOLOADER_WITH_GLOO
    if (backend == Backend::GLOO) {
        // Initialize Gloo
        auto device = gloo::transport::tcp::CreateDevice("localhost");
        g_dist_state.gloo_context = std::make_shared<gloo::rendezvous::Context>(rank, world_size);
        g_dist_state.gloo_context->connectFullMesh(*device);
    }
#endif

    if (backend == Backend::MPI) {
        throw std::runtime_error("MPI backend not yet implemented");
    }

    g_dist_state.initialized = true;
}

void DistributedPipeline::init_from_env(Backend backend) {
    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");
    const char* master_addr_env = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");

    if (!rank_env || !world_size_env) {
        throw std::runtime_error("RANK and WORLD_SIZE environment variables must be set");
    }

    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);
    std::string master_addr = master_addr_env ? master_addr_env : "localhost";
    int master_port = master_port_env ? std::atoi(master_port_env) : 29500;

    init(backend, rank, world_size, master_addr, master_port);
}

void DistributedPipeline::finalize() {
    if (!g_dist_state.initialized) return;

#ifdef TURBOLOADER_WITH_NCCL
    if (g_dist_state.backend == Backend::NCCL) {
        ncclCommDestroy(g_dist_state.nccl_comm);
    }
#endif

    g_dist_state.initialized = false;
}

DistributedPipeline::DistributedPipeline(
    const std::vector<std::string>& tar_paths,
    const Config& config)
    : pimpl_(std::make_unique<Impl>(tar_paths, config)) {}

DistributedPipeline::~DistributedPipeline() = default;
DistributedPipeline::DistributedPipeline(DistributedPipeline&&) noexcept = default;
DistributedPipeline& DistributedPipeline::operator=(DistributedPipeline&&) noexcept = default;

void DistributedPipeline::start() {
    pimpl_->start();
}

void DistributedPipeline::stop() {
    pimpl_->stop();
}

std::vector<Sample> DistributedPipeline::next_batch(size_t batch_size) {
    return pimpl_->next_batch(batch_size);
}

size_t DistributedPipeline::total_samples() const {
    return pimpl_->sampler_.samples_per_rank();
}

size_t DistributedPipeline::global_total_samples() const {
    return pimpl_->shard_indices_.size() * pimpl_->config_.world_size;
}

void DistributedPipeline::barrier() {
#ifdef TURBOLOADER_WITH_NCCL
    if (g_dist_state.backend == Backend::NCCL) {
        // NCCL doesn't have barrier, use allreduce with dummy data
        int dummy = 0;
        ncclAllReduce(&dummy, &dummy, 1, ncclInt, ncclSum, g_dist_state.nccl_comm, nullptr);
    }
#endif

#ifdef TURBOLOADER_WITH_GLOO
    if (g_dist_state.backend == Backend::GLOO) {
        g_dist_state.gloo_context->barrier();
    }
#endif
}

int DistributedPipeline::rank() const {
    return pimpl_->config_.rank;
}

int DistributedPipeline::world_size() const {
    return pimpl_->config_.world_size;
}

bool DistributedPipeline::is_initialized() {
    return g_dist_state.initialized;
}

// DistributedSampler implementation
DistributedSampler::DistributedSampler(size_t total_samples, int rank, int world_size, bool shuffle)
    : total_samples_(total_samples),
      rank_(rank),
      world_size_(world_size),
      shuffle_(shuffle) {

    // Calculate samples per rank (evenly distributed)
    num_samples_ = (total_samples_ + world_size_ - 1) / world_size_;
}

std::vector<size_t> DistributedSampler::get_indices() const {
    std::vector<size_t> indices;
    indices.reserve(num_samples_);

    // Generate all indices
    std::vector<size_t> all_indices(total_samples_);
    for (size_t i = 0; i < total_samples_; ++i) {
        all_indices[i] = i;
    }

    // Shuffle if enabled
    if (shuffle_) {
        std::mt19937 rng(epoch_);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
    }

    // Shard indices for this rank
    for (size_t i = rank_; i < total_samples_; i += world_size_) {
        indices.push_back(all_indices[i]);
        if (indices.size() >= num_samples_) break;
    }

    // Pad if necessary (to ensure all ranks have same number of samples)
    while (indices.size() < num_samples_) {
        indices.push_back(all_indices[indices.size() % total_samples_]);
    }

    return indices;
}

void DistributedSampler::set_epoch(int epoch) {
    epoch_ = epoch;
}

// MultiGpuTransfer implementation
struct MultiGpuTransfer::Impl {
    Config config;
    void* staging_buffer{nullptr};
    size_t staging_size{0};

    Impl(const Config& cfg) : config(cfg) {
#ifdef TURBOLOADER_WITH_CUDA
        if (config.pin_memory) {
            // Allocate pinned staging buffer
            cudaHostAlloc(&staging_buffer, config.staging_buffer_size, cudaHostAllocDefault);
            staging_size = config.staging_buffer_size;
        }
#endif
    }

    ~Impl() {
#ifdef TURBOLOADER_WITH_CUDA
        if (staging_buffer) {
            cudaFreeHost(staging_buffer);
        }
#endif
    }
};

MultiGpuTransfer::MultiGpuTransfer(const Config& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

MultiGpuTransfer::~MultiGpuTransfer() = default;

std::vector<void*> MultiGpuTransfer::cpu_to_gpu(const std::vector<Sample>& batch, int device_id) {
    std::vector<void*> gpu_ptrs;

#ifdef TURBOLOADER_WITH_CUDA
    cudaSetDevice(device_id);

    for (const auto& sample : batch) {
        size_t size = sample.data.at(".jpg").size();
        void* gpu_ptr = nullptr;
        cudaMalloc(&gpu_ptr, size);
        cudaMemcpy(gpu_ptr, sample.data.at(".jpg").data(), size, cudaMemcpyHostToDevice);
        gpu_ptrs.push_back(gpu_ptr);
    }
#endif

    return gpu_ptrs;
}

std::vector<void*> MultiGpuTransfer::gpu_to_gpu(
    const std::vector<void*>& gpu_ptrs,
    int src_device,
    int dst_device) {

    std::vector<void*> dst_ptrs;

#ifdef TURBOLOADER_WITH_CUDA
    // Check if peer-to-peer is available
    if (pimpl_->config.use_p2p && can_access_peer(src_device, dst_device)) {
        // Use peer-to-peer transfer (fastest)
        cudaSetDevice(dst_device);
        cudaDeviceEnablePeerAccess(src_device, 0);

        for (auto src_ptr : gpu_ptrs) {
            void* dst_ptr = nullptr;
            // Get size (we'd need to track this separately in practice)
            size_t size = 256 * 256 * 3;  // Example size
            cudaMalloc(&dst_ptr, size);
            cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, size);
            dst_ptrs.push_back(dst_ptr);
        }
    } else {
        // Fallback to CPU staging
        // TODO: Implement CPU staging transfer
    }
#endif

    return dst_ptrs;
}

void MultiGpuTransfer::broadcast(std::vector<void*>& gpu_ptrs, size_t size) {
#ifdef TURBOLOADER_WITH_NCCL
    if (g_dist_state.backend == DistributedPipeline::Backend::NCCL) {
        for (auto ptr : gpu_ptrs) {
            ncclBroadcast(ptr, ptr, size, ncclUint8, 0, g_dist_state.nccl_comm, nullptr);
        }
    }
#endif
}

bool MultiGpuTransfer::can_access_peer(int src_device, int dst_device) {
#ifdef TURBOLOADER_WITH_CUDA
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
    return can_access != 0;
#else
    return false;
#endif
}

}  // namespace distributed
}  // namespace turboloader
