/**
 * @file pipeline_v2.hpp
 * @brief High-performance data loading pipeline for v2.0
 *
 * Integrates all v2.0 components into a complete pipeline:
 * - Per-worker TAR readers (eliminates mutex bottleneck)
 * - SIMD-accelerated JPEG decoding (libjpeg-turbo)
 * - Lock-free queues between stages
 * - Object pooling for zero-allocation operation
 *
 * Architecture:
 * ```
 * [Worker 0] --\
 * [Worker 1] ----> [Lock-free Queue] --> [Main Thread] --> Batches
 * [Worker 2] --/
 * [Worker 3]
 * ```
 *
 * Each worker:
 * 1. Reads JPEG samples from its TAR partition (zero-copy mmap)
 * 2. Decodes JPEGs using SIMD-accelerated decoder
 * 3. Pushes decoded samples to lock-free queue
 *
 * Main thread:
 * 1. Pops samples from queue
 * 2. Assembles into batches
 * 3. Returns batches to user
 */

#pragma once

#include "../core_v2/spsc_ring_buffer.hpp"
#include "../core_v2/object_pool.hpp"
#include "../core_v2/sample_v2.hpp"
#include "../io_v2/tar_reader_v2.hpp"
#include "../decode_v2/jpeg_decoder_v2.hpp"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace turboloader {
namespace v2 {

/**
 * @brief Configuration for pipeline
 */
struct PipelineConfig {
    std::string tar_path;           // Path to TAR file
    size_t num_workers = 4;         // Number of worker threads
    size_t batch_size = 32;         // Samples per batch
    size_t queue_size = 256;        // Size of lock-free queues (must be power of 2)
    size_t buffer_pool_size = 128;  // Number of buffers to pool
    bool shuffle = false;           // Enable shuffling (future feature)
    size_t prefetch = 2;            // Number of batches to prefetch
};

/**
 * @brief Worker thread that reads, decodes, and queues samples
 *
 * Each worker:
 * - Has its own TAR reader (no mutex contention)
 * - Has its own JPEG decoder (no mutex contention)
 * - Pushes to dedicated SPSC queue (lock-free)
 */
class DataWorker {
public:
    /**
     * @brief Construct data worker
     *
     * @param config Pipeline configuration
     * @param worker_id Worker ID (0-based)
     * @param buffer_pool Shared buffer pool for decoded RGB data
     */
    DataWorker(
        const PipelineConfig& config,
        size_t worker_id,
        BufferPool* buffer_pool
    ) : config_(config),
        worker_id_(worker_id),
        buffer_pool_(buffer_pool),
        running_(false),
        samples_processed_(0) {

        // Create per-worker TAR reader
        tar_reader_ = std::make_unique<TarReaderV2>(
            config.tar_path,
            worker_id,
            config.num_workers
        );

        // Create per-worker JPEG decoder
        decoder_ = std::make_unique<JPEGDecoderV2>(buffer_pool);

        // Create lock-free queue for this worker
        queue_ = std::make_unique<SPSCRingBuffer<SampleV2, 256>>();
    }

    /**
     * @brief Start worker thread
     */
    void start() {
        running_ = true;
        thread_ = std::thread(&DataWorker::run, this);
    }

    /**
     * @brief Stop worker thread
     */
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    /**
     * @brief Try to pop a sample from worker's queue
     *
     * @param sample Output sample
     * @return true if sample was popped
     */
    bool try_pop(SampleV2& sample) {
        return queue_->try_pop(sample);
    }

    /**
     * @brief Check if worker has finished all samples
     *
     * @return true if worker is done and queue is empty
     */
    bool is_finished() const {
        return !running_ && queue_->empty();
    }

    /**
     * @brief Get number of samples processed
     *
     * @return Samples processed by this worker
     */
    size_t samples_processed() const {
        return samples_processed_.load();
    }

    /**
     * @brief Get number of samples assigned to this worker
     *
     * @return Total samples for this worker
     */
    size_t num_samples() const {
        return tar_reader_->num_samples();
    }

private:
    /**
     * @brief Worker thread main loop
     */
    void run() {
        size_t total_samples = tar_reader_->num_samples();

        for (size_t i = 0; i < total_samples && running_; ++i) {
            // Get zero-copy view of JPEG data from TAR
            auto jpeg_data = tar_reader_->get_sample(i);
            const auto& entry = tar_reader_->get_entry(i);

            // Create sample with zero-copy JPEG data
            SampleV2 sample(entry.index, jpeg_data);

            // Decode JPEG
            try {
                decoder_->decode_sample(sample);
            } catch (const std::exception& e) {
                // Skip corrupted JPEGs
                continue;
            }

            // Push to lock-free queue (wait if full)
            while (running_ && !queue_->try_push(std::move(sample))) {
                std::this_thread::yield();
            }

            samples_processed_++;
        }

        running_ = false;
    }

    PipelineConfig config_;
    size_t worker_id_;
    BufferPool* buffer_pool_;

    std::unique_ptr<TarReaderV2> tar_reader_;
    std::unique_ptr<JPEGDecoderV2> decoder_;
    std::unique_ptr<SPSCRingBuffer<SampleV2, 256>> queue_;

    std::thread thread_;
    std::atomic<bool> running_;
    std::atomic<size_t> samples_processed_;
};

/**
 * @brief Main data loading pipeline
 *
 * Creates worker threads and coordinates batch assembly.
 */
class PipelineV2 {
public:
    /**
     * @brief Construct pipeline
     *
     * @param config Pipeline configuration
     */
    explicit PipelineV2(const PipelineConfig& config)
        : config_(config),
          current_worker_(0) {

        // Create buffer pool
        buffer_pool_ = std::make_unique<BufferPool>(
            256 * 256 * 3,           // Buffer size for 256x256 RGB
            config.buffer_pool_size,  // Pool size
            config.buffer_pool_size * 2  // Max size
        );

        // Create workers
        for (size_t i = 0; i < config.num_workers; ++i) {
            workers_.push_back(std::make_unique<DataWorker>(
                config, i, buffer_pool_.get()
            ));
        }

        // Start all workers
        for (auto& worker : workers_) {
            worker->start();
        }
    }

    /**
     * @brief Destructor - stops all workers
     */
    ~PipelineV2() {
        stop();
    }

    /**
     * @brief Get next batch of samples
     *
     * @return Batch of decoded samples (or empty if done)
     *
     * This method:
     * 1. Collects samples from worker queues (round-robin)
     * 2. Assembles them into a batch
     * 3. Returns when batch is full or all workers are done
     */
    BatchV2 next_batch() {
        BatchV2 batch(config_.batch_size);

        // Collect samples from workers (round-robin)
        size_t failed_attempts = 0;
        const size_t MAX_ATTEMPTS = config_.num_workers * 100;  // More patient waiting

        while (batch.size() < config_.batch_size && failed_attempts < MAX_ATTEMPTS) {
            // Check if all workers are finished
            if (is_finished()) {
                break;
            }

            SampleV2 sample;

            // Try current worker
            if (workers_[current_worker_]->try_pop(sample)) {
                batch.add(std::move(sample));
                failed_attempts = 0;  // Reset on success
            } else {
                failed_attempts++;
                // Move to next worker
                current_worker_ = (current_worker_ + 1) % workers_.size();

                // Only yield if we've tried all workers
                if (failed_attempts % config_.num_workers == 0) {
                    std::this_thread::yield();
                }
            }
        }

        return batch;
    }

    /**
     * @brief Check if pipeline is finished
     *
     * @return true if all workers are done
     */
    bool is_finished() const {
        for (const auto& worker : workers_) {
            if (!worker->is_finished()) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Stop all workers
     */
    void stop() {
        for (auto& worker : workers_) {
            worker->stop();
        }
    }

    /**
     * @brief Get total samples processed
     *
     * @return Total samples processed across all workers
     */
    size_t total_samples_processed() const {
        size_t total = 0;
        for (const auto& worker : workers_) {
            total += worker->samples_processed();
        }
        return total;
    }

    /**
     * @brief Get total samples in dataset
     *
     * @return Total samples across all workers
     */
    size_t total_samples() const {
        size_t total = 0;
        for (const auto& worker : workers_) {
            total += worker->num_samples();
        }
        return total;
    }

private:
    PipelineConfig config_;
    std::unique_ptr<BufferPool> buffer_pool_;
    std::vector<std::unique_ptr<DataWorker>> workers_;
    size_t current_worker_;
};

} // namespace v2
} // namespace turboloader
