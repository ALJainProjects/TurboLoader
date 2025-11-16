#pragma once

#include "turboloader/core/lock_free_queue.hpp"
#include "turboloader/core/thread_safe_queue.hpp"
#include "turboloader/core/memory_pool.hpp"
#include "turboloader/core/thread_pool.hpp"
#include "turboloader/readers/tar_reader.hpp"
#include "turboloader/decoders/jpeg_decoder.hpp"
#include "turboloader/transforms/image_transform.hpp"
#include "turboloader/transforms/simd_transforms.hpp"
#include <atomic>
#include <memory>
#include <vector>

namespace turboloader {

/**
 * Data sample (generic container)
 */
struct Sample {
    std::unordered_map<std::string, std::vector<uint8_t>> data;
    size_t index{0};

    // Decoded image data (if decode_jpeg is enabled)
    int width{0};
    int height{0};
    int channels{0};

    // Transformed image data (if SIMD transforms enabled)
    std::vector<float> transformed_data;  // Float output from SIMD transforms
    bool is_transformed{false};
};

/**
 * High-performance data loading pipeline
 *
 * Architecture:
 *   [TAR Files] -> [Reader Thread] -> [Queue] -> [Worker Threads] -> [Output Queue]
 *
 * Flow:
 * 1. Reader thread reads from TAR files (I/O bound)
 * 2. Pushes raw samples to processing queue
 * 3. Worker threads decode/transform (CPU bound)
 * 4. Pushes processed samples to output queue
 * 5. User iterates output queue
 */
class Pipeline {
public:
    struct Config {
        size_t num_workers{4};
        size_t queue_size{256};
        size_t prefetch_factor{2};
        bool shuffle{false};
        size_t shuffle_buffer_size{1000};
        bool decode_jpeg{false};  // Enable JPEG decoding

        // SIMD Transform options
        bool enable_simd_transforms{false};
        transforms::TransformConfig transform_config{};

        // Legacy transform options (deprecated - use transform_config instead)
        bool enable_resize{false};
        int resize_width{224};
        int resize_height{224};
        bool enable_normalize{false};
    };

    /**
     * Create pipeline for TAR datasets
     * @param tar_paths List of TAR file paths
     * @param config Pipeline configuration
     */
    Pipeline(const std::vector<std::string>& tar_paths, const Config& config);

    ~Pipeline();

    // Non-copyable, non-movable
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * Start pipeline
     */
    void start();

    /**
     * Stop pipeline
     */
    void stop();

    /**
     * Get next batch of samples
     * @param batch_size Number of samples to fetch
     * @return Batch of samples, or empty if no more data
     */
    std::vector<Sample> next_batch(size_t batch_size);

    /**
     * Get total number of samples across all TAR files
     */
    size_t total_samples() const { return total_samples_; }

    /**
     * Reset pipeline for next epoch
     */
    void reset();

private:
    Config config_;
    std::vector<std::unique_ptr<TarReader>> readers_;
    mutable std::vector<std::unique_ptr<std::mutex>> reader_mutexes_;  // One mutex per reader for thread-safe access
    size_t total_samples_{0};

    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<ThreadSafeQueue<Sample>> output_queue_;

    std::atomic<bool> running_{false};
    std::atomic<size_t> current_sample_{0};

    std::vector<size_t> sample_indices_;  // For shuffling

    // SIMD transform pipeline
    std::unique_ptr<transforms::TransformPipeline> transform_pipeline_;

    // Reader thread
    void reader_loop();
    std::thread reader_thread_;

    // Load sample from TAR
    Sample load_sample(size_t global_index);
};

}  // namespace turboloader
