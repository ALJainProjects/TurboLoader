#include "turboloader/pipeline/pipeline.hpp"
#include <algorithm>
#include <random>

namespace turboloader {

Pipeline::Pipeline(const std::vector<std::string>& tar_paths, const Config& config)
    : config_(config) {

    // Open all TAR files
    readers_.reserve(tar_paths.size());
    for (const auto& path : tar_paths) {
        auto reader = std::make_unique<TarReader>(path);
        if (!reader->is_open()) {
            throw std::runtime_error("Failed to open TAR file: " + path);
        }
        total_samples_ += reader->num_samples();
        readers_.push_back(std::move(reader));
    }

    // Create thread pool
    thread_pool_ = std::make_unique<ThreadPool>(config_.num_workers);

    // Create output queue
    output_queue_ = std::make_unique<LockFreeSPMCQueue<Sample>>(config_.queue_size);

    // Initialize sample indices
    sample_indices_.resize(total_samples_);
    for (size_t i = 0; i < total_samples_; ++i) {
        sample_indices_[i] = i;
    }

    // Shuffle if requested
    if (config_.shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(sample_indices_.begin(), sample_indices_.end(), g);
    }
}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (running_) {
        return;
    }

    running_ = true;
    current_sample_ = 0;

    // Start reader thread
    reader_thread_ = std::thread([this]() { reader_loop(); });
}

void Pipeline::stop() {
    if (!running_) {
        return;
    }

    running_ = false;

    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    thread_pool_->wait();
}

void Pipeline::reset() {
    stop();

    current_sample_ = 0;

    // Re-shuffle if needed
    if (config_.shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(sample_indices_.begin(), sample_indices_.end(), g);
    }

    // Clear output queue
    while (output_queue_->try_pop()) {}
}

std::vector<Sample> Pipeline::next_batch(size_t batch_size) {
    std::vector<Sample> batch;
    batch.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        auto sample = output_queue_->try_pop();
        if (!sample) {
            break;
        }
        batch.push_back(std::move(*sample));
    }

    return batch;
}

void Pipeline::reader_loop() {
    while (running_) {
        size_t idx = current_sample_.fetch_add(1);

        if (idx >= total_samples_) {
            break;  // Done with epoch
        }

        // Get actual index (accounting for shuffle)
        size_t actual_idx = sample_indices_[idx];

        // Submit to thread pool for processing
        thread_pool_->submit([this, actual_idx]() {
            try {
                Sample sample = load_sample(actual_idx);

                // Push to output queue (spin if full)
                while (running_ && !output_queue_->try_push(std::move(sample))) {
                    std::this_thread::yield();
                }
            } catch (...) {
                // TODO: Error handling
            }
        });
    }
}

Sample Pipeline::load_sample(size_t global_index) {
    // Find which TAR file contains this sample
    size_t current_offset = 0;

    for (const auto& reader : readers_) {
        size_t num_samples = reader->num_samples();

        if (global_index < current_offset + num_samples) {
            // Found the right TAR file
            size_t local_index = global_index - current_offset;
            const auto& tar_sample = reader->get_sample(local_index);

            // Create output sample
            Sample sample;
            sample.index = global_index;

            // Read all files in this sample
            for (const auto& [ext, entry] : tar_sample.files) {
                auto data_span = reader->read_file(entry);

                std::vector<uint8_t> data(data_span.begin(), data_span.end());
                sample.data[ext] = std::move(data);
            }

            return sample;
        }

        current_offset += num_samples;
    }

    throw std::out_of_range("Sample index out of range");
}

}  // namespace turboloader
