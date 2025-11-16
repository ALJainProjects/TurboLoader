#include "turboloader/pipeline/pipeline.hpp"
#include <algorithm>
#include <random>
#include <thread>
#include <iostream>

namespace turboloader {

Pipeline::Pipeline(const std::vector<std::string>& tar_paths, const Config& config)
    : config_(config) {

    // Open all TAR files
    readers_.reserve(tar_paths.size());
    reader_mutexes_.reserve(tar_paths.size());
    for (const auto& path : tar_paths) {
        auto reader = std::make_unique<TarReader>(path);
        if (!reader->is_open()) {
            throw std::runtime_error("Failed to open TAR file: " + path);
        }
        total_samples_ += reader->num_samples();
        readers_.push_back(std::move(reader));
        reader_mutexes_.push_back(std::make_unique<std::mutex>());
    }

    // Create thread pool
    thread_pool_ = std::make_unique<ThreadPool>(config_.num_workers);

    // Create output queue (using ThreadSafeQueue for Sample to avoid move-related races)
    output_queue_ = std::make_unique<ThreadSafeQueue<Sample>>(config_.queue_size);

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

    // Create SIMD transform pipeline if enabled
    if (config_.enable_simd_transforms) {
        transform_pipeline_ = std::make_unique<transforms::TransformPipeline>(config_.transform_config);
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
        // Try non-blocking pop first
        auto sample = output_queue_->try_pop();

        if (!sample) {
            // Queue is empty - no more data available
            break;
        }

        batch.push_back(std::move(*sample));
    }

    return batch;
}

void Pipeline::reader_loop() {
    // Submit all work to thread pool
    while (running_) {
        size_t idx = current_sample_.fetch_add(1);

        if (idx >= total_samples_) {
            break;  // Done submitting
        }

        // Get actual index (accounting for shuffle)
        size_t actual_idx = sample_indices_[idx];

        // Submit to thread pool for processing
        thread_pool_->submit([this, actual_idx, idx]() {
            // Each thread gets its own decoder and transforms
            static thread_local JpegDecoder decoder;
            static thread_local std::unique_ptr<ResizeTransform> resize_transform;
            static thread_local std::unique_ptr<NormalizeTransform> normalize_transform;

            // Lazy init legacy transforms
            if (config_.enable_resize && !resize_transform) {
                resize_transform = std::make_unique<ResizeTransform>(
                    config_.resize_width, config_.resize_height);
            }
            if (config_.enable_normalize && !normalize_transform) {
                normalize_transform = std::make_unique<NormalizeTransform>();
            }

            try {
                Sample sample = load_sample(actual_idx);

                // Decode JPEG if requested
                if (config_.decode_jpeg) {
                    for (auto& [ext, data] : sample.data) {
                        if (ext == "jpg" && JpegDecoder::is_jpeg(data)) {
                            auto decoded = decoder.decode(data);
                            sample.width = decoded.width;
                            sample.height = decoded.height;
                            sample.channels = decoded.channels;
                            data = std::move(decoded.data);
                            break;  // Only decode first JPEG
                        }
                    }
                }

                // Apply SIMD transforms if enabled
                if (config_.enable_simd_transforms && sample.width > 0 && transform_pipeline_) {
                    auto& data = sample.data["jpg"];

                    // Get output dimensions
                    int out_w, out_h;
                    transform_pipeline_->get_output_dims(out_w, out_h);

                    // Allocate output buffer
                    size_t out_size = out_w * out_h * sample.channels;
                    sample.transformed_data.resize(out_size);

                    // Apply all transforms in one pass
                    transform_pipeline_->transform(
                        data.data(), sample.width, sample.height, sample.channels,
                        sample.transformed_data.data()
                    );

                    // Update dimensions
                    sample.width = out_w;
                    sample.height = out_h;
                    sample.is_transformed = true;
                }
                // Apply legacy transforms if requested (deprecated)
                else if (config_.enable_resize && sample.width > 0) {
                    auto& data = sample.data["jpg"];
                    auto transformed = resize_transform->apply(data, sample.width, sample.height);
                    sample.width = transformed.width;
                    sample.height = transformed.height;
                    data = std::move(transformed.data);

                    if (config_.enable_normalize) {
                        auto normalized = normalize_transform->apply(data, sample.width, sample.height);
                        data = std::move(normalized.data);
                    }
                }

                // Push to output queue (spin if full)
                while (running_ && !output_queue_->try_push(std::move(sample))) {
                    std::this_thread::yield();
                }
            } catch (const std::exception& e) {
                // Silently skip samples that fail to load/decode
                // This prevents the pipeline from crashing on corrupt data
                std::cerr << "Sample processing error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Sample processing error: unknown" << std::endl;
            }
        });
    }

    // Wait for all tasks to complete
    thread_pool_->wait();
}

Sample Pipeline::load_sample(size_t global_index) {
    // Find which TAR file contains this sample
    size_t current_offset = 0;

    for (size_t reader_idx = 0; reader_idx < readers_.size(); ++reader_idx) {
        const auto& reader = readers_[reader_idx];
        size_t num_samples = reader->num_samples();

        if (global_index < current_offset + num_samples) {
            // Found the right TAR file
            size_t local_index = global_index - current_offset;

            // Lock this reader for thread-safe access
            std::lock_guard<std::mutex> lock(*reader_mutexes_[reader_idx]);

            const auto& tar_sample = reader->get_sample(local_index);

            // Create output sample
            Sample sample;
            sample.index = global_index;

            // Read all files in this sample
            for (const auto& [ext, entry] : tar_sample.files) {
                auto data_span = reader->read_file(entry);

                // Copy data while holding the lock to ensure mmap'd memory stays valid
                std::vector<uint8_t> data(data_span.size());
                std::memcpy(data.data(), data_span.data(), data_span.size());
                sample.data[ext] = std::move(data);
            }

            return sample;
        }

        current_offset += num_samples;
    }

    throw std::out_of_range("Sample index out of range");
}

}  // namespace turboloader
