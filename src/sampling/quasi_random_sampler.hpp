/**
 * @file quasi_random_sampler.hpp
 * @brief FFCV-style quasi-random sampling for memory-efficient shuffling (v2.17.0)
 *
 * Provides memory-efficient shuffling for datasets that don't fit in RAM.
 * Instead of loading the entire dataset into memory, we:
 * 1. Divide the dataset into "pages" (contiguous chunks)
 * 2. Shuffle pages, not individual samples
 * 3. Keep only a few pages in memory at a time
 * 4. Shuffle within each page
 *
 * This provides "quasi-random" access that's nearly as random as full shuffling
 * but uses constant memory regardless of dataset size.
 *
 * Features:
 * - Constant memory usage O(page_size * num_buffer_pages)
 * - Near-random sample order with page-level and intra-page shuffling
 * - Reproducible with seed
 * - Compatible with distributed training (each rank gets different pages)
 *
 * Usage:
 * ```cpp
 * QuasiRandomSampler sampler(dataset_size, 8 * 1024 * 1024);  // 8MB pages
 * sampler.set_buffer_pages(4);  // Keep 4 pages in memory
 *
 * for (size_t idx : sampler.get_indices_for_epoch(epoch, seed)) {
 *     // Process sample at idx
 * }
 * ```
 */

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace turboloader {
namespace sampling {

/**
 * @brief Ordering options for data loading
 */
enum class OrderOption {
    SEQUENTIAL,     // No shuffling, samples in order
    RANDOM,         // Full random shuffle (requires all indices in memory)
    QUASI_RANDOM    // Page-based shuffling (constant memory)
};

/**
 * @brief Memory-efficient quasi-random sampler
 *
 * Implements FFCV-style quasi-random shuffling that uses constant memory
 * regardless of dataset size.
 */
class QuasiRandomSampler {
public:
    /**
     * @brief Create sampler for dataset
     * @param dataset_size Total number of samples in dataset
     * @param page_size_bytes Size of each page in bytes (default 8MB)
     * @param avg_sample_size Average size of each sample in bytes (for page calculation)
     */
    QuasiRandomSampler(
        size_t dataset_size,
        size_t page_size_bytes = 8 * 1024 * 1024,
        size_t avg_sample_size = 100 * 1024  // 100KB default
    ) : dataset_size_(dataset_size),
        page_size_bytes_(page_size_bytes),
        avg_sample_size_(avg_sample_size) {

        // Calculate samples per page
        samples_per_page_ = std::max(size_t(1), page_size_bytes_ / avg_sample_size_);

        // Calculate number of pages
        num_pages_ = (dataset_size_ + samples_per_page_ - 1) / samples_per_page_;

        // Default buffer pages
        buffer_pages_ = std::min(size_t(4), num_pages_);
    }

    /**
     * @brief Set number of pages to keep in buffer
     * @param num_pages Number of pages (more = more random, more memory)
     */
    void set_buffer_pages(size_t num_pages) {
        buffer_pages_ = std::min(num_pages, num_pages_);
    }

    /**
     * @brief Set page size in bytes
     */
    void set_page_size(size_t bytes) {
        page_size_bytes_ = bytes;
        samples_per_page_ = std::max(size_t(1), page_size_bytes_ / avg_sample_size_);
        num_pages_ = (dataset_size_ + samples_per_page_ - 1) / samples_per_page_;
        buffer_pages_ = std::min(buffer_pages_, num_pages_);
    }

    /**
     * @brief Set average sample size (for page calculation)
     */
    void set_avg_sample_size(size_t bytes) {
        avg_sample_size_ = bytes;
        samples_per_page_ = std::max(size_t(1), page_size_bytes_ / avg_sample_size_);
        num_pages_ = (dataset_size_ + samples_per_page_ - 1) / samples_per_page_;
    }

    /**
     * @brief Get shuffled indices for an epoch
     *
     * Uses quasi-random shuffling:
     * 1. Shuffle page order
     * 2. For each buffer of pages, shuffle samples within
     * 3. Yield samples from buffer before loading next pages
     *
     * @param epoch Epoch number (for deterministic but different shuffling)
     * @param seed Base random seed
     * @return Vector of sample indices in quasi-random order
     */
    std::vector<size_t> get_indices_for_epoch(size_t epoch, uint64_t seed = 42) const {
        std::vector<size_t> indices;
        indices.reserve(dataset_size_);

        // Create RNG with epoch-specific seed
        std::mt19937_64 rng(seed + epoch * 0x9E3779B97F4A7C15ULL);

        // Create and shuffle page order
        std::vector<size_t> page_order(num_pages_);
        std::iota(page_order.begin(), page_order.end(), 0);
        std::shuffle(page_order.begin(), page_order.end(), rng);

        // Process pages in buffers
        for (size_t buf_start = 0; buf_start < num_pages_; buf_start += buffer_pages_) {
            size_t buf_end = std::min(buf_start + buffer_pages_, num_pages_);

            // Collect all sample indices from pages in this buffer
            std::vector<size_t> buffer_indices;
            buffer_indices.reserve((buf_end - buf_start) * samples_per_page_);

            for (size_t p = buf_start; p < buf_end; ++p) {
                size_t page_idx = page_order[p];
                size_t page_start = page_idx * samples_per_page_;
                size_t page_end = std::min(page_start + samples_per_page_, dataset_size_);

                for (size_t i = page_start; i < page_end; ++i) {
                    buffer_indices.push_back(i);
                }
            }

            // Shuffle within buffer
            std::shuffle(buffer_indices.begin(), buffer_indices.end(), rng);

            // Add to output
            indices.insert(indices.end(), buffer_indices.begin(), buffer_indices.end());
        }

        return indices;
    }

    /**
     * @brief Get indices for distributed training
     *
     * Each rank gets a disjoint subset of pages.
     *
     * @param epoch Epoch number
     * @param seed Base random seed
     * @param rank Current process rank (0-indexed)
     * @param world_size Total number of processes
     * @return Vector of sample indices for this rank
     */
    std::vector<size_t> get_indices_for_rank(
        size_t epoch,
        uint64_t seed,
        int rank,
        int world_size
    ) const {
        std::vector<size_t> indices;

        // Create RNG with epoch-specific seed (same across all ranks)
        std::mt19937_64 rng(seed + epoch * 0x9E3779B97F4A7C15ULL);

        // Create and shuffle page order (same across all ranks)
        std::vector<size_t> page_order(num_pages_);
        std::iota(page_order.begin(), page_order.end(), 0);
        std::shuffle(page_order.begin(), page_order.end(), rng);

        // Each rank gets every world_size-th page
        std::vector<size_t> my_pages;
        for (size_t i = rank; i < num_pages_; i += world_size) {
            my_pages.push_back(page_order[i]);
        }

        // Reserve space
        size_t expected_samples = (dataset_size_ / world_size) + samples_per_page_;
        indices.reserve(expected_samples);

        // Process my pages in buffers
        size_t my_buffer_pages = std::min(buffer_pages_, my_pages.size());

        for (size_t buf_start = 0; buf_start < my_pages.size(); buf_start += my_buffer_pages) {
            size_t buf_end = std::min(buf_start + my_buffer_pages, my_pages.size());

            std::vector<size_t> buffer_indices;
            buffer_indices.reserve((buf_end - buf_start) * samples_per_page_);

            for (size_t p = buf_start; p < buf_end; ++p) {
                size_t page_idx = my_pages[p];
                size_t page_start = page_idx * samples_per_page_;
                size_t page_end = std::min(page_start + samples_per_page_, dataset_size_);

                for (size_t i = page_start; i < page_end; ++i) {
                    buffer_indices.push_back(i);
                }
            }

            // Shuffle within buffer (rank-specific RNG state)
            std::mt19937_64 buffer_rng(seed + epoch * 0x9E3779B97F4A7C15ULL + rank + buf_start);
            std::shuffle(buffer_indices.begin(), buffer_indices.end(), buffer_rng);

            indices.insert(indices.end(), buffer_indices.begin(), buffer_indices.end());
        }

        return indices;
    }

    // ========================================================================
    // Iterator Interface
    // ========================================================================

    /**
     * @brief Iterator for quasi-random sample indices
     */
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const size_t*;
        using reference = const size_t&;

        Iterator(const QuasiRandomSampler* sampler, size_t epoch, uint64_t seed, size_t pos = 0)
            : sampler_(sampler), epoch_(epoch), seed_(seed), pos_(pos) {
            if (sampler_ && pos_ == 0) {
                indices_ = sampler_->get_indices_for_epoch(epoch_, seed_);
            }
        }

        // Default constructor for end iterator
        Iterator() : sampler_(nullptr), epoch_(0), seed_(0), pos_(0) {}

        reference operator*() const { return indices_[pos_]; }
        pointer operator->() const { return &indices_[pos_]; }

        Iterator& operator++() {
            ++pos_;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            if (!sampler_ && !other.sampler_) return true;
            if (!sampler_ || !other.sampler_) return pos_ >= indices_.size() && other.pos_ >= other.indices_.size();
            return pos_ == other.pos_;
        }

        bool operator!=(const Iterator& other) const { return !(*this == other); }

    private:
        const QuasiRandomSampler* sampler_;
        size_t epoch_;
        uint64_t seed_;
        size_t pos_;
        std::vector<size_t> indices_;
    };

    Iterator begin(size_t epoch = 0, uint64_t seed = 42) const {
        return Iterator(this, epoch, seed, 0);
    }

    Iterator end() const {
        return Iterator();
    }

    // ========================================================================
    // Getters
    // ========================================================================

    size_t dataset_size() const { return dataset_size_; }
    size_t num_pages() const { return num_pages_; }
    size_t samples_per_page() const { return samples_per_page_; }
    size_t buffer_pages() const { return buffer_pages_; }
    size_t page_size_bytes() const { return page_size_bytes_; }

    /**
     * @brief Estimate memory usage in bytes
     */
    size_t estimated_memory_usage() const {
        return buffer_pages_ * samples_per_page_ * sizeof(size_t);
    }

    /**
     * @brief Get string description of sampler configuration
     */
    std::string describe() const {
        return "QuasiRandomSampler: " +
               std::to_string(dataset_size_) + " samples, " +
               std::to_string(num_pages_) + " pages (" +
               std::to_string(samples_per_page_) + " samples/page), " +
               std::to_string(buffer_pages_) + " buffer pages, ~" +
               std::to_string(estimated_memory_usage() / 1024) + " KB memory";
    }

private:
    size_t dataset_size_;
    size_t page_size_bytes_;
    size_t avg_sample_size_;
    size_t samples_per_page_;
    size_t num_pages_;
    size_t buffer_pages_;
};

/**
 * @brief Simple sequential sampler (no shuffling)
 */
class SequentialSampler {
public:
    explicit SequentialSampler(size_t dataset_size) : dataset_size_(dataset_size) {}

    std::vector<size_t> get_indices() const {
        std::vector<size_t> indices(dataset_size_);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    size_t dataset_size() const { return dataset_size_; }

private:
    size_t dataset_size_;
};

/**
 * @brief Full random sampler (requires all indices in memory)
 */
class RandomSampler {
public:
    explicit RandomSampler(size_t dataset_size) : dataset_size_(dataset_size) {}

    std::vector<size_t> get_indices_for_epoch(size_t epoch, uint64_t seed = 42) const {
        std::vector<size_t> indices(dataset_size_);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937_64 rng(seed + epoch * 0x9E3779B97F4A7C15ULL);
        std::shuffle(indices.begin(), indices.end(), rng);

        return indices;
    }

    size_t dataset_size() const { return dataset_size_; }

private:
    size_t dataset_size_;
};

}  // namespace sampling
}  // namespace turboloader
