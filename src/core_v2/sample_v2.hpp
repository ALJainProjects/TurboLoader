/**
 * @file sample_v2.hpp
 * @brief Zero-copy sample structure for v2.0 pipeline
 *
 * Optimized sample structure that uses std::span for zero-copy views
 * into memory-mapped regions. Eliminates unnecessary data copies
 * during the pipeline processing.
 *
 * Design:
 * - Uses std::span<const uint8_t> for JPEG data (points into mmap)
 * - Uses pooled std::vector<uint8_t> for decoded RGB data
 * - Move-only semantics to prevent accidental copies
 * - Minimal memory footprint
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace turboloader {
namespace v2 {

/**
 * @brief Zero-copy sample structure
 *
 * Represents a single training sample with zero-copy semantics
 */
struct SampleV2 {
    /**
     * @brief Sample index in dataset
     */
    size_t index = 0;

    /**
     * @brief Zero-copy view of compressed JPEG data
     *
     * Points directly into memory-mapped TAR file.
     * MUST NOT outlive the mmap region!
     */
    std::span<const uint8_t> jpeg_data;

    /**
     * @brief Decoded RGB image data (from object pool)
     *
     * Layout: RGBRGBRGB... (row-major, channels-last)
     * Size: width * height * channels bytes
     */
    std::vector<uint8_t> decoded_rgb;

    /**
     * @brief Image dimensions
     */
    int width = 0;
    int height = 0;
    int channels = 0;

    /**
     * @brief Default constructor
     */
    SampleV2() = default;

    /**
     * @brief Construct from JPEG data view
     *
     * @param idx Sample index
     * @param data Zero-copy view of JPEG bytes
     */
    SampleV2(size_t idx, std::span<const uint8_t> data)
        : index(idx), jpeg_data(data) {}

    /**
     * @brief Move constructor
     */
    SampleV2(SampleV2&& other) noexcept = default;

    /**
     * @brief Move assignment
     */
    SampleV2& operator=(SampleV2&& other) noexcept = default;

    /**
     * @brief Deleted copy constructor (prevent accidental copies)
     */
    SampleV2(const SampleV2&) = delete;

    /**
     * @brief Deleted copy assignment (prevent accidental copies)
     */
    SampleV2& operator=(const SampleV2&) = delete;

    /**
     * @brief Check if sample has been decoded
     *
     * @return true if decoded_rgb contains data
     */
    bool is_decoded() const {
        return !decoded_rgb.empty();
    }

    /**
     * @brief Get size of JPEG data
     *
     * @return Number of bytes in compressed JPEG
     */
    size_t jpeg_size() const {
        return jpeg_data.size();
    }

    /**
     * @brief Get size of decoded RGB data
     *
     * @return Number of bytes in decoded image
     */
    size_t decoded_size() const {
        return decoded_rgb.size();
    }

    /**
     * @brief Get memory usage of this sample
     *
     * @return Total bytes used (JPEG view size + decoded RGB size)
     */
    size_t memory_usage() const {
        return jpeg_size() + decoded_size();
    }

    /**
     * @brief Clear decoded data (keeps JPEG view)
     *
     * Useful for freeing memory while keeping sample valid
     */
    void clear_decoded() {
        decoded_rgb.clear();
        decoded_rgb.shrink_to_fit();
        width = 0;
        height = 0;
        channels = 0;
    }
};

/**
 * @brief Batch of samples
 *
 * Represents a batch for training. Owns the samples.
 */
struct BatchV2 {
    /**
     * @brief Samples in this batch
     */
    std::vector<SampleV2> samples;

    /**
     * @brief Default constructor
     */
    BatchV2() = default;

    /**
     * @brief Construct with reserved capacity
     *
     * @param capacity Number of samples to reserve space for
     */
    explicit BatchV2(size_t capacity) {
        samples.reserve(capacity);
    }

    /**
     * @brief Move constructor
     */
    BatchV2(BatchV2&& other) noexcept = default;

    /**
     * @brief Move assignment
     */
    BatchV2& operator=(BatchV2&& other) noexcept = default;

    /**
     * @brief Deleted copy constructor
     */
    BatchV2(const BatchV2&) = delete;

    /**
     * @brief Deleted copy assignment
     */
    BatchV2& operator=(const BatchV2&) = delete;

    /**
     * @brief Add a sample to the batch
     *
     * @param sample Sample to add (will be moved)
     */
    void add(SampleV2&& sample) {
        samples.push_back(std::move(sample));
    }

    /**
     * @brief Get batch size
     *
     * @return Number of samples in batch
     */
    size_t size() const {
        return samples.size();
    }

    /**
     * @brief Check if batch is empty
     *
     * @return true if no samples
     */
    bool empty() const {
        return samples.empty();
    }

    /**
     * @brief Clear all samples
     */
    void clear() {
        samples.clear();
    }

    /**
     * @brief Get total memory usage of batch
     *
     * @return Total bytes used by all samples
     */
    size_t memory_usage() const {
        size_t total = 0;
        for (const auto& sample : samples) {
            total += sample.memory_usage();
        }
        return total;
    }

    /**
     * @brief Access sample by index
     *
     * @param idx Index (must be < size())
     * @return Reference to sample
     */
    SampleV2& operator[](size_t idx) {
        return samples[idx];
    }

    /**
     * @brief Access sample by index (const)
     *
     * @param idx Index (must be < size())
     * @return Const reference to sample
     */
    const SampleV2& operator[](size_t idx) const {
        return samples[idx];
    }

    /**
     * @brief Begin iterator
     */
    auto begin() { return samples.begin(); }

    /**
     * @brief End iterator
     */
    auto end() { return samples.end(); }

    /**
     * @brief Begin const iterator
     */
    auto begin() const { return samples.begin(); }

    /**
     * @brief End const iterator
     */
    auto end() const { return samples.end(); }
};

} // namespace v2
} // namespace turboloader
