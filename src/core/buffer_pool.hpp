/**
 * @file buffer_pool.hpp
 * @brief Thread-safe buffer pool for memory reuse
 *
 * Provides efficient buffer allocation by reusing previously allocated
 * buffers. Reduces allocation overhead in hot paths like image transforms.
 *
 * Features:
 * - Thread-safe with mutex protection
 * - Size-bucketed allocation for efficient reuse
 * - Automatic cleanup of unused buffers
 * - Statistics tracking for debugging
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <algorithm>

namespace turboloader {

/**
 * @brief Thread-safe size-aware buffer pool for memory reuse
 *
 * Pools buffers by size buckets to enable efficient reuse.
 * When a buffer is released, it goes back to the pool for future use.
 *
 * Note: Named SizedBufferPool to avoid conflict with BufferPool in object_pool.hpp
 */
class SizedBufferPool {
public:
    /**
     * @brief Construct a buffer pool
     * @param max_buffers_per_bucket Maximum buffers to keep per size bucket
     * @param max_buffer_size Maximum individual buffer size to pool (larger buffers not pooled)
     */
    explicit SizedBufferPool(size_t max_buffers_per_bucket = 16,
                             size_t max_buffer_size = 64 * 1024 * 1024)  // 64MB default
        : max_buffers_per_bucket_(max_buffers_per_bucket),
          max_buffer_size_(max_buffer_size) {}

    ~SizedBufferPool() = default;

    // Non-copyable, non-movable
    SizedBufferPool(const SizedBufferPool&) = delete;
    SizedBufferPool& operator=(const SizedBufferPool&) = delete;
    SizedBufferPool(SizedBufferPool&&) = delete;
    SizedBufferPool& operator=(SizedBufferPool&&) = delete;

    /**
     * @brief Acquire a buffer of at least the specified size
     *
     * First tries to find a suitable buffer in the pool.
     * If none available, allocates a new buffer.
     *
     * @param size Minimum buffer size needed
     * @return Unique pointer to buffer (caller owns)
     */
    std::unique_ptr<uint8_t[]> acquire(size_t size) {
        if (size == 0) {
            return nullptr;
        }

        // Round up to bucket size for better reuse
        size_t bucket_size = round_to_bucket(size);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.acquire_calls++;

            // Find a buffer in the appropriate bucket
            auto it = std::find_if(pool_.begin(), pool_.end(),
                [bucket_size](const PooledBuffer& buf) {
                    return buf.size >= bucket_size && !buf.in_use;
                });

            if (it != pool_.end()) {
                // Found a suitable buffer
                it->in_use = true;
                stats_.cache_hits++;

                // Move the buffer out and mark for removal
                auto result = std::move(it->data);
                pool_.erase(it);
                return result;
            }

            stats_.cache_misses++;
        }

        // No suitable buffer found, allocate new
        stats_.allocations++;
        return std::make_unique<uint8_t[]>(bucket_size);
    }

    /**
     * @brief Release a buffer back to the pool for reuse
     *
     * If the pool bucket is full or buffer is too large, the buffer is freed.
     *
     * @param buffer Buffer to release (transfers ownership)
     * @param size Size of the buffer
     */
    void release(std::unique_ptr<uint8_t[]> buffer, size_t size) {
        if (!buffer || size == 0) {
            return;
        }

        // Don't pool very large buffers
        if (size > max_buffer_size_) {
            stats_.oversized_releases++;
            return;  // Buffer freed when unique_ptr goes out of scope
        }

        size_t bucket_size = round_to_bucket(size);

        std::lock_guard<std::mutex> lock(mutex_);
        stats_.release_calls++;

        // Count buffers in this bucket
        size_t bucket_count = std::count_if(pool_.begin(), pool_.end(),
            [bucket_size](const PooledBuffer& buf) {
                return buf.size == bucket_size;
            });

        // Only pool if we haven't hit the limit for this bucket
        if (bucket_count < max_buffers_per_bucket_) {
            pool_.push_back({std::move(buffer), bucket_size, false});
            stats_.pooled_buffers++;
        } else {
            stats_.bucket_full_releases++;
            // Buffer freed when unique_ptr goes out of scope
        }
    }

    /**
     * @brief Clear all pooled buffers
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
        stats_.clears++;
    }

    /**
     * @brief Get current number of pooled buffers
     */
    size_t pooled_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

    /**
     * @brief Get total memory used by pooled buffers
     */
    size_t pooled_memory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (const auto& buf : pool_) {
            total += buf.size;
        }
        return total;
    }

    /**
     * @brief Statistics for debugging and monitoring
     */
    struct Stats {
        size_t acquire_calls = 0;
        size_t release_calls = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        size_t allocations = 0;
        size_t pooled_buffers = 0;
        size_t oversized_releases = 0;
        size_t bucket_full_releases = 0;
        size_t clears = 0;

        float hit_rate() const {
            if (acquire_calls == 0) return 0.0f;
            return static_cast<float>(cache_hits) / acquire_calls;
        }
    };

    /**
     * @brief Get pool statistics
     */
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    /**
     * @brief Reset statistics
     */
    void reset_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = Stats{};
    }

private:
    struct PooledBuffer {
        std::unique_ptr<uint8_t[]> data;
        size_t size;
        bool in_use;
    };

    /**
     * @brief Round size up to nearest power of 2 bucket
     *
     * This ensures better buffer reuse by grouping similar sizes.
     */
    static size_t round_to_bucket(size_t size) {
        // Minimum bucket size is 4KB
        constexpr size_t MIN_BUCKET = 4096;
        if (size <= MIN_BUCKET) return MIN_BUCKET;

        // Round up to next power of 2
        size_t bucket = MIN_BUCKET;
        while (bucket < size) {
            bucket *= 2;
        }
        return bucket;
    }

    mutable std::mutex mutex_;
    std::vector<PooledBuffer> pool_;
    size_t max_buffers_per_bucket_;
    size_t max_buffer_size_;
    Stats stats_;
};

/**
 * @brief RAII wrapper for pooled buffers
 *
 * Automatically releases buffer back to pool when destroyed.
 */
class PooledBufferGuard {
public:
    PooledBufferGuard(SizedBufferPool& pool, size_t size)
        : pool_(pool), size_(size), buffer_(pool.acquire(size)) {}

    ~PooledBufferGuard() {
        if (buffer_) {
            pool_.release(std::move(buffer_), size_);
        }
    }

    // Non-copyable
    PooledBufferGuard(const PooledBufferGuard&) = delete;
    PooledBufferGuard& operator=(const PooledBufferGuard&) = delete;

    // Movable
    PooledBufferGuard(PooledBufferGuard&& other) noexcept
        : pool_(other.pool_), size_(other.size_), buffer_(std::move(other.buffer_)) {
        other.size_ = 0;
    }

    uint8_t* get() { return buffer_.get(); }
    const uint8_t* get() const { return buffer_.get(); }
    size_t size() const { return size_; }

    /**
     * @brief Release ownership without returning to pool
     */
    std::unique_ptr<uint8_t[]> release() {
        size_ = 0;
        return std::move(buffer_);
    }

private:
    SizedBufferPool& pool_;
    size_t size_;
    std::unique_ptr<uint8_t[]> buffer_;
};

// Global buffer pool instance for transforms
inline SizedBufferPool& get_resize_buffer_pool() {
    static SizedBufferPool pool(32, 128 * 1024 * 1024);  // 32 buffers/bucket, 128MB max
    return pool;
}

} // namespace turboloader
