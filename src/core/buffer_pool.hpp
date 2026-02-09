/**
 * @file buffer_pool.hpp
 * @brief Unified thread-safe buffer pool for memory reuse
 *
 * Consolidates buffer pooling for both:
 * - Raw byte arrays (transforms, resize operations)
 * - Vector buffers (image decoders)
 *
 * Features:
 * - Thread-safe with mutex protection
 * - Size-bucketed allocation for efficient reuse
 * - Auto-releasing pooled pointers for vectors
 * - Statistics tracking for debugging
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <algorithm>
#include <functional>

namespace turboloader {

/**
 * @brief Unified thread-safe buffer pool for memory reuse
 *
 * Supports two modes:
 * 1. Raw buffers: acquire(size) / release() for transforms
 * 2. Vector buffers: acquire_vector() for decoders (auto-releasing)
 */
class BufferPool {
public:
    /**
     * @brief Construct a buffer pool
     * @param max_buffers_per_bucket Maximum buffers to keep per size bucket
     * @param max_buffer_size Maximum individual buffer size to pool
     * @param default_vector_size Default size for vector buffers
     */
    explicit BufferPool(size_t max_buffers_per_bucket = 16,
                        size_t max_buffer_size = 64 * 1024 * 1024,
                        size_t default_vector_size = 256 * 256 * 3)
        : max_buffers_per_bucket_(max_buffers_per_bucket),
          max_buffer_size_(max_buffer_size),
          default_vector_size_(default_vector_size) {}

    ~BufferPool() = default;

    // Non-copyable, non-movable
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;

    // =========================================================================
    // Raw Buffer Interface (for transforms)
    // =========================================================================

    /**
     * @brief Acquire a raw buffer of at least the specified size
     * @param size Minimum buffer size needed
     * @return Unique pointer to buffer (caller owns)
     */
    std::unique_ptr<uint8_t[]> acquire(size_t size) {
        if (size == 0) return nullptr;

        size_t bucket_size = round_to_bucket(size);

        {
            std::lock_guard<std::mutex> lock(raw_mutex_);
            stats_.acquire_calls++;

            auto it = std::find_if(raw_pool_.begin(), raw_pool_.end(),
                [bucket_size](const RawBuffer& buf) {
                    return buf.size >= bucket_size;
                });

            if (it != raw_pool_.end()) {
                stats_.cache_hits++;
                auto result = std::move(it->data);
                raw_pool_.erase(it);
                return result;
            }

            stats_.cache_misses++;
        }

        stats_.allocations++;
        return std::make_unique<uint8_t[]>(bucket_size);
    }

    /**
     * @brief Release a raw buffer back to the pool
     * @param buffer Buffer to release
     * @param size Size of the buffer
     */
    void release(std::unique_ptr<uint8_t[]> buffer, size_t size) {
        if (!buffer || size == 0) return;
        if (size > max_buffer_size_) {
            stats_.oversized_releases++;
            return;
        }

        size_t bucket_size = round_to_bucket(size);

        std::lock_guard<std::mutex> lock(raw_mutex_);
        stats_.release_calls++;

        size_t bucket_count = std::count_if(raw_pool_.begin(), raw_pool_.end(),
            [bucket_size](const RawBuffer& buf) { return buf.size == bucket_size; });

        if (bucket_count < max_buffers_per_bucket_) {
            raw_pool_.push_back({std::move(buffer), bucket_size});
            stats_.pooled_buffers++;
        } else {
            stats_.bucket_full_releases++;
        }
    }

    // =========================================================================
    // Vector Buffer Interface (for decoders) - Compatible with old BufferPool
    // =========================================================================

    /**
     * @brief Pooled pointer that auto-releases vector back to pool
     */
    class PooledVector {
    public:
        PooledVector(std::vector<uint8_t>&& vec, BufferPool* pool)
            : vec_(std::move(vec)), pool_(pool) {}

        ~PooledVector() {
            if (pool_) {
                pool_->release_vector(std::move(vec_));
            }
        }

        // Non-copyable
        PooledVector(const PooledVector&) = delete;
        PooledVector& operator=(const PooledVector&) = delete;

        // Movable
        PooledVector(PooledVector&& other) noexcept
            : vec_(std::move(other.vec_)), pool_(other.pool_) {
            other.pool_ = nullptr;
        }

        PooledVector& operator=(PooledVector&& other) noexcept {
            if (this != &other) {
                if (pool_) pool_->release_vector(std::move(vec_));
                vec_ = std::move(other.vec_);
                pool_ = other.pool_;
                other.pool_ = nullptr;
            }
            return *this;
        }

        std::vector<uint8_t>& operator*() { return vec_; }
        const std::vector<uint8_t>& operator*() const { return vec_; }
        std::vector<uint8_t>* operator->() { return &vec_; }
        const std::vector<uint8_t>* operator->() const { return &vec_; }

    private:
        std::vector<uint8_t> vec_;
        BufferPool* pool_;
    };

    /**
     * @brief Acquire a vector buffer from pool (auto-releasing)
     *
     * This is the interface expected by decoders. The returned PooledVector
     * automatically releases the buffer back to the pool when destroyed.
     *
     * @return PooledVector that auto-releases on destruction
     */
    PooledVector acquire_vector() {
        std::lock_guard<std::mutex> lock(vec_mutex_);
        stats_.vector_acquire_calls++;

        if (!vec_pool_.empty()) {
            stats_.vector_cache_hits++;
            auto vec = std::move(vec_pool_.back());
            vec_pool_.pop_back();
            vec.clear();  // Clear but keep capacity
            return PooledVector(std::move(vec), this);
        }

        stats_.vector_cache_misses++;
        std::vector<uint8_t> vec;
        vec.reserve(default_vector_size_);
        return PooledVector(std::move(vec), this);
    }

    /**
     * @brief Alias for acquire_vector() - backward compatibility
     */
    PooledVector acquire() {
        return acquire_vector();
    }

    // =========================================================================
    // Pool Management
    // =========================================================================

    /**
     * @brief Clear all pooled buffers
     */
    void clear() {
        {
            std::lock_guard<std::mutex> lock(raw_mutex_);
            raw_pool_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(vec_mutex_);
            vec_pool_.clear();
        }
        stats_.clears++;
    }

    /**
     * @brief Get number of pooled raw buffers
     */
    size_t raw_pooled_count() const {
        std::lock_guard<std::mutex> lock(raw_mutex_);
        return raw_pool_.size();
    }

    /**
     * @brief Get number of pooled vector buffers
     */
    size_t vector_pooled_count() const {
        std::lock_guard<std::mutex> lock(vec_mutex_);
        return vec_pool_.size();
    }

    /**
     * @brief Get total pooled count (both raw and vector)
     */
    size_t pooled_count() const {
        return raw_pooled_count() + vector_pooled_count();
    }

    /**
     * @brief Get total memory used by pooled raw buffers
     */
    size_t pooled_memory() const {
        std::lock_guard<std::mutex> lock(raw_mutex_);
        size_t total = 0;
        for (const auto& buf : raw_pool_) {
            total += buf.size;
        }
        return total;
    }

    /**
     * @brief Statistics for debugging and monitoring
     */
    struct Stats {
        // Raw buffer stats
        size_t acquire_calls = 0;
        size_t release_calls = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        size_t allocations = 0;
        size_t pooled_buffers = 0;
        size_t oversized_releases = 0;
        size_t bucket_full_releases = 0;

        // Vector buffer stats
        size_t vector_acquire_calls = 0;
        size_t vector_cache_hits = 0;
        size_t vector_cache_misses = 0;

        size_t clears = 0;

        float raw_hit_rate() const {
            if (acquire_calls == 0) return 0.0f;
            return static_cast<float>(cache_hits) / acquire_calls;
        }

        float vector_hit_rate() const {
            if (vector_acquire_calls == 0) return 0.0f;
            return static_cast<float>(vector_cache_hits) / vector_acquire_calls;
        }
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lock1(raw_mutex_);
        std::lock_guard<std::mutex> lock2(vec_mutex_);
        return stats_;
    }

    void reset_stats() {
        std::lock_guard<std::mutex> lock1(raw_mutex_);
        std::lock_guard<std::mutex> lock2(vec_mutex_);
        stats_ = Stats{};
    }

    /**
     * @brief Get number of available buffers (backward compatibility)
     */
    size_t size() const {
        return vector_pooled_count();
    }

private:
    void release_vector(std::vector<uint8_t>&& vec) {
        std::lock_guard<std::mutex> lock(vec_mutex_);
        if (vec_pool_.size() < max_buffers_per_bucket_ * 4) {  // Higher limit for vectors
            // Discard oversized buffers to prevent memory bloat from outlier images.
            // If a single large image expanded the vector beyond 2x the default,
            // let it deallocate rather than keeping it in the pool forever.
            if (vec.capacity() > default_vector_size_ * 2) {
                stats_.oversized_releases++;
                return;  // Let the vector deallocate
            }
            vec_pool_.push_back(std::move(vec));
        }
    }

    struct RawBuffer {
        std::unique_ptr<uint8_t[]> data;
        size_t size;
    };

    static size_t round_to_bucket(size_t size) {
        constexpr size_t MIN_BUCKET = 4096;
        if (size <= MIN_BUCKET) return MIN_BUCKET;
        size_t bucket = MIN_BUCKET;
        while (bucket < size) bucket *= 2;
        return bucket;
    }

    // Raw buffer pool
    mutable std::mutex raw_mutex_;
    std::vector<RawBuffer> raw_pool_;

    // Vector buffer pool
    mutable std::mutex vec_mutex_;
    std::vector<std::vector<uint8_t>> vec_pool_;

    // Configuration
    size_t max_buffers_per_bucket_;
    size_t max_buffer_size_;
    size_t default_vector_size_;

    // Statistics
    Stats stats_;
};

/**
 * @brief RAII wrapper for pooled raw buffers
 */
class PooledBufferGuard {
public:
    PooledBufferGuard(BufferPool& pool, size_t size)
        : pool_(pool), size_(size), buffer_(pool.acquire(size)) {}

    ~PooledBufferGuard() {
        if (buffer_) {
            pool_.release(std::move(buffer_), size_);
        }
    }

    PooledBufferGuard(const PooledBufferGuard&) = delete;
    PooledBufferGuard& operator=(const PooledBufferGuard&) = delete;

    PooledBufferGuard(PooledBufferGuard&& other) noexcept
        : pool_(other.pool_), size_(other.size_), buffer_(std::move(other.buffer_)) {
        other.size_ = 0;
    }

    uint8_t* get() { return buffer_.get(); }
    const uint8_t* get() const { return buffer_.get(); }
    size_t size() const { return size_; }

    std::unique_ptr<uint8_t[]> release() {
        size_ = 0;
        return std::move(buffer_);
    }

private:
    BufferPool& pool_;
    size_t size_;
    std::unique_ptr<uint8_t[]> buffer_;
};

// Global buffer pool instance for transforms
inline BufferPool& get_global_buffer_pool() {
    static BufferPool pool(32, 128 * 1024 * 1024, 256 * 256 * 3);
    return pool;
}

// Alias for backward compatibility with resize transforms
inline BufferPool& get_resize_buffer_pool() {
    return get_global_buffer_pool();
}

} // namespace turboloader
