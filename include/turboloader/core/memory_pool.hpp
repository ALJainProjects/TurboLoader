#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace turboloader {

/**
 * High-performance arena/pool allocator for ML data loading
 *
 * Optimizations:
 * - Batch allocations reduce malloc overhead
 * - Aligned allocations for SIMD operations
 * - Thread-local pools reduce lock contention
 * - Reusable memory reduces GC pressure
 *
 * Usage:
 *   MemoryPool pool(1024 * 1024);  // 1MB blocks
 *   void* ptr = pool.allocate(4096, 64);  // 4KB, 64-byte aligned
 *   pool.reset();  // Reuse all memory
 */
class MemoryPool {
public:
    /**
     * @param block_size Size of each memory block to allocate
     * @param alignment Default alignment (must be power of 2)
     */
    explicit MemoryPool(size_t block_size = 1024 * 1024,
                       size_t alignment = 64);

    ~MemoryPool();

    // Non-copyable, non-movable (has mutex)
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    /**
     * Allocate memory from the pool
     * @param size Bytes to allocate
     * @param alignment Alignment requirement (default: pool alignment)
     * @return Aligned pointer, or nullptr if allocation fails
     */
    void* allocate(size_t size, size_t alignment = 0);

    /**
     * Reset pool - invalidates all previous allocations
     * Memory is reused, not freed (fast reset for next epoch)
     */
    void reset();

    /**
     * Free all memory (actual deallocation)
     */
    void clear();

    /**
     * Get total bytes allocated from system
     */
    size_t total_bytes() const { return total_bytes_; }

    /**
     * Get bytes currently used
     */
    size_t used_bytes() const { return used_bytes_; }

private:
    struct Block {
        std::unique_ptr<uint8_t[]> data;
        size_t size;
        size_t used;

        Block(size_t s) : data(new uint8_t[s]), size(s), used(0) {}
    };

    size_t block_size_;
    size_t default_alignment_;

    std::vector<Block> blocks_;
    size_t current_block_;
    size_t total_bytes_;
    size_t used_bytes_;

    std::mutex mutex_;  // For thread safety

    // Helper: align pointer
    static size_t align_up(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // Allocate new block
    void allocate_new_block(size_t min_size);
};

/**
 * RAII wrapper for memory pool allocations
 */
template <typename T>
class PoolAllocator {
public:
    using value_type = T;

    explicit PoolAllocator(MemoryPool& pool) : pool_(&pool) {}

    template <typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}

    T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        void* ptr = pool_->allocate(bytes, alignof(T));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T*, size_t) {
        // Pool allocator doesn't support individual deallocation
        // Memory is reclaimed on reset() or clear()
    }

    template <typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return pool_ == other.pool_;
    }

    template <typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }

private:
    MemoryPool* pool_;

    template <typename U>
    friend class PoolAllocator;
};

}  // namespace turboloader
