#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>

namespace turboloader {

/**
 * Lock-free Single-Producer Multiple-Consumer (SPMC) queue
 *
 * Optimized for ML data loading where:
 * - One producer thread reads/decodes data
 * - Multiple consumer threads (training workers) fetch batches
 *
 * Based on ring buffer with atomic operations for synchronization.
 * Cache-line padding prevents false sharing between producer/consumer.
 */
template <typename T>
class LockFreeSPMCQueue {
public:
    /**
     * @param capacity Queue capacity (must be power of 2 for fast modulo)
     */
    explicit LockFreeSPMCQueue(size_t capacity);

    ~LockFreeSPMCQueue();

    // Non-copyable, non-movable (atomic members)
    LockFreeSPMCQueue(const LockFreeSPMCQueue&) = delete;
    LockFreeSPMCQueue& operator=(const LockFreeSPMCQueue&) = delete;

    /**
     * Try to push item (producer only - single thread)
     * @return true if successful, false if queue is full
     */
    bool try_push(T&& item);
    bool try_push(const T& item);

    /**
     * Try to pop item (consumer - multiple threads)
     * @return item if available, std::nullopt if queue is empty
     */
    std::optional<T> try_pop();

    /**
     * Get current queue size (approximate - may be stale)
     */
    size_t size() const;

    /**
     * Check if queue is empty (approximate)
     */
    bool empty() const;

    /**
     * Get queue capacity
     */
    size_t capacity() const { return capacity_; }

    /**
     * Wait for queue to have data (blocking)
     * Returns false if timeout
     */
    bool wait_for_data(std::chrono::milliseconds timeout = std::chrono::milliseconds(100));

private:
    // Slot in the ring buffer
    struct alignas(64) Slot {  // Cache line alignment
        std::atomic<uint64_t> sequence{0};
        T data;
    };

    size_t capacity_;
    size_t mask_;  // capacity - 1, for fast modulo

    // Cache-line aligned to prevent false sharing
    alignas(64) std::atomic<uint64_t> head_{0};  // Producer position
    alignas(64) std::atomic<uint64_t> tail_{0};  // Consumer position

    std::unique_ptr<Slot[]> buffer_;

    // Helper to get slot index
    size_t index(uint64_t pos) const {
        return pos & mask_;
    }
};

// Implementation

template <typename T>
LockFreeSPMCQueue<T>::LockFreeSPMCQueue(size_t capacity)
    : capacity_(capacity)
    , mask_(capacity - 1)
    , buffer_(new Slot[capacity]) {

    // Capacity must be power of 2
    if (capacity == 0 || (capacity & mask_) != 0) {
        throw std::invalid_argument("Capacity must be power of 2");
    }

    // Initialize sequences
    for (size_t i = 0; i < capacity_; ++i) {
        buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }
}

template <typename T>
LockFreeSPMCQueue<T>::~LockFreeSPMCQueue() = default;

template <typename T>
bool LockFreeSPMCQueue<T>::try_push(T&& item) {
    uint64_t pos = head_.load(std::memory_order_relaxed);
    Slot& slot = buffer_[index(pos)];

    uint64_t seq = slot.sequence.load(std::memory_order_acquire);

    // Check if slot is available for writing
    if (seq != pos) {
        return false;  // Queue is full
    }

    // Write data
    slot.data = std::move(item);

    // Make data visible to consumers
    slot.sequence.store(pos + 1, std::memory_order_release);

    // Move head forward
    head_.store(pos + 1, std::memory_order_relaxed);

    return true;
}

template <typename T>
bool LockFreeSPMCQueue<T>::try_push(const T& item) {
    uint64_t pos = head_.load(std::memory_order_relaxed);
    Slot& slot = buffer_[index(pos)];

    uint64_t seq = slot.sequence.load(std::memory_order_acquire);

    // Check if slot is available for writing
    if (seq != pos) {
        return false;  // Queue is full
    }

    // Write data
    slot.data = item;

    // Make data visible to consumers
    slot.sequence.store(pos + 1, std::memory_order_release);

    // Move head forward
    head_.store(pos + 1, std::memory_order_relaxed);

    return true;
}

template <typename T>
std::optional<T> LockFreeSPMCQueue<T>::try_pop() {
    while (true) {
        uint64_t pos = tail_.load(std::memory_order_relaxed);
        Slot& slot = buffer_[index(pos)];

        uint64_t seq = slot.sequence.load(std::memory_order_acquire);

        // Check if slot has data
        int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);

        if (diff == 0) {
            // Slot has data, try to claim it
            if (tail_.compare_exchange_weak(pos, pos + 1,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
                // Successfully claimed, read data
                T data = std::move(slot.data);

                // Mark slot as available for writing
                slot.sequence.store(pos + capacity_, std::memory_order_release);

                return data;
            }
            // CAS failed, another consumer got it, retry
        } else if (diff < 0) {
            // Queue is empty
            return std::nullopt;
        } else {
            // seq > pos + 1, means we're lagging, retry
            // This can happen with multiple consumers racing
        }
    }
}

template <typename T>
size_t LockFreeSPMCQueue<T>::size() const {
    uint64_t head = head_.load(std::memory_order_relaxed);
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    return static_cast<size_t>(head - tail);
}

template <typename T>
bool LockFreeSPMCQueue<T>::empty() const {
    return size() == 0;
}

}  // namespace turboloader
