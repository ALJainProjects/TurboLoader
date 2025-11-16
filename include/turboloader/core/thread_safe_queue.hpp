#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace turboloader {

/**
 * Thread-safe FIFO queue with mutex protection
 * Optimized for complex objects like Sample that are expensive to move
 */
template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t capacity) : capacity_(capacity) {}

    /**
     * Push an item onto the queue (blocking if full)
     */
    void push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this] { return queue_.size() < capacity_; });
        queue_.push(std::move(item));
        cv_not_empty_.notify_one();
    }

    /**
     * Try to push an item (non-blocking)
     * @return true if successful, false if queue is full
     */
    bool try_push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= capacity_) {
            return false;
        }
        queue_.push(std::move(item));
        cv_not_empty_.notify_one();
        return true;
    }

    /**
     * Pop an item from the queue (blocking until available)
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this] { return !queue_.empty(); });

        T item = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return item;
    }

    /**
     * Try to pop an item (non-blocking)
     * @return Item if available, nullopt if queue is empty
     */
    std::optional<T> try_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return item;
    }

    /**
     * Get current queue size
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * Check if queue is empty
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * Clear all items from queue
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        cv_not_full_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::queue<T> queue_;
    size_t capacity_;
};

}  // namespace turboloader
