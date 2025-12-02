/**
 * @file hybrid_wait.hpp
 * @brief Hybrid wait strategy combining spinning with condition variable fallback
 *
 * Phase 3 optimization: Replace pure busy-wait with hybrid strategy
 *
 * Strategy:
 * 1. Spin briefly (configurable iterations)
 * 2. Fall back to condition variable with timeout
 * 3. Atomic flag to avoid CV notification overhead when no waiters
 *
 * This reduces CPU usage while maintaining low latency for fast operations.
 *
 * Performance characteristics:
 * - Fast path (no contention): ~10-20ns (pure spin)
 * - Slow path (contention): ~100us (CV wait)
 * - CPU efficiency: Much better than pure spin-wait
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>

namespace turboloader {

/**
 * @brief Configuration for hybrid wait strategy
 */
struct HybridWaitConfig {
    /// Number of spin iterations before falling back to CV
    size_t spin_iterations = 32;

    /// Timeout for condition variable wait (microseconds)
    size_t cv_timeout_us = 100;

    /// Maximum total wait time before giving up (milliseconds)
    /// Set to 0 for infinite wait
    size_t max_wait_ms = 0;
};

/**
 * @brief Hybrid wait utility for producer-consumer synchronization
 *
 * Provides efficient waiting with configurable spin + CV fallback.
 */
class HybridWait {
public:
    /**
     * @brief Construct hybrid wait with configuration
     *
     * @param config Wait strategy configuration
     */
    explicit HybridWait(const HybridWaitConfig& config = HybridWaitConfig())
        : config_(config), waiters_(0) {}

    /**
     * @brief Wait until condition is satisfied
     *
     * @tparam Predicate Callable returning bool
     * @param pred Condition to wait for (called until returns true)
     * @param running Atomic flag to check for shutdown
     * @return true if condition satisfied, false if shutdown requested
     *
     * Strategy:
     * 1. Spin for spin_iterations
     * 2. Fall back to CV wait with timeout
     * 3. Repeat until condition satisfied or shutdown
     */
    template<typename Predicate>
    bool wait_for(Predicate&& pred, const std::atomic<bool>& running) {
        // Fast path: check condition immediately
        if (pred()) {
            return true;
        }

        // Check if we should continue
        if (!running.load(std::memory_order_acquire)) {
            return false;
        }

        // Phase 1: Spin wait
        for (size_t i = 0; i < config_.spin_iterations; ++i) {
            if (pred()) {
                return true;
            }
            if (!running.load(std::memory_order_acquire)) {
                return false;
            }
            // Pause instruction for spin-wait optimization
            spin_pause();
        }

        // Phase 2: Fall back to condition variable
        auto deadline = (config_.max_wait_ms > 0)
            ? std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.max_wait_ms)
            : std::chrono::steady_clock::time_point::max();

        while (running.load(std::memory_order_acquire)) {
            if (pred()) {
                return true;
            }

            // Check deadline
            if (config_.max_wait_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                return pred();  // One final check
            }

            // Register as waiter
            waiters_.fetch_add(1, std::memory_order_release);

            {
                std::unique_lock<std::mutex> lock(mutex_);

                // Double-check condition under lock
                if (pred()) {
                    waiters_.fetch_sub(1, std::memory_order_release);
                    return true;
                }

                // Wait with timeout
                cv_.wait_for(lock, std::chrono::microseconds(config_.cv_timeout_us));
            }

            waiters_.fetch_sub(1, std::memory_order_release);
        }

        return pred();  // Final check on shutdown
    }

    /**
     * @brief Notify one waiting thread
     *
     * Only notifies if there are actually waiters to avoid overhead.
     */
    void notify_one() {
        if (waiters_.load(std::memory_order_acquire) > 0) {
            cv_.notify_one();
        }
    }

    /**
     * @brief Notify all waiting threads
     *
     * Only notifies if there are actually waiters to avoid overhead.
     */
    void notify_all() {
        if (waiters_.load(std::memory_order_acquire) > 0) {
            cv_.notify_all();
        }
    }

    /**
     * @brief Get number of currently waiting threads
     */
    size_t waiter_count() const {
        return waiters_.load(std::memory_order_relaxed);
    }

private:
    /**
     * @brief Platform-specific spin-wait pause instruction
     *
     * Reduces power consumption and improves performance during spin-wait
     */
    static void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        // x86: PAUSE instruction
        __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(_M_ARM64)
        // ARM64: YIELD instruction
        __asm__ __volatile__("yield" ::: "memory");
#else
        // Fallback: use std::this_thread::yield() but only occasionally
        // to avoid excessive context switches
        std::this_thread::yield();
#endif
    }

    HybridWaitConfig config_;
    std::atomic<size_t> waiters_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

/**
 * @brief Helper function for hybrid push-wait on a queue
 *
 * @tparam Queue Queue type with try_push method
 * @tparam T Element type
 * @param queue Queue to push to
 * @param item Item to push (will be moved)
 * @param running Atomic flag to check for shutdown
 * @param wait HybridWait instance for waiting
 * @return true if pushed, false if shutdown
 */
template<typename Queue, typename T>
bool hybrid_push(Queue& queue, T&& item,
                 const std::atomic<bool>& running,
                 HybridWait& wait) {
    // Fast path: try immediate push
    if (queue.try_push(std::forward<T>(item))) {
        return true;
    }

    // Slow path: wait for space
    bool success = wait.wait_for(
        [&]() { return queue.try_push(std::forward<T>(item)); },
        running
    );

    return success;
}

/**
 * @brief Helper function for hybrid pop-wait on a queue
 *
 * @tparam Queue Queue type with try_pop method
 * @tparam T Element type
 * @param queue Queue to pop from
 * @param item Output item
 * @param running Atomic flag to check for shutdown
 * @param wait HybridWait instance for waiting
 * @return true if popped, false if shutdown
 */
template<typename Queue, typename T>
bool hybrid_pop(Queue& queue, T& item,
                const std::atomic<bool>& running,
                HybridWait& wait) {
    // Fast path: try immediate pop
    if (queue.try_pop(item)) {
        return true;
    }

    // Slow path: wait for item
    bool success = wait.wait_for(
        [&]() { return queue.try_pop(item); },
        running
    );

    return success;
}

} // namespace turboloader
