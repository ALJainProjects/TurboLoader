#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace turboloader {

/**
 * High-performance thread pool for data loading pipeline
 *
 * Features:
 * - Work stealing for load balancing
 * - Thread affinity for cache locality
 * - Priority queue for urgent tasks
 * - Graceful shutdown
 *
 * Unlike Python multiprocessing, this uses shared memory threads
 * avoiding serialization overhead and memory duplication.
 */
class ThreadPool {
public:
    using Task = std::function<void()>;

    /**
     * @param num_threads Number of worker threads (0 = hardware concurrency)
     * @param set_affinity Enable CPU affinity for cache locality
     */
    explicit ThreadPool(size_t num_threads = 0, bool set_affinity = false);

    ~ThreadPool();

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * Submit task to thread pool
     * @return future for task result
     */
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * Submit task with priority (higher = more urgent)
     */
    template <typename F, typename... Args>
    auto submit_priority(int priority, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * Wait for all tasks to complete
     */
    void wait();

    /**
     * Get number of threads
     */
    size_t num_threads() const { return workers_.size(); }

    /**
     * Get number of pending tasks (approximate)
     */
    size_t pending_tasks() const;

private:
    struct PriorityTask {
        int priority;
        Task task;

        bool operator<(const PriorityTask& other) const {
            return priority < other.priority;  // Higher priority first
        }
    };

    std::vector<std::thread> workers_;
    std::priority_queue<PriorityTask> tasks_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};

    void worker_loop(size_t thread_id);
};

// Template implementations

template <typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    return submit_priority(0, std::forward<F>(f), std::forward<Args>(args)...);
}

template <typename F, typename... Args>
auto ThreadPool::submit_priority(int priority, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {

    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (stop_) {
            throw std::runtime_error("Cannot submit to stopped thread pool");
        }

        tasks_.push({priority, [task]() { (*task)(); }});
    }

    cv_.notify_one();
    return result;
}

}  // namespace turboloader
