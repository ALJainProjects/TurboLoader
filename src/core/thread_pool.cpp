#include "turboloader/core/thread_pool.hpp"
#include <algorithm>
#include <iostream>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace turboloader {

ThreadPool::ThreadPool(size_t num_threads, bool set_affinity) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 4;  // Fallback
        }
    }

    workers_.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this, i, set_affinity]() {
            if (set_affinity) {
#ifdef __linux__
                // Set CPU affinity for better cache locality
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
            }
            worker_loop(i);
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until queue is empty and no active tasks
    cv_.wait(lock, [this]() {
        return tasks_.empty() && active_tasks_ == 0;
    });
}

size_t ThreadPool::pending_tasks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

void ThreadPool::worker_loop(size_t thread_id) {
    (void)thread_id;  // Unused for now, could be used for logging

    while (true) {
        PriorityTask task;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            cv_.wait(lock, [this]() {
                return stop_ || !tasks_.empty();
            });

            if (stop_ && tasks_.empty()) {
                return;
            }

            if (!tasks_.empty()) {
                task = std::move(const_cast<PriorityTask&>(tasks_.top()));
                tasks_.pop();
            } else {
                continue;
            }
        }

        // Execute task outside lock
        active_tasks_++;
        try {
            task.task();
        } catch (const std::exception& e) {
            // Catch and silently swallow exceptions to prevent thread termination
            // In production, this would log to stderr or a logging system
            std::cerr << "Thread pool task exception: " << e.what() << std::endl;
        } catch (...) {
            // Catch all other exceptions
            std::cerr << "Thread pool task exception: unknown error" << std::endl;
        }
        active_tasks_--;

        // Notify waiters
        cv_.notify_all();
    }
}

}  // namespace turboloader
