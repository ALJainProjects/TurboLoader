#pragma once

// A small persistent thread pool providing a blocking parallel_for. This replaces
// OpenMP (`#pragma omp parallel for`), which TurboLoader cannot use because linking
// a second OpenMP runtime crashes alongside PyTorch on macOS. The pool is created
// once (global singleton) so there is no per-call thread spawn cost.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace turboloader {

class ThreadPool {
public:
    explicit ThreadPool(unsigned num_workers) {
        for (unsigned i = 0; i < num_workers; ++i) {
            threads_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    // Process-wide pool. Sized to the hardware so the data-parallel assembly can
    // use every core; the calling thread participates too (so workers = cores - 1).
    static ThreadPool& global() {
        static ThreadPool pool(pool_workers());
        return pool;
    }

    // Number of threads that execute a parallel_for (workers + the caller).
    size_t concurrency() const { return threads_.size() + 1; }

    // Run body(i) for i in [0, n). Blocks until every index is processed. The
    // calling thread participates. Indices are handed out in small contiguous
    // chunks via an atomic cursor (cheap dynamic load balancing).
    void parallel_for(size_t n, const std::function<void(size_t)>& body) {
        const size_t nthreads = threads_.size() + 1;
        if (n == 0) return;
        if (n == 1 || nthreads == 1) {
            for (size_t i = 0; i < n; ++i) body(i);
            return;
        }
        // One dispatch at a time: body_/cursor_/remaining_/generation_ are shared, so two
        // concurrent callers corrupt the dispatch state (observed as a deadlock when two
        // loaders' prefetch producers fill batches simultaneously). Serializing here keeps
        // every caller correct; the second caller simply waits its turn. Uncontended cost
        // is a single cheap mutex acquire.
        std::lock_guard<std::mutex> dispatch_lock(dispatch_mu_);
        {
            std::unique_lock<std::mutex> lk(mu_);
            body_ = &body;
            total_ = n;
            cursor_.store(0, std::memory_order_relaxed);
            remaining_ = threads_.size();  // workers we must wait for
            ++generation_;
        }
        cv_.notify_all();
        run_chunks();  // caller participates
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [this] { return remaining_ == 0; });
    }

private:
    static unsigned pool_workers() {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc < 2) hc = 2;
        return hc - 1;  // + the calling thread == hc
    }

    void run_chunks() {
        constexpr size_t kChunk = 2;
        const size_t total = total_;
        const std::function<void(size_t)>* body = body_;
        for (;;) {
            size_t start = cursor_.fetch_add(kChunk, std::memory_order_relaxed);
            if (start >= total) break;
            size_t end = std::min(start + kChunk, total);
            for (size_t i = start; i < end; ++i) (*body)(i);
        }
    }

    void worker_loop() {
        uint64_t last_gen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this, last_gen] { return stop_ || generation_ != last_gen; });
                if (stop_) return;
                last_gen = generation_;
            }
            run_chunks();
            {
                std::unique_lock<std::mutex> lk(done_mu_);
                if (--remaining_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex dispatch_mu_;  // serializes whole parallel_for dispatches (single-caller pool)
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    uint64_t generation_ = 0;
    const std::function<void(size_t)>* body_ = nullptr;
    size_t total_ = 0;
    std::atomic<size_t> cursor_{0};

    std::mutex done_mu_;
    std::condition_variable done_cv_;
    size_t remaining_ = 0;
};

// Convenience: run body(i) for i in [0, n) on the global pool.
inline void parallel_for(size_t n, const std::function<void(size_t)>& body) {
    ThreadPool::global().parallel_for(n, body);
}

}  // namespace turboloader
