/**
 * @file test_hybrid_wait.cpp
 * @brief Tests for Phase 3: Hybrid wait strategy
 *
 * Tests:
 * 1. Basic wait functionality
 * 2. Spin phase behavior
 * 3. CV fallback behavior
 * 4. Multi-threaded producer-consumer
 * 5. Shutdown handling
 * 6. Performance comparison with busy-wait
 */

#include "../src/core/hybrid_wait.hpp"
#include "../src/core/spsc_ring_buffer.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Test basic wait functionality
 */
void test_basic_wait() {
    std::cout << BOLD << "\n[TEST] Basic Wait Functionality" << RESET << std::endl;

    HybridWait wait;
    std::atomic<bool> running{true};
    std::atomic<int> value{0};

    // Test immediate satisfaction
    value.store(42);
    bool result = wait.wait_for(
        [&]() { return value.load() == 42; },
        running
    );
    assert(result);
    std::cout << "  " << GREEN << "✓" << RESET << " Immediate satisfaction" << std::endl;

    // Test with delay
    value.store(0);
    std::thread setter([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        value.store(100);
        wait.notify_one();
    });

    result = wait.wait_for(
        [&]() { return value.load() == 100; },
        running
    );
    setter.join();
    assert(result);
    assert(value.load() == 100);
    std::cout << "  " << GREEN << "✓" << RESET << " Delayed satisfaction" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test spin phase behavior
 */
void test_spin_phase() {
    std::cout << BOLD << "\n[TEST] Spin Phase Behavior" << RESET << std::endl;

    // Configure with many spin iterations
    HybridWaitConfig config;
    config.spin_iterations = 1000;
    config.cv_timeout_us = 1000;

    HybridWait wait(config);
    std::atomic<bool> running{true};
    std::atomic<int> counter{0};
    std::atomic<int> checks{0};

    // This should complete within spin phase (no CV needed)
    std::thread setter([&]() {
        // Short delay - should complete during spin
        for (int i = 0; i < 100; ++i) {
            std::this_thread::yield();
        }
        counter.store(1);
    });

    auto start = std::chrono::high_resolution_clock::now();
    bool result = wait.wait_for(
        [&]() {
            checks.fetch_add(1);
            return counter.load() == 1;
        },
        running
    );
    auto end = std::chrono::high_resolution_clock::now();
    setter.join();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    assert(result);
    std::cout << "  Checks performed: " << checks.load() << std::endl;
    std::cout << "  Duration: " << duration_us << " us" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Spin phase completed" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test CV fallback behavior
 */
void test_cv_fallback() {
    std::cout << BOLD << "\n[TEST] CV Fallback Behavior" << RESET << std::endl;

    // Configure with few spin iterations to force CV fallback
    HybridWaitConfig config;
    config.spin_iterations = 2;
    config.cv_timeout_us = 1000;  // 1ms

    HybridWait wait(config);
    std::atomic<bool> running{true};
    std::atomic<int> value{0};

    // This should require CV wait
    std::thread setter([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        value.store(1);
        wait.notify_one();
    });

    auto start = std::chrono::high_resolution_clock::now();
    bool result = wait.wait_for(
        [&]() { return value.load() == 1; },
        running
    );
    auto end = std::chrono::high_resolution_clock::now();
    setter.join();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    assert(result);
    assert(duration_ms >= 40);  // Should have waited ~50ms
    std::cout << "  Duration: " << duration_ms << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " CV fallback completed" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test shutdown handling
 */
void test_shutdown() {
    std::cout << BOLD << "\n[TEST] Shutdown Handling" << RESET << std::endl;

    HybridWait wait;
    std::atomic<bool> running{true};
    std::atomic<int> value{0};

    // Never satisfy the condition, but shutdown
    std::thread stopper([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        running.store(false);
        wait.notify_all();
    });

    auto start = std::chrono::high_resolution_clock::now();
    bool result = wait.wait_for(
        [&]() { return value.load() == 100; },  // Will never be true
        running
    );
    auto end = std::chrono::high_resolution_clock::now();
    stopper.join();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    assert(!result);  // Should return false due to shutdown
    std::cout << "  Duration: " << duration_ms << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Shutdown handled correctly" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test with SPSC queue using hybrid_push/hybrid_pop
 */
void test_queue_integration() {
    std::cout << BOLD << "\n[TEST] Queue Integration" << RESET << std::endl;

    SPSCRingBuffer<int, 16> queue;
    HybridWait producer_wait;
    HybridWait consumer_wait;
    std::atomic<bool> running{true};

    const int num_items = 1000;
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    int sum_produced = 0;
    int sum_consumed = 0;

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_items && running.load(); ++i) {
            int item = i;
            sum_produced += item;

            // Use hybrid push
            bool pushed = hybrid_push(queue, std::move(item), running, producer_wait);
            if (pushed) {
                produced.fetch_add(1);
                consumer_wait.notify_one();
            }
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        for (int i = 0; i < num_items && running.load(); ++i) {
            int item;

            // Use hybrid pop
            bool popped = hybrid_pop(queue, item, running, consumer_wait);
            if (popped) {
                sum_consumed += item;
                consumed.fetch_add(1);
                producer_wait.notify_one();
            }
        }
    });

    producer.join();
    consumer.join();

    std::cout << "  Produced: " << produced.load() << std::endl;
    std::cout << "  Consumed: " << consumed.load() << std::endl;
    std::cout << "  Sum produced: " << sum_produced << std::endl;
    std::cout << "  Sum consumed: " << sum_consumed << std::endl;

    assert(produced.load() == num_items);
    assert(consumed.load() == num_items);
    assert(sum_produced == sum_consumed);

    std::cout << "  " << GREEN << "✓" << RESET << " Queue integration successful" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test waiter count tracking
 */
void test_waiter_count() {
    std::cout << BOLD << "\n[TEST] Waiter Count Tracking" << RESET << std::endl;

    HybridWaitConfig config;
    config.spin_iterations = 2;
    config.cv_timeout_us = 10000;  // 10ms

    HybridWait wait(config);
    std::atomic<bool> running{true};
    std::atomic<int> value{0};
    std::atomic<bool> thread_started{false};

    assert(wait.waiter_count() == 0);
    std::cout << "  Initial waiter count: 0" << std::endl;

    // Start a waiter thread
    std::thread waiter([&]() {
        thread_started.store(true);
        wait.wait_for(
            [&]() { return value.load() == 1; },
            running
        );
    });

    // Wait for thread to start and enter wait
    while (!thread_started.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Should have waiter now
    size_t count = wait.waiter_count();
    std::cout << "  Waiter count during wait: " << count << std::endl;

    // Release waiter
    value.store(1);
    wait.notify_one();
    waiter.join();

    // Count should be back to 0
    assert(wait.waiter_count() == 0);
    std::cout << "  Waiter count after release: 0" << std::endl;

    std::cout << "  " << GREEN << "✓" << RESET << " Waiter count tracking correct" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Performance comparison: busy-wait vs hybrid wait
 */
void test_performance_comparison() {
    std::cout << BOLD << "\n[TEST] Performance Comparison" << RESET << std::endl;

    const int iterations = 10000;
    std::atomic<bool> running{true};

    // Test 1: Pure busy-wait (baseline)
    {
        SPSCRingBuffer<int, 256> queue;
        std::atomic<int> consumed{0};

        auto start = std::chrono::high_resolution_clock::now();

        std::thread producer([&]() {
            for (int i = 0; i < iterations && running.load(); ++i) {
                int item = i;
                while (running.load() && !queue.try_push(std::move(item))) {
                    std::this_thread::yield();
                }
            }
        });

        std::thread consumer([&]() {
            for (int i = 0; i < iterations && running.load(); ++i) {
                int item;
                while (running.load() && !queue.try_pop(item)) {
                    std::this_thread::yield();
                }
                consumed.fetch_add(1);
            }
        });

        producer.join();
        consumer.join();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "  Busy-wait: " << iterations << " items in " << duration_us << " us"
                  << " (" << (iterations * 1000000.0 / duration_us) << " ops/s)" << std::endl;
    }

    // Test 2: Hybrid wait
    {
        SPSCRingBuffer<int, 256> queue;
        HybridWait producer_wait;
        HybridWait consumer_wait;
        std::atomic<int> consumed{0};

        auto start = std::chrono::high_resolution_clock::now();

        std::thread producer([&]() {
            for (int i = 0; i < iterations && running.load(); ++i) {
                int item = i;
                hybrid_push(queue, std::move(item), running, producer_wait);
                consumer_wait.notify_one();
            }
        });

        std::thread consumer([&]() {
            for (int i = 0; i < iterations && running.load(); ++i) {
                int item;
                hybrid_pop(queue, item, running, consumer_wait);
                producer_wait.notify_one();
                consumed.fetch_add(1);
            }
        });

        producer.join();
        consumer.join();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "  Hybrid wait: " << iterations << " items in " << duration_us << " us"
                  << " (" << (iterations * 1000000.0 / duration_us) << " ops/s)" << std::endl;
    }

    std::cout << "  " << GREEN << "✓" << RESET << " Performance comparison complete" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test max wait timeout
 */
void test_max_wait_timeout() {
    std::cout << BOLD << "\n[TEST] Max Wait Timeout" << RESET << std::endl;

    HybridWaitConfig config;
    config.spin_iterations = 2;
    config.cv_timeout_us = 1000;
    config.max_wait_ms = 100;  // 100ms max wait

    HybridWait wait(config);
    std::atomic<bool> running{true};
    std::atomic<int> value{0};

    auto start = std::chrono::high_resolution_clock::now();
    bool result = wait.wait_for(
        [&]() { return value.load() == 100; },  // Will never be true
        running
    );
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    assert(!result);  // Should timeout
    assert(duration_ms >= 90 && duration_ms <= 150);  // Should be around 100ms

    std::cout << "  Timeout duration: " << duration_ms << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Max wait timeout works" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

int main() {
    std::cout << BOLD << "\n========================================" << std::endl;
    std::cout << "TurboLoader Phase 3: Hybrid Wait Tests" << std::endl;
    std::cout << "========================================" << RESET << std::endl;

    try {
        test_basic_wait();
        test_spin_phase();
        test_cv_fallback();
        test_shutdown();
        test_queue_integration();
        test_waiter_count();
        test_performance_comparison();
        test_max_wait_timeout();

        std::cout << BOLD << "\n========================================" << std::endl;
        std::cout << GREEN << "ALL TESTS PASSED" << RESET << std::endl;
        std::cout << "========================================" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "ERROR: " << e.what() << RESET << std::endl;
        return 1;
    }
}
