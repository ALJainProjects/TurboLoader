/**
 * @file test_hybrid_wait_strategy.cpp
 * @brief Tests for HybridWaitStrategy (Phase 4.1)
 *
 * Tests the hybrid wait strategy for efficient queue operations:
 * - Spin-yield-sleep phases
 * - Timeout behavior
 * - CPU efficiency (no busy-wait)
 */

#include <gtest/gtest.h>
#include "../src/core/spsc_ring_buffer.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

using namespace turboloader;

class HybridWaitTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(HybridWaitTest, ImmediateTrue) {
    // Condition that's immediately true should return fast
    auto start = std::chrono::high_resolution_clock::now();

    HybridWaitStrategy::wait([&] { return true; });

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Should complete in well under 1ms
    EXPECT_LT(ns, 1000000);  // < 1ms
}

TEST_F(HybridWaitTest, DelayedTrue) {
    // Condition that becomes true after a short delay
    std::atomic<bool> flag{false};

    std::thread setter([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        flag.store(true);
    });

    auto start = std::chrono::high_resolution_clock::now();

    HybridWaitStrategy::wait([&] { return flag.load(); });

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    setter.join();

    // Should complete shortly after flag is set (~50ms + small overhead)
    EXPECT_GE(ms, 45);  // At least 45ms (flag delay minus some scheduling variance)
    EXPECT_LT(ms, 200);  // Should not take too long after flag is set
}

TEST_F(HybridWaitTest, TimeoutExpires) {
    // Condition that never becomes true - should timeout
    auto start = std::chrono::high_resolution_clock::now();

    bool result = HybridWaitStrategy::wait_for([&] { return false; }, 100);  // 100ms timeout

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_FALSE(result);
    EXPECT_GE(ms, 90);   // Should wait at least close to timeout
    EXPECT_LT(ms, 200);  // Should not overshoot by too much
}

TEST_F(HybridWaitTest, TimeoutNotNeeded) {
    // Condition that becomes true before timeout
    std::atomic<bool> flag{false};

    std::thread setter([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        flag.store(true);
    });

    auto start = std::chrono::high_resolution_clock::now();

    bool result = HybridWaitStrategy::wait_for([&] { return flag.load(); }, 500);  // 500ms timeout

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    setter.join();

    EXPECT_TRUE(result);
    EXPECT_LT(ms, 200);  // Should not wait for full timeout
}

// ============================================================================
// SPSC Queue Integration Tests
// ============================================================================

TEST_F(HybridWaitTest, SPSCQueuePush) {
    // Test hybrid wait with actual SPSC queue
    SPSCRingBuffer<int, 4> queue;

    // Fill the queue
    EXPECT_TRUE(queue.try_push(1));
    EXPECT_TRUE(queue.try_push(2));
    EXPECT_TRUE(queue.try_push(3));
    // Queue should now be full (capacity 4, but one slot reserved)

    // Start a consumer that will pop after delay
    std::thread consumer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        int val;
        queue.try_pop(val);
    });

    // Producer should wait until consumer pops
    auto start = std::chrono::high_resolution_clock::now();

    HybridWaitStrategy::wait([&] {
        return queue.try_push(4);
    });

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    consumer.join();

    EXPECT_GE(ms, 40);  // Should wait for consumer
    EXPECT_LT(ms, 200);  // Should not wait too long
}

// ============================================================================
// CPU Efficiency Test
// ============================================================================

TEST_F(HybridWaitTest, CPUEfficiencyDuringSleep) {
    // Verify that CPU usage is low during sleep phase
    // (This is hard to test directly, so we verify timing behavior)

    std::atomic<int> counter{0};
    std::atomic<bool> done{false};

    // Start a thread that counts iterations
    std::thread worker([&] {
        while (!done.load()) {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Wait for 100ms with false condition (will go into sleep phase)
    HybridWaitStrategy::wait_for([&] { return false; }, 100);

    done.store(true);
    worker.join();

    // Worker should have executed many times (not blocked by busy-wait)
    // If hybrid wait was busy-waiting, worker would be starved
    EXPECT_GT(counter.load(), 50);  // Should have many iterations in 100ms
}

// ============================================================================
// Performance Benchmark
// ============================================================================

TEST_F(HybridWaitTest, BenchmarkFastCondition) {
    // Benchmark for condition that becomes true quickly
    const int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        std::atomic<int> counter{0};
        HybridWaitStrategy::wait([&] {
            return counter.fetch_add(1) >= 1;  // True on second call
        });
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "\n=== HybridWait Benchmark (fast condition) ===" << std::endl;
    std::cout << "  " << iterations << " waits: " << ns / 1000 << " us total" << std::endl;
    std::cout << "  " << ns / iterations << " ns/wait" << std::endl;

    // Should be fast for conditions that become true quickly
    EXPECT_LT(ns / iterations, 10000);  // < 10us per wait
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "=== TurboLoader HybridWaitStrategy Tests ===" << std::endl;

    return RUN_ALL_TESTS();
}
