/**
 * @file test_spsc_ring_buffer.cpp
 * @brief Unit tests for SPSCRingBuffer
 */

#include "../../src/core_v2/spsc_ring_buffer.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

using namespace turboloader::v2;

// Test fixture helpers
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " << #name << "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

// Basic functionality tests
TEST(basic_push_pop) {
    SPSCRingBuffer<int, 8> queue;

    // Initially empty
    assert(queue.empty());
    assert(queue.size() == 0);

    // Push one item
    assert(queue.try_push(42));
    assert(!queue.empty());
    assert(queue.size() == 1);

    // Pop it back
    int value = 0;
    assert(queue.try_pop(value));
    assert(value == 42);
    assert(queue.empty());
    assert(queue.size() == 0);
}

TEST(push_until_full) {
    SPSCRingBuffer<int, 8> queue;

    // Fill the queue (capacity - 1 due to sentinel)
    for (int i = 0; i < 7; ++i) {
        assert(queue.try_push(i));
    }

    // Queue should be full
    assert(!queue.try_push(999));
    assert(queue.size() == 7);

    // Pop all items
    for (int i = 0; i < 7; ++i) {
        int value;
        assert(queue.try_pop(value));
        assert(value == i);
    }

    assert(queue.empty());
}

TEST(pop_from_empty) {
    SPSCRingBuffer<int, 8> queue;

    int value = 42;
    assert(!queue.try_pop(value));
    assert(value == 42);  // Unchanged
}

TEST(wraparound) {
    SPSCRingBuffer<int, 8> queue;

    // Fill, empty, fill again to test wraparound
    for (int iter = 0; iter < 3; ++iter) {
        for (int i = 0; i < 7; ++i) {
            assert(queue.try_push(i + iter * 100));
        }

        for (int i = 0; i < 7; ++i) {
            int value;
            assert(queue.try_pop(value));
            assert(value == i + iter * 100);
        }

        assert(queue.empty());
    }
}

TEST(move_semantics) {
    SPSCRingBuffer<std::vector<int>, 8> queue;

    std::vector<int> data = {1, 2, 3, 4, 5};
    auto* original_ptr = data.data();

    // Move into queue
    assert(queue.try_push(std::move(data)));
    assert(data.empty());  // Moved from

    // Move out of queue
    std::vector<int> result;
    assert(queue.try_pop(result));
    assert(result.size() == 5);
    assert(result.data() == original_ptr);  // Same allocation
}

// Concurrency tests
TEST(single_producer_single_consumer) {
    constexpr size_t NUM_ITEMS = 10000;
    SPSCRingBuffer<int, 256> queue;

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> items_consumed{0};

    // Producer thread
    std::thread producer([&]() {
        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            while (!queue.try_push(static_cast<int>(i))) {
                std::this_thread::yield();
            }
        }
        producer_done = true;
    });

    // Consumer thread
    std::thread consumer([&]() {
        size_t expected = 0;
        while (expected < NUM_ITEMS) {
            int value;
            if (queue.try_pop(value)) {
                assert(value == static_cast<int>(expected));
                ++expected;
                ++items_consumed;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    assert(items_consumed == NUM_ITEMS);
    assert(queue.empty());
}

TEST(high_throughput) {
    constexpr size_t NUM_ITEMS = 1000000;
    SPSCRingBuffer<int, 1024> queue;

    std::atomic<bool> start{false};

    // Producer thread
    std::thread producer([&]() {
        while (!start) std::this_thread::yield();

        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            while (!queue.try_push(static_cast<int>(i))) {
                std::this_thread::yield();
            }
        }
    });

    // Consumer thread
    std::thread consumer([&]() {
        while (!start) std::this_thread::yield();

        for (size_t i = 0; i < NUM_ITEMS; ++i) {
            int value;
            while (!queue.try_pop(value)) {
                std::this_thread::yield();
            }
            assert(value == static_cast<int>(i));
        }
    });

    // Start both threads
    auto start_time = std::chrono::high_resolution_clock::now();
    start = true;

    producer.join();
    consumer.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double throughput = NUM_ITEMS / (duration.count() / 1000.0);
    std::cout << "\n  Throughput: " << throughput / 1e6 << "M items/sec ";
}

TEST(bursts) {
    SPSCRingBuffer<int, 64> queue;

    // Simulate bursty producer
    std::thread producer([&]() {
        for (int burst = 0; burst < 100; ++burst) {
            // Burst of pushes
            for (int i = 0; i < 50; ++i) {
                while (!queue.try_push(burst * 50 + i)) {
                    std::this_thread::yield();
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Steady consumer
    std::thread consumer([&]() {
        for (int i = 0; i < 5000; ++i) {
            int value;
            while (!queue.try_pop(value)) {
                std::this_thread::yield();
            }
            assert(value == i);
        }
    });

    producer.join();
    consumer.join();
    assert(queue.empty());
}

// Stress tests
TEST(alternating_operations) {
    SPSCRingBuffer<int, 8> queue;

    // Alternate push/pop many times
    for (int i = 0; i < 10000; ++i) {
        assert(queue.try_push(i));
        int value;
        assert(queue.try_pop(value));
        assert(value == i);
        assert(queue.empty());
    }
}

TEST(large_objects) {
    struct LargeObject {
        std::array<int, 1024> data;
        int id;

        LargeObject(int i = 0) : id(i) {
            data.fill(i);
        }
    };

    SPSCRingBuffer<LargeObject, 16> queue;

    // Push/pop large objects
    for (int i = 0; i < 10; ++i) {
        assert(queue.try_push(LargeObject(i)));
    }

    for (int i = 0; i < 10; ++i) {
        LargeObject obj;
        assert(queue.try_pop(obj));
        assert(obj.id == i);
        assert(obj.data[0] == i);
    }
}

int main() {
    std::cout << "=== SPSC Ring Buffer Tests ===" << std::endl;

    RUN_TEST(basic_push_pop);
    RUN_TEST(push_until_full);
    RUN_TEST(pop_from_empty);
    RUN_TEST(wraparound);
    RUN_TEST(move_semantics);
    RUN_TEST(single_producer_single_consumer);
    RUN_TEST(high_throughput);
    RUN_TEST(bursts);
    RUN_TEST(alternating_operations);
    RUN_TEST(large_objects);

    std::cout << "\nAll SPSC Ring Buffer tests PASSED!" << std::endl;
    return 0;
}
