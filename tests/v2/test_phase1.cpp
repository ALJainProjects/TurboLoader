/**
 * @file test_phase1.cpp
 * @brief Combined unit tests for Phase 1 components
 *
 * Tests: SPSC Ring Buffer, Object Pool, and Sample V2
 */

#include "../../src/core_v2/spsc_ring_buffer.hpp"
#include "../../src/core_v2/object_pool.hpp"
#include "../../src/core_v2/sample_v2.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <array>

using namespace turboloader::v2;

// Test framework macros
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "  " << #name << "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

// ============================================================================
// SPSC Ring Buffer Tests
// ============================================================================

TEST(spsc_basic) {
    SPSCRingBuffer<int, 8> queue;
    assert(queue.empty());
    assert(queue.try_push(42));
    int val;
    assert(queue.try_pop(val));
    assert(val == 42);
    assert(queue.empty());
}

TEST(spsc_concurrent) {
    constexpr size_t NUM = 10000;
    SPSCRingBuffer<int, 256> queue;

    std::thread producer([&]() {
        for (size_t i = 0; i < NUM; ++i) {
            while (!queue.try_push(static_cast<int>(i))) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (size_t i = 0; i < NUM; ++i) {
            int value;
            while (!queue.try_pop(value)) {
                std::this_thread::yield();
            }
            assert(value == static_cast<int>(i));
        }
    });

    producer.join();
    consumer.join();
    assert(queue.empty());
}

TEST(spsc_move_semantics) {
    SPSCRingBuffer<std::vector<int>, 8> queue;
    std::vector<int> data = {1, 2, 3};
    assert(queue.try_push(std::move(data)));
    assert(data.empty());  // Moved from

    std::vector<int> result;
    assert(queue.try_pop(result));
    assert(result.size() == 3);
}

// ============================================================================
// Object Pool Tests
// ============================================================================

TEST(pool_acquire_release) {
    ObjectPool<std::vector<int>> pool(4, 16);

    // Acquire object
    auto obj = pool.acquire();
    assert(obj != nullptr);
    obj->push_back(42);

    // Release (happens automatically via deleter)
    obj.reset();

    // Acquire again - should get same object from pool
    auto obj2 = pool.acquire();
    assert(obj2 != nullptr);
}

TEST(pool_multiple_objects) {
    ObjectPool<int> pool(4, 16);

    std::vector<decltype(pool.acquire())> objects;
    for (int i = 0; i < 10; ++i) {
        objects.push_back(pool.acquire());
        *objects.back() = i;
    }

    // Verify all objects have correct values
    for (size_t i = 0; i < objects.size(); ++i) {
        assert(*objects[i] == static_cast<int>(i));
    }
}

TEST(pool_buffer_specialization) {
    BufferPool pool(1024, 4, 16);

    auto buf1 = pool.acquire();
    buf1->resize(512, 0xFF);

    auto buf2 = pool.acquire();
    assert(buf2->empty());  // Should be cleared

    // Release buf1 and acquire again
    buf1.reset();
    auto buf3 = pool.acquire();
    assert(buf3->empty());  // Should be cleared even though reused
}

TEST(pool_concurrent) {
    ObjectPool<int> pool(16, 64);
    std::atomic<int> sum{0};

    auto worker = [&]() {
        for (int i = 0; i < 1000; ++i) {
            auto obj = pool.acquire();
            *obj = 1;
            sum += *obj;
        }
    };

    std::thread t1(worker);
    std::thread t2(worker);
    std::thread t3(worker);
    std::thread t4(worker);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    assert(sum == 4000);
}

// ============================================================================
// Sample V2 Tests
// ============================================================================

TEST(sample_basic) {
    SampleV2 sample;
    assert(sample.index == 0);
    assert(sample.jpeg_size() == 0);
    assert(sample.decoded_size() == 0);
    assert(!sample.is_decoded());
}

TEST(sample_with_jpeg_data) {
    std::vector<uint8_t> jpeg_bytes = {0xFF, 0xD8, 0xFF, 0xE0};
    std::span<const uint8_t> jpeg_span(jpeg_bytes);

    SampleV2 sample(42, jpeg_span);
    assert(sample.index == 42);
    assert(sample.jpeg_size() == 4);
    assert(!sample.is_decoded());
}

TEST(sample_decoded) {
    SampleV2 sample;
    sample.decoded_rgb.resize(256 * 256 * 3, 128);
    sample.width = 256;
    sample.height = 256;
    sample.channels = 3;

    assert(sample.is_decoded());
    assert(sample.decoded_size() == 256 * 256 * 3);

    sample.clear_decoded();
    assert(!sample.is_decoded());
    assert(sample.width == 0);
}

TEST(sample_move_semantics) {
    SampleV2 sample1;
    sample1.index = 100;
    sample1.decoded_rgb.resize(1000, 42);

    // Move construct
    SampleV2 sample2(std::move(sample1));
    assert(sample2.index == 100);
    assert(sample2.decoded_rgb.size() == 1000);

    // Move assign
    SampleV2 sample3;
    sample3 = std::move(sample2);
    assert(sample3.index == 100);
}

TEST(batch_basic) {
    BatchV2 batch(32);
    assert(batch.empty());
    assert(batch.size() == 0);

    batch.add(SampleV2());
    assert(batch.size() == 1);
    assert(!batch.empty());

    batch.clear();
    assert(batch.empty());
}

TEST(batch_iteration) {
    BatchV2 batch;
    for (int i = 0; i < 10; ++i) {
        SampleV2 sample;
        sample.index = i;
        batch.add(std::move(sample));
    }

    assert(batch.size() == 10);

    // Index access
    assert(batch[0].index == 0);
    assert(batch[9].index == 9);

    // Iterator access
    size_t count = 0;
    for (const auto& sample : batch) {
        assert(sample.index == count);
        ++count;
    }
    assert(count == 10);
}

TEST(batch_memory_usage) {
    BatchV2 batch;

    for (int i = 0; i < 5; ++i) {
        SampleV2 sample;
        sample.decoded_rgb.resize(1000, i);
        batch.add(std::move(sample));
    }

    size_t total = batch.memory_usage();
    assert(total == 5 * 1000);  // 5 samples * 1000 bytes each
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(integration_queue_and_pool) {
    // Simulate pipeline: queue of samples with pooled buffers
    SPSCRingBuffer<SampleV2, 16> queue;
    BufferPool pool(1024, 8, 32);

    std::thread producer([&]() {
        for (int i = 0; i < 100; ++i) {
            SampleV2 sample;
            sample.index = i;

            // Get buffer from pool
            auto buf = pool.acquire();
            buf->resize(128, static_cast<uint8_t>(i % 256));
            sample.decoded_rgb = std::move(*buf);
            sample.width = 16;
            sample.height = 8;
            sample.channels = 1;

            while (!queue.try_push(std::move(sample))) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (int i = 0; i < 100; ++i) {
            SampleV2 sample;
            while (!queue.try_pop(sample)) {
                std::this_thread::yield();
            }

            assert(sample.index == static_cast<size_t>(i));
            assert(sample.decoded_size() == 128);
            assert(sample.is_decoded());
        }
    });

    producer.join();
    consumer.join();
}

TEST(integration_batch_processing) {
    SPSCRingBuffer<BatchV2, 8> batch_queue;

    std::thread producer([&]() {
        for (int batch_idx = 0; batch_idx < 10; ++batch_idx) {
            BatchV2 batch(32);
            for (int i = 0; i < 32; ++i) {
                SampleV2 sample;
                sample.index = batch_idx * 32 + i;
                batch.add(std::move(sample));
            }

            while (!batch_queue.try_push(std::move(batch))) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (int batch_idx = 0; batch_idx < 10; ++batch_idx) {
            BatchV2 batch;
            while (!batch_queue.try_pop(batch)) {
                std::this_thread::yield();
            }

            assert(batch.size() == 32);
            for (size_t i = 0; i < batch.size(); ++i) {
                assert(batch[i].index == static_cast<size_t>(batch_idx * 32 + i));
            }
        }
    });

    producer.join();
    consumer.join();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  TurboLoader v2.0 - Phase 1 Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nSPSC Ring Buffer:" << std::endl;
    RUN_TEST(spsc_basic);
    RUN_TEST(spsc_concurrent);
    RUN_TEST(spsc_move_semantics);

    std::cout << "\nObject Pool:" << std::endl;
    RUN_TEST(pool_acquire_release);
    RUN_TEST(pool_multiple_objects);
    RUN_TEST(pool_buffer_specialization);
    RUN_TEST(pool_concurrent);

    std::cout << "\nSample V2:" << std::endl;
    RUN_TEST(sample_basic);
    RUN_TEST(sample_with_jpeg_data);
    RUN_TEST(sample_decoded);
    RUN_TEST(sample_move_semantics);

    std::cout << "\nBatch V2:" << std::endl;
    RUN_TEST(batch_basic);
    RUN_TEST(batch_iteration);
    RUN_TEST(batch_memory_usage);

    std::cout << "\nIntegration Tests:" << std::endl;
    RUN_TEST(integration_queue_and_pool);
    RUN_TEST(integration_batch_processing);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  ALL PHASE 1 TESTS PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
