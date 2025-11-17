/**
 * @file test_smart_batching.cpp
 * @brief Unit tests for Smart Batching (v1.2.0)
 *
 * Tests:
 * 1. Bucket creation and sample assignment
 * 2. Size-based grouping functionality
 * 3. Batch retrieval and flushing
 * 4. Statistics tracking
 * 5. Configuration options (strict sizing, bucket limits)
 * 6. Thread safety with concurrent additions
 */

#include "../src/pipeline/smart_batching.hpp"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

using namespace turboloader::pipeline;

// ANSI color codes for pretty output
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

// Simple test sample type
struct TestSample {
    int id;
    size_t width;
    size_t height;

    TestSample(int id, size_t w, size_t h) : id(id), width(w), height(h) {}
};

// Test counter for pass/fail tracking
static int tests_passed = 0;
static int tests_failed = 0;

void test_passed(const char* name) {
    printf("%s[PASS]%s %s\n", GREEN, RESET, name);
    tests_passed++;
}

void test_failed(const char* name, const char* reason) {
    printf("%s[FAIL]%s %s: %s\n", RED, RESET, name, reason);
    tests_failed++;
}

/**
 * Test 1: Basic bucket creation and sample addition
 */
void test_bucket_creation() {
    const char* test_name = "Bucket creation and sample addition";

    try {
        SampleBucket<TestSample> bucket(224, 224, 100);

        // Add sample with exact size
        TestSample sample1(1, 224, 224);
        bool added = bucket.try_add(sample1, 224, 224, 32, 32);

        if (!added) {
            test_failed(test_name, "Failed to add sample with exact size");
            return;
        }

        if (bucket.size() != 1) {
            test_failed(test_name, "Bucket size incorrect after adding sample");
            return;
        }

        // Add sample within tolerance (224 ± 32)
        TestSample sample2(2, 240, 240);
        added = bucket.try_add(sample2, 240, 240, 32, 32);

        if (!added) {
            test_failed(test_name, "Failed to add sample within tolerance");
            return;
        }

        if (bucket.size() != 2) {
            test_failed(test_name, "Bucket size incorrect after second sample");
            return;
        }

        // Try to add sample outside tolerance
        TestSample sample3(3, 300, 300);
        added = bucket.try_add(sample3, 300, 300, 32, 32);

        if (added) {
            test_failed(test_name, "Should not add sample outside tolerance");
            return;
        }

        if (bucket.size() != 2) {
            test_failed(test_name, "Bucket size changed after rejected sample");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 2: Bucket capacity limits
 */
void test_bucket_capacity() {
    const char* test_name = "Bucket capacity limits";

    try {
        SampleBucket<TestSample> bucket(224, 224, 10);  // Max 10 samples

        // Add 10 samples (should succeed)
        for (int i = 0; i < 10; ++i) {
            TestSample sample(i, 224, 224);
            bool added = bucket.try_add(sample, 224, 224, 32, 32);
            if (!added) {
                test_failed(test_name, "Failed to add sample within capacity");
                return;
            }
        }

        if (bucket.size() != 10) {
            test_failed(test_name, "Bucket size incorrect after filling");
            return;
        }

        // Try to add 11th sample (should fail)
        TestSample sample11(11, 224, 224);
        bool added = bucket.try_add(sample11, 224, 224, 32, 32);

        if (added) {
            test_failed(test_name, "Should not exceed capacity");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 3: Bucket flush functionality
 */
void test_bucket_flush() {
    const char* test_name = "Bucket flush functionality";

    try {
        SampleBucket<TestSample> bucket(224, 224, 100);

        // Add 5 samples
        for (int i = 0; i < 5; ++i) {
            TestSample sample(i, 224, 224);
            bucket.try_add(sample, 224, 224, 32, 32);
        }

        // Check ready status (min_size = 3)
        if (!bucket.is_ready(3)) {
            test_failed(test_name, "Bucket should be ready with min_size=3");
            return;
        }

        // Flush bucket
        auto samples = bucket.flush();

        if (samples.size() != 5) {
            test_failed(test_name, "Flush should return all 5 samples");
            return;
        }

        if (bucket.size() != 0) {
            test_failed(test_name, "Bucket should be empty after flush");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 4: SmartBatcher basic functionality
 */
void test_smart_batcher_basic() {
    const char* test_name = "SmartBatcher basic functionality";

    try {
        SmartBatchConfig config;
        config.bucket_width_step = 32;
        config.bucket_height_step = 32;
        config.min_bucket_size = 4;
        config.max_bucket_size = 16;

        SmartBatcher<TestSample> batcher(config);

        // Add samples with similar sizes (should go to same bucket)
        for (int i = 0; i < 8; ++i) {
            TestSample sample(i, 224, 224);
            bool added = batcher.add_sample(sample, 224, 224);
            if (!added) {
                test_failed(test_name, "Failed to add sample to batcher");
                return;
            }
        }

        // Check stats
        auto stats = batcher.get_stats();
        if (stats.num_buckets != 1) {
            test_failed(test_name, "Should have exactly 1 bucket");
            return;
        }

        if (stats.total_samples != 8) {
            test_failed(test_name, "Should have 8 total samples");
            return;
        }

        // Get ready batches (min_bucket_size = 4)
        auto batches = batcher.get_ready_batches();

        if (batches.size() != 1) {
            test_failed(test_name, "Should have 1 ready batch");
            return;
        }

        if (batches[0].size() != 8) {
            test_failed(test_name, "Batch should contain all 8 samples");
            return;
        }

        // Verify bucket is now empty
        stats = batcher.get_stats();
        if (stats.total_samples != 0) {
            test_failed(test_name, "Bucket should be empty after getting batches");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 5: Multiple buckets with different sizes
 */
void test_multiple_buckets() {
    const char* test_name = "Multiple buckets with different sizes";

    try {
        SmartBatchConfig config;
        config.bucket_width_step = 64;
        config.bucket_height_step = 64;
        config.min_bucket_size = 2;
        config.max_bucket_size = 32;

        SmartBatcher<TestSample> batcher(config);

        // Add samples with 3 different size groups
        // Group 1: 224x224 (3 samples)
        for (int i = 0; i < 3; ++i) {
            TestSample sample(i, 224, 224);
            batcher.add_sample(sample, 224, 224);
        }

        // Group 2: 320x320 (4 samples)
        for (int i = 10; i < 14; ++i) {
            TestSample sample(i, 320, 320);
            batcher.add_sample(sample, 320, 320);
        }

        // Group 3: 512x512 (5 samples)
        for (int i = 20; i < 25; ++i) {
            TestSample sample(i, 512, 512);
            batcher.add_sample(sample, 512, 512);
        }

        // Check stats
        auto stats = batcher.get_stats();
        if (stats.num_buckets != 3) {
            test_failed(test_name, "Should have 3 buckets");
            return;
        }

        if (stats.total_samples != 12) {
            test_failed(test_name, "Should have 12 total samples");
            return;
        }

        // Get ready batches
        auto batches = batcher.get_ready_batches();

        if (batches.size() != 3) {
            test_failed(test_name, "Should have 3 ready batches");
            return;
        }

        // Verify batch sizes
        std::vector<size_t> batch_sizes;
        for (const auto& batch : batches) {
            batch_sizes.push_back(batch.size());
        }
        std::sort(batch_sizes.begin(), batch_sizes.end());

        if (batch_sizes[0] != 3 || batch_sizes[1] != 4 || batch_sizes[2] != 5) {
            test_failed(test_name, "Batch sizes incorrect");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 6: Bucket size bucketing algorithm
 */
void test_bucket_key_calculation() {
    const char* test_name = "Bucket key calculation";

    try {
        SmartBatchConfig config;
        config.bucket_width_step = 32;
        config.bucket_height_step = 32;
        config.min_bucket_size = 1;

        SmartBatcher<TestSample> batcher(config);

        // Samples with exact bucket boundary (224 = 7*32)
        // and samples within tolerance (±32)
        TestSample s1(1, 224, 224);  // Exact bucket center
        TestSample s2(2, 220, 220);  // Within tolerance (224-32=192, 224+32=256)
        TestSample s3(3, 240, 240);  // Within tolerance

        batcher.add_sample(s1, 224, 224);
        batcher.add_sample(s2, 220, 220);
        batcher.add_sample(s3, 240, 240);

        auto stats = batcher.get_stats();

        // Note: 220 rounds to bucket 192 (6*32), 224 to 224 (7*32), 240 to 224 (7*32)
        // So we expect 2 buckets: one for 220, one for 224 and 240
        if (stats.num_buckets > 2) {
            test_failed(test_name, "Should have at most 2 buckets");
            return;
        }

        if (stats.total_samples != 3) {
            test_failed(test_name, "All 3 samples should be added");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 7: Flush all buckets
 */
void test_flush_all() {
    const char* test_name = "Flush all buckets";

    try {
        SmartBatchConfig config;
        config.min_bucket_size = 10;  // High threshold
        config.max_bucket_size = 32;

        SmartBatcher<TestSample> batcher(config);

        // Add samples that won't meet min_bucket_size threshold
        for (int i = 0; i < 5; ++i) {
            TestSample sample(i, 224, 224);
            batcher.add_sample(sample, 224, 224);
        }

        // get_ready_batches should return nothing (below threshold)
        auto ready_batches = batcher.get_ready_batches();
        if (ready_batches.size() != 0) {
            test_failed(test_name, "Should have no ready batches (below threshold)");
            return;
        }

        // flush_all should return all batches regardless of size
        auto all_batches = batcher.flush_all();
        if (all_batches.size() != 1) {
            test_failed(test_name, "flush_all should return 1 batch");
            return;
        }

        if (all_batches[0].size() != 5) {
            test_failed(test_name, "Batch should have 5 samples");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 8: Max buckets limit
 */
void test_max_buckets_limit() {
    const char* test_name = "Max buckets limit";

    try {
        SmartBatchConfig config;
        config.bucket_width_step = 64;
        config.bucket_height_step = 64;
        config.max_buckets = 3;
        config.enable_dynamic_buckets = true;

        SmartBatcher<TestSample> batcher(config);

        // Try to create 5 different size buckets
        std::vector<size_t> sizes = {128, 256, 384, 512, 640};

        int added_count = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            TestSample sample(i, sizes[i], sizes[i]);
            if (batcher.add_sample(sample, sizes[i], sizes[i])) {
                added_count++;
            }
        }

        auto stats = batcher.get_stats();
        if (stats.num_buckets > 3) {
            test_failed(test_name, "Should not exceed max_buckets limit");
            return;
        }

        // Should have created exactly 3 buckets
        if (stats.num_buckets != 3) {
            test_failed(test_name, "Should have exactly 3 buckets");
            return;
        }

        // Should have added exactly 3 samples (one per bucket)
        if (stats.total_samples != 3) {
            test_failed(test_name, "Should have 3 samples (max_buckets limit)");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 9: Thread safety - concurrent additions
 */
void test_thread_safety() {
    const char* test_name = "Thread safety - concurrent additions";

    try {
        SmartBatchConfig config;
        config.bucket_width_step = 32;
        config.bucket_height_step = 32;
        config.min_bucket_size = 1;
        config.max_bucket_size = 1000;

        SmartBatcher<TestSample> batcher(config);

        const int num_threads = 8;
        const int samples_per_thread = 50;

        std::vector<std::thread> threads;

        // Launch threads to add samples concurrently
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&batcher, t, samples_per_thread]() {
                for (int i = 0; i < samples_per_thread; ++i) {
                    int id = t * samples_per_thread + i;
                    TestSample sample(id, 224, 224);
                    batcher.add_sample(sample, 224, 224);
                }
            });
        }

        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }

        // Check that all samples were added
        auto stats = batcher.get_stats();
        int expected_samples = num_threads * samples_per_thread;

        if (stats.total_samples != static_cast<size_t>(expected_samples)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Expected %d samples, got %zu",
                     expected_samples, stats.total_samples);
            test_failed(test_name, msg);
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

/**
 * Test 10: Statistics accuracy
 */
void test_statistics() {
    const char* test_name = "Statistics accuracy";

    try {
        SmartBatchConfig config;
        config.bucket_width_step = 64;
        config.bucket_height_step = 64;
        config.min_bucket_size = 1;
        config.max_bucket_size = 100;

        SmartBatcher<TestSample> batcher(config);

        // Bucket 1: 5 samples of size 224x224
        for (int i = 0; i < 5; ++i) {
            TestSample sample(i, 224, 224);
            batcher.add_sample(sample, 224, 224);
        }

        // Bucket 2: 10 samples of size 512x512
        for (int i = 0; i < 10; ++i) {
            TestSample sample(i + 10, 512, 512);
            batcher.add_sample(sample, 512, 512);
        }

        // Bucket 3: 3 samples of size 768x768
        for (int i = 0; i < 3; ++i) {
            TestSample sample(i + 20, 768, 768);
            batcher.add_sample(sample, 768, 768);
        }

        auto stats = batcher.get_stats();

        if (stats.num_buckets != 3) {
            test_failed(test_name, "num_buckets incorrect");
            return;
        }

        if (stats.total_samples != 18) {
            test_failed(test_name, "total_samples incorrect");
            return;
        }

        if (stats.min_bucket_size != 3) {
            test_failed(test_name, "min_bucket_size incorrect");
            return;
        }

        if (stats.max_bucket_size != 10) {
            test_failed(test_name, "max_bucket_size incorrect");
            return;
        }

        // avg = (5 + 10 + 3) / 3 = 6
        if (stats.avg_bucket_size != 6) {
            test_failed(test_name, "avg_bucket_size incorrect");
            return;
        }

        test_passed(test_name);

    } catch (const std::exception& e) {
        test_failed(test_name, e.what());
    }
}

int main() {
    printf("\n");
    printf("%s========================================%s\n", BOLD, RESET);
    printf("%s  SMART BATCHING TESTS (v1.2.0)       %s\n", BOLD, RESET);
    printf("%s========================================%s\n", BOLD, RESET);
    printf("\n");

    // Run all tests
    test_bucket_creation();
    test_bucket_capacity();
    test_bucket_flush();
    test_smart_batcher_basic();
    test_multiple_buckets();
    test_bucket_key_calculation();
    test_flush_all();
    test_max_buckets_limit();
    test_thread_safety();
    test_statistics();

    // Print summary
    printf("\n");
    printf("%s========================================%s\n", BOLD, RESET);
    printf("%s  TEST SUMMARY                         %s\n", BOLD, RESET);
    printf("%s========================================%s\n", BOLD, RESET);
    printf("Total tests: %d\n", tests_passed + tests_failed);
    printf("%sPassed: %d%s\n", GREEN, tests_passed, RESET);

    if (tests_failed > 0) {
        printf("%sFailed: %d%s\n", RED, tests_failed, RESET);
    } else {
        printf("Failed: 0\n");
    }

    printf("%s========================================%s\n", BOLD, RESET);
    printf("\n");

    return tests_failed > 0 ? 1 : 0;
}
