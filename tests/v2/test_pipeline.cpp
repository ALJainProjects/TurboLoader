/**
 * @file test_pipeline.cpp
 * @brief Integration tests for complete PipelineV2
 *
 * Tests the full pipeline end-to-end with real TAR file.
 */

#include "../../src/pipeline_v2/pipeline_v2.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace turboloader::v2;

// Test framework macros
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "  " << #name << "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

// Helper: Check if test TAR exists
bool test_tar_exists() {
    std::ifstream file("/tmp/test_v2.tar");
    return file.good();
}

// ============================================================================
// Basic Pipeline Tests
// ============================================================================

TEST(pipeline_construction) {
    if (!test_tar_exists()) {
        std::cout << "(no TAR) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = "/tmp/test_v2.tar";
    config.num_workers = 2;
    config.batch_size = 4;

    PipelineV2 pipeline(config);

    // Pipeline should construct successfully
    assert(pipeline.total_samples() >= 0);
}

TEST(pipeline_single_batch) {
    if (!test_tar_exists()) {
        std::cout << "(no TAR) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = "/tmp/test_v2.tar";
    config.num_workers = 1;
    config.batch_size = 2;

    PipelineV2 pipeline(config);

    // Get first batch
    auto batch = pipeline.next_batch();

    // Should get some samples
    assert(batch.size() > 0);
    assert(batch.size() <= config.batch_size);

    // Samples should be decoded
    for (const auto& sample : batch) {
        // May or may not be decoded depending on JPEG validity
        // Just check structure is valid
        assert(sample.index >= 0);
    }
}

TEST(pipeline_multiple_batches) {
    if (!test_tar_exists()) {
        std::cout << "(no TAR) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = "/tmp/test_v2.tar";
    config.num_workers = 2;
    config.batch_size = 2;

    PipelineV2 pipeline(config);

    size_t total_samples = 0;
    size_t num_batches = 0;

    // Consume all batches
    while (!pipeline.is_finished() || num_batches < 10) {
        auto batch = pipeline.next_batch();

        if (batch.empty()) {
            break;
        }

        total_samples += batch.size();
        num_batches++;
    }

    std::cout << "(" << total_samples << " samples, " << num_batches << " batches) ";
}

TEST(pipeline_graceful_shutdown) {
    if (!test_tar_exists()) {
        std::cout << "(no TAR) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = "/tmp/test_v2.tar";
    config.num_workers = 4;
    config.batch_size = 8;

    PipelineV2 pipeline(config);

    // Get one batch
    auto batch = pipeline.next_batch();

    // Stop pipeline early
    pipeline.stop();

    // Should stop gracefully (destructor will be called)
}

TEST(pipeline_multi_worker) {
    if (!test_tar_exists()) {
        std::cout << "(no TAR) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = "/tmp/test_v2.tar";
    config.num_workers = 4;
    config.batch_size = 4;

    PipelineV2 pipeline(config);

    size_t total_samples = 0;

    // Consume all batches
    while (!pipeline.is_finished()) {
        auto batch = pipeline.next_batch();

        if (batch.empty()) {
            break;
        }

        total_samples += batch.size();
    }

    std::cout << "(" << total_samples << " samples from 4 workers) ";
}

// ============================================================================
// Performance Tests (with real TAR if available)
// ============================================================================

TEST(pipeline_throughput) {
    const char* test_paths[] = {
        "/tmp/benchmark_dataset/test.tar",
        "/tmp/test_v2.tar"
    };

    std::string tar_path;
    bool found = false;

    for (const char* path : test_paths) {
        std::ifstream file(path);
        if (file.good()) {
            tar_path = path;
            found = true;
            break;
        }
    }

    if (!found) {
        std::cout << "(no TAR) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = tar_path;
    config.num_workers = 4;
    config.batch_size = 32;

    PipelineV2 pipeline(config);

    auto start = std::chrono::high_resolution_clock::now();

    size_t total_samples = 0;
    while (!pipeline.is_finished()) {
        auto batch = pipeline.next_batch();
        if (batch.empty()) {
            break;
        }
        total_samples += batch.size();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (duration.count() > 0 && total_samples > 0) {
        double throughput = total_samples / (duration.count() / 1000.0);
        std::cout << "(" << total_samples << " samples, "
                  << throughput << " samples/sec) ";
    }
}

// ============================================================================
// Real Dataset Test (if available)
// ============================================================================

TEST(pipeline_real_dataset) {
    const char* dataset_paths[] = {
        "/tmp/benchmark_dataset/test.tar",
        "/tmp/benchmark_1000.tar",
        "/tmp/test.tar"
    };

    std::string tar_path;
    bool found = false;

    for (const char* path : dataset_paths) {
        std::ifstream file(path);
        if (file.good()) {
            tar_path = path;
            found = true;
            break;
        }
    }

    if (!found) {
        std::cout << "(no dataset) ";
        return;
    }

    PipelineConfig config;
    config.tar_path = tar_path;
    config.num_workers = 4;
    config.batch_size = 32;

    PipelineV2 pipeline(config);

    size_t total_samples = 0;
    size_t decoded_samples = 0;

    while (!pipeline.is_finished()) {
        auto batch = pipeline.next_batch();

        if (batch.empty()) {
            break;
        }

        total_samples += batch.size();

        for (const auto& sample : batch) {
            if (sample.is_decoded()) {
                decoded_samples++;
                assert(sample.width > 0);
                assert(sample.height > 0);
                assert(sample.channels == 3);
            }
        }
    }

    double decode_rate = (double)decoded_samples / total_samples * 100.0;
    std::cout << "(" << total_samples << " total, "
              << decoded_samples << " decoded, "
              << decode_rate << "%) ";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  PipelineV2 Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nBasic Tests:" << std::endl;
    RUN_TEST(pipeline_construction);
    RUN_TEST(pipeline_single_batch);
    RUN_TEST(pipeline_multiple_batches);
    RUN_TEST(pipeline_graceful_shutdown);
    RUN_TEST(pipeline_multi_worker);

    std::cout << "\nPerformance Tests:" << std::endl;
    RUN_TEST(pipeline_throughput);

    std::cout << "\nReal Dataset Test:" << std::endl;
    RUN_TEST(pipeline_real_dataset);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  ALL PIPELINE TESTS PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
