/**
 * @file test_prefetch_pipeline.cpp
 * @brief Unit tests for prefetching pipeline
 *
 * Tests correctness, thread safety, and performance of the prefetch pipeline.
 */

#include "../src/pipeline/prefetch_pipeline.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <thread>

using namespace turboloader;

// Helper function to print test result
void print_result(const char* test_name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

// Test 1: Basic prefetch functionality
bool test_basic_prefetch() {
    try {
        // Create a simple test TAR file first
        // For now, we'll use an existing test dataset
        const std::string test_tar = "/tmp/test_prefetch.tar";

        // Create minimal pipeline config
        UnifiedPipelineConfig base_config;
        base_config.data_path = test_tar;
        base_config.batch_size = 4;
        base_config.num_workers = 2;
        base_config.format = DataFormat::TAR;
        base_config.shuffle = false;

        PrefetchPipelineConfig config;
        config.base_pipeline_config = base_config;
        config.num_prefetch_batches = 2;  // Double buffering

        PrefetchPipeline pipeline(config);
        pipeline.start();

        // Get first batch - should be instant after prefetch
        auto batch1 = pipeline.next_batch();

        if (batch1.empty()) {
            std::cerr << "  Failed: First batch is empty" << std::endl;
            return false;
        }

        if (batch1.size() != base_config.batch_size) {
            std::cerr << "  Failed: Expected batch size " << base_config.batch_size
                      << ", got " << batch1.size() << std::endl;
            return false;
        }

        pipeline.stop();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  Exception: " << e.what() << std::endl;
        return false;
    }
}

// Test 2: Prefetch buffer size
bool test_prefetch_buffer_size() {
    try {
        const std::string test_tar = "/tmp/test_prefetch.tar";

        UnifiedPipelineConfig base_config;
        base_config.data_path = test_tar;
        base_config.batch_size = 4;
        base_config.num_workers = 2;
        base_config.format = DataFormat::TAR;
        base_config.shuffle = false;

        PrefetchPipelineConfig config;
        config.base_pipeline_config = base_config;
        config.num_prefetch_batches = 3;  // Triple buffering

        PrefetchPipeline pipeline(config);
        pipeline.start();

        // Give prefetch thread time to fill buffer
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check buffer size
        size_t buffer_size = pipeline.prefetch_buffer_size();

        if (buffer_size == 0) {
            std::cerr << "  Failed: Prefetch buffer is empty" << std::endl;
            return false;
        }

        if (buffer_size > config.num_prefetch_batches) {
            std::cerr << "  Failed: Buffer size exceeds max ("
                      << buffer_size << " > " << config.num_prefetch_batches << ")" << std::endl;
            return false;
        }

        std::cout << "  Buffer size: " << buffer_size << "/" << config.num_prefetch_batches << std::endl;

        pipeline.stop();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  Exception: " << e.what() << std::endl;
        return false;
    }
}

// Test 3: Zero-wait next_batch (already prefetched)
bool test_zero_wait_next_batch() {
    try {
        const std::string test_tar = "/tmp/test_prefetch.tar";

        UnifiedPipelineConfig base_config;
        base_config.data_path = test_tar;
        base_config.batch_size = 4;
        base_config.num_workers = 2;
        base_config.format = DataFormat::TAR;
        base_config.shuffle = false;

        PrefetchPipelineConfig config;
        config.base_pipeline_config = base_config;
        config.num_prefetch_batches = 2;

        PrefetchPipeline pipeline(config);
        pipeline.start();

        // Give prefetch thread time to prepare first batch
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Measure time for next_batch (should be ~instant)
        auto start = std::chrono::high_resolution_clock::now();
        auto batch = pipeline.next_batch();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if (batch.empty()) {
            std::cerr << "  Failed: Batch is empty" << std::endl;
            return false;
        }

        // Prefetched batch should return in < 1ms (usually < 100us)
        if (duration_us > 1000) {
            std::cerr << "  Warning: next_batch took " << duration_us << "us (expected < 1000us)" << std::endl;
        }

        std::cout << "  next_batch latency: " << duration_us << " microseconds" << std::endl;

        pipeline.stop();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  Exception: " << e.what() << std::endl;
        return false;
    }
}

// Test 4: Thread safety - concurrent access
bool test_thread_safety() {
    try {
        const std::string test_tar = "/tmp/test_prefetch.tar";

        UnifiedPipelineConfig base_config;
        base_config.data_path = test_tar;
        base_config.batch_size = 4;
        base_config.num_workers = 2;
        base_config.format = DataFormat::TAR;
        base_config.shuffle = false;

        PrefetchPipelineConfig config;
        config.base_pipeline_config = base_config;
        config.num_prefetch_batches = 2;

        PrefetchPipeline pipeline(config);
        pipeline.start();

        // Multiple threads trying to get batches concurrently
        std::atomic<int> successful_reads{0};
        std::atomic<bool> error_occurred{false};

        auto consumer_thread = [&]() {
            try {
                for (int i = 0; i < 5; ++i) {
                    auto batch = pipeline.next_batch();
                    if (!batch.empty()) {
                        successful_reads++;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            } catch (...) {
                error_occurred = true;
            }
        };

        // Launch multiple consumer threads
        std::thread t1(consumer_thread);
        std::thread t2(consumer_thread);

        t1.join();
        t2.join();

        pipeline.stop();

        if (error_occurred) {
            std::cerr << "  Failed: Exception in consumer threads" << std::endl;
            return false;
        }

        std::cout << "  Successful concurrent reads: " << successful_reads.load() << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "  Exception: " << e.what() << std::endl;
        return false;
    }
}

// Test 5: Finish signal propagation
bool test_finish_signal() {
    try {
        const std::string test_tar = "/tmp/test_prefetch_small.tar";  // Small dataset

        UnifiedPipelineConfig base_config;
        base_config.data_source = test_tar;
        base_config.batch_size = 4;
        base_config.num_workers = 1;
        base_config.format = DataFormat::TAR;
        base_config.shuffle = false;

        PrefetchPipelineConfig config;
        config.base_pipeline_config = base_config;
        config.num_prefetch_batches = 2;

        PrefetchPipeline pipeline(config);
        pipeline.start();

        // Consume all batches
        int batch_count = 0;
        while (!pipeline.is_finished()) {
            auto batch = pipeline.next_batch();
            if (batch.empty()) {
                break;
            }
            batch_count++;

            // Safety: max 1000 batches
            if (batch_count > 1000) {
                std::cerr << "  Failed: Too many batches, infinite loop?" << std::endl;
                return false;
            }
        }

        // Verify pipeline is finished
        if (!pipeline.is_finished()) {
            std::cerr << "  Failed: Pipeline not marked as finished" << std::endl;
            return false;
        }

        // Verify next_batch returns empty after finish
        auto batch = pipeline.next_batch();
        if (!batch.empty()) {
            std::cerr << "  Failed: next_batch should return empty after finish" << std::endl;
            return false;
        }

        std::cout << "  Consumed " << batch_count << " batches before finish" << std::endl;

        pipeline.stop();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "  Exception: " << e.what() << std::endl;
        return false;
    }
}

// Performance comparison: Prefetch vs No-Prefetch
void benchmark_prefetch_performance() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Prefetch Pipeline Performance Benchmark" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    const std::string test_tar = "/tmp/test_prefetch.tar";
    const size_t num_batches = 100;

    try {
        // Benchmark 1: Without prefetching (baseline)
        std::cout << "\n[Baseline] Without Prefetching:" << std::endl;
        {
            UnifiedPipelineConfig config;
            config.data_path = test_tar;
            config.batch_size = 8;
            config.num_workers = 2;
            config.format = DataFormat::TAR;
            config.shuffle = false;

            UnifiedPipeline pipeline(config);
            pipeline.start();

            auto start = std::chrono::high_resolution_clock::now();

            size_t total_samples = 0;
            for (size_t i = 0; i < num_batches; ++i) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;
                total_samples += batch.size();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            double throughput = (total_samples * 1000.0) / duration_ms;

            std::cout << "  Total samples: " << total_samples << std::endl;
            std::cout << "  Duration: " << duration_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << throughput << " samples/s" << std::endl;

            pipeline.stop();
        }

        // Benchmark 2: With prefetching (2 batches)
        std::cout << "\n[Optimized] With Prefetching (2 batches):" << std::endl;
        {
            UnifiedPipelineConfig base_config;
            base_config.data_path = test_tar;
            base_config.batch_size = 8;
            base_config.num_workers = 2;
            base_config.format = DataFormat::TAR;
            base_config.shuffle = false;

            PrefetchPipelineConfig config;
            config.base_pipeline_config = base_config;
            config.num_prefetch_batches = 2;

            PrefetchPipeline pipeline(config);
            pipeline.start();

            // Warmup: let prefetch fill buffer
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto start = std::chrono::high_resolution_clock::now();

            size_t total_samples = 0;
            for (size_t i = 0; i < num_batches; ++i) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;
                total_samples += batch.size();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            double throughput = (total_samples * 1000.0) / duration_ms;

            std::cout << "  Total samples: " << total_samples << std::endl;
            std::cout << "  Duration: " << duration_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << throughput << " samples/s" << std::endl;

            pipeline.stop();
        }

        // Benchmark 3: With prefetching (4 batches)
        std::cout << "\n[Maximum] With Prefetching (4 batches):" << std::endl;
        {
            UnifiedPipelineConfig base_config;
            base_config.data_path = test_tar;
            base_config.batch_size = 8;
            base_config.num_workers = 2;
            base_config.format = DataFormat::TAR;
            base_config.shuffle = false;

            PrefetchPipelineConfig config;
            config.base_pipeline_config = base_config;
            config.num_prefetch_batches = 4;

            PrefetchPipeline pipeline(config);
            pipeline.start();

            // Warmup: let prefetch fill buffer
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto start = std::chrono::high_resolution_clock::now();

            size_t total_samples = 0;
            for (size_t i = 0; i < num_batches; ++i) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;
                total_samples += batch.size();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            double throughput = (total_samples * 1000.0) / duration_ms;

            std::cout << "  Total samples: " << total_samples << std::endl;
            std::cout << "  Duration: " << duration_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                      << throughput << " samples/s" << std::endl;

            pipeline.stop();
        }

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }

    std::cout << std::string(80, '=') << std::endl;
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "Prefetch Pipeline Unit Tests" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::cout << "\nNOTE: These tests require test TAR files in /tmp/" << std::endl;
    std::cout << "      Run setup script first: python3 tests/create_test_dataset.py" << std::endl;
    std::cout << std::endl;

    // Run correctness tests
    int passed = 0;
    int total = 5;

    std::cout << "Running correctness tests..." << std::endl;
    std::cout << std::endl;

    if (test_basic_prefetch()) passed++;
    print_result("test_basic_prefetch", test_basic_prefetch());

    if (test_prefetch_buffer_size()) passed++;
    print_result("test_prefetch_buffer_size", test_prefetch_buffer_size());

    if (test_zero_wait_next_batch()) passed++;
    print_result("test_zero_wait_next_batch", test_zero_wait_next_batch());

    if (test_thread_safety()) passed++;
    print_result("test_thread_safety", test_thread_safety());

    if (test_finish_signal()) passed++;
    print_result("test_finish_signal", test_finish_signal());

    std::cout << "\nTest Summary: " << passed << "/" << total << " tests passed" << std::endl;

    // Run performance benchmarks
    if (passed == total) {
        benchmark_prefetch_performance();
    } else {
        std::cout << "\nSkipping benchmarks due to test failures." << std::endl;
    }

    return (passed == total) ? 0 : 1;
}
