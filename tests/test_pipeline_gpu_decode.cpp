/**
 * @file test_pipeline_gpu_decode.cpp
 * @brief Test GPU-accelerated JPEG decoding integration into UnifiedPipeline
 *
 * This test verifies that the nvJPEG GPU decoder is properly integrated into
 * the main UnifiedPipeline for TAR-based image loading.
 *
 * Tests:
 * 1. Pipeline with GPU decode enabled (use_gpu_decode = true)
 * 2. Pipeline with GPU decode disabled (use_gpu_decode = false)
 * 3. Automatic CPU fallback when GPU unavailable
 * 4. Per-worker GPU decoder instances
 * 5. GPU vs CPU decode output consistency
 * 6. Performance benchmarking (GPU vs CPU)
 */

#include "../src/pipeline/pipeline.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

using namespace turboloader;

// ANSI color codes
#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[32m"
#define COLOR_RED "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"
#define COLOR_BOLD "\033[1m"

// Helper: Create test JPEG using Python PIL
bool create_test_jpeg(const std::string& path, int width, int height, uint8_t r = 255, uint8_t g = 0, uint8_t b = 0) {
    std::string cmd = "python3 -c \"from PIL import Image; "
                     "img = Image.new('RGB', (" + std::to_string(width) + ", " +
                     std::to_string(height) + "), color=(" + std::to_string(r) + ", " +
                     std::to_string(g) + ", " + std::to_string(b) + ")); "
                     "img.save('" + path + "', quality=90)\" 2>/dev/null";
    return system(cmd.c_str()) == 0;
}

// Helper: Create test TAR archive
bool create_test_tar(const std::string& tar_path, int num_images, int width, int height) {
    std::cout << "  Creating test TAR with " << num_images << " images (" << width << "x" << height << ")..." << std::endl;

    // Create test images
    std::vector<std::string> image_paths;
    for (int i = 0; i < num_images; ++i) {
        std::string img_path = "/tmp/test_gpu_pipeline_" + std::to_string(i) + ".jpg";
        // Vary colors to make images distinguishable
        uint8_t r = (i * 30) % 256;
        uint8_t g = (i * 60) % 256;
        uint8_t b = (i * 90) % 256;
        if (!create_test_jpeg(img_path, width, height, r, g, b)) {
            std::cout << COLOR_YELLOW << "  ⚠ Could not create test JPEG" << COLOR_RESET << std::endl;
            return false;
        }
        image_paths.push_back(img_path);
    }

    // Create TAR archive using Python tarfile
    std::string cmd = "python3 -c \"import tarfile; "
                     "tar = tarfile.open('" + tar_path + "', 'w'); ";
    for (int i = 0; i < num_images; ++i) {
        cmd += "tar.add('" + image_paths[i] + "', arcname='img_" + std::to_string(i) + ".jpg'); ";
    }
    cmd += "tar.close()\" 2>/dev/null";

    bool success = system(cmd.c_str()) == 0;

    // Cleanup individual image files
    for (const auto& path : image_paths) {
        remove(path.c_str());
    }

    return success;
}

//==============================================================================
// Test 1: Pipeline with GPU Decode Enabled
//==============================================================================
void test_pipeline_gpu_enabled() {
    std::cout << COLOR_CYAN << "\n[TEST] Pipeline with GPU Decode Enabled" << COLOR_RESET << std::endl;

    std::string tar_path = "/tmp/test_pipeline_gpu_enabled.tar";
    const int num_images = 10;

    if (!create_test_tar(tar_path, num_images, 128, 128)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test TAR" << COLOR_RESET << std::endl;
        return;
    }

    try {
        UnifiedPipelineConfig config;
        config.data_path = tar_path;
        config.format = DataFormat::TAR;
        config.batch_size = 4;
        config.num_workers = 2;
        config.use_gpu_decode = true;  // Enable GPU decode

        std::cout << "  Configuration:" << std::endl;
        std::cout << "    use_gpu_decode: " << (config.use_gpu_decode ? "true" : "false") << std::endl;
        std::cout << "    num_workers: " << config.num_workers << std::endl;
        std::cout << "    batch_size: " << config.batch_size << std::endl;

        UnifiedPipeline pipeline(config);
        pipeline.start();

        int total_samples = 0;
        int batches = 0;

        while (!pipeline.is_finished()) {
            auto batch = pipeline.next_batch();
            if (batch.empty()) break;

            total_samples += batch.size();
            batches++;

            // Verify samples
            for (const auto& sample : batch.samples) {
                assert(sample.width == 128);
                assert(sample.height == 128);
                assert(sample.channels == 3);
                assert(sample.image_data.size() == 128 * 128 * 3);
            }
        }

        pipeline.stop();

        std::cout << "  Results:" << std::endl;
        std::cout << "    Total samples: " << total_samples << std::endl;
        std::cout << "    Batches: " << batches << std::endl;

        assert(total_samples == num_images);
        std::cout << COLOR_GREEN << "  ✓ PASSED" << COLOR_RESET << std::endl;

    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "  ✗ Exception: " << e.what() << COLOR_RESET << std::endl;
        remove(tar_path.c_str());
        throw;
    }

    remove(tar_path.c_str());
}

//==============================================================================
// Test 2: Pipeline with GPU Decode Disabled
//==============================================================================
void test_pipeline_gpu_disabled() {
    std::cout << COLOR_CYAN << "\n[TEST] Pipeline with GPU Decode Disabled (CPU-only)" << COLOR_RESET << std::endl;

    std::string tar_path = "/tmp/test_pipeline_gpu_disabled.tar";
    const int num_images = 10;

    if (!create_test_tar(tar_path, num_images, 128, 128)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test TAR" << COLOR_RESET << std::endl;
        return;
    }

    try {
        UnifiedPipelineConfig config;
        config.data_path = tar_path;
        config.format = DataFormat::TAR;
        config.batch_size = 4;
        config.num_workers = 2;
        config.use_gpu_decode = false;  // Disable GPU decode (CPU-only)

        std::cout << "  Configuration:" << std::endl;
        std::cout << "    use_gpu_decode: " << (config.use_gpu_decode ? "true" : "false") << std::endl;
        std::cout << "    num_workers: " << config.num_workers << std::endl;
        std::cout << "    batch_size: " << config.batch_size << std::endl;

        UnifiedPipeline pipeline(config);
        pipeline.start();

        int total_samples = 0;
        int batches = 0;

        while (!pipeline.is_finished()) {
            auto batch = pipeline.next_batch();
            if (batch.empty()) break;

            total_samples += batch.size();
            batches++;

            // Verify samples
            for (const auto& sample : batch.samples) {
                assert(sample.width == 128);
                assert(sample.height == 128);
                assert(sample.channels == 3);
                assert(sample.image_data.size() == 128 * 128 * 3);
            }
        }

        pipeline.stop();

        std::cout << "  Results:" << std::endl;
        std::cout << "    Total samples: " << total_samples << std::endl;
        std::cout << "    Batches: " << batches << std::endl;

        assert(total_samples == num_images);
        std::cout << COLOR_GREEN << "  ✓ PASSED - CPU decode works correctly" << COLOR_RESET << std::endl;

    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "  ✗ Exception: " << e.what() << COLOR_RESET << std::endl;
        remove(tar_path.c_str());
        throw;
    }

    remove(tar_path.c_str());
}

//==============================================================================
// Test 3: GPU vs CPU Output Consistency
//==============================================================================
void test_gpu_cpu_consistency() {
    std::cout << COLOR_CYAN << "\n[TEST] GPU vs CPU Decode Output Consistency" << COLOR_RESET << std::endl;

    std::string tar_path = "/tmp/test_pipeline_consistency.tar";
    const int num_images = 5;

    if (!create_test_tar(tar_path, num_images, 64, 64)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test TAR" << COLOR_RESET << std::endl;
        return;
    }

    try {
        // Run with GPU decode enabled
        std::vector<std::vector<uint8_t>> gpu_results;
        {
            UnifiedPipelineConfig config;
            config.data_path = tar_path;
            config.format = DataFormat::TAR;
            config.batch_size = 10;
            config.num_workers = 1;  // Single worker for deterministic ordering
            config.use_gpu_decode = true;

            UnifiedPipeline pipeline(config);
            pipeline.start();

            while (!pipeline.is_finished()) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;

                for (const auto& sample : batch.samples) {
                    gpu_results.push_back(sample.image_data);
                }
            }

            pipeline.stop();
        }

        // Run with CPU decode (GPU disabled)
        std::vector<std::vector<uint8_t>> cpu_results;
        {
            UnifiedPipelineConfig config;
            config.data_path = tar_path;
            config.format = DataFormat::TAR;
            config.batch_size = 10;
            config.num_workers = 1;  // Single worker for deterministic ordering
            config.use_gpu_decode = false;

            UnifiedPipeline pipeline(config);
            pipeline.start();

            while (!pipeline.is_finished()) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;

                for (const auto& sample : batch.samples) {
                    cpu_results.push_back(sample.image_data);
                }
            }

            pipeline.stop();
        }

        std::cout << "  GPU results: " << gpu_results.size() << " images" << std::endl;
        std::cout << "  CPU results: " << cpu_results.size() << " images" << std::endl;

        assert(gpu_results.size() == cpu_results.size());
        assert(gpu_results.size() == num_images);

        // Compare decoded outputs (allow small differences due to GPU/CPU rounding)
        int total_pixels_compared = 0;
        int pixels_with_differences = 0;
        const int max_diff_threshold = 2;  // Allow ±2 difference per pixel channel

        for (size_t i = 0; i < gpu_results.size(); ++i) {
            const auto& gpu_data = gpu_results[i];
            const auto& cpu_data = cpu_results[i];

            assert(gpu_data.size() == cpu_data.size());

            for (size_t j = 0; j < gpu_data.size(); ++j) {
                int diff = std::abs(static_cast<int>(gpu_data[j]) - static_cast<int>(cpu_data[j]));
                total_pixels_compared++;

                if (diff > 0) {
                    pixels_with_differences++;
                    if (diff > max_diff_threshold) {
                        std::cout << COLOR_RED << "  ✗ Large difference at image " << i << ", pixel " << j
                                  << ": GPU=" << static_cast<int>(gpu_data[j])
                                  << " CPU=" << static_cast<int>(cpu_data[j])
                                  << " (diff=" << diff << ")" << COLOR_RESET << std::endl;
                        assert(false);
                    }
                }
            }
        }

        double diff_percentage = 100.0 * pixels_with_differences / total_pixels_compared;
        std::cout << "  Pixel differences: " << pixels_with_differences << "/" << total_pixels_compared
                  << " (" << diff_percentage << "%)" << std::endl;

        // Allow up to 5% pixel differences due to GPU/CPU rounding
        assert(diff_percentage < 5.0);

        std::cout << COLOR_GREEN << "  ✓ PASSED - GPU and CPU outputs are consistent" << COLOR_RESET << std::endl;

    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "  ✗ Exception: " << e.what() << COLOR_RESET << std::endl;
        remove(tar_path.c_str());
        throw;
    }

    remove(tar_path.c_str());
}

//==============================================================================
// Test 4: Multi-Worker GPU Decode
//==============================================================================
void test_multi_worker_gpu() {
    std::cout << COLOR_CYAN << "\n[TEST] Multi-Worker GPU Decode (Per-Worker Decoders)" << COLOR_RESET << std::endl;

    std::string tar_path = "/tmp/test_pipeline_multiworker.tar";
    const int num_images = 32;

    if (!create_test_tar(tar_path, num_images, 64, 64)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test TAR" << COLOR_RESET << std::endl;
        return;
    }

    try {
        UnifiedPipelineConfig config;
        config.data_path = tar_path;
        config.format = DataFormat::TAR;
        config.batch_size = 8;
        config.num_workers = 4;  // Multiple workers (each gets own GPU decoder)
        config.use_gpu_decode = true;

        std::cout << "  Configuration:" << std::endl;
        std::cout << "    use_gpu_decode: true" << std::endl;
        std::cout << "    num_workers: " << config.num_workers << std::endl;
        std::cout << "    batch_size: " << config.batch_size << std::endl;

        UnifiedPipeline pipeline(config);
        pipeline.start();

        int total_samples = 0;
        int batches = 0;

        auto start = std::chrono::high_resolution_clock::now();

        while (!pipeline.is_finished()) {
            auto batch = pipeline.next_batch();
            if (batch.empty()) break;

            total_samples += batch.size();
            batches++;

            // Verify samples
            for (const auto& sample : batch.samples) {
                assert(sample.width == 64);
                assert(sample.height == 64);
                assert(sample.channels == 3);
                assert(sample.image_data.size() == 64 * 64 * 3);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        pipeline.stop();

        std::cout << "  Results:" << std::endl;
        std::cout << "    Total samples: " << total_samples << std::endl;
        std::cout << "    Batches: " << batches << std::endl;
        std::cout << "    Time: " << duration.count() << " ms" << std::endl;

        if (duration.count() > 0) {
            double throughput = (total_samples * 1000.0) / duration.count();
            std::cout << "    Throughput: " << static_cast<int>(throughput) << " img/s" << std::endl;
        }

        assert(total_samples == num_images);
        std::cout << COLOR_GREEN << "  ✓ PASSED - Multi-worker GPU decode successful" << COLOR_RESET << std::endl;

    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "  ✗ Exception: " << e.what() << COLOR_RESET << std::endl;
        remove(tar_path.c_str());
        throw;
    }

    remove(tar_path.c_str());
}

//==============================================================================
// Benchmark: GPU vs CPU Performance
//==============================================================================
void benchmark_gpu_vs_cpu() {
    std::cout << COLOR_BLUE << "\n[BENCHMARK] GPU vs CPU Decode Performance" << COLOR_RESET << std::endl;

    std::string tar_path = "/tmp/test_pipeline_benchmark.tar";
    const int num_images = 100;

    if (!create_test_tar(tar_path, num_images, 256, 256)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test TAR" << COLOR_RESET << std::endl;
        return;
    }

    std::cout << "  Dataset: " << num_images << " images (256x256)" << std::endl;

    try {
        // Benchmark GPU decode
        double gpu_time_ms = 0;
        int gpu_samples = 0;
        {
            UnifiedPipelineConfig config;
            config.data_path = tar_path;
            config.format = DataFormat::TAR;
            config.batch_size = 16;
            config.num_workers = 4;
            config.use_gpu_decode = true;

            UnifiedPipeline pipeline(config);
            pipeline.start();

            auto start = std::chrono::high_resolution_clock::now();

            while (!pipeline.is_finished()) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;
                gpu_samples += batch.size();
            }

            auto end = std::chrono::high_resolution_clock::now();
            gpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            pipeline.stop();
        }

        // Benchmark CPU decode
        double cpu_time_ms = 0;
        int cpu_samples = 0;
        {
            UnifiedPipelineConfig config;
            config.data_path = tar_path;
            config.format = DataFormat::TAR;
            config.batch_size = 16;
            config.num_workers = 4;
            config.use_gpu_decode = false;

            UnifiedPipeline pipeline(config);
            pipeline.start();

            auto start = std::chrono::high_resolution_clock::now();

            while (!pipeline.is_finished()) {
                auto batch = pipeline.next_batch();
                if (batch.empty()) break;
                cpu_samples += batch.size();
            }

            auto end = std::chrono::high_resolution_clock::now();
            cpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

            pipeline.stop();
        }

        std::cout << "\n  GPU Decode (use_gpu_decode=true):" << std::endl;
        std::cout << "    Samples: " << gpu_samples << std::endl;
        std::cout << "    Time: " << gpu_time_ms << " ms" << std::endl;
        if (gpu_time_ms > 0) {
            double gpu_throughput = (gpu_samples * 1000.0) / gpu_time_ms;
            std::cout << "    Throughput: " << static_cast<int>(gpu_throughput) << " img/s" << std::endl;
        }

        std::cout << "\n  CPU Decode (use_gpu_decode=false):" << std::endl;
        std::cout << "    Samples: " << cpu_samples << std::endl;
        std::cout << "    Time: " << cpu_time_ms << " ms" << std::endl;
        if (cpu_time_ms > 0) {
            double cpu_throughput = (cpu_samples * 1000.0) / cpu_time_ms;
            std::cout << "    Throughput: " << static_cast<int>(cpu_throughput) << " img/s" << std::endl;
        }

        if (gpu_time_ms > 0 && cpu_time_ms > 0) {
            double speedup = cpu_time_ms / gpu_time_ms;
            std::cout << "\n  Speedup: " << speedup << "x ";
            if (speedup > 1.0) {
                std::cout << COLOR_GREEN << "⚡ GPU FASTER" << COLOR_RESET;
            } else if (speedup < 1.0) {
                std::cout << COLOR_YELLOW << "(CPU faster on this system)" << COLOR_RESET;
            }
            std::cout << std::endl;
        }

        std::cout << COLOR_GREEN << "\n  ✓ BENCHMARK COMPLETE" << COLOR_RESET << std::endl;

    } catch (const std::exception& e) {
        std::cout << COLOR_RED << "  ✗ Exception: " << e.what() << COLOR_RESET << std::endl;
        remove(tar_path.c_str());
        throw;
    }

    remove(tar_path.c_str());
}

//==============================================================================
// Main Test Runner
//==============================================================================
int main() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << COLOR_BOLD << "UNIFIED PIPELINE GPU DECODE INTEGRATION TEST SUITE" << COLOR_RESET << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    try {
        // Functional tests
        test_pipeline_gpu_enabled();
        test_pipeline_gpu_disabled();
        test_gpu_cpu_consistency();
        test_multi_worker_gpu();

        // Performance benchmark
        benchmark_gpu_vs_cpu();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << COLOR_GREEN << COLOR_BOLD << "ALL TESTS PASSED ✓" << COLOR_RESET << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n" << std::string(80, '=') << std::endl;
        std::cerr << COLOR_RED << "TEST FAILED: " << e.what() << COLOR_RESET << std::endl;
        std::cerr << std::string(80, '=') << std::endl;
        return 1;
    }
}
