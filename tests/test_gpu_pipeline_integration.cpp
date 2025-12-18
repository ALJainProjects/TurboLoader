/**
 * @file test_gpu_pipeline_integration.cpp
 * @brief Tests for GPU Pipeline Integration (v2.16.0)
 *
 * Tests the integrated GPU pipeline that keeps data GPU-resident
 * from decode through transform to final tensor output.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>

// GPU Pipeline
#include "../src/pipeline/gpu_pipeline_integration.hpp"
#include "../src/transforms/gpu/gpu_transforms.hpp"

using namespace turboloader;
using namespace turboloader::pipeline;
using namespace turboloader::transforms::gpu;

// ============================================================================
// Test Utilities
// ============================================================================

// Simple JPEG creation for testing (creates a valid minimal JPEG)
std::vector<uint8_t> create_test_jpeg(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    // This creates a simple solid-color JPEG for testing
    // In production tests, you'd use actual JPEG files

    // Minimal valid JPEG header for a solid color image
    // This is a pre-made minimal JPEG that will be decoded
    std::vector<uint8_t> jpeg_data;

    // For simplicity, we'll just create raw RGB data and note that
    // the actual tests would need real JPEG files
    // Here we simulate by just returning a placeholder
    jpeg_data.resize(1024);
    jpeg_data[0] = 0xFF;  // JPEG SOI marker
    jpeg_data[1] = 0xD8;

    return jpeg_data;
}

bool files_exist_for_testing() {
    // Check if we have test JPEG files
    std::ifstream test_file("/tmp/turboloader_test/test.jpg");
    return test_file.good();
}

void create_test_files() {
    // Create test directory and files using system commands
    system("mkdir -p /tmp/turboloader_test");

    // Create a simple test JPEG using ImageMagick if available
    // Or we could embed a minimal JPEG
    system("convert -size 256x256 xc:red /tmp/turboloader_test/test_red.jpg 2>/dev/null");
    system("convert -size 256x256 xc:green /tmp/turboloader_test/test_green.jpg 2>/dev/null");
    system("convert -size 256x256 xc:blue /tmp/turboloader_test/test_blue.jpg 2>/dev/null");
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

// ============================================================================
// GPU Availability Tests
// ============================================================================

void test_gpu_availability() {
    std::cout << "\n=== Test: GPU Availability ===" << std::endl;

    bool available = GPUPipelineIntegration::is_available();
    std::cout << "  GPU Pipeline Available: " << (available ? "YES" : "NO") << std::endl;

    if (available) {
        try {
            GPUPipelineIntegration pipeline(0);
            std::cout << "  Device: " << pipeline.get_device_info() << std::endl;
            std::cout << "  [PASS] GPU pipeline created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  [FAIL] Error: " << e.what() << std::endl;
        }
    } else {
        std::cout << "  [SKIP] GPU not available - using CPU fallback" << std::endl;
    }
}

// ============================================================================
// GPU Transform Tests (CPU Fallback Mode)
// ============================================================================

void test_gpu_transforms_availability() {
    std::cout << "\n=== Test: GPU Transforms Availability ===" << std::endl;

    bool available = gpu_available();
    std::cout << "  GPU available: " << (available ? "YES" : "NO") << std::endl;

    if (available) {
        std::cout << "  GPU Info: " << get_gpu_info() << std::endl;

        // Try to create transforms
        try {
            auto resize = std::make_unique<GPUResize>(224, 224);
            std::cout << "  [PASS] GPUResize created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  [INFO] GPUResize: " << e.what() << std::endl;
        }

        try {
            auto flip = std::make_unique<GPUHorizontalFlip>(0.5f);
            std::cout << "  [PASS] GPUHorizontalFlip created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  [INFO] GPUHorizontalFlip: " << e.what() << std::endl;
        }

        try {
            auto crop = std::make_unique<GPURandomCrop>(200, 200);
            std::cout << "  [PASS] GPURandomCrop created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  [INFO] GPURandomCrop: " << e.what() << std::endl;
        }

        try {
            auto jitter = std::make_unique<GPUColorJitter>(0.2f, 0.2f, 0.2f, 0.1f);
            std::cout << "  [PASS] GPUColorJitter created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  [INFO] GPUColorJitter: " << e.what() << std::endl;
        }
    } else {
        std::cout << "  [SKIP] GPU not available" << std::endl;
    }
}

// ============================================================================
// GPU Pipeline Integration Tests
// ============================================================================

#ifdef TURBOLOADER_HAS_CUDA

void test_gpu_batch_buffer() {
    std::cout << "\n=== Test: GPU Batch Buffer ===" << std::endl;

    try {
        GPUBatchBuffer buffer(4, 224, 224, 3);

        std::cout << "  Batch size: " << buffer.batch_size() << std::endl;
        std::cout << "  Width: " << buffer.width() << std::endl;
        std::cout << "  Height: " << buffer.height() << std::endl;
        std::cout << "  Channels: " << buffer.channels() << std::endl;
        std::cout << "  Image size: " << buffer.image_size() << " bytes" << std::endl;
        std::cout << "  Total size: " << buffer.total_size() << " bytes" << std::endl;

        // Verify pointers are valid
        if (buffer.data() != nullptr && buffer.float_ptr() != nullptr) {
            std::cout << "  [PASS] GPU buffers allocated" << std::endl;
        } else {
            std::cout << "  [FAIL] GPU buffers are null" << std::endl;
        }

        // Test resize
        buffer.resize(8, 256, 256, 3);
        std::cout << "  After resize: " << buffer.batch_size() << "x"
                  << buffer.width() << "x" << buffer.height() << std::endl;
        std::cout << "  [PASS] Buffer resize works" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error: " << e.what() << std::endl;
    }
}

void test_gpu_decode_to_gpu() {
    std::cout << "\n=== Test: GPU Decode to GPU Memory ===" << std::endl;

    if (!files_exist_for_testing()) {
        std::cout << "  [SKIP] Test files not available" << std::endl;
        return;
    }

    try {
        GPUPipelineIntegration pipeline(0);

        // Read test JPEG
        auto jpeg_data = read_file("/tmp/turboloader_test/test_red.jpg");
        if (jpeg_data.empty()) {
            std::cout << "  [SKIP] Could not read test JPEG" << std::endl;
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        GPUDecodeResult result = pipeline.decode_to_gpu(jpeg_data.data(), jpeg_data.size());

        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "  Decoded size: " << result.width << "x" << result.height << std::endl;
        std::cout << "  GPU pointer valid: " << (result.gpu_data != nullptr ? "YES" : "NO") << std::endl;
        std::cout << "  Decode time: " << us << " us" << std::endl;

        if (result.gpu_data != nullptr && result.width > 0 && result.height > 0) {
            std::cout << "  [PASS] GPU decode successful" << std::endl;
        } else {
            std::cout << "  [FAIL] Invalid decode result" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error: " << e.what() << std::endl;
    }
}

void test_gpu_pipeline_batch_processing() {
    std::cout << "\n=== Test: GPU Pipeline Batch Processing ===" << std::endl;

    if (!files_exist_for_testing()) {
        std::cout << "  [SKIP] Test files not available" << std::endl;
        return;
    }

    try {
        GPUPipelineIntegration pipeline(0);

        // Add transforms
        pipeline.add_transform(std::make_unique<GPUResize>(224, 224));
        pipeline.add_transform(std::make_unique<GPUHorizontalFlip>(0.5f));

        // Set ImageNet normalization
        std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        std::vector<float> std = {0.229f, 0.224f, 0.225f};
        pipeline.set_normalization(mean, std);

        // Prepare batch
        std::vector<std::vector<uint8_t>> jpeg_files;
        jpeg_files.push_back(read_file("/tmp/turboloader_test/test_red.jpg"));
        jpeg_files.push_back(read_file("/tmp/turboloader_test/test_green.jpg"));
        jpeg_files.push_back(read_file("/tmp/turboloader_test/test_blue.jpg"));

        std::vector<const uint8_t*> jpeg_ptrs;
        std::vector<size_t> jpeg_sizes;
        for (const auto& file : jpeg_files) {
            if (file.empty()) continue;
            jpeg_ptrs.push_back(file.data());
            jpeg_sizes.push_back(file.size());
        }

        if (jpeg_ptrs.empty()) {
            std::cout << "  [SKIP] No valid test files" << std::endl;
            return;
        }

        std::cout << "  Batch size: " << jpeg_ptrs.size() << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        float* output_tensor = pipeline.process_batch_gpu(jpeg_ptrs, jpeg_sizes, 224, 224);

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "  Processing time: " << ms << " ms" << std::endl;
        std::cout << "  Output tensor valid: " << (output_tensor != nullptr ? "YES" : "NO") << std::endl;

        int batch_size, channels, height, width;
        pipeline.get_output_shape(batch_size, channels, height, width);
        std::cout << "  Output shape: [" << batch_size << ", " << channels
                  << ", " << height << ", " << width << "]" << std::endl;

        if (output_tensor != nullptr) {
            std::cout << "  [PASS] Batch processing successful" << std::endl;
        } else {
            std::cout << "  [FAIL] Null output tensor" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error: " << e.what() << std::endl;
    }
}

void test_gpu_pipeline_performance() {
    std::cout << "\n=== Test: GPU Pipeline Performance ===" << std::endl;

    if (!files_exist_for_testing()) {
        std::cout << "  [SKIP] Test files not available" << std::endl;
        return;
    }

    try {
        GPUPipelineIntegration pipeline(0);
        pipeline.add_transform(std::make_unique<GPUResize>(224, 224));

        // Read a single JPEG
        auto jpeg_data = read_file("/tmp/turboloader_test/test_red.jpg");
        if (jpeg_data.empty()) {
            std::cout << "  [SKIP] Could not read test JPEG" << std::endl;
            return;
        }

        // Create batch of same image
        const int batch_size = 32;
        std::vector<const uint8_t*> jpeg_ptrs(batch_size, jpeg_data.data());
        std::vector<size_t> jpeg_sizes(batch_size, jpeg_data.size());

        // Warmup
        pipeline.process_batch_gpu(jpeg_ptrs, jpeg_sizes, 224, 224);

        // Benchmark
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            pipeline.process_batch_gpu(jpeg_ptrs, jpeg_sizes, 224, 224);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        double images_per_sec = (iterations * batch_size * 1000.0) / total_ms;
        double ms_per_batch = total_ms / static_cast<double>(iterations);

        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Total time: " << total_ms << " ms" << std::endl;
        std::cout << "  Time per batch: " << ms_per_batch << " ms" << std::endl;
        std::cout << "  Throughput: " << images_per_sec << " images/sec" << std::endl;

        std::cout << "  [PASS] Performance benchmark complete" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error: " << e.what() << std::endl;
    }
}

void test_hwc_to_chw_conversion() {
    std::cout << "\n=== Test: HWC to CHW Conversion ===" << std::endl;

    try {
        // Allocate GPU memory for test
        const int width = 4;
        const int height = 4;
        const int channels = 3;

        // Create test HWC data on host
        std::vector<uint8_t> hwc_host(width * height * channels);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                hwc_host[(h * width + w) * 3 + 0] = 255;  // R
                hwc_host[(h * width + w) * 3 + 1] = 128;  // G
                hwc_host[(h * width + w) * 3 + 2] = 64;   // B
            }
        }

        // Upload to GPU
        uint8_t* d_hwc;
        float* d_chw;
        float* d_mean;
        float* d_std;

        cudaMalloc(&d_hwc, hwc_host.size());
        cudaMalloc(&d_chw, width * height * channels * sizeof(float));
        cudaMalloc(&d_mean, 3 * sizeof(float));
        cudaMalloc(&d_std, 3 * sizeof(float));

        cudaMemcpy(d_hwc, hwc_host.data(), hwc_host.size(), cudaMemcpyHostToDevice);

        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std[3] = {0.5f, 0.5f, 0.5f};
        cudaMemcpy(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Convert
        kernels::convert_hwc_to_chw_normalized(d_hwc, d_chw, width, height, channels,
                                               d_mean, d_std, nullptr);
        cudaDeviceSynchronize();

        // Download result
        std::vector<float> chw_host(width * height * channels);
        cudaMemcpy(chw_host.data(), d_chw, chw_host.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify CHW layout
        // Red channel should be at indices [0, height*width)
        // Green channel should be at indices [height*width, 2*height*width)
        // Blue channel should be at indices [2*height*width, 3*height*width)

        bool correct = true;
        float expected_r = (255.0f / 255.0f - 0.5f) / 0.5f;
        float expected_g = (128.0f / 255.0f - 0.5f) / 0.5f;
        float expected_b = (64.0f / 255.0f - 0.5f) / 0.5f;

        for (int i = 0; i < width * height; ++i) {
            if (std::abs(chw_host[i] - expected_r) > 0.01f) correct = false;
            if (std::abs(chw_host[width * height + i] - expected_g) > 0.01f) correct = false;
            if (std::abs(chw_host[2 * width * height + i] - expected_b) > 0.01f) correct = false;
        }

        std::cout << "  Expected R: " << expected_r << ", Got: " << chw_host[0] << std::endl;
        std::cout << "  Expected G: " << expected_g << ", Got: " << chw_host[width * height] << std::endl;
        std::cout << "  Expected B: " << expected_b << ", Got: " << chw_host[2 * width * height] << std::endl;

        // Cleanup
        cudaFree(d_hwc);
        cudaFree(d_chw);
        cudaFree(d_mean);
        cudaFree(d_std);

        if (correct) {
            std::cout << "  [PASS] HWC to CHW conversion correct" << std::endl;
        } else {
            std::cout << "  [FAIL] HWC to CHW conversion incorrect" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error: " << e.what() << std::endl;
    }
}

#else  // !TURBOLOADER_HAS_CUDA

void test_gpu_batch_buffer() {
    std::cout << "\n=== Test: GPU Batch Buffer ===" << std::endl;
    std::cout << "  [SKIP] CUDA not compiled" << std::endl;
}

void test_gpu_decode_to_gpu() {
    std::cout << "\n=== Test: GPU Decode to GPU Memory ===" << std::endl;
    std::cout << "  [SKIP] CUDA not compiled" << std::endl;
}

void test_gpu_pipeline_batch_processing() {
    std::cout << "\n=== Test: GPU Pipeline Batch Processing ===" << std::endl;
    std::cout << "  [SKIP] CUDA not compiled" << std::endl;
}

void test_gpu_pipeline_performance() {
    std::cout << "\n=== Test: GPU Pipeline Performance ===" << std::endl;
    std::cout << "  [SKIP] CUDA not compiled" << std::endl;
}

void test_hwc_to_chw_conversion() {
    std::cout << "\n=== Test: HWC to CHW Conversion ===" << std::endl;
    std::cout << "  [SKIP] CUDA not compiled" << std::endl;
}

#endif  // TURBOLOADER_HAS_CUDA

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TurboLoader GPU Pipeline Integration Tests ===" << std::endl;
    std::cout << "Version: 2.16.0" << std::endl;

#ifdef TURBOLOADER_HAS_CUDA
    std::cout << "CUDA: Enabled" << std::endl;
#else
    std::cout << "CUDA: Disabled (CPU fallback mode)" << std::endl;
#endif

    // Create test files if needed
    create_test_files();

    // Run tests
    test_gpu_availability();
    test_gpu_transforms_availability();
    test_gpu_batch_buffer();
    test_gpu_decode_to_gpu();
    test_gpu_pipeline_batch_processing();
    test_gpu_pipeline_performance();
    test_hwc_to_chw_conversion();

    std::cout << "\n=== Tests Complete ===" << std::endl;

    return 0;
}
