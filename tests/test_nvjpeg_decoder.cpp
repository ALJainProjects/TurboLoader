/**
 * @file test_nvjpeg_decoder.cpp
 * @brief Comprehensive test suite for GPU-accelerated JPEG decoder
 *
 * Tests:
 * 1. GPU availability detection
 * 2. Device information retrieval
 * 3. Single image decode (CPU fallback)
 * 4. Batch decode operations
 * 5. GPU/CPU automatic fallback behavior
 * 6. Error handling (invalid JPEG data)
 * 7. Result structure validation
 * 8. Performance benchmarks (CPU decode)
 *
 * Note: GPU tests will be skipped if CUDA/nvJPEG not available.
 *       All tests will run on CPU fallback path to ensure correctness.
 */

#include "../src/decode/nvjpeg_decoder.hpp"
#include "../src/decode/jpeg_decoder.hpp"
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace turboloader;

// ANSI color codes for output
#define COLOR_RESET "\033[0m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"

// Helper: Create a minimal valid JPEG (1x1 pixel, red)
std::vector<uint8_t> create_minimal_jpeg() {
    // This is a valid 1x1 red pixel JPEG
    return {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x09, 0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        0xD2, 0xCF, 0x20, 0xFF, 0xD9
    };
}

// Helper: Load JPEG file from disk
std::vector<uint8_t> load_jpeg_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    return data;
}

// Helper: Create test JPEG using PIL (if available)
bool create_test_jpeg(const std::string& path, int width, int height) {
    // Try to create a test image using Python PIL
    std::string cmd = "python3 -c \"from PIL import Image; "
                     "img = Image.new('RGB', (" + std::to_string(width) + ", " +
                     std::to_string(height) + "), color=(255, 0, 0)); "
                     "img.save('" + path + "')\" 2>/dev/null";

    return system(cmd.c_str()) == 0;
}

//==============================================================================
// Test 1: GPU Availability Detection
//==============================================================================
void test_gpu_availability() {
    std::cout << COLOR_CYAN << "\n[TEST] GPU Availability Detection" << COLOR_RESET << std::endl;

    NvJpegDecoder decoder;

    std::cout << "  GPU Available: " << (decoder.is_available() ? "YES" : "NO") << std::endl;

    if (decoder.is_available()) {
        std::cout << COLOR_GREEN << "  ✓ PASSED - GPU detected" << COLOR_RESET << std::endl;
    } else {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - GPU not available (CPU fallback will be used)" << COLOR_RESET << std::endl;
    }
}

//==============================================================================
// Test 2: Device Information
//==============================================================================
void test_device_info() {
    std::cout << COLOR_CYAN << "\n[TEST] Device Information" << COLOR_RESET << std::endl;

    NvJpegDecoder decoder;
    std::string info = decoder.get_device_info();

    std::cout << "  Device: " << info << std::endl;

    assert(!info.empty());
    std::cout << COLOR_GREEN << "  ✓ PASSED" << COLOR_RESET << std::endl;
}

//==============================================================================
// Test 3: Single Image Decode (Minimal JPEG)
//==============================================================================
void test_single_decode_minimal() {
    std::cout << COLOR_CYAN << "\n[TEST] Single Image Decode (Minimal JPEG)" << COLOR_RESET << std::endl;

    NvJpegDecoder decoder;

    // Create minimal 1x1 JPEG
    auto jpeg_data = create_minimal_jpeg();

    NvJpegResult result;
    bool success = decoder.decode(jpeg_data.data(), jpeg_data.size(), result);

    std::cout << "  Decode Success: " << (success ? "YES" : "NO") << std::endl;
    if (success) {
        std::cout << "  Image Size: " << result.width << "x" << result.height << std::endl;
        std::cout << "  Channels: " << result.channels << std::endl;
        std::cout << "  Data Size: " << result.data.size() << " bytes" << std::endl;
        std::cout << "  GPU Decoded: " << (result.gpu_decoded ? "YES" : "NO (CPU fallback)") << std::endl;
        std::cout << "  Decode Time: " << result.decode_time_ms << " ms" << std::endl;

        // Validate
        assert(result.width == 1);
        assert(result.height == 1);
        assert(result.channels == 3);
        assert(result.data.size() == 3);  // 1x1x3 = 3 bytes
        assert(result.is_success());
    }

    assert(success);
    std::cout << COLOR_GREEN << "  ✓ PASSED" << COLOR_RESET << std::endl;
}

//==============================================================================
// Test 4: Single Image Decode (Real JPEG File)
//==============================================================================
void test_single_decode_real_file() {
    std::cout << COLOR_CYAN << "\n[TEST] Single Image Decode (Real JPEG File)" << COLOR_RESET << std::endl;

    // Create a test JPEG file
    std::string test_file = "/tmp/test_nvjpeg_64x64.jpg";

    if (!create_test_jpeg(test_file, 64, 64)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test JPEG (Python PIL not available)" << COLOR_RESET << std::endl;
        return;
    }

    // Load the file
    auto jpeg_data = load_jpeg_file(test_file);
    if (jpeg_data.empty()) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not load test JPEG" << COLOR_RESET << std::endl;
        return;
    }

    std::cout << "  JPEG File Size: " << jpeg_data.size() << " bytes" << std::endl;

    // Decode
    NvJpegDecoder decoder;
    NvJpegResult result;
    bool success = decoder.decode(jpeg_data.data(), jpeg_data.size(), result);

    std::cout << "  Decode Success: " << (success ? "YES" : "NO") << std::endl;
    if (success) {
        std::cout << "  Image Size: " << result.width << "x" << result.height << std::endl;
        std::cout << "  Channels: " << result.channels << std::endl;
        std::cout << "  Data Size: " << result.data.size() << " bytes" << std::endl;
        std::cout << "  GPU Decoded: " << (result.gpu_decoded ? "YES" : "NO (CPU fallback)") << std::endl;
        std::cout << "  Decode Time: " << result.decode_time_ms << " ms" << std::endl;

        // Validate
        assert(result.width == 64);
        assert(result.height == 64);
        assert(result.channels == 3);
        assert(result.data.size() == 64 * 64 * 3);
        assert(result.is_success());
    }

    assert(success);
    std::cout << COLOR_GREEN << "  ✓ PASSED" << COLOR_RESET << std::endl;
}

//==============================================================================
// Test 5: Batch Decode
//==============================================================================
void test_batch_decode() {
    std::cout << COLOR_CYAN << "\n[TEST] Batch Decode (4 Images)" << COLOR_RESET << std::endl;

    // Create test images
    std::vector<std::string> test_files;
    std::vector<std::vector<uint8_t>> jpeg_data_storage;
    std::vector<const uint8_t*> jpeg_data_ptrs;
    std::vector<size_t> jpeg_sizes;

    // Create 4 test images of different sizes
    int sizes[] = {32, 64, 128, 256};

    for (int i = 0; i < 4; ++i) {
        std::string path = "/tmp/test_nvjpeg_batch_" + std::to_string(sizes[i]) + ".jpg";
        if (!create_test_jpeg(path, sizes[i], sizes[i])) {
            std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test JPEGs" << COLOR_RESET << std::endl;
            return;
        }

        auto data = load_jpeg_file(path);
        if (data.empty()) {
            std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not load test JPEGs" << COLOR_RESET << std::endl;
            return;
        }

        jpeg_data_storage.push_back(std::move(data));
        jpeg_data_ptrs.push_back(jpeg_data_storage.back().data());
        jpeg_sizes.push_back(jpeg_data_storage.back().size());
    }

    std::cout << "  Created 4 test images: 32x32, 64x64, 128x128, 256x256" << std::endl;

    // Batch decode
    NvJpegDecoder decoder;
    std::vector<NvJpegResult> results;

    auto start = std::chrono::high_resolution_clock::now();
    bool success = decoder.decode_batch(jpeg_data_ptrs, jpeg_sizes, results);
    auto end = std::chrono::high_resolution_clock::now();
    double batch_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "  Batch Decode Success: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "  Batch Decode Time: " << batch_time_ms << " ms" << std::endl;
    std::cout << "  Average per Image: " << (batch_time_ms / 4.0) << " ms" << std::endl;

    // Validate results
    assert(success);
    assert(results.size() == 4);

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << "  Image " << (i + 1) << ": " << result.width << "x" << result.height
                  << " (" << (result.gpu_decoded ? "GPU" : "CPU") << ")" << std::endl;

        assert(result.width == sizes[i]);
        assert(result.height == sizes[i]);
        assert(result.channels == 3);
        assert(result.data.size() == static_cast<size_t>(sizes[i] * sizes[i] * 3));
        assert(result.is_success());
    }

    std::cout << COLOR_GREEN << "  ✓ PASSED" << COLOR_RESET << std::endl;
}

//==============================================================================
// Test 6: Error Handling (Invalid JPEG Data)
//==============================================================================
void test_error_handling() {
    std::cout << COLOR_CYAN << "\n[TEST] Error Handling (Invalid JPEG)" << COLOR_RESET << std::endl;

    NvJpegDecoder decoder;

    // Invalid JPEG data
    std::vector<uint8_t> invalid_data = {0x00, 0x01, 0x02, 0x03, 0x04};

    NvJpegResult result;
    bool success = decoder.decode(invalid_data.data(), invalid_data.size(), result);

    std::cout << "  Decode Success: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "  Error Message: " << (result.error_message.empty() ? "(none)" : result.error_message) << std::endl;

    // Should fail
    assert(!success);
    assert(!result.is_success());
    assert(!result.error_message.empty());

    std::cout << COLOR_GREEN << "  ✓ PASSED - Error correctly detected" << COLOR_RESET << std::endl;
}

//==============================================================================
// Test 7: Result Structure Validation
//==============================================================================
void test_result_structure() {
    std::cout << COLOR_CYAN << "\n[TEST] Result Structure Validation" << COLOR_RESET << std::endl;

    // Test NvJpegResult default values
    NvJpegResult result;

    assert(result.width == 0);
    assert(result.height == 0);
    assert(result.channels == 3);
    assert(result.gpu_decoded == false);
    assert(result.decode_time_ms == 0.0);
    assert(result.data.empty());
    assert(result.error_message.empty());
    assert(result.is_success());  // Empty error = success

    std::cout << "  Default Result Structure: OK" << std::endl;

    // Test with error
    result.error_message = "Test error";
    assert(!result.is_success());

    std::cout << "  Error Detection: OK" << std::endl;

    std::cout << COLOR_GREEN << "  ✓ PASSED" << COLOR_RESET << std::endl;
}

//==============================================================================
// Benchmark 1: CPU Decode Performance
//==============================================================================
void benchmark_cpu_decode() {
    std::cout << COLOR_BLUE << "\n[BENCHMARK] CPU Decode Performance" << COLOR_RESET << std::endl;

    // Create test image
    std::string test_file = "/tmp/test_nvjpeg_benchmark.jpg";
    if (!create_test_jpeg(test_file, 256, 256)) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test JPEG" << COLOR_RESET << std::endl;
        return;
    }

    auto jpeg_data = load_jpeg_file(test_file);
    if (jpeg_data.empty()) {
        std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not load test JPEG" << COLOR_RESET << std::endl;
        return;
    }

    std::cout << "  Image: 256x256 JPEG" << std::endl;
    std::cout << "  File Size: " << jpeg_data.size() << " bytes" << std::endl;

    NvJpegDecoder decoder;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        NvJpegResult result;
        decoder.decode(jpeg_data.data(), jpeg_data.size(), result);
    }

    // Benchmark: 100 decodes
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        NvJpegResult result;
        decoder.decode(jpeg_data.data(), jpeg_data.size(), result);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_ms = total_time_ms / num_iterations;
    double throughput_fps = 1000.0 / avg_time_ms;

    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Total Time: " << total_time_ms << " ms" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms/image" << std::endl;
    std::cout << "  Throughput: " << throughput_fps << " images/sec" << std::endl;

    std::cout << COLOR_GREEN << "  ✓ BENCHMARK COMPLETE" << COLOR_RESET << std::endl;
}

//==============================================================================
// Benchmark 2: Batch vs Single Decode Comparison
//==============================================================================
void benchmark_batch_vs_single() {
    std::cout << COLOR_BLUE << "\n[BENCHMARK] Batch vs Single Decode" << COLOR_RESET << std::endl;

    // Create batch of test images
    const int batch_size = 16;
    std::vector<std::vector<uint8_t>> jpeg_data_storage;
    std::vector<const uint8_t*> jpeg_data_ptrs;
    std::vector<size_t> jpeg_sizes;

    for (int i = 0; i < batch_size; ++i) {
        std::string path = "/tmp/test_nvjpeg_batch_cmp_" + std::to_string(i) + ".jpg";
        if (!create_test_jpeg(path, 128, 128)) {
            std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not create test JPEGs" << COLOR_RESET << std::endl;
            return;
        }

        auto data = load_jpeg_file(path);
        if (data.empty()) {
            std::cout << COLOR_YELLOW << "  ⚠ SKIPPED - Could not load test JPEGs" << COLOR_RESET << std::endl;
            return;
        }

        jpeg_data_storage.push_back(std::move(data));
        jpeg_data_ptrs.push_back(jpeg_data_storage.back().data());
        jpeg_sizes.push_back(jpeg_data_storage.back().size());
    }

    std::cout << "  Batch Size: " << batch_size << " images (128x128 each)" << std::endl;

    NvJpegDecoder decoder;

    // Method 1: Single decode (sequential)
    auto start_single = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < batch_size; ++i) {
        NvJpegResult result;
        decoder.decode(jpeg_data_ptrs[i], jpeg_sizes[i], result);
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    double single_time_ms = std::chrono::duration<double, std::milli>(end_single - start_single).count();

    // Method 2: Batch decode
    auto start_batch = std::chrono::high_resolution_clock::now();
    std::vector<NvJpegResult> results;
    decoder.decode_batch(jpeg_data_ptrs, jpeg_sizes, results);
    auto end_batch = std::chrono::high_resolution_clock::now();
    double batch_time_ms = std::chrono::duration<double, std::milli>(end_batch - start_batch).count();

    std::cout << "\n  Single Decode (Sequential):" << std::endl;
    std::cout << "    Total Time: " << single_time_ms << " ms" << std::endl;
    std::cout << "    Per Image: " << (single_time_ms / batch_size) << " ms" << std::endl;

    std::cout << "\n  Batch Decode:" << std::endl;
    std::cout << "    Total Time: " << batch_time_ms << " ms" << std::endl;
    std::cout << "    Per Image: " << (batch_time_ms / batch_size) << " ms" << std::endl;

    double speedup = single_time_ms / batch_time_ms;
    std::cout << "\n  Batch Speedup: " << speedup << "x";
    if (speedup > 1.0) {
        std::cout << " ⚡ FASTER";
    }
    std::cout << std::endl;

    std::cout << COLOR_GREEN << "  ✓ BENCHMARK COMPLETE" << COLOR_RESET << std::endl;
}

//==============================================================================
// Main Test Runner
//==============================================================================
int main() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "nvJPEG DECODER TEST SUITE" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    try {
        // Basic tests
        test_gpu_availability();
        test_device_info();
        test_single_decode_minimal();
        test_single_decode_real_file();
        test_batch_decode();
        test_error_handling();
        test_result_structure();

        // Benchmarks
        benchmark_cpu_decode();
        benchmark_batch_vs_single();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << COLOR_GREEN << "ALL TESTS PASSED ✓" << COLOR_RESET << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n" << std::string(80, '=') << std::endl;
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        std::cerr << std::string(80, '=') << std::endl;
        return 1;
    }
}
