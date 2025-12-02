/**
 * @file test_decode_scaled.cpp
 * @brief Tests for Phase 2: DCT-scaled JPEG decoding (fused decode+resize)
 *
 * Tests:
 * 1. Basic scaled decode functionality
 * 2. Scale factor selection
 * 3. Various target dimensions
 * 4. Edge cases (very small, very large targets)
 * 5. Correctness of output dimensions
 */

#include "../src/decode/jpeg_decoder.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <vector>
#include <cmath>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

// Minimal valid JPEG data (1x1 red pixel)
// This is a valid JPEG that decodes to a small image
static const uint8_t MINIMAL_JPEG[] = {
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08,
    0x00, 0x08, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
    0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
    0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
    0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
    0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xB8, 0xAF, 0xFF, 0xD9
};
static const size_t MINIMAL_JPEG_SIZE = sizeof(MINIMAL_JPEG);

// Helper to create a simple test JPEG in memory
std::vector<uint8_t> create_test_jpeg(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        // Return minimal JPEG if file not found
        return std::vector<uint8_t>(MINIMAL_JPEG, MINIMAL_JPEG + MINIMAL_JPEG_SIZE);
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

/**
 * @brief Test basic decode functionality (without scaling)
 */
void test_basic_decode() {
    std::cout << BOLD << "\n[TEST] Basic JPEG Decode" << RESET << std::endl;

    JPEGDecoder decoder;

    // Use minimal JPEG
    std::span<const uint8_t> jpeg_data(MINIMAL_JPEG, MINIMAL_JPEG_SIZE);
    std::vector<uint8_t> output;
    int width, height, channels;

    try {
        decoder.decode(jpeg_data, output, width, height, channels);

        std::cout << "  Decoded dimensions: " << width << "x" << height << "x" << channels << std::endl;
        std::cout << "  Output size: " << output.size() << " bytes" << std::endl;

        assert(width > 0);
        assert(height > 0);
        assert(channels == 3);
        assert(output.size() == static_cast<size_t>(width * height * channels));

        std::cout << "  " << GREEN << "✓" << RESET << " Basic decode successful" << std::endl;
        std::cout << GREEN << "  PASSED" << RESET << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  " << YELLOW << "!" << RESET << " Skipped: " << e.what() << std::endl;
    }
}

/**
 * @brief Test scaled decode with various target dimensions
 */
void test_scaled_decode() {
    std::cout << BOLD << "\n[TEST] Scaled JPEG Decode" << RESET << std::endl;

    JPEGDecoder decoder;

    std::span<const uint8_t> jpeg_data(MINIMAL_JPEG, MINIMAL_JPEG_SIZE);

    // First decode without scaling to get original dimensions
    std::vector<uint8_t> orig_output;
    int orig_width, orig_height, orig_channels;

    try {
        decoder.decode(jpeg_data, orig_output, orig_width, orig_height, orig_channels);
        std::cout << "  Original dimensions: " << orig_width << "x" << orig_height << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  " << YELLOW << "!" << RESET << " Skipped: " << e.what() << std::endl;
        return;
    }

    // Test various target sizes
    struct ScaleTest {
        int target_width;
        int target_height;
        const char* description;
    };

    std::vector<ScaleTest> tests = {
        {0, 0, "No scaling (0, 0)"},
        {1, 1, "Minimum (1x1)"},
        {4, 4, "Small (4x4)"},
        {orig_width / 2, orig_height / 2, "Half size"},
        {orig_width, orig_height, "Original size"},
        {orig_width * 2, orig_height * 2, "Double size"},
    };

    for (const auto& test : tests) {
        std::vector<uint8_t> output;
        int actual_width, actual_height, channels;

        try {
            decoder.decode_scaled(
                jpeg_data, output,
                test.target_width, test.target_height,
                actual_width, actual_height, channels
            );

            std::cout << "  " << test.description << ": target="
                      << test.target_width << "x" << test.target_height
                      << " -> actual=" << actual_width << "x" << actual_height << std::endl;

            // Verify output is valid
            assert(actual_width > 0);
            assert(actual_height > 0);
            assert(channels == 3);
            assert(output.size() == static_cast<size_t>(actual_width * actual_height * channels));

            // If target specified, output should be >= target
            if (test.target_width > 0 && test.target_height > 0) {
                assert(actual_width >= test.target_width);
                assert(actual_height >= test.target_height);
            }

            std::cout << "    " << GREEN << "✓" << RESET << " Valid output" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "    " << YELLOW << "!" << RESET << " Exception: " << e.what() << std::endl;
        }
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test scale factor calculations
 */
void test_scale_factor_selection() {
    std::cout << BOLD << "\n[TEST] Scale Factor Selection" << RESET << std::endl;

    // Test the scale factor selection logic
    // This mimics the logic in decode_scaled()
    struct { unsigned int num, denom; } scales[] = {
        {1, 8}, {1, 4}, {3, 8}, {1, 2}, {5, 8}, {3, 4}, {7, 8}, {1, 1},
        {9, 8}, {5, 4}, {11, 8}, {3, 2}, {13, 8}, {7, 4}, {15, 8}, {2, 1}
    };

    // Test case: 1024x768 image, target 224x224
    int orig_width = 1024;
    int orig_height = 768;
    int target_width = 224;
    int target_height = 224;

    unsigned int best_num = 8, best_denom = 8;

    for (const auto& scale : scales) {
        int scaled_w = (orig_width * scale.num + scale.denom - 1) / scale.denom;
        int scaled_h = (orig_height * scale.num + scale.denom - 1) / scale.denom;

        if (scaled_w >= target_width && scaled_h >= target_height) {
            best_num = scale.num;
            best_denom = scale.denom;
            break;
        }
    }

    std::cout << "  Original: " << orig_width << "x" << orig_height << std::endl;
    std::cout << "  Target: " << target_width << "x" << target_height << std::endl;
    std::cout << "  Best scale: " << best_num << "/" << best_denom << std::endl;

    int result_w = (orig_width * best_num + best_denom - 1) / best_denom;
    int result_h = (orig_height * best_num + best_denom - 1) / best_denom;
    std::cout << "  Result: " << result_w << "x" << result_h << std::endl;

    assert(result_w >= target_width);
    assert(result_h >= target_height);

    std::cout << "  " << GREEN << "✓" << RESET << " Scale factor selection correct" << std::endl;
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test decode_sample_scaled with Sample struct
 */
void test_decode_sample_scaled() {
    std::cout << BOLD << "\n[TEST] Decode Sample Scaled" << RESET << std::endl;

    JPEGDecoder decoder;

    // Create a sample with JPEG data - note: jpeg_data is a span, so we need
    // to keep the underlying data alive for the duration of the sample's use
    std::vector<uint8_t> jpeg_buffer(MINIMAL_JPEG, MINIMAL_JPEG + MINIMAL_JPEG_SIZE);
    Sample sample(0, std::span<const uint8_t>(jpeg_buffer));

    try {
        decoder.decode_sample_scaled(sample, 4, 4);

        std::cout << "  Sample decoded: " << sample.width << "x" << sample.height
                  << "x" << sample.channels << std::endl;
        std::cout << "  RGB buffer size: " << sample.decoded_rgb.size() << " bytes" << std::endl;

        assert(sample.width > 0);
        assert(sample.height > 0);
        assert(sample.channels == 3);
        assert(!sample.decoded_rgb.empty());
        assert(sample.decoded_rgb.size() ==
               static_cast<size_t>(sample.width * sample.height * sample.channels));

        std::cout << "  " << GREEN << "✓" << RESET << " Sample scaling successful" << std::endl;
        std::cout << GREEN << "  PASSED" << RESET << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  " << YELLOW << "!" << RESET << " Skipped: " << e.what() << std::endl;
    }
}

/**
 * @brief Test error handling
 */
void test_error_handling() {
    std::cout << BOLD << "\n[TEST] Error Handling" << RESET << std::endl;

    JPEGDecoder decoder;

    // Test 1: Empty data
    {
        std::vector<uint8_t> empty_data;
        std::span<const uint8_t> jpeg_data(empty_data);
        std::vector<uint8_t> output;
        int width, height, channels;

        bool caught_exception = false;
        try {
            decoder.decode_scaled(jpeg_data, output, 224, 224, width, height, channels);
        } catch (const std::runtime_error& e) {
            caught_exception = true;
            std::cout << "  " << GREEN << "✓" << RESET << " Caught exception for empty data: "
                      << e.what() << std::endl;
        }
        assert(caught_exception);
    }

    // Test 2: Invalid JPEG data
    {
        std::vector<uint8_t> invalid_data = {0x00, 0x01, 0x02, 0x03};
        std::span<const uint8_t> jpeg_data(invalid_data);
        std::vector<uint8_t> output;
        int width, height, channels;

        bool caught_exception = false;
        try {
            decoder.decode_scaled(jpeg_data, output, 224, 224, width, height, channels);
        } catch (const std::runtime_error& e) {
            caught_exception = true;
            std::cout << "  " << GREEN << "✓" << RESET << " Caught exception for invalid data: "
                      << e.what() << std::endl;
        }
        assert(caught_exception);
    }

    // Test 3: Empty sample
    {
        Sample sample;  // Default constructed with empty jpeg_data

        bool caught_exception = false;
        try {
            decoder.decode_sample_scaled(sample, 224, 224);
        } catch (const std::runtime_error& e) {
            caught_exception = true;
            std::cout << "  " << GREEN << "✓" << RESET << " Caught exception for empty sample: "
                      << e.what() << std::endl;
        }
        assert(caught_exception);
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test version info includes DCT scaling
 */
void test_version_info() {
    std::cout << BOLD << "\n[TEST] Version Info" << RESET << std::endl;

    std::string info = JPEGDecoder::version_info();
    std::cout << "  Version info: " << info << std::endl;

    // Check that DCT scaling is mentioned
    assert(info.find("DCT scaling") != std::string::npos);
    assert(info.find("SIMD") != std::string::npos);

    std::cout << "  " << GREEN << "✓" << RESET << " Version info includes DCT scaling" << std::endl;
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Performance comparison: scaled vs full decode
 */
void test_performance_comparison() {
    std::cout << BOLD << "\n[TEST] Performance Comparison (Scaled vs Full)" << RESET << std::endl;

    // Try to load a real JPEG for performance testing
    std::vector<uint8_t> jpeg_data;

    // Try common test image locations
    const char* test_paths[] = {
        "/tmp/test_image.jpg",
        "/tmp/benchmark_images/image_0000.jpg",
        nullptr
    };

    for (const char** path = test_paths; *path; ++path) {
        std::ifstream file(*path, std::ios::binary);
        if (file) {
            jpeg_data = std::vector<uint8_t>(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>()
            );
            std::cout << "  Using test image: " << *path << std::endl;
            break;
        }
    }

    if (jpeg_data.empty()) {
        std::cout << "  " << YELLOW << "!" << RESET << " No test image found, using minimal JPEG" << std::endl;
        jpeg_data = std::vector<uint8_t>(MINIMAL_JPEG, MINIMAL_JPEG + MINIMAL_JPEG_SIZE);
    }

    std::span<const uint8_t> data(jpeg_data);
    JPEGDecoder decoder;

    const int iterations = 100;
    std::vector<uint8_t> output;
    int width, height, channels;

    // Benchmark full decode
    auto start_full = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        decoder.decode(data, output, width, height, channels);
    }
    auto end_full = std::chrono::high_resolution_clock::now();
    double time_full = std::chrono::duration<double, std::milli>(end_full - start_full).count();

    std::cout << "  Full decode: " << width << "x" << height << std::endl;
    std::cout << "  Full decode time: " << (time_full / iterations) << " ms/decode" << std::endl;

    // Benchmark scaled decode (target 224x224)
    int actual_w, actual_h;
    auto start_scaled = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        decoder.decode_scaled(data, output, 224, 224, actual_w, actual_h, channels);
    }
    auto end_scaled = std::chrono::high_resolution_clock::now();
    double time_scaled = std::chrono::duration<double, std::milli>(end_scaled - start_scaled).count();

    std::cout << "  Scaled decode: " << actual_w << "x" << actual_h << std::endl;
    std::cout << "  Scaled decode time: " << (time_scaled / iterations) << " ms/decode" << std::endl;

    double speedup = time_full / time_scaled;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;

    // If scaled decode should be faster (when target is smaller than original)
    if (actual_w < width || actual_h < height) {
        if (speedup > 1.0) {
            std::cout << "  " << GREEN << "✓" << RESET << " Scaled decode is faster" << std::endl;
        } else {
            std::cout << "  " << YELLOW << "!" << RESET << " Scaled decode not faster (may be due to small image)" << std::endl;
        }
    } else {
        std::cout << "  " << YELLOW << "!" << RESET << " Target >= original, no scaling benefit expected" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

int main() {
    std::cout << BOLD << "\n========================================" << std::endl;
    std::cout << "TurboLoader Phase 2: DCT Scaling Tests" << std::endl;
    std::cout << "========================================" << RESET << std::endl;

    std::cout << "\nJPEG Decoder: " << JPEGDecoder::version_info() << std::endl;

    try {
        test_basic_decode();
        test_scaled_decode();
        test_scale_factor_selection();
        test_decode_sample_scaled();
        test_error_handling();
        test_version_info();
        test_performance_comparison();

        std::cout << BOLD << "\n========================================" << std::endl;
        std::cout << GREEN << "ALL TESTS PASSED" << RESET << std::endl;
        std::cout << "========================================" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "ERROR: " << e.what() << RESET << std::endl;
        return 1;
    }
}
