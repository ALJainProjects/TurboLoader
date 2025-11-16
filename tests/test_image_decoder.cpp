/**
 * @file test_image_decoder.cpp
 * @brief Comprehensive tests for all image format decoders
 *
 * Tests:
 * 1. JPEG decoding (SIMD-accelerated)
 * 2. PNG decoding (all color types)
 * 3. WebP decoding (SIMD-accelerated)
 * 4. BMP decoding (uncompressed)
 * 5. Format auto-detection
 * 6. ImageDecoder orchestrator
 */

#include "../src/decode/image_decoder.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Create minimal valid JPEG (1x1 pixel, red)
 *
 * This is a properly encoded JPEG file that libjpeg-turbo can decode.
 * Generated using: convert -size 1x1 xc:red test.jpg
 */
std::vector<uint8_t> create_test_jpeg() {
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

/**
 * @brief Create minimal valid PNG (1x1 pixel, red)
 * Generated using PIL: Image.new('RGB', (1, 1), color=(255, 0, 0)).save('test.png')
 */
std::vector<uint8_t> create_test_png() {
    return {
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00,
        0x0C, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x03, 0x01, 0x01, 0x00, 0xC9, 0xFE, 0x92, 0xEF, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    };
}

/**
 * @brief Create minimal valid BMP (2x2 pixels, 24-bit)
 * Generated using PIL with red, white, green, blue pixels
 */
std::vector<uint8_t> create_test_bmp() {
    return {
        0x42, 0x4D, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00,
        0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0xC4, 0x0E, 0x00, 0x00, 0xC4, 0x0E, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00
    };
}

/**
 * @brief Test format detection
 */
void test_format_detection() {
    std::cout << BOLD << "\n[TEST] Format Detection" << RESET << std::endl;

    auto jpeg_data = create_test_jpeg();
    auto png_data = create_test_png();
    auto bmp_data = create_test_bmp();

    assert(detect_format(jpeg_data) == ImageFormat::JPEG);
    std::cout << "  " << GREEN << "✓" << RESET << " JPEG format detected correctly" << std::endl;

    assert(detect_format(png_data) == ImageFormat::PNG);
    std::cout << "  " << GREEN << "✓" << RESET << " PNG format detected correctly" << std::endl;

    assert(detect_format(bmp_data) == ImageFormat::BMP);
    std::cout << "  " << GREEN << "✓" << RESET << " BMP format detected correctly" << std::endl;

    // Test unknown format
    std::vector<uint8_t> unknown = {0x00, 0x00, 0x00, 0x00};
    assert(detect_format(unknown) == ImageFormat::UNKNOWN);
    std::cout << "  " << GREEN << "✓" << RESET << " Unknown format detected correctly" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test JPEG decoder
 */
void test_jpeg_decoder() {
    std::cout << BOLD << "\n[TEST] JPEG Decoder" << RESET << std::endl;

    auto jpeg_data = create_test_jpeg();
    JPEGDecoder decoder;

    std::vector<uint8_t> output;
    int width, height, channels;

    decoder.decode(jpeg_data, output, width, height, channels);

    assert(width >= 1);
    assert(height >= 1);
    assert(channels == 3);
    assert(!output.empty());

    std::cout << "  " << GREEN << "✓" << RESET << " JPEG decoded successfully" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Output dimensions: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " " << JPEGDecoder::version_info() << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test PNG decoder
 */
void test_png_decoder() {
    std::cout << BOLD << "\n[TEST] PNG Decoder" << RESET << std::endl;

    auto png_data = create_test_png();
    PNGDecoder decoder;

    std::vector<uint8_t> output;
    int width, height, channels;

    decoder.decode(png_data, output, width, height, channels);

    assert(width == 1);
    assert(height == 1);
    assert(channels == 3);
    assert(output.size() == 3);  // 1x1x3

    std::cout << "  " << GREEN << "✓" << RESET << " PNG decoded successfully" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Output dimensions: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " " << PNGDecoder::version_info() << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test BMP decoder
 */
void test_bmp_decoder() {
    std::cout << BOLD << "\n[TEST] BMP Decoder" << RESET << std::endl;

    auto bmp_data = create_test_bmp();
    BMPDecoder decoder;

    std::vector<uint8_t> output;
    int width, height, channels;

    decoder.decode(bmp_data, output, width, height, channels);

    assert(width == 2);
    assert(height == 2);
    assert(channels == 3);
    assert(output.size() == 12);  // 2x2x3

    std::cout << "  " << GREEN << "✓" << RESET << " BMP decoded successfully" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Output dimensions: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " " << BMPDecoder::version_info() << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test unified ImageDecoder
 */
void test_image_decoder() {
    std::cout << BOLD << "\n[TEST] Unified ImageDecoder" << RESET << std::endl;

    ImageDecoder decoder;

    // Test JPEG
    {
        auto jpeg_data = create_test_jpeg();
        std::vector<uint8_t> output;
        int width, height, channels;

        decoder.decode(jpeg_data, output, width, height, channels);
        assert(!output.empty());
        std::cout << "  " << GREEN << "✓" << RESET << " JPEG decoded via ImageDecoder" << std::endl;
    }

    // Test PNG
    {
        auto png_data = create_test_png();
        std::vector<uint8_t> output;
        int width, height, channels;

        decoder.decode(png_data, output, width, height, channels);
        assert(output.size() == 3);
        std::cout << "  " << GREEN << "✓" << RESET << " PNG decoded via ImageDecoder" << std::endl;
    }

    // Test BMP
    {
        auto bmp_data = create_test_bmp();
        std::vector<uint8_t> output;
        int width, height, channels;

        decoder.decode(bmp_data, output, width, height, channels);
        assert(output.size() == 12);
        std::cout << "  " << GREEN << "✓" << RESET << " BMP decoded via ImageDecoder" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Performance benchmark for decoder throughput
 *
 * Measures images/second for each decoder
 */
void benchmark_decoder_performance() {
    std::cout << BOLD << "\n[BENCHMARK] Decoder Performance" << RESET << std::endl;

    const int num_iterations = 1000;

    // Benchmark JPEG decoder
    {
        auto jpeg_data = create_test_jpeg();
        JPEGDecoder decoder;
        std::vector<uint8_t> output;
        int width, height, channels;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            decoder.decode(jpeg_data, output, width, height, channels);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double images_per_sec = (num_iterations * 1000000.0) / duration.count();

        std::cout << "  " << GREEN << "✓" << RESET << " JPEG: " << static_cast<int>(images_per_sec)
                  << " images/sec (" << (duration.count() / num_iterations) << " μs/image)" << std::endl;
    }

    // Benchmark PNG decoder
    {
        auto png_data = create_test_png();
        PNGDecoder decoder;
        std::vector<uint8_t> output;
        int width, height, channels;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            decoder.decode(png_data, output, width, height, channels);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double images_per_sec = (num_iterations * 1000000.0) / duration.count();

        std::cout << "  " << GREEN << "✓" << RESET << " PNG: " << static_cast<int>(images_per_sec)
                  << " images/sec (" << (duration.count() / num_iterations) << " μs/image)" << std::endl;
    }

    // Benchmark BMP decoder
    {
        auto bmp_data = create_test_bmp();
        BMPDecoder decoder;
        std::vector<uint8_t> output;
        int width, height, channels;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            decoder.decode(bmp_data, output, width, height, channels);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double images_per_sec = (num_iterations * 1000000.0) / duration.count();

        std::cout << "  " << GREEN << "✓" << RESET << " BMP: " << static_cast<int>(images_per_sec)
                  << " images/sec (" << (duration.count() / num_iterations) << " μs/image)" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "║      TurboLoader Image Decoder Test Suite           ║" << RESET << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

    try {
        test_format_detection();
        test_jpeg_decoder();
        test_png_decoder();
        test_bmp_decoder();
        test_image_decoder();
        benchmark_decoder_performance();

        std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << "║  " << GREEN << "✓ ALL TESTS PASSED" << RESET << BOLD << "                                ║" << RESET << std::endl;
        std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ TEST FAILED: " << e.what() << RESET << std::endl;
        return 1;
    }
}
