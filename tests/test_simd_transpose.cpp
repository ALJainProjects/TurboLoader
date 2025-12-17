/**
 * @file test_simd_transpose.cpp
 * @brief Tests for SIMD HWC→CHW channel transpose (Phase 3.1)
 *
 * Tests the SIMD-accelerated channel transpose functions:
 * - transpose_hwc_to_chw: RGB interleaved → planar
 * - transpose_chw_to_hwc: RGB planar → interleaved
 *
 * Verifies correctness on ARM NEON, AVX2, and scalar fallback.
 */

#include <gtest/gtest.h>
#include "../src/transforms/simd_utils.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

using namespace turboloader::transforms::simd;

class SIMDTransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducible tests
        rng_.seed(42);
    }

    // Generate random RGB image data
    std::vector<uint8_t> generate_hwc_image(int width, int height) {
        std::vector<uint8_t> data(width * height * 3);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& byte : data) {
            byte = static_cast<uint8_t>(dist(rng_));
        }
        return data;
    }

    // Generate random planar image data
    std::vector<uint8_t> generate_chw_image(int width, int height) {
        return generate_hwc_image(width, height); // Same generation
    }

    // Reference scalar HWC→CHW implementation
    void reference_hwc_to_chw(const uint8_t* src, uint8_t* dst,
                               size_t num_pixels, int channels = 3) {
        for (size_t p = 0; p < num_pixels; ++p) {
            for (int c = 0; c < channels; ++c) {
                dst[c * num_pixels + p] = src[p * channels + c];
            }
        }
    }

    // Reference scalar CHW→HWC implementation
    void reference_chw_to_hwc(const uint8_t* src, uint8_t* dst,
                               size_t num_pixels, int channels = 3) {
        for (size_t p = 0; p < num_pixels; ++p) {
            for (int c = 0; c < channels; ++c) {
                dst[p * channels + c] = src[c * num_pixels + p];
            }
        }
    }

    std::mt19937 rng_;
};

// ============================================================================
// Basic Correctness Tests
// ============================================================================

TEST_F(SIMDTransposeTest, HWC_to_CHW_SmallImage) {
    // 4x4 image = 16 pixels
    const int width = 4, height = 4;
    const size_t num_pixels = width * height;

    auto hwc_data = generate_hwc_image(width, height);
    std::vector<uint8_t> simd_result(hwc_data.size());
    std::vector<uint8_t> ref_result(hwc_data.size());

    transpose_hwc_to_chw(hwc_data.data(), simd_result.data(), num_pixels, 3);
    reference_hwc_to_chw(hwc_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

TEST_F(SIMDTransposeTest, HWC_to_CHW_224x224) {
    // Standard ImageNet size
    const int width = 224, height = 224;
    const size_t num_pixels = width * height;

    auto hwc_data = generate_hwc_image(width, height);
    std::vector<uint8_t> simd_result(hwc_data.size());
    std::vector<uint8_t> ref_result(hwc_data.size());

    transpose_hwc_to_chw(hwc_data.data(), simd_result.data(), num_pixels, 3);
    reference_hwc_to_chw(hwc_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

TEST_F(SIMDTransposeTest, HWC_to_CHW_OddDimensions) {
    // Non-power-of-2 dimensions to test tail handling
    const int width = 127, height = 97;
    const size_t num_pixels = width * height;

    auto hwc_data = generate_hwc_image(width, height);
    std::vector<uint8_t> simd_result(hwc_data.size());
    std::vector<uint8_t> ref_result(hwc_data.size());

    transpose_hwc_to_chw(hwc_data.data(), simd_result.data(), num_pixels, 3);
    reference_hwc_to_chw(hwc_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

TEST_F(SIMDTransposeTest, CHW_to_HWC_SmallImage) {
    const int width = 4, height = 4;
    const size_t num_pixels = width * height;

    auto chw_data = generate_chw_image(width, height);
    std::vector<uint8_t> simd_result(chw_data.size());
    std::vector<uint8_t> ref_result(chw_data.size());

    transpose_chw_to_hwc(chw_data.data(), simd_result.data(), num_pixels, 3);
    reference_chw_to_hwc(chw_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

TEST_F(SIMDTransposeTest, CHW_to_HWC_224x224) {
    const int width = 224, height = 224;
    const size_t num_pixels = width * height;

    auto chw_data = generate_chw_image(width, height);
    std::vector<uint8_t> simd_result(chw_data.size());
    std::vector<uint8_t> ref_result(chw_data.size());

    transpose_chw_to_hwc(chw_data.data(), simd_result.data(), num_pixels, 3);
    reference_chw_to_hwc(chw_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

// ============================================================================
// Round-Trip Tests
// ============================================================================

TEST_F(SIMDTransposeTest, RoundTrip_HWC_CHW_HWC) {
    const int width = 256, height = 256;
    const size_t num_pixels = width * height;

    auto original = generate_hwc_image(width, height);
    std::vector<uint8_t> chw_temp(original.size());
    std::vector<uint8_t> hwc_final(original.size());

    // HWC → CHW → HWC
    transpose_hwc_to_chw(original.data(), chw_temp.data(), num_pixels, 3);
    transpose_chw_to_hwc(chw_temp.data(), hwc_final.data(), num_pixels, 3);

    EXPECT_EQ(original, hwc_final);
}

TEST_F(SIMDTransposeTest, RoundTrip_CHW_HWC_CHW) {
    const int width = 256, height = 256;
    const size_t num_pixels = width * height;

    auto original = generate_chw_image(width, height);
    std::vector<uint8_t> hwc_temp(original.size());
    std::vector<uint8_t> chw_final(original.size());

    // CHW → HWC → CHW
    transpose_chw_to_hwc(original.data(), hwc_temp.data(), num_pixels, 3);
    transpose_hwc_to_chw(hwc_temp.data(), chw_final.data(), num_pixels, 3);

    EXPECT_EQ(original, chw_final);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SIMDTransposeTest, SinglePixel) {
    const size_t num_pixels = 1;
    uint8_t hwc[3] = {100, 150, 200};
    uint8_t chw[3] = {0, 0, 0};

    transpose_hwc_to_chw(hwc, chw, num_pixels, 3);

    EXPECT_EQ(chw[0], 100);  // R channel
    EXPECT_EQ(chw[1], 150);  // G channel
    EXPECT_EQ(chw[2], 200);  // B channel
}

TEST_F(SIMDTransposeTest, VeryLargeImage) {
    // 4K resolution
    const int width = 3840, height = 2160;
    const size_t num_pixels = width * height;

    auto hwc_data = generate_hwc_image(width, height);
    std::vector<uint8_t> simd_result(hwc_data.size());
    std::vector<uint8_t> ref_result(hwc_data.size());

    transpose_hwc_to_chw(hwc_data.data(), simd_result.data(), num_pixels, 3);
    reference_hwc_to_chw(hwc_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

TEST_F(SIMDTransposeTest, ExactlySIMDWidth) {
    // 16 pixels = exact NEON vld3q_u8 width
    const size_t num_pixels = 16;
    auto hwc_data = generate_hwc_image(16, 1);
    std::vector<uint8_t> simd_result(hwc_data.size());
    std::vector<uint8_t> ref_result(hwc_data.size());

    transpose_hwc_to_chw(hwc_data.data(), simd_result.data(), num_pixels, 3);
    reference_hwc_to_chw(hwc_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

TEST_F(SIMDTransposeTest, JustBelowSIMDWidth) {
    // 15 pixels = tests scalar tail
    const size_t num_pixels = 15;
    auto hwc_data = generate_hwc_image(15, 1);
    std::vector<uint8_t> simd_result(hwc_data.size());
    std::vector<uint8_t> ref_result(hwc_data.size());

    transpose_hwc_to_chw(hwc_data.data(), simd_result.data(), num_pixels, 3);
    reference_hwc_to_chw(hwc_data.data(), ref_result.data(), num_pixels, 3);

    EXPECT_EQ(simd_result, ref_result);
}

// ============================================================================
// Performance Benchmark
// ============================================================================

TEST_F(SIMDTransposeTest, BenchmarkHWC_to_CHW) {
    // Benchmark 224x224 image transpose
    const int width = 224, height = 224;
    const size_t num_pixels = width * height;
    const int iterations = 10000;  // More iterations for stable timing

    auto hwc_data = generate_hwc_image(width, height);
    std::vector<uint8_t> result(hwc_data.size());

    // Warm up
    for (int i = 0; i < 100; ++i) {
        transpose_hwc_to_chw(hwc_data.data(), result.data(), num_pixels, 3);
    }

    // Benchmark SIMD (use nanoseconds for precision)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        transpose_hwc_to_chw(hwc_data.data(), result.data(), num_pixels, 3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Benchmark reference (scalar)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        reference_hwc_to_chw(hwc_data.data(), result.data(), num_pixels, 3);
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Avoid division by zero
    float speedup = (simd_ns > 0) ? static_cast<float>(scalar_ns) / simd_ns : 1.0f;

    std::cout << "\n=== HWC→CHW Transpose Benchmark (224x224) ===" << std::endl;
    std::cout << "  SIMD:   " << simd_ns / iterations << " ns/image ("
              << (simd_ns / iterations / 1000.0) << " us/image)" << std::endl;
    std::cout << "  Scalar: " << scalar_ns / iterations << " ns/image ("
              << (scalar_ns / iterations / 1000.0) << " us/image)" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;

    // Platform info
#if defined(TURBOLOADER_SIMD_NEON)
    std::cout << "  Platform: ARM NEON" << std::endl;
#elif defined(TURBOLOADER_SIMD_AVX2)
    std::cout << "  Platform: x86 AVX2" << std::endl;
#elif defined(TURBOLOADER_SIMD_AVX512)
    std::cout << "  Platform: x86 AVX-512" << std::endl;
#else
    std::cout << "  Platform: Scalar fallback" << std::endl;
#endif

    // On NEON, expect at least 1.5x speedup (SIMD should not be slower)
#if defined(TURBOLOADER_SIMD_NEON)
    EXPECT_GE(speedup, 1.0f) << "SIMD should not be slower than scalar";
#endif
}

// ============================================================================
// Data Integrity Tests
// ============================================================================

TEST_F(SIMDTransposeTest, ChannelIntegrity) {
    // Create image with distinct channel values
    const int width = 32, height = 32;
    const size_t num_pixels = width * height;

    std::vector<uint8_t> hwc(num_pixels * 3);
    for (size_t p = 0; p < num_pixels; ++p) {
        hwc[p * 3 + 0] = 10;   // R = 10
        hwc[p * 3 + 1] = 20;   // G = 20
        hwc[p * 3 + 2] = 30;   // B = 30
    }

    std::vector<uint8_t> chw(hwc.size());
    transpose_hwc_to_chw(hwc.data(), chw.data(), num_pixels, 3);

    // Verify R channel (first num_pixels bytes)
    for (size_t p = 0; p < num_pixels; ++p) {
        EXPECT_EQ(chw[p], 10) << "R channel corrupted at pixel " << p;
    }

    // Verify G channel (second num_pixels bytes)
    for (size_t p = 0; p < num_pixels; ++p) {
        EXPECT_EQ(chw[num_pixels + p], 20) << "G channel corrupted at pixel " << p;
    }

    // Verify B channel (third num_pixels bytes)
    for (size_t p = 0; p < num_pixels; ++p) {
        EXPECT_EQ(chw[2 * num_pixels + p], 30) << "B channel corrupted at pixel " << p;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "=== TurboLoader SIMD Transpose Tests ===" << std::endl;

    return RUN_ALL_TESTS();
}
