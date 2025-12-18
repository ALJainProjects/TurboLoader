/**
 * @file test_simd_bilinear.cpp
 * @brief Tests for SIMD-accelerated bilinear interpolation (Phase 3.2)
 *
 * Tests the SIMD-accelerated bilinear resize functions:
 * - resize_bilinear_simd: SIMD-optimized resize
 * - Verifies correctness on ARM NEON, AVX2, and scalar fallback
 * - Benchmarks against scalar implementation
 */

#include <gtest/gtest.h>
#include "../src/transforms/simd_utils.hpp"
#include "../src/transforms/resize_transform.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace turboloader::transforms::simd;

class SIMDBilinearTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);
    }

    // Generate random RGB image data
    std::vector<uint8_t> generate_rgb_image(int width, int height) {
        std::vector<uint8_t> data(width * height * 3);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& byte : data) {
            byte = static_cast<uint8_t>(dist(rng_));
        }
        return data;
    }

    // Generate gradient image for visual verification
    std::vector<uint8_t> generate_gradient_image(int width, int height) {
        std::vector<uint8_t> data(width * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                data[idx + 0] = static_cast<uint8_t>(x * 255 / (width - 1));   // R = horizontal gradient
                data[idx + 1] = static_cast<uint8_t>(y * 255 / (height - 1)); // G = vertical gradient
                data[idx + 2] = 128;  // B = constant
            }
        }
        return data;
    }

    // Reference scalar bilinear resize implementation
    void reference_bilinear_resize(const uint8_t* src, uint8_t* dst,
                                    int src_width, int src_height,
                                    int dst_width, int dst_height,
                                    int channels = 3) {
        const float x_ratio = static_cast<float>(src_width - 1) / std::max(1, dst_width - 1);
        const float y_ratio = static_cast<float>(src_height - 1) / std::max(1, dst_height - 1);

        for (int y = 0; y < dst_height; ++y) {
            float src_y = y * y_ratio;
            int y0 = static_cast<int>(src_y);
            int y1 = std::min(y0 + 1, src_height - 1);
            float dy = src_y - y0;

            for (int x = 0; x < dst_width; ++x) {
                float src_x = x * x_ratio;
                int x0 = static_cast<int>(src_x);
                int x1 = std::min(x0 + 1, src_width - 1);
                float dx = src_x - x0;

                for (int c = 0; c < channels; ++c) {
                    float p00 = src[(y0 * src_width + x0) * channels + c];
                    float p10 = src[(y0 * src_width + x1) * channels + c];
                    float p01 = src[(y1 * src_width + x0) * channels + c];
                    float p11 = src[(y1 * src_width + x1) * channels + c];

                    float top = p00 * (1.0f - dx) + p10 * dx;
                    float bot = p01 * (1.0f - dx) + p11 * dx;
                    float val = top * (1.0f - dy) + bot * dy;

                    dst[(y * dst_width + x) * channels + c] =
                        static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
                }
            }
        }
    }

    // Calculate max absolute difference between two images
    int max_diff(const uint8_t* a, const uint8_t* b, size_t size) {
        int max_d = 0;
        for (size_t i = 0; i < size; ++i) {
            int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
            max_d = std::max(max_d, d);
        }
        return max_d;
    }

    std::mt19937 rng_;
};

// ============================================================================
// Basic Correctness Tests
// ============================================================================

TEST_F(SIMDBilinearTest, Downscale_224_to_112) {
    const int src_w = 224, src_h = 224;
    const int dst_w = 112, dst_h = 112;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    // Allow small rounding differences (max 1 due to float precision)
    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1) << "SIMD and reference results differ by more than 1";
}

TEST_F(SIMDBilinearTest, Upscale_112_to_224) {
    const int src_w = 112, src_h = 112;
    const int dst_w = 224, dst_h = 224;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1) << "SIMD and reference results differ by more than 1";
}

TEST_F(SIMDBilinearTest, AsymmetricResize) {
    const int src_w = 320, src_h = 240;
    const int dst_w = 224, dst_h = 224;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1) << "SIMD and reference results differ by more than 1";
}

TEST_F(SIMDBilinearTest, NonMultipleOfFourWidth) {
    // Test width that's not a multiple of 4 (SIMD vector size)
    const int src_w = 100, src_h = 100;
    const int dst_w = 127, dst_h = 97;  // Odd dimensions

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1) << "SIMD and reference results differ by more than 1";
}

TEST_F(SIMDBilinearTest, SmallImage) {
    const int src_w = 8, src_h = 8;
    const int dst_w = 16, dst_h = 16;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1) << "SIMD and reference results differ by more than 1";
}

TEST_F(SIMDBilinearTest, LargeImage_4K_to_224) {
    // 4K resolution downscale to 224x224
    const int src_w = 3840, src_h = 2160;
    const int dst_w = 224, dst_h = 224;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1) << "SIMD and reference results differ by more than 1";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SIMDBilinearTest, SameSize) {
    // No actual resize - should still work
    const int w = 64, h = 64;

    auto src = generate_rgb_image(w, h);
    std::vector<uint8_t> simd_result(w * h * 3);
    std::vector<uint8_t> ref_result(w * h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), w, h, w, h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), w, h, w, h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1);
}

TEST_F(SIMDBilinearTest, SingleRow) {
    const int src_w = 100, src_h = 1;
    const int dst_w = 224, dst_h = 1;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1);
}

TEST_F(SIMDBilinearTest, SingleColumn) {
    const int src_w = 1, src_h = 100;
    const int dst_w = 1, dst_h = 224;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_result(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), simd_result.data(), src_w, src_h, dst_w, dst_h, 3);
    reference_bilinear_resize(src.data(), ref_result.data(), src_w, src_h, dst_w, dst_h, 3);

    int diff = max_diff(simd_result.data(), ref_result.data(), simd_result.size());
    EXPECT_LE(diff, 1);
}

TEST_F(SIMDBilinearTest, GradientPreservation) {
    // Test that gradients are preserved after resize
    const int src_w = 256, src_h = 256;
    const int dst_w = 128, dst_h = 128;

    auto src = generate_gradient_image(src_w, src_h);
    std::vector<uint8_t> result(dst_w * dst_h * 3);

    resize_bilinear_simd(src.data(), result.data(), src_w, src_h, dst_w, dst_h, 3);

    // Check that the output still has smooth gradients
    // Top-left should be dark red (low R, low G)
    EXPECT_LT(result[0], 10);  // R near 0
    EXPECT_LT(result[1], 10);  // G near 0
    EXPECT_NEAR(result[2], 128, 5);  // B constant

    // Top-right should be bright red
    int top_right = (dst_w - 1) * 3;
    EXPECT_GT(result[top_right], 245);  // R near 255
    EXPECT_LT(result[top_right + 1], 10);  // G near 0

    // Bottom-left should be dark green
    int bot_left = (dst_h - 1) * dst_w * 3;
    EXPECT_LT(result[bot_left], 10);  // R near 0
    EXPECT_GT(result[bot_left + 1], 245);  // G near 255
}

// ============================================================================
// ResizeTransform Integration Test
// ============================================================================

TEST_F(SIMDBilinearTest, ResizeTransformIntegration) {
    // Test via the ResizeTransform class
    const int src_w = 256, src_h = 256;
    const int dst_w = 224, dst_h = 224;

    auto src_data = generate_rgb_image(src_w, src_h);

    turboloader::transforms::ImageData input(
        src_data.data(), src_w, src_h, 3, src_w * 3, false
    );

    turboloader::transforms::ResizeTransform resize(dst_w, dst_h,
        turboloader::transforms::InterpolationMode::BILINEAR);

    auto output = resize.apply(input);

    EXPECT_EQ(output->width, dst_w);
    EXPECT_EQ(output->height, dst_h);
    EXPECT_EQ(output->channels, 3);

    // Verify output is valid (no zero/uninitialized regions)
    int non_zero_count = 0;
    for (size_t i = 0; i < output->size_bytes(); ++i) {
        if (output->data[i] > 0) non_zero_count++;
    }
    EXPECT_GT(non_zero_count, output->size_bytes() / 4);  // At least 25% non-zero
}

// ============================================================================
// Performance Benchmark
// ============================================================================

TEST_F(SIMDBilinearTest, BenchmarkResize) {
    const int src_w = 512, src_h = 512;
    const int dst_w = 224, dst_h = 224;
    const int iterations = 1000;

    auto src = generate_rgb_image(src_w, src_h);
    std::vector<uint8_t> simd_dst(dst_w * dst_h * 3);
    std::vector<uint8_t> ref_dst(dst_w * dst_h * 3);

    // Warm up
    for (int i = 0; i < 50; ++i) {
        resize_bilinear_simd(src.data(), simd_dst.data(), src_w, src_h, dst_w, dst_h, 3);
    }

    // Benchmark SIMD
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        resize_bilinear_simd(src.data(), simd_dst.data(), src_w, src_h, dst_w, dst_h, 3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Benchmark reference scalar
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        reference_bilinear_resize(src.data(), ref_dst.data(), src_w, src_h, dst_w, dst_h, 3);
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    float speedup = (simd_ns > 0) ? static_cast<float>(scalar_ns) / simd_ns : 1.0f;

    std::cout << "\n=== Bilinear Resize Benchmark (512x512 -> 224x224) ===" << std::endl;
    std::cout << "  SIMD:   " << simd_ns / iterations << " ns/image ("
              << (simd_ns / iterations / 1000.0) << " us/image)" << std::endl;
    std::cout << "  Scalar: " << scalar_ns / iterations << " ns/image ("
              << (scalar_ns / iterations / 1000.0) << " us/image)" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;

#if defined(TURBOLOADER_SIMD_NEON)
    std::cout << "  Platform: ARM NEON" << std::endl;
    EXPECT_GE(speedup, 1.0f) << "SIMD should not be slower than scalar";
#elif defined(TURBOLOADER_SIMD_AVX2)
    std::cout << "  Platform: x86 AVX2" << std::endl;
    EXPECT_GE(speedup, 1.0f) << "SIMD should not be slower than scalar";
#elif defined(TURBOLOADER_SIMD_AVX512)
    std::cout << "  Platform: x86 AVX-512" << std::endl;
#else
    std::cout << "  Platform: Scalar fallback" << std::endl;
#endif
}

TEST_F(SIMDBilinearTest, Benchmark_ImageNet_Standard) {
    // Standard ImageNet resize: various sizes -> 224x224
    const int dst_w = 224, dst_h = 224;
    const int iterations = 500;

    std::vector<std::pair<int, int>> src_sizes = {
        {256, 256}, {384, 384}, {512, 512}, {640, 480}, {1024, 768}
    };

    std::cout << "\n=== ImageNet Resize Benchmarks (various -> 224x224) ===" << std::endl;

    for (auto [src_w, src_h] : src_sizes) {
        auto src = generate_rgb_image(src_w, src_h);
        std::vector<uint8_t> dst(dst_w * dst_h * 3);

        // Benchmark SIMD only
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            resize_bilinear_simd(src.data(), dst.data(), src_w, src_h, dst_w, dst_h, 3);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        std::cout << "  " << src_w << "x" << src_h << " -> 224x224: "
                  << ns / iterations / 1000.0 << " us/image" << std::endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "=== TurboLoader SIMD Bilinear Resize Tests ===" << std::endl;

#if defined(TURBOLOADER_SIMD_NEON)
    std::cout << "Running on ARM NEON" << std::endl;
#elif defined(TURBOLOADER_SIMD_AVX2)
    std::cout << "Running on x86 AVX2" << std::endl;
#elif defined(TURBOLOADER_SIMD_AVX512)
    std::cout << "Running on x86 AVX-512" << std::endl;
#else
    std::cout << "Running on Scalar (no SIMD)" << std::endl;
#endif

    return RUN_ALL_TESTS();
}
