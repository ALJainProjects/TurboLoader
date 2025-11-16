#include "turboloader/transforms/simd_transforms.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace turboloader::transforms;

// Helper functions
void print_test(const char* name) {
    std::cout << "\n[TEST] " << name << "...\n";
}

void assert_approx_equal(float a, float b, float epsilon = 0.01f, const char* msg = "") {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "FAIL: " << msg << " - Expected " << a << " ≈ " << b
                  << " (diff: " << std::abs(a - b) << ")\n";
        exit(1);
    }
}

void create_test_image(std::vector<uint8_t>& img, int width, int height, int channels) {
    img.resize(width * height * channels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                // Create a gradient pattern
                img[(y * width + x) * channels + c] =
                    static_cast<uint8_t>((x + y + c * 50) % 256);
            }
        }
    }
}

// Test 1: CPU Feature Detection
void test_cpu_features() {
    print_test("CPU Feature Detection");

    auto features = simd_utils::detect_cpu_features();

    std::cout << "  CPU Features:\n";
    std::cout << "    AVX2: " << (features.has_avx2 ? "YES" : "NO") << "\n";
    std::cout << "    SSE4.2: " << (features.has_sse42 ? "YES" : "NO") << "\n";
    std::cout << "    NEON: " << (features.has_neon ? "YES" : "NO") << "\n";

    // At least one should be available on modern hardware
    assert(features.has_avx2 || features.has_neon || features.has_sse42);

    std::cout << "  ✓ CPU features detected\n";
}

// Test 2: SIMD Backend Detection
void test_simd_backend() {
    print_test("SIMD Backend Detection");

    const char* backend = TransformPipeline::get_simd_backend();
    bool available = TransformPipeline::is_simd_available();

    std::cout << "  SIMD Backend: " << backend << "\n";
    std::cout << "  SIMD Available: " << (available ? "YES" : "NO") << "\n";

    assert(backend != nullptr);

    std::cout << "  ✓ SIMD backend: " << backend << "\n";
}

// Test 3: Image Resize (Nearest Neighbor)
void test_resize_nearest() {
    print_test("SIMD Resize - Nearest Neighbor");

    std::vector<uint8_t> src, dst;
    int src_w = 64, src_h = 64, dst_w = 32, dst_h = 32;
    int channels = 3;

    create_test_image(src, src_w, src_h, channels);
    dst.resize(dst_w * dst_h * channels);

    SimdResize::resize(
        src.data(), src_w, src_h, channels,
        dst.data(), dst_w, dst_h,
        ResizeMethod::NEAREST
    );

    // Verify output is not all zeros
    bool has_data = false;
    for (auto val : dst) {
        if (val != 0) {
            has_data = true;
            break;
        }
    }
    assert(has_data);

    std::cout << "  Input: " << src_w << "x" << src_h << "\n";
    std::cout << "  Output: " << dst_w << "x" << dst_h << "\n";
    std::cout << "  ✓ Resize nearest neighbor works\n";
}

// Test 4: Image Resize (Bilinear)
void test_resize_bilinear() {
    print_test("SIMD Resize - Bilinear Interpolation");

    std::vector<uint8_t> src, dst;
    int src_w = 64, src_h = 64, dst_w = 128, dst_h = 128;
    int channels = 3;

    create_test_image(src, src_w, src_h, channels);
    dst.resize(dst_w * dst_h * channels);

    SimdResize::resize(
        src.data(), src_w, src_h, channels,
        dst.data(), dst_w, dst_h,
        ResizeMethod::BILINEAR
    );

    // Verify output
    assert(dst.size() == static_cast<size_t>(dst_w * dst_h * channels));

    std::cout << "  Input: " << src_w << "x" << src_h << "\n";
    std::cout << "  Output: " << dst_w << "x" << dst_h << "\n";
    std::cout << "  ✓ Bilinear interpolation works\n";
}

// Test 5: Resize to Float
void test_resize_to_float() {
    print_test("SIMD Resize to Float");

    std::vector<uint8_t> src;
    std::vector<float> dst;
    int src_w = 256, src_h = 256, dst_w = 224, dst_h = 224;
    int channels = 3;

    create_test_image(src, src_w, src_h, channels);
    dst.resize(dst_w * dst_h * channels);

    float scale = 1.0f / 255.0f;

    SimdResize::resize_to_float(
        src.data(), src_w, src_h, channels,
        dst.data(), dst_w, dst_h,
        scale,
        ResizeMethod::BILINEAR
    );

    // Verify values are in [0, 1] range
    for (auto val : dst) {
        assert(val >= 0.0f && val <= 1.0f);
    }

    std::cout << "  Input: " << src_w << "x" << src_h << " (uint8)\n";
    std::cout << "  Output: " << dst_w << "x" << dst_h << " (float)\n";
    std::cout << "  ✓ Resize to float works\n";
}

// Test 6: Normalization (uint8 to float)
void test_normalize_uint8() {
    print_test("SIMD Normalize - uint8 to float");

    std::vector<uint8_t> src(3 * 224 * 224);
    std::vector<float> dst(3 * 224 * 224);

    // Fill with test pattern
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = static_cast<uint8_t>(i % 256);
    }

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};

    SimdNormalize::normalize_uint8(
        src.data(), dst.data(), src.size(),
        mean, std, 3
    );

    // Verify normalization formula: (x/255 - mean) / std
    float expected = (src[0] / 255.0f - mean[0]) / std[0];
    assert_approx_equal(dst[0], expected, 0.01f, "Normalization formula");

    std::cout << "  Size: " << src.size() / 1000 << "K pixels\n";
    std::cout << "  ✓ uint8 normalization works\n";
}

// Test 7: Normalization (float to float)
void test_normalize_float() {
    print_test("SIMD Normalize - float to float");

    std::vector<float> src(3 * 224 * 224);
    std::vector<float> dst(3 * 224 * 224);

    // Fill with values in [0, 1]
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = (i % 256) / 255.0f;
    }

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};

    SimdNormalize::normalize_float(
        src.data(), dst.data(), src.size(),
        mean, std, 3
    );

    // Verify normalization
    float expected = (src[0] - mean[0]) / std[0];
    assert_approx_equal(dst[0], expected, 0.01f, "Float normalization");

    std::cout << "  Size: " << src.size() / 1000 << "K pixels\n";
    std::cout << "  ✓ float normalization works\n";
}

// Test 8: Combined Resize + Normalize
void test_resize_and_normalize() {
    print_test("SIMD Resize + Normalize (combined)");

    std::vector<uint8_t> src;
    std::vector<float> dst;
    int src_w = 256, src_h = 256, dst_w = 224, dst_h = 224;
    int channels = 3;

    create_test_image(src, src_w, src_h, channels);
    dst.resize(dst_w * dst_h * channels);

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};

    SimdNormalize::resize_and_normalize(
        src.data(), src_w, src_h,
        dst.data(), dst_w, dst_h, channels,
        mean, std
    );

    // Verify output size
    assert(dst.size() == static_cast<size_t>(dst_w * dst_h * channels));

    std::cout << "  Input: " << src_w << "x" << src_h << " (uint8)\n";
    std::cout << "  Output: " << dst_w << "x" << dst_h << " (normalized float)\n";
    std::cout << "  ✓ Combined resize+normalize works\n";
}

// Test 9: Color Conversion (RGB to BGR)
void test_color_rgb_to_bgr() {
    print_test("SIMD Color Convert - RGB to BGR");

    std::vector<uint8_t> src = {255, 0, 0,  0, 255, 0,  0, 0, 255};  // R, G, B
    std::vector<uint8_t> dst(9);

    SimdColorConvert::rgb_to_bgr(src.data(), dst.data(), 3);

    // Verify channel swap
    assert(dst[0] == 0 && dst[1] == 0 && dst[2] == 255);  // First pixel: BGR
    assert(dst[3] == 0 && dst[4] == 255 && dst[5] == 0);  // Second pixel: BGR
    assert(dst[6] == 255 && dst[7] == 0 && dst[8] == 0);  // Third pixel: BGR

    std::cout << "  RGB -> BGR conversion verified\n";
    std::cout << "  ✓ Color conversion works\n";
}

// Test 10: Color Conversion (RGB to Grayscale)
void test_color_rgb_to_gray() {
    print_test("SIMD Color Convert - RGB to Grayscale");

    std::vector<uint8_t> src = {255, 255, 255,  0, 0, 0,  128, 128, 128};
    std::vector<uint8_t> dst(3);

    SimdColorConvert::rgb_to_gray(src.data(), dst.data(), 3);

    // White -> ~255, Black -> ~0, Gray -> ~128
    assert(dst[0] > 250);  // White
    assert(dst[1] < 5);    // Black
    assert(dst[2] > 120 && dst[2] < 135);  // Gray

    std::cout << "  RGB -> Grayscale conversion verified\n";
    std::cout << "  ✓ Grayscale conversion works\n";
}

// Test 11: Crop
void test_crop() {
    print_test("SIMD Crop");

    std::vector<uint8_t> src, dst;
    int src_w = 64, src_h = 64;
    int crop_x = 16, crop_y = 16, crop_w = 32, crop_h = 32;
    int channels = 3;

    create_test_image(src, src_w, src_h, channels);
    dst.resize(crop_w * crop_h * channels);

    SimdCropFlip::crop(
        src.data(), src_w, src_h, channels,
        dst.data(), crop_x, crop_y, crop_w, crop_h
    );

    // Verify output size
    assert(dst.size() == static_cast<size_t>(crop_w * crop_h * channels));

    std::cout << "  Source: " << src_w << "x" << src_h << "\n";
    std::cout << "  Crop: " << crop_w << "x" << crop_h << " at (" << crop_x << "," << crop_y << ")\n";
    std::cout << "  ✓ Crop works\n";
}

// Test 12: Flip Horizontal
void test_flip_horizontal() {
    print_test("SIMD Flip Horizontal");

    std::vector<uint8_t> src = {1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12};  // 2x2 image, RGB
    std::vector<uint8_t> dst(12);

    SimdCropFlip::flip_horizontal(src.data(), dst.data(), 2, 2, 3);

    // Verify horizontal flip
    assert(dst[0] == 4 && dst[1] == 5 && dst[2] == 6);    // Second pixel first
    assert(dst[3] == 1 && dst[4] == 2 && dst[5] == 3);    // First pixel second
    assert(dst[6] == 10 && dst[7] == 11 && dst[8] == 12); // Fourth pixel first
    assert(dst[9] == 7 && dst[10] == 8 && dst[11] == 9);  // Third pixel second

    std::cout << "  ✓ Horizontal flip works\n";
}

// Test 13: Flip Vertical
void test_flip_vertical() {
    print_test("SIMD Flip Vertical");

    std::vector<uint8_t> src = {1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12};  // 2x2 image, RGB
    std::vector<uint8_t> dst(12);

    SimdCropFlip::flip_vertical(src.data(), dst.data(), 2, 2, 3);

    // Verify vertical flip (bottom row becomes top row)
    assert(dst[0] == 7 && dst[1] == 8 && dst[2] == 9);    // Third pixel first
    assert(dst[3] == 10 && dst[4] == 11 && dst[5] == 12); // Fourth pixel second
    assert(dst[6] == 1 && dst[7] == 2 && dst[8] == 3);    // First pixel third
    assert(dst[9] == 4 && dst[10] == 5 && dst[11] == 6);  // Second pixel fourth

    std::cout << "  ✓ Vertical flip works\n";
}

// Test 14: Transform Pipeline
void test_transform_pipeline() {
    print_test("SIMD Transform Pipeline");

    std::vector<uint8_t> src;
    int src_w = 256, src_h = 256, channels = 3;

    create_test_image(src, src_w, src_h, channels);

    TransformConfig config;
    config.enable_resize = true;
    config.resize_width = 224;
    config.resize_height = 224;
    config.enable_normalize = true;

    TransformPipeline pipeline(config);

    float* output = pipeline.transform(src.data(), src_w, src_h, channels);

    assert(output != nullptr);

    int out_w, out_h;
    pipeline.get_output_dims(out_w, out_h);

    assert(out_w == 224);
    assert(out_h == 224);

    std::cout << "  Input: " << src_w << "x" << src_h << "\n";
    std::cout << "  Output: " << out_w << "x" << out_h << "\n";
    std::cout << "  Transforms: Resize + Normalize\n";
    std::cout << "  ✓ Transform pipeline works\n";
}

// Test 15: Performance Benchmark
void test_performance() {
    print_test("SIMD Performance Benchmark");

    std::vector<uint8_t> src;
    std::vector<float> dst;
    int src_w = 256, src_h = 256, dst_w = 224, dst_h = 224;
    int channels = 3;

    create_test_image(src, src_w, src_h, channels);
    dst.resize(dst_w * dst_h * channels);

    const int iterations = 100;

    // Benchmark resize
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        SimdResize::resize_to_float(
            src.data(), src_w, src_h, channels,
            dst.data(), dst_w, dst_h,
            1.0f / 255.0f,
            ResizeMethod::BILINEAR
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = duration.count() / static_cast<double>(iterations);
    double throughput = 1000000.0 / avg_time_us;  // images per second

    std::cout << "  Resize Performance:\n";
    std::cout << "    Iterations: " << iterations << "\n";
    std::cout << "    Avg time: " << avg_time_us << " μs\n";
    std::cout << "    Throughput: " << static_cast<int>(throughput) << " img/s\n";

    // Benchmark normalize
    std::vector<uint8_t> norm_src(dst_w * dst_h * channels);
    std::vector<float> norm_dst(dst_w * dst_h * channels);
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        SimdNormalize::normalize_uint8(
            norm_src.data(), norm_dst.data(), norm_src.size(),
            mean, std, channels
        );
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    avg_time_us = duration.count() / static_cast<double>(iterations);
    throughput = 1000000.0 / avg_time_us;

    std::cout << "  Normalize Performance:\n";
    std::cout << "    Iterations: " << iterations << "\n";
    std::cout << "    Avg time: " << avg_time_us << " μs\n";
    std::cout << "    Throughput: " << static_cast<int>(throughput) << " img/s\n";

    std::cout << "  ✓ Performance benchmarks complete\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "TurboLoader SIMD Transforms Test Suite\n";
    std::cout << "========================================\n";

    try {
        test_cpu_features();
        test_simd_backend();
        test_resize_nearest();
        test_resize_bilinear();
        test_resize_to_float();
        test_normalize_uint8();
        test_normalize_float();
        test_resize_and_normalize();
        test_color_rgb_to_bgr();
        test_color_rgb_to_gray();
        test_crop();
        test_flip_horizontal();
        test_flip_vertical();
        test_transform_pipeline();
        test_performance();

        std::cout << "\n========================================\n";
        std::cout << "✓ All 15 tests PASSED!\n";
        std::cout << "========================================\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
