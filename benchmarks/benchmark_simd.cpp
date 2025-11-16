/**
 * SIMD Transform Benchmark
 *
 * Compares SIMD-accelerated transforms vs scalar implementations
 * Measures throughput and speedup for different operations
 */

#include "turboloader/transforms/simd_transforms.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace turboloader::transforms;
using namespace std::chrono;

// Benchmark configuration
constexpr int NUM_ITERATIONS = 1000;
constexpr int WARMUP_ITERATIONS = 100;

// Test image sizes
struct TestSize {
    int width;
    int height;
    const char* name;
};

const TestSize TEST_SIZES[] = {
    {64, 64, "64x64 (tiny)"},
    {256, 256, "256x256 (small)"},
    {512, 512, "512x512 (medium)"},
    {1024, 1024, "1024x1024 (large)"},
};

// Scalar implementations for comparison
namespace scalar {

void resize_bilinear(const uint8_t* src, int src_w, int src_h, int ch,
                     uint8_t* dst, int dst_w, int dst_h) {
    float x_ratio = static_cast<float>(src_w) / dst_w;
    float y_ratio = static_cast<float>(src_h) / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, src_w - 1);
            int y1 = std::min(y0 + 1, src_h - 1);

            float dx = src_x - x0;
            float dy = src_y - y0;

            for (int c = 0; c < ch; ++c) {
                float v00 = src[(y0 * src_w + x0) * ch + c];
                float v01 = src[(y0 * src_w + x1) * ch + c];
                float v10 = src[(y1 * src_w + x0) * ch + c];
                float v11 = src[(y1 * src_w + x1) * ch + c];

                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float v = v0 * (1 - dy) + v1 * dy;

                dst[(y * dst_w + x) * ch + c] = static_cast<uint8_t>(v);
            }
        }
    }
}

void normalize(const uint8_t* src, float* dst, size_t size,
              const float* mean, const float* std, int channels) {
    for (size_t i = 0; i < size; ++i) {
        int c = i % channels;
        float val = static_cast<float>(src[i]) / 255.0f;
        dst[i] = (val - mean[c]) / std[c];
    }
}

} // namespace scalar

// Benchmark helper
template<typename Func>
double benchmark(Func&& func, int iterations) {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    return static_cast<double>(duration) / iterations;
}

// Create test image
std::vector<uint8_t> create_test_image(int width, int height, int channels) {
    std::vector<uint8_t> img(width * height * channels);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = static_cast<uint8_t>((i * 73 + 17) % 256);  // Pseudo-random
    }
    return img;
}

void benchmark_resize() {
    std::cout << "\n========================================\n";
    std::cout << "SIMD Resize Benchmark (Bilinear)\n";
    std::cout << "========================================\n\n";

    for (const auto& size : TEST_SIZES) {
        int src_w = size.width;
        int src_h = size.height;
        int dst_w = 224;  // Standard ML input size
        int dst_h = 224;
        int channels = 3;

        auto src = create_test_image(src_w, src_h, channels);
        std::vector<uint8_t> dst_simd(dst_w * dst_h * channels);
        std::vector<uint8_t> dst_scalar(dst_w * dst_h * channels);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            SimdResize::resize(src.data(), src_w, src_h, channels,
                             dst_simd.data(), dst_w, dst_h,
                             ResizeMethod::BILINEAR);
        }

        // Benchmark SIMD
        double simd_time = benchmark([&]() {
            SimdResize::resize(src.data(), src_w, src_h, channels,
                             dst_simd.data(), dst_w, dst_h,
                             ResizeMethod::BILINEAR);
        }, NUM_ITERATIONS);

        // Benchmark scalar
        double scalar_time = benchmark([&]() {
            scalar::resize_bilinear(src.data(), src_w, src_h, channels,
                                   dst_scalar.data(), dst_w, dst_h);
        }, NUM_ITERATIONS);

        double speedup = scalar_time / simd_time;
        double throughput = 1e6 / simd_time;  // images per second

        std::cout << "Size: " << size.name << " -> 224x224\n";
        std::cout << "  SIMD:    " << std::fixed << std::setprecision(2)
                  << simd_time << " μs\n";
        std::cout << "  Scalar:  " << scalar_time << " μs\n";
        std::cout << "  Speedup: " << speedup << "x\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
                  << throughput << " img/s\n\n";
    }
}

void benchmark_normalize() {
    std::cout << "\n========================================\n";
    std::cout << "SIMD Normalize Benchmark\n";
    std::cout << "========================================\n\n";

    for (const auto& size : TEST_SIZES) {
        int width = size.width;
        int height = size.height;
        int channels = 3;
        size_t total_size = width * height * channels;

        auto src = create_test_image(width, height, channels);
        std::vector<float> dst_simd(total_size);
        std::vector<float> dst_scalar(total_size);

        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std[3] = {0.229f, 0.224f, 0.225f};

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            SimdNormalize::normalize_uint8(src.data(), dst_simd.data(),
                                          total_size, mean, std, channels);
        }

        // Benchmark SIMD
        double simd_time = benchmark([&]() {
            SimdNormalize::normalize_uint8(src.data(), dst_simd.data(),
                                          total_size, mean, std, channels);
        }, NUM_ITERATIONS);

        // Benchmark scalar
        double scalar_time = benchmark([&]() {
            scalar::normalize(src.data(), dst_scalar.data(),
                            total_size, mean, std, channels);
        }, NUM_ITERATIONS);

        double speedup = scalar_time / simd_time;
        double throughput = 1e6 / simd_time;  // operations per second

        std::cout << "Size: " << size.name << "\n";
        std::cout << "  SIMD:    " << std::fixed << std::setprecision(2)
                  << simd_time << " μs\n";
        std::cout << "  Scalar:  " << scalar_time << " μs\n";
        std::cout << "  Speedup: " << speedup << "x\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
                  << throughput << " img/s\n\n";
    }
}

void benchmark_combined_resize_normalize() {
    std::cout << "\n========================================\n";
    std::cout << "SIMD Combined Resize + Normalize\n";
    std::cout << "========================================\n\n";

    for (const auto& size : TEST_SIZES) {
        int src_w = size.width;
        int src_h = size.height;
        int dst_w = 224;
        int dst_h = 224;
        int channels = 3;

        auto src = create_test_image(src_w, src_h, channels);
        std::vector<float> dst_simd(dst_w * dst_h * channels);
        std::vector<uint8_t> dst_temp(dst_w * dst_h * channels);
        std::vector<float> dst_scalar(dst_w * dst_h * channels);

        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std[3] = {0.229f, 0.224f, 0.225f};

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            SimdNormalize::resize_and_normalize(src.data(), src_w, src_h,
                                               dst_simd.data(), dst_w, dst_h,
                                               channels, mean, std);
        }

        // Benchmark SIMD (combined)
        double simd_time = benchmark([&]() {
            SimdNormalize::resize_and_normalize(src.data(), src_w, src_h,
                                               dst_simd.data(), dst_w, dst_h,
                                               channels, mean, std);
        }, NUM_ITERATIONS);

        // Benchmark scalar (separate operations)
        double scalar_time = benchmark([&]() {
            scalar::resize_bilinear(src.data(), src_w, src_h, channels,
                                   dst_temp.data(), dst_w, dst_h);
            scalar::normalize(dst_temp.data(), dst_scalar.data(),
                            dst_w * dst_h * channels, mean, std, channels);
        }, NUM_ITERATIONS);

        double speedup = scalar_time / simd_time;
        double throughput = 1e6 / simd_time;

        std::cout << "Size: " << size.name << " -> 224x224 + normalize\n";
        std::cout << "  SIMD (combined): " << std::fixed << std::setprecision(2)
                  << simd_time << " μs\n";
        std::cout << "  Scalar (2-pass): " << scalar_time << " μs\n";
        std::cout << "  Speedup: " << speedup << "x\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
                  << throughput << " img/s\n\n";
    }
}

void benchmark_color_convert() {
    std::cout << "\n========================================\n";
    std::cout << "SIMD Color Conversion Benchmark\n";
    std::cout << "========================================\n\n";

    int width = 1024;
    int height = 1024;
    size_t pixels = width * height;

    auto src = create_test_image(width, height, 3);
    std::vector<uint8_t> dst(pixels * 3);

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        SimdColorConvert::rgb_to_bgr(src.data(), dst.data(), pixels);
    }

    // RGB to BGR
    double rgb_bgr_time = benchmark([&]() {
        SimdColorConvert::rgb_to_bgr(src.data(), dst.data(), pixels);
    }, NUM_ITERATIONS);

    std::cout << "RGB to BGR (1024x1024):\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2)
              << rgb_bgr_time << " μs\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
              << 1e6 / rgb_bgr_time << " img/s\n\n";
}

void print_system_info() {
    std::cout << "========================================\n";
    std::cout << "System Information\n";
    std::cout << "========================================\n\n";

    auto features = simd_utils::detect_cpu_features();
    std::cout << "SIMD Backend: " << TransformPipeline::get_simd_backend() << "\n";
    std::cout << "SIMD Available: "
              << (TransformPipeline::is_simd_available() ? "YES" : "NO") << "\n";
    std::cout << "\nCPU Features:\n";
    std::cout << "  AVX2:    " << (features.has_avx2 ? "YES" : "NO") << "\n";
    std::cout << "  AVX-512: " << (features.has_avx512 ? "YES" : "NO") << "\n";
    std::cout << "  NEON:    " << (features.has_neon ? "YES" : "NO") << "\n";
    std::cout << "  SSE4.2:  " << (features.has_sse42 ? "YES" : "NO") << "\n";
    std::cout << "\n";
}

int main() {
    std::cout << "TurboLoader SIMD Transform Benchmark\n";
    std::cout << "====================================\n\n";

    print_system_info();

    benchmark_resize();
    benchmark_normalize();
    benchmark_combined_resize_normalize();
    benchmark_color_convert();

    std::cout << "\n========================================\n";
    std::cout << "Benchmark Complete!\n";
    std::cout << "========================================\n";

    return 0;
}
