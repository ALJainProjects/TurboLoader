#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <memory>

namespace turboloader {
namespace transforms {

/**
 * SIMD-accelerated image transformations
 *
 * Features:
 * - AVX2 (x86_64) and NEON (ARM) SIMD backends
 * - 4-8x faster than scalar implementations
 * - Zero-copy in-place transformations where possible
 * - Batch processing for maximum throughput
 *
 * Supported transforms:
 * - Resize (bilinear interpolation)
 * - Normalize (mean/std subtraction)
 * - Color space conversion (RGB/BGR/YUV)
 * - Crop and padding
 * - Flip (horizontal/vertical)
 */

enum class ResizeMethod {
    NEAREST,     // Nearest neighbor (fastest)
    BILINEAR,    // Bilinear interpolation (good quality/speed)
    BICUBIC      // Bicubic interpolation (best quality, slower)
};

enum class ColorSpace {
    RGB,
    BGR,
    YUV,
    GRAY
};

struct TransformConfig {
    // Resize
    bool enable_resize{false};
    int resize_width{224};
    int resize_height{224};
    ResizeMethod resize_method{ResizeMethod::BILINEAR};

    // Normalize
    bool enable_normalize{false};
    float mean[3]{0.485f, 0.456f, 0.406f};  // ImageNet means
    float std[3]{0.229f, 0.224f, 0.225f};   // ImageNet stds

    // Color conversion
    bool enable_color_convert{false};
    ColorSpace src_color{ColorSpace::RGB};
    ColorSpace dst_color{ColorSpace::RGB};

    // Crop
    bool enable_crop{false};
    int crop_x{0};
    int crop_y{0};
    int crop_width{224};
    int crop_height{224};

    // Flip
    bool enable_flip_horizontal{false};
    bool enable_flip_vertical{false};

    // Output format
    bool output_float{true};  // If false, output uint8
};

/**
 * SIMD-accelerated image resize
 *
 * Uses SIMD instructions for fast bilinear interpolation
 */
class SimdResize {
public:
    /**
     * Resize image using SIMD
     * @param src Source image data (H x W x C)
     * @param src_width Source width
     * @param src_height Source height
     * @param channels Number of channels (1, 3, or 4)
     * @param dst Destination buffer (must be pre-allocated)
     * @param dst_width Destination width
     * @param dst_height Destination height
     * @param method Resize method
     */
    static void resize(
        const uint8_t* src,
        int src_width,
        int src_height,
        int channels,
        uint8_t* dst,
        int dst_width,
        int dst_height,
        ResizeMethod method = ResizeMethod::BILINEAR
    );

    /**
     * Resize to float output (combined resize + normalize)
     * @param scale Scale factor (e.g., 1.0/255.0)
     */
    static void resize_to_float(
        const uint8_t* src,
        int src_width,
        int src_height,
        int channels,
        float* dst,
        int dst_width,
        int dst_height,
        float scale = 1.0f / 255.0f,
        ResizeMethod method = ResizeMethod::BILINEAR
    );

private:
    static void resize_bilinear_simd(
        const uint8_t* src, int src_w, int src_h, int ch,
        uint8_t* dst, int dst_w, int dst_h
    );

    static void resize_nearest_simd(
        const uint8_t* src, int src_w, int src_h, int ch,
        uint8_t* dst, int dst_w, int dst_h
    );
};

/**
 * SIMD-accelerated normalization
 *
 * Performs (x - mean) / std using SIMD
 */
class SimdNormalize {
public:
    /**
     * Normalize image (per-channel mean/std)
     * @param src Source data (uint8 or float)
     * @param dst Destination (float)
     * @param size Number of elements
     * @param mean Per-channel means
     * @param std Per-channel standard deviations
     * @param channels Number of channels
     */
    static void normalize_uint8(
        const uint8_t* src,
        float* dst,
        size_t size,
        const float* mean,
        const float* std,
        int channels
    );

    static void normalize_float(
        const float* src,
        float* dst,
        size_t size,
        const float* mean,
        const float* std,
        int channels
    );

    /**
     * Combined resize + normalize (optimized)
     */
    static void resize_and_normalize(
        const uint8_t* src,
        int src_width,
        int src_height,
        float* dst,
        int dst_width,
        int dst_height,
        int channels,
        const float* mean,
        const float* std
    );
};

/**
 * SIMD-accelerated color space conversion
 */
class SimdColorConvert {
public:
    /**
     * Convert RGB to BGR (or vice versa) - simple channel swap
     */
    static void rgb_to_bgr(const uint8_t* src, uint8_t* dst, size_t pixels);

    /**
     * Convert RGB to YUV
     */
    static void rgb_to_yuv(const uint8_t* src, uint8_t* dst, size_t pixels);

    /**
     * Convert YUV to RGB
     */
    static void yuv_to_rgb(const uint8_t* src, uint8_t* dst, size_t pixels);

    /**
     * Convert RGB to grayscale
     */
    static void rgb_to_gray(const uint8_t* src, uint8_t* dst, size_t pixels);
};

/**
 * SIMD-accelerated crop and flip
 */
class SimdCropFlip {
public:
    /**
     * Crop region from image
     */
    static void crop(
        const uint8_t* src,
        int src_width,
        int src_height,
        int channels,
        uint8_t* dst,
        int crop_x,
        int crop_y,
        int crop_width,
        int crop_height
    );

    /**
     * Flip image horizontally
     */
    static void flip_horizontal(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels
    );

    /**
     * Flip image vertically
     */
    static void flip_vertical(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels
    );
};

/**
 * High-level transform pipeline
 *
 * Applies multiple transforms efficiently using SIMD
 */
class TransformPipeline {
public:
    explicit TransformPipeline(const TransformConfig& config);
    ~TransformPipeline();

    // Non-copyable, movable
    TransformPipeline(const TransformPipeline&) = delete;
    TransformPipeline& operator=(const TransformPipeline&) = delete;
    TransformPipeline(TransformPipeline&&) noexcept;
    TransformPipeline& operator=(TransformPipeline&&) noexcept;

    /**
     * Apply all configured transforms
     * @param src Source image (uint8, H x W x C)
     * @param src_width Source width
     * @param src_height Source height
     * @param channels Number of channels
     * @param dst Destination buffer (auto-allocated if nullptr)
     * @return Pointer to output data
     */
    float* transform(
        const uint8_t* src,
        int src_width,
        int src_height,
        int channels,
        float* dst = nullptr
    );

    /**
     * Get output dimensions after transforms
     */
    void get_output_dims(int& width, int& height) const;

    /**
     * Check if SIMD is available on this platform
     */
    static bool is_simd_available();

    /**
     * Get SIMD backend name (AVX2, NEON, or Scalar)
     */
    static const char* get_simd_backend();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * SIMD utilities and benchmarking
 */
namespace simd_utils {

/**
 * Check CPU features
 */
struct CpuFeatures {
    bool has_avx2{false};
    bool has_avx512{false};
    bool has_neon{false};
    bool has_sse42{false};
};

CpuFeatures detect_cpu_features();

/**
 * Benchmark SIMD vs scalar implementation
 */
struct BenchmarkResult {
    double simd_time_ms;
    double scalar_time_ms;
    double speedup;
};

BenchmarkResult benchmark_resize(int src_w, int src_h, int dst_w, int dst_h);
BenchmarkResult benchmark_normalize(size_t num_pixels);

} // namespace simd_utils

} // namespace transforms
} // namespace turboloader
