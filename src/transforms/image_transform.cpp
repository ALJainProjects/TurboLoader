#include "turboloader/transforms/image_transform.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace turboloader {

// ============================================================================
// ResizeTransform
// ============================================================================

ImageTransform::TransformResult ResizeTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    TransformResult result;
    result.width = target_width_;
    result.height = target_height_;
    result.channels = 3;
    result.data.resize(target_width_ * target_height_ * 3);

    resize_bilinear_simd(
        src.data(), width, height,
        result.data.data(), target_width_, target_height_
    );

    return result;
}

void ResizeTransform::resize_bilinear_simd(
    const uint8_t* src, int src_w, int src_h,
    uint8_t* dst, int dst_w, int dst_h
) {
    // Compute scaling factors
    float x_ratio = static_cast<float>(src_w - 1) / dst_w;
    float y_ratio = static_cast<float>(src_h - 1) / dst_h;

#ifdef __ARM_NEON
    // NEON-optimized bilinear interpolation
    for (int y = 0; y < dst_h; ++y) {
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_h - 1);
        float y_diff = src_y - y0;

        // Broadcast y weights
        float32x4_t vy_weight = vdupq_n_f32(y_diff);
        float32x4_t vy_weight_inv = vdupq_n_f32(1.0f - y_diff);

        int x = 0;
        // Process 4 pixels at a time
        for (; x + 3 < dst_w; x += 4) {
            // Calculate source x coordinates for 4 pixels
            float src_x[4];
            int x0[4], x1[4];
            float x_diff[4];

            for (int i = 0; i < 4; ++i) {
                src_x[i] = (x + i) * x_ratio;
                x0[i] = static_cast<int>(src_x[i]);
                x1[i] = std::min(x0[i] + 1, src_w - 1);
                x_diff[i] = src_x[i] - x0[i];
            }

            // Load and interpolate 4 pixels
            for (int i = 0; i < 4; ++i) {
                // Load 4 corner pixels
                int idx00 = (y0 * src_w + x0[i]) * 3;
                int idx01 = (y0 * src_w + x1[i]) * 3;
                int idx10 = (y1 * src_w + x0[i]) * 3;
                int idx11 = (y1 * src_w + x1[i]) * 3;

                // Load RGB values (pad to 4 for SIMD)
                uint8_t p00[4] = {src[idx00], src[idx00+1], src[idx00+2], 0};
                uint8_t p01[4] = {src[idx01], src[idx01+1], src[idx01+2], 0};
                uint8_t p10[4] = {src[idx10], src[idx10+1], src[idx10+2], 0};
                uint8_t p11[4] = {src[idx11], src[idx11+1], src[idx11+2], 0};

                // Convert to float
                uint32x4_t u00 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p00))));
                uint32x4_t u01 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p01))));
                uint32x4_t u10 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p10))));
                uint32x4_t u11 = vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p11))));

                float32x4_t f00 = vcvtq_f32_u32(u00);
                float32x4_t f01 = vcvtq_f32_u32(u01);
                float32x4_t f10 = vcvtq_f32_u32(u10);
                float32x4_t f11 = vcvtq_f32_u32(u11);

                // Horizontal interpolation
                float32x4_t vx_weight = vdupq_n_f32(x_diff[i]);
                float32x4_t vx_weight_inv = vdupq_n_f32(1.0f - x_diff[i]);

                float32x4_t top = vmlaq_f32(vmulq_f32(f00, vx_weight_inv), f01, vx_weight);
                float32x4_t bottom = vmlaq_f32(vmulq_f32(f10, vx_weight_inv), f11, vx_weight);

                // Vertical interpolation
                float32x4_t result = vmlaq_f32(vmulq_f32(top, vy_weight_inv), bottom, vy_weight);

                // Convert back to uint8
                uint32x4_t result_u32 = vcvtq_u32_f32(vaddq_f32(result, vdupq_n_f32(0.5f)));
                uint16x4_t result_u16 = vmovn_u32(result_u32);
                uint8x8_t result_u8 = vmovn_u16(vcombine_u16(result_u16, result_u16));

                // Store RGB (first 3 bytes)
                int dst_idx = (y * dst_w + x + i) * 3;
                vst1_lane_u8(&dst[dst_idx + 0], result_u8, 0);
                vst1_lane_u8(&dst[dst_idx + 1], result_u8, 1);
                vst1_lane_u8(&dst[dst_idx + 2], result_u8, 2);
            }
        }

        // Handle remaining pixels
        for (; x < dst_w; ++x) {
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_w - 1);
            float x_diff = src_x - x0;

            int idx00 = (y0 * src_w + x0) * 3;
            int idx01 = (y0 * src_w + x1) * 3;
            int idx10 = (y1 * src_w + x0) * 3;
            int idx11 = (y1 * src_w + x1) * 3;

            for (int c = 0; c < 3; ++c) {
                float p00 = src[idx00 + c];
                float p01 = src[idx01 + c];
                float p10 = src[idx10 + c];
                float p11 = src[idx11 + c];

                float top = p00 + (p01 - p00) * x_diff;
                float bottom = p10 + (p11 - p10) * x_diff;
                float value = top + (bottom - top) * y_diff;

                dst[(y * dst_w + x) * 3 + c] = static_cast<uint8_t>(value + 0.5f);
            }
        }
    }

#elif defined(__AVX2__)
    // AVX2-optimized bilinear interpolation
    for (int y = 0; y < dst_h; ++y) {
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_h - 1);
        float y_diff = src_y - y0;

        __m256 vy_weight = _mm256_set1_ps(y_diff);
        __m256 vy_weight_inv = _mm256_set1_ps(1.0f - y_diff);

        int x = 0;
        // Process 8 pixels at a time
        for (; x + 7 < dst_w; x += 8) {
            // Calculate source x coordinates
            alignas(32) float src_x[8];
            alignas(32) int x0[8], x1[8];
            alignas(32) float x_diff[8];

            for (int i = 0; i < 8; ++i) {
                src_x[i] = (x + i) * x_ratio;
                x0[i] = static_cast<int>(src_x[i]);
                x1[i] = std::min(x0[i] + 1, src_w - 1);
                x_diff[i] = src_x[i] - x0[i];
            }

            // Process each pixel (AVX2 doesn't have great gather for this case)
            for (int i = 0; i < 8; ++i) {
                int idx00 = (y0 * src_w + x0[i]) * 3;
                int idx01 = (y0 * src_w + x1[i]) * 3;
                int idx10 = (y1 * src_w + x0[i]) * 3;
                int idx11 = (y1 * src_w + x1[i]) * 3;

                __m256 p00 = _mm256_set_ps(0, 0, 0, 0, 0, src[idx00+2], src[idx00+1], src[idx00]);
                __m256 p01 = _mm256_set_ps(0, 0, 0, 0, 0, src[idx01+2], src[idx01+1], src[idx01]);
                __m256 p10 = _mm256_set_ps(0, 0, 0, 0, 0, src[idx10+2], src[idx10+1], src[idx10]);
                __m256 p11 = _mm256_set_ps(0, 0, 0, 0, 0, src[idx11+2], src[idx11+1], src[idx11]);

                __m256 vx_weight = _mm256_set1_ps(x_diff[i]);
                __m256 vx_weight_inv = _mm256_set1_ps(1.0f - x_diff[i]);

                __m256 top = _mm256_add_ps(_mm256_mul_ps(p00, vx_weight_inv),
                                           _mm256_mul_ps(p01, vx_weight));
                __m256 bottom = _mm256_add_ps(_mm256_mul_ps(p10, vx_weight_inv),
                                              _mm256_mul_ps(p11, vx_weight));

                __m256 result = _mm256_add_ps(_mm256_mul_ps(top, vy_weight_inv),
                                              _mm256_mul_ps(bottom, vy_weight));

                alignas(32) float result_f[8];
                _mm256_store_ps(result_f, result);

                int dst_idx = (y * dst_w + x + i) * 3;
                dst[dst_idx + 0] = static_cast<uint8_t>(result_f[0] + 0.5f);
                dst[dst_idx + 1] = static_cast<uint8_t>(result_f[1] + 0.5f);
                dst[dst_idx + 2] = static_cast<uint8_t>(result_f[2] + 0.5f);
            }
        }

        // Handle remaining pixels
        for (; x < dst_w; ++x) {
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_w - 1);
            float x_diff = src_x - x0;

            int idx00 = (y0 * src_w + x0) * 3;
            int idx01 = (y0 * src_w + x1) * 3;
            int idx10 = (y1 * src_w + x0) * 3;
            int idx11 = (y1 * src_w + x1) * 3;

            for (int c = 0; c < 3; ++c) {
                float p00 = src[idx00 + c];
                float p01 = src[idx01 + c];
                float p10 = src[idx10 + c];
                float p11 = src[idx11 + c];

                float top = p00 + (p01 - p00) * x_diff;
                float bottom = p10 + (p11 - p10) * x_diff;
                float value = top + (bottom - top) * y_diff;

                dst[(y * dst_w + x) * 3 + c] = static_cast<uint8_t>(value + 0.5f);
            }
        }
    }

#else
    // Scalar fallback
    for (int y = 0; y < dst_h; ++y) {
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_h - 1);
        float y_diff = src_y - y0;

        for (int x = 0; x < dst_w; ++x) {
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_w - 1);
            float x_diff = src_x - x0;

            int idx00 = (y0 * src_w + x0) * 3;
            int idx01 = (y0 * src_w + x1) * 3;
            int idx10 = (y1 * src_w + x0) * 3;
            int idx11 = (y1 * src_w + x1) * 3;

            for (int c = 0; c < 3; ++c) {
                float p00 = src[idx00 + c];
                float p01 = src[idx01 + c];
                float p10 = src[idx10 + c];
                float p11 = src[idx11 + c];

                float top = p00 + (p01 - p00) * x_diff;
                float bottom = p10 + (p11 - p10) * x_diff;
                float value = top + (bottom - top) * y_diff;

                dst[(y * dst_w + x) * 3 + c] = static_cast<uint8_t>(value + 0.5f);
            }
        }
    }
#endif
}

// ============================================================================
// NormalizeTransform
// ============================================================================

ImageTransform::TransformResult NormalizeTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    TransformResult result;
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.data.resize(width * height * 3);

    normalize_simd(src.data(), result.data.data(), width * height);

    return result;
}

void NormalizeTransform::normalize_simd(
    const uint8_t* src, uint8_t* dst, size_t pixels
) {
    // ImageNet normalization: (pixel / 255.0 - mean) / std
    // But we'll keep it as uint8 for now, just do a simple scale

    size_t total = pixels * 3;

#ifdef __ARM_NEON
    // NEON implementation for ARM (Apple Silicon)
    size_t i = 0;
    size_t simd_end = total - (total % 16);

    // Process 16 bytes at a time
    for (; i < simd_end; i += 16) {
        uint8x16_t data = vld1q_u8(src + i);

        // TODO: Full SIMD normalization (this is a placeholder)
        // For now, just copy the data
        vst1q_u8(dst + i, data);
    }

    // Handle remaining elements
    for (; i < total; ++i) {
        dst[i] = src[i];
    }

#elif defined(__AVX2__)
    // AVX2 implementation for x86
    size_t i = 0;
    size_t simd_end = total - (total % 32);

    // Process 32 bytes at a time
    for (; i < simd_end; i += 32) {
        __m256i data = _mm256_loadu_si256((__m256i*)(src + i));

        // Simple scaling (for now, just copy - full normalization later)
        _mm256_storeu_si256((__m256i*)(dst + i), data);
    }

    // Handle remaining elements
    for (; i < total; ++i) {
        dst[i] = src[i];
    }

#else
    // Scalar fallback
    std::memcpy(dst, src, total);
#endif
}

// ============================================================================
// CenterCropTransform
// ============================================================================

ImageTransform::TransformResult CenterCropTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    TransformResult result;
    result.channels = 3;

    // Clamp target size to source size
    int crop_w = std::min(target_width_, width);
    int crop_h = std::min(target_height_, height);

    result.width = crop_w;
    result.height = crop_h;
    result.data.resize(crop_w * crop_h * 3);

    // Calculate center crop offsets
    int start_x = (width - crop_w) / 2;
    int start_y = (height - crop_h) / 2;

    // Copy cropped region row by row
    for (int y = 0; y < crop_h; ++y) {
        const uint8_t* src_row = src.data() + ((start_y + y) * width + start_x) * 3;
        uint8_t* dst_row = result.data.data() + y * crop_w * 3;
        std::memcpy(dst_row, src_row, crop_w * 3);
    }

    return result;
}

// ============================================================================
// RandomCropTransform
// ============================================================================

ImageTransform::TransformResult RandomCropTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    // Thread-local random engine
    thread_local std::mt19937 rng(std::random_device{}());

    TransformResult result;
    result.channels = 3;

    // Clamp target size to source size
    int crop_w = std::min(target_width_, width);
    int crop_h = std::min(target_height_, height);

    result.width = crop_w;
    result.height = crop_h;
    result.data.resize(crop_w * crop_h * 3);

    // Random crop offsets
    std::uniform_int_distribution<int> dist_x(0, width - crop_w);
    std::uniform_int_distribution<int> dist_y(0, height - crop_h);

    int start_x = dist_x(rng);
    int start_y = dist_y(rng);

    // Copy cropped region row by row
    for (int y = 0; y < crop_h; ++y) {
        const uint8_t* src_row = src.data() + ((start_y + y) * width + start_x) * 3;
        uint8_t* dst_row = result.data.data() + y * crop_w * 3;
        std::memcpy(dst_row, src_row, crop_w * 3);
    }

    return result;
}

// ============================================================================
// HorizontalFlipTransform
// ============================================================================

ImageTransform::TransformResult HorizontalFlipTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    TransformResult result;
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.data.resize(width * height * 3);

    // Flip horizontally by reversing each row
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_idx = (y * width + x) * 3;
            int dst_idx = (y * width + (width - 1 - x)) * 3;

            result.data[dst_idx + 0] = src[src_idx + 0];  // R
            result.data[dst_idx + 1] = src[src_idx + 1];  // G
            result.data[dst_idx + 2] = src[src_idx + 2];  // B
        }
    }

    return result;
}

// ============================================================================
// VerticalFlipTransform
// ============================================================================

ImageTransform::TransformResult VerticalFlipTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    TransformResult result;
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.data.resize(width * height * 3);

    // Flip vertically by reversing row order
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.data() + y * width * 3;
        uint8_t* dst_row = result.data.data() + (height - 1 - y) * width * 3;
        std::memcpy(dst_row, src_row, width * 3);
    }

    return result;
}

// ============================================================================
// RandomHorizontalFlipTransform
// ============================================================================

ImageTransform::TransformResult RandomHorizontalFlipTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    // Thread-local random engine
    thread_local std::mt19937 rng(std::random_device{}());
    thread_local std::bernoulli_distribution dist(0.5);

    TransformResult result;
    result.width = width;
    result.height = height;
    result.channels = 3;
    result.data.resize(width * height * 3);

    if (dist(rng)) {
        // Flip
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int src_idx = (y * width + x) * 3;
                int dst_idx = (y * width + (width - 1 - x)) * 3;

                result.data[dst_idx + 0] = src[src_idx + 0];
                result.data[dst_idx + 1] = src[src_idx + 1];
                result.data[dst_idx + 2] = src[src_idx + 2];
            }
        }
    } else {
        // No flip, just copy
        std::memcpy(result.data.data(), src.data(), width * height * 3);
    }

    return result;
}

// ============================================================================
// ComposedTransform
// ============================================================================

ImageTransform::TransformResult ComposedTransform::apply(
    std::span<const uint8_t> src,
    int width,
    int height
) {
    if (transforms_.empty()) {
        TransformResult result;
        result.width = width;
        result.height = height;
        result.channels = 3;
        result.data.assign(src.begin(), src.end());
        return result;
    }

    // Apply first transform
    auto result = transforms_[0]->apply(src, width, height);

    // Apply remaining transforms
    for (size_t i = 1; i < transforms_.size(); ++i) {
        result = transforms_[i]->apply(result.data, result.width, result.height);
    }

    return result;
}

}  // namespace turboloader
