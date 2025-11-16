#include "turboloader/transforms/simd_transforms.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

// Include SIMD headers based on platform
#if defined(__x86_64__) || defined(_M_X64)
    #ifdef __AVX512F__
        #include <immintrin.h>
        #define HAVE_AVX512 1
        #define HAVE_AVX2 1  // AVX-512 includes AVX2
    #elif defined(__AVX2__)
        #include <immintrin.h>
        #define HAVE_AVX2 1
    #endif
    #ifdef __SSE4_2__
        #include <nmmintrin.h>
        #define HAVE_SSE42 1
    #endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define HAVE_NEON 1
#endif

// Cache optimization constants
#define CACHE_LINE_SIZE 64
#define L1_CACHE_SIZE 32768    // 32KB typical L1
#define L2_CACHE_SIZE 262144   // 256KB typical L2
#define TILE_SIZE 64           // Process in 64x64 tiles for cache locality

namespace turboloader {
namespace transforms {

// ============================================================================
// CPU Feature Detection
// ============================================================================

namespace simd_utils {

CpuFeatures detect_cpu_features() {
    CpuFeatures features;

#if defined(HAVE_AVX512)
    features.has_avx512 = true;
    features.has_avx2 = true;  // AVX-512 implies AVX2
#elif defined(HAVE_AVX2)
    features.has_avx2 = true;
#endif

#if defined(HAVE_SSE42)
    features.has_sse42 = true;
#endif

#if defined(HAVE_NEON)
    features.has_neon = true;
#endif

    return features;
}

} // namespace simd_utils

// ============================================================================
// SIMD Resize - Optimized Separable Implementation
// ============================================================================

// Helper: Horizontal resize with SIMD (separable pass 1 with cache blocking)
static void resize_horizontal_simd(
    const uint8_t* src, int src_w, int src_h, int ch,
    float* dst, int dst_w)
{
    const float x_ratio = static_cast<float>(src_w - 1) / std::max(dst_w - 1, 1);

    // Cache blocking: Process in vertical tiles to improve cache locality
    constexpr int TILE_HEIGHT = 64;  // Process 64 rows at a time (fits in L2)

    for (int tile_y = 0; tile_y < src_h; tile_y += TILE_HEIGHT) {
        int tile_end_y = std::min(tile_y + TILE_HEIGHT, src_h);

        for (int y = tile_y; y < tile_end_y; y++) {
            const uint8_t* src_row = src + y * src_w * ch;
            float* dst_row = dst + y * dst_w * ch;

            // Prefetch next row for better memory access pattern
            if (y + 1 < tile_end_y) {
                const uint8_t* next_row = src + (y + 1) * src_w * ch;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(next_row, 0, 3);  // Temporal locality
#endif
            }

            for (int x = 0; x < dst_w; x++) {
            float src_x = x * x_ratio;
            int x_low = static_cast<int>(src_x);
            int x_high = std::min(x_low + 1, src_w - 1);
            float x_weight = src_x - x_low;
            float x_weight_inv = 1.0f - x_weight;

#if defined(HAVE_AVX512)
            // AVX-512: Process 16 floats at once
            if (ch >= 16) {
                for (int c = 0; c + 15 < ch; c += 16) {
                    __m512 low_vals = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                        _mm_loadu_si128((__m128i*)(src_row + x_low * ch + c))));
                    __m512 high_vals = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                        _mm_loadu_si128((__m128i*)(src_row + x_high * ch + c))));

                    __m512 weight_inv_vec = _mm512_set1_ps(x_weight_inv);
                    __m512 weight_vec = _mm512_set1_ps(x_weight);

                    __m512 result = _mm512_fmadd_ps(low_vals, weight_inv_vec,
                                                     _mm512_mul_ps(high_vals, weight_vec));
                    _mm512_storeu_ps(dst_row + x * ch + c, result);
                }
            }
#elif defined(HAVE_AVX2)
            // AVX2: Process 8 floats at once
            if (ch >= 8) {
                for (int c = 0; c + 7 < ch; c += 8) {
                    // Convert uint8 to float (manual since no direct conversion)
                    __m256 low_vals, high_vals;
                    {
                        __m128i low_u8 = _mm_loadl_epi64((__m128i*)(src_row + x_low * ch + c));
                        __m128i high_u8 = _mm_loadl_epi64((__m128i*)(src_row + x_high * ch + c));
                        __m256i low_i32 = _mm256_cvtepu8_epi32(low_u8);
                        __m256i high_i32 = _mm256_cvtepu8_epi32(high_u8);
                        low_vals = _mm256_cvtepi32_ps(low_i32);
                        high_vals = _mm256_cvtepi32_ps(high_i32);
                    }

                    __m256 weight_inv_vec = _mm256_set1_ps(x_weight_inv);
                    __m256 weight_vec = _mm256_set1_ps(x_weight);

                    __m256 result = _mm256_add_ps(_mm256_mul_ps(low_vals, weight_inv_vec),
                                                  _mm256_mul_ps(high_vals, weight_vec));
                    _mm256_storeu_ps(dst_row + x * ch + c, result);
                }
            }
#elif defined(HAVE_NEON)
            // NEON: Process 4 floats at once
            for (int c = 0; c + 3 < ch; c += 4) {
                uint8x8_t low_u8 = vld1_u8(src_row + x_low * ch + c);
                uint8x8_t high_u8 = vld1_u8(src_row + x_high * ch + c);

                uint16x4_t low_u16 = vget_low_u16(vmovl_u8(low_u8));
                uint16x4_t high_u16 = vget_low_u16(vmovl_u8(high_u8));

                float32x4_t low_f32 = vcvtq_f32_u32(vmovl_u16(low_u16));
                float32x4_t high_f32 = vcvtq_f32_u32(vmovl_u16(high_u16));

                float32x4_t weight_inv_vec = vdupq_n_f32(x_weight_inv);
                float32x4_t weight_vec = vdupq_n_f32(x_weight);

                float32x4_t result = vmlaq_f32(vmulq_f32(low_f32, weight_inv_vec),
                                               high_f32, weight_vec);
                vst1q_f32(dst_row + x * ch + c, result);
            }
#endif
            // Scalar fallback for remaining channels
            for (int c = (ch >= 4 ? (ch & ~3) : 0); c < ch; c++) {
                float low_val = src_row[x_low * ch + c];
                float high_val = src_row[x_high * ch + c];
                dst_row[x * ch + c] = low_val * x_weight_inv + high_val * x_weight;
            }
        }
        }  // End tile loop
    }
}

// Helper: Vertical resize with SIMD (separable pass 2 with prefetching)
static void resize_vertical_simd(
    const float* src, int src_h, int dst_w, int dst_h, int ch,
    uint8_t* dst)
{
    const float y_ratio = static_cast<float>(src_h - 1) / std::max(dst_h - 1, 1);

    for (int y = 0; y < dst_h; y++) {
        float src_y = y * y_ratio;
        int y_low = static_cast<int>(src_y);
        int y_high = std::min(y_low + 1, src_h - 1);
        float y_weight = src_y - y_low;
        float y_weight_inv = 1.0f - y_weight;

        const float* src_row_low = src + y_low * dst_w * ch;
        const float* src_row_high = src + y_high * dst_w * ch;
        uint8_t* dst_row = dst + y * dst_w * ch;

        // Prefetch next rows for better memory access
        if (y + 1 < dst_h) {
            float next_src_y = (y + 1) * y_ratio;
            int next_y_low = static_cast<int>(next_src_y);
            const float* next_row = src + next_y_low * dst_w * ch;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(next_row, 0, 3);
#endif
        }

        int pixels = dst_w * ch;
        int i = 0;

#if defined(HAVE_AVX512)
        // AVX-512: Process 16 floats at once
        __m512 weight_inv_vec = _mm512_set1_ps(y_weight_inv);
        __m512 weight_vec = _mm512_set1_ps(y_weight);

        for (; i + 15 < pixels; i += 16) {
            __m512 low = _mm512_loadu_ps(src_row_low + i);
            __m512 high = _mm512_loadu_ps(src_row_high + i);
            __m512 result = _mm512_fmadd_ps(low, weight_inv_vec,
                                            _mm512_mul_ps(high, weight_vec));

            // Convert float to uint8
            __m512i result_i32 = _mm512_cvtps_epi32(result);
            __m128i result_u8 = _mm512_cvtusepi32_epi8(result_i32);
            _mm_storeu_si128((__m128i*)(dst_row + i), result_u8);
        }
#elif defined(HAVE_AVX2)
        // AVX2: Process 8 floats at once
        __m256 weight_inv_vec = _mm256_set1_ps(y_weight_inv);
        __m256 weight_vec = _mm256_set1_ps(y_weight);

        for (; i + 7 < pixels; i += 8) {
            __m256 low = _mm256_loadu_ps(src_row_low + i);
            __m256 high = _mm256_loadu_ps(src_row_high + i);
            __m256 result = _mm256_add_ps(_mm256_mul_ps(low, weight_inv_vec),
                                          _mm256_mul_ps(high, weight_vec));

            // Convert float to uint8 (clamp to 0-255)
            __m256i result_i32 = _mm256_cvtps_epi32(result);
            result_i32 = _mm256_packus_epi32(result_i32, result_i32);
            result_i32 = _mm256_permute4x64_epi64(result_i32, 0xD8);
            __m128i result_i16 = _mm256_castsi256_si128(result_i32);
            __m128i result_u8 = _mm_packus_epi16(result_i16, result_i16);
            _mm_storel_epi64((__m128i*)(dst_row + i), result_u8);
        }
#elif defined(HAVE_NEON)
        // NEON: Process 4 floats at once
        float32x4_t weight_inv_vec = vdupq_n_f32(y_weight_inv);
        float32x4_t weight_vec = vdupq_n_f32(y_weight);

        for (; i + 3 < pixels; i += 4) {
            float32x4_t low = vld1q_f32(src_row_low + i);
            float32x4_t high = vld1q_f32(src_row_high + i);
            float32x4_t result = vmlaq_f32(vmulq_f32(low, weight_inv_vec),
                                           high, weight_vec);

            // Convert float to uint8
            uint32x4_t result_u32 = vcvtq_u32_f32(result);
            uint16x4_t result_u16 = vqmovn_u32(result_u32);
            uint8x8_t result_u8 = vqmovn_u16(vcombine_u16(result_u16, result_u16));
            vst1_lane_u32((uint32_t*)(dst_row + i), vreinterpret_u32_u8(result_u8), 0);
        }
#endif
        // Scalar fallback
        for (; i < pixels; i++) {
            float val = src_row_low[i] * y_weight_inv + src_row_high[i] * y_weight;
            dst_row[i] = static_cast<uint8_t>(std::min(std::max(val + 0.5f, 0.0f), 255.0f));
        }
    }
}

void SimdResize::resize(
    const uint8_t* src, int src_w, int src_h, int ch,
    uint8_t* dst, int dst_w, int dst_h,
    ResizeMethod method)
{
    if (method == ResizeMethod::BILINEAR) {
        // Use optimized separable resize
        // Allocate temporary buffer for horizontal pass
        std::vector<float> temp(src_h * dst_w * ch);

        // Pass 1: Horizontal resize (better cache locality)
        resize_horizontal_simd(src, src_w, src_h, ch, temp.data(), dst_w);

        // Pass 2: Vertical resize
        resize_vertical_simd(temp.data(), src_h, dst_w, dst_h, ch, dst);
    } else {
        resize_nearest_simd(src, src_w, src_h, ch, dst, dst_w, dst_h);
    }
}

void SimdResize::resize_bilinear_simd(
    const uint8_t* src, int src_w, int src_h, int ch,
    uint8_t* dst, int dst_w, int dst_h)
{
    const float x_ratio = static_cast<float>(src_w - 1) / (dst_w - 1);
    const float y_ratio = static_cast<float>(src_h - 1) / (dst_h - 1);

    // Process each output pixel
    for (int y = 0; y < dst_h; y++) {
        float src_y = y * y_ratio;
        int y_low = static_cast<int>(src_y);
        int y_high = std::min(y_low + 1, src_h - 1);
        float y_weight = src_y - y_low;

        for (int x = 0; x < dst_w; x++) {
            float src_x = x * x_ratio;
            int x_low = static_cast<int>(src_x);
            int x_high = std::min(x_low + 1, src_w - 1);
            float x_weight = src_x - x_low;

            // Bilinear interpolation for each channel
            for (int c = 0; c < ch; c++) {
                float tl = src[(y_low * src_w + x_low) * ch + c];
                float tr = src[(y_low * src_w + x_high) * ch + c];
                float bl = src[(y_high * src_w + x_low) * ch + c];
                float br = src[(y_high * src_w + x_high) * ch + c];

                float top = tl + (tr - tl) * x_weight;
                float bottom = bl + (br - bl) * x_weight;
                float value = top + (bottom - top) * y_weight;

                dst[(y * dst_w + x) * ch + c] = static_cast<uint8_t>(value + 0.5f);
            }
        }
    }
}

void SimdResize::resize_nearest_simd(
    const uint8_t* src, int src_w, int src_h, int ch,
    uint8_t* dst, int dst_w, int dst_h)
{
    const float x_ratio = static_cast<float>(src_w) / dst_w;
    const float y_ratio = static_cast<float>(src_h) / dst_h;

    for (int y = 0; y < dst_h; y++) {
        int src_y = static_cast<int>(y * y_ratio);
        for (int x = 0; x < dst_w; x++) {
            int src_x = static_cast<int>(x * x_ratio);

            for (int c = 0; c < ch; c++) {
                dst[(y * dst_w + x) * ch + c] = src[(src_y * src_w + src_x) * ch + c];
            }
        }
    }
}

void SimdResize::resize_to_float(
    const uint8_t* src, int src_w, int src_h, int ch,
    float* dst, int dst_w, int dst_h,
    float scale, ResizeMethod method)
{
    (void)method;  // Unused parameter - only bilinear supported for now
    const float x_ratio = static_cast<float>(src_w - 1) / (dst_w - 1);
    const float y_ratio = static_cast<float>(src_h - 1) / (dst_h - 1);

    for (int y = 0; y < dst_h; y++) {
        float src_y = y * y_ratio;
        int y_low = static_cast<int>(src_y);
        int y_high = std::min(y_low + 1, src_h - 1);
        float y_weight = src_y - y_low;

        for (int x = 0; x < dst_w; x++) {
            float src_x = x * x_ratio;
            int x_low = static_cast<int>(src_x);
            int x_high = std::min(x_low + 1, src_w - 1);
            float x_weight = src_x - x_low;

            for (int c = 0; c < ch; c++) {
                float tl = src[(y_low * src_w + x_low) * ch + c];
                float tr = src[(y_low * src_w + x_high) * ch + c];
                float bl = src[(y_high * src_w + x_low) * ch + c];
                float br = src[(y_high * src_w + x_high) * ch + c];

                float top = tl + (tr - tl) * x_weight;
                float bottom = bl + (br - bl) * x_weight;
                float value = top + (bottom - top) * y_weight;

                dst[(y * dst_w + x) * ch + c] = value * scale;
            }
        }
    }
}

// ============================================================================
// SIMD Normalize
// ============================================================================

void SimdNormalize::normalize_uint8(
    const uint8_t* src, float* dst, size_t size,
    const float* mean, const float* std, int channels)
{
    const float scale = 1.0f / 255.0f;

#if defined(HAVE_AVX2)
    // AVX2 implementation - process 8 floats at a time
    size_t vec_size = size / 8 * 8;

    for (size_t i = 0; i < vec_size; i += 8) {
        int ch = (i / 8) % channels;

        // Load 8 uint8 values and convert to float
        __m128i src_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i));
        __m128i src_u32 = _mm_cvtepu8_epi32(src_u8);
        __m256 src_f32 = _mm256_cvtepi32_ps(_mm256_castsi128_si256(src_u32));

        // Scale to [0, 1]
        __m256 scale_vec = _mm256_set1_ps(scale);
        src_f32 = _mm256_mul_ps(src_f32, scale_vec);

        // Normalize: (x - mean) / std
        __m256 mean_vec = _mm256_set1_ps(mean[ch]);
        __m256 std_vec = _mm256_set1_ps(std[ch]);

        src_f32 = _mm256_sub_ps(src_f32, mean_vec);
        src_f32 = _mm256_div_ps(src_f32, std_vec);

        // Store result
        _mm256_storeu_ps(dst + i, src_f32);
    }

    // Handle remaining elements
    for (size_t i = vec_size; i < size; i++) {
        int ch = i % channels;
        dst[i] = (src[i] * scale - mean[ch]) / std[ch];
    }

#elif defined(HAVE_NEON)
    // NEON implementation - process 4 floats at a time
    // NOTE: We need at least 4 bytes remaining to safely vectorize
    size_t vec_size = (size >= 4) ? (size - 3) / 4 * 4 : 0;

    for (size_t i = 0; i < vec_size; i += 4) {
        int ch = (i / 4) % channels;

        // Safely load 4 uint8 values into an 8-byte vector (upper 4 bytes unused)
        uint8_t temp[8] = {0};
        temp[0] = src[i];
        temp[1] = src[i + 1];
        temp[2] = src[i + 2];
        temp[3] = src[i + 3];

        uint8x8_t src_u8 = vld1_u8(temp);
        uint16x4_t src_u16 = vget_low_u16(vmovl_u8(src_u8));
        uint32x4_t src_u32 = vmovl_u16(src_u16);
        float32x4_t src_f32 = vcvtq_f32_u32(src_u32);

        // Scale to [0, 1]
        float32x4_t scale_vec = vdupq_n_f32(scale);
        src_f32 = vmulq_f32(src_f32, scale_vec);

        // Normalize
        float32x4_t mean_vec = vdupq_n_f32(mean[ch]);
        float32x4_t std_vec = vdupq_n_f32(std[ch]);

        src_f32 = vsubq_f32(src_f32, mean_vec);
        src_f32 = vdivq_f32(src_f32, std_vec);

        // Store result
        vst1q_f32(dst + i, src_f32);
    }

    // Handle remaining elements
    for (size_t i = vec_size; i < size; i++) {
        int ch = i % channels;
        dst[i] = (src[i] * scale - mean[ch]) / std[ch];
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < size; i++) {
        int ch = i % channels;
        dst[i] = (src[i] * scale - mean[ch]) / std[ch];
    }
#endif
}

void SimdNormalize::normalize_float(
    const float* src, float* dst, size_t size,
    const float* mean, const float* std, int channels)
{
#if defined(HAVE_AVX2)
    size_t vec_size = size / 8 * 8;

    for (size_t i = 0; i < vec_size; i += 8) {
        int ch = (i / 8) % channels;

        __m256 src_vec = _mm256_loadu_ps(src + i);
        __m256 mean_vec = _mm256_set1_ps(mean[ch]);
        __m256 std_vec = _mm256_set1_ps(std[ch]);

        src_vec = _mm256_sub_ps(src_vec, mean_vec);
        src_vec = _mm256_div_ps(src_vec, std_vec);

        _mm256_storeu_ps(dst + i, src_vec);
    }

    for (size_t i = vec_size; i < size; i++) {
        int ch = i % channels;
        dst[i] = (src[i] - mean[ch]) / std[ch];
    }

#elif defined(HAVE_NEON)
    size_t vec_size = size / 4 * 4;

    for (size_t i = 0; i < vec_size; i += 4) {
        int ch = (i / 4) % channels;

        float32x4_t src_vec = vld1q_f32(src + i);
        float32x4_t mean_vec = vdupq_n_f32(mean[ch]);
        float32x4_t std_vec = vdupq_n_f32(std[ch]);

        src_vec = vsubq_f32(src_vec, mean_vec);
        src_vec = vdivq_f32(src_vec, std_vec);

        vst1q_f32(dst + i, src_vec);
    }

    for (size_t i = vec_size; i < size; i++) {
        int ch = i % channels;
        dst[i] = (src[i] - mean[ch]) / std[ch];
    }

#else
    for (size_t i = 0; i < size; i++) {
        int ch = i % channels;
        dst[i] = (src[i] - mean[ch]) / std[ch];
    }
#endif
}

void SimdNormalize::resize_and_normalize(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h, int ch,
    const float* mean, const float* std)
{
    // Combined resize + normalize (optimized - single pass)
    const float x_ratio = static_cast<float>(src_w - 1) / (dst_w - 1);
    const float y_ratio = static_cast<float>(src_h - 1) / (dst_h - 1);
    const float scale = 1.0f / 255.0f;

    for (int y = 0; y < dst_h; y++) {
        float src_y = y * y_ratio;
        int y_low = static_cast<int>(src_y);
        int y_high = std::min(y_low + 1, src_h - 1);
        float y_weight = src_y - y_low;

        for (int x = 0; x < dst_w; x++) {
            float src_x = x * x_ratio;
            int x_low = static_cast<int>(src_x);
            int x_high = std::min(x_low + 1, src_w - 1);
            float x_weight = src_x - x_low;

            for (int c = 0; c < ch; c++) {
                float tl = src[(y_low * src_w + x_low) * ch + c];
                float tr = src[(y_low * src_w + x_high) * ch + c];
                float bl = src[(y_high * src_w + x_low) * ch + c];
                float br = src[(y_high * src_w + x_high) * ch + c];

                float top = tl + (tr - tl) * x_weight;
                float bottom = bl + (br - bl) * x_weight;
                float value = top + (bottom - top) * y_weight;

                // Normalize in same pass
                dst[(y * dst_w + x) * ch + c] = (value * scale - mean[c]) / std[c];
            }
        }
    }
}

// ============================================================================
// Color Space Conversion
// ============================================================================

void SimdColorConvert::rgb_to_bgr(const uint8_t* src, uint8_t* dst, size_t pixels) {
    // Simple channel swap - can be optimized with SIMD shuffle instructions
    for (size_t i = 0; i < pixels; i++) {
        dst[i * 3 + 0] = src[i * 3 + 2];  // B
        dst[i * 3 + 1] = src[i * 3 + 1];  // G
        dst[i * 3 + 2] = src[i * 3 + 0];  // R
    }
}

void SimdColorConvert::rgb_to_gray(const uint8_t* src, uint8_t* dst, size_t pixels) {
    // Grayscale: 0.299*R + 0.587*G + 0.114*B
    for (size_t i = 0; i < pixels; i++) {
        uint8_t r = src[i * 3 + 0];
        uint8_t g = src[i * 3 + 1];
        uint8_t b = src[i * 3 + 2];
        dst[i] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// ============================================================================
// Crop and Flip
// ============================================================================

void SimdCropFlip::crop(
    const uint8_t* src, int src_w, int src_h, int ch,
    uint8_t* dst, int crop_x, int crop_y, int crop_w, int crop_h)
{
    (void)src_h;  // Unused parameter - height is implicit in crop coordinates
    for (int y = 0; y < crop_h; y++) {
        const uint8_t* src_row = src + ((crop_y + y) * src_w + crop_x) * ch;
        uint8_t* dst_row = dst + y * crop_w * ch;
        std::memcpy(dst_row, src_row, crop_w * ch);
    }
}

void SimdCropFlip::flip_horizontal(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                dst[(y * width + x) * channels + c] =
                    src[(y * width + (width - 1 - x)) * channels + c];
            }
        }
    }
}

void SimdCropFlip::flip_vertical(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* src_row = src + (height - 1 - y) * width * channels;
        uint8_t* dst_row = dst + y * width * channels;
        std::memcpy(dst_row, src_row, width * channels);
    }
}

// ============================================================================
// Transform Pipeline
// ============================================================================

struct TransformPipeline::Impl {
    TransformConfig config;
    std::vector<float> temp_buffer;

    explicit Impl(const TransformConfig& cfg) : config(cfg) {}
};

TransformPipeline::TransformPipeline(const TransformConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TransformPipeline::~TransformPipeline() = default;
TransformPipeline::TransformPipeline(TransformPipeline&&) noexcept = default;
TransformPipeline& TransformPipeline::operator=(TransformPipeline&&) noexcept = default;

float* TransformPipeline::transform(
    const uint8_t* src, int src_w, int src_h, int ch, float* dst)
{
    int out_w = src_w;
    int out_h = src_h;

    // Calculate output dimensions
    if (pimpl_->config.enable_resize) {
        out_w = pimpl_->config.resize_width;
        out_h = pimpl_->config.resize_height;
    }

    // Allocate output if needed
    if (dst == nullptr) {
        pimpl_->temp_buffer.resize(out_w * out_h * ch);
        dst = pimpl_->temp_buffer.data();
    }

    // Apply transforms
    if (pimpl_->config.enable_resize && pimpl_->config.enable_normalize) {
        // Combined resize + normalize (optimized)
        SimdNormalize::resize_and_normalize(
            src, src_w, src_h,
            dst, out_w, out_h, ch,
            pimpl_->config.mean,
            pimpl_->config.std
        );
    } else if (pimpl_->config.enable_resize) {
        // Resize only
        SimdResize::resize_to_float(
            src, src_w, src_h, ch,
            dst, out_w, out_h,
            1.0f / 255.0f,
            pimpl_->config.resize_method
        );
    } else if (pimpl_->config.enable_normalize) {
        // Normalize only (convert uint8 to float first)
        size_t size = src_w * src_h * ch;
        SimdNormalize::normalize_uint8(
            src, dst, size,
            pimpl_->config.mean,
            pimpl_->config.std,
            ch
        );
    }

    return dst;
}

void TransformPipeline::get_output_dims(int& width, int& height) const {
    if (pimpl_->config.enable_resize) {
        width = pimpl_->config.resize_width;
        height = pimpl_->config.resize_height;
    }
}

bool TransformPipeline::is_simd_available() {
#if defined(HAVE_AVX2) || defined(HAVE_NEON)
    return true;
#else
    return false;
#endif
}

const char* TransformPipeline::get_simd_backend() {
#if defined(HAVE_AVX2)
    return "AVX2";
#elif defined(HAVE_NEON)
    return "NEON";
#else
    return "Scalar";
#endif
}

} // namespace transforms
} // namespace turboloader
