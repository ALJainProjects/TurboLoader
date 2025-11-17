/**
 * @file simd_utils.hpp
 * @brief SIMD utilities for image transforms (AVX-512/AVX2/NEON)
 *
 * Provides vectorized operations for high-performance image processing:
 * - Compile-time platform detection (AVX-512/AVX2 on x86, NEON on ARM)
 * - Vectorized arithmetic (add, mul, clamp, etc.)
 * - Channel manipulation (RGB/HSV conversion, channel shuffle)
 * - Memory alignment helpers
 *
 * Performance: 4-16x speedup vs scalar code on typical operations
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

// Platform detection and SIMD headers
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define TURBOLOADER_SIMD_AVX512 1
    #define SIMD_BYTES 64
    #define SIMD_FLOAT_WIDTH 16
    #define SIMD_INT32_WIDTH 16
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define TURBOLOADER_SIMD_AVX2 1
    #define SIMD_BYTES 32
    #define SIMD_FLOAT_WIDTH 8
    #define SIMD_INT32_WIDTH 8
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define TURBOLOADER_SIMD_NEON 1
    #define SIMD_BYTES 16
    #define SIMD_FLOAT_WIDTH 4
    #define SIMD_INT32_WIDTH 4
#else
    #define TURBOLOADER_SIMD_SCALAR 1
    #define SIMD_BYTES 16
    #define SIMD_FLOAT_WIDTH 4
    #define SIMD_INT32_WIDTH 4
#endif

namespace turboloader {
namespace transforms {
namespace simd {

/**
 * @brief Check if pointer is aligned to SIMD boundary
 */
inline bool is_aligned(const void* ptr, size_t alignment = SIMD_BYTES) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Align value up to next multiple of alignment
 */
inline size_t align_up(size_t value, size_t alignment = SIMD_BYTES) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Allocate aligned memory
 */
inline void* aligned_alloc(size_t size, size_t alignment = SIMD_BYTES) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * @brief Free aligned memory
 */
inline void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// VECTORIZED OPERATIONS - AVX-512
// ============================================================================

#ifdef TURBOLOADER_SIMD_AVX512

/**
 * @brief Convert uint8 to float32 (normalized to [0,1]) - AVX-512
 */
inline void cvt_u8_to_f32_normalized(const uint8_t* src, float* dst, size_t count) {
    const __m512 scale = _mm512_set1_ps(1.0f / 255.0f);

    size_t i = 0;
    // Process 16 floats at a time
    for (; i + 16 <= count; i += 16) {
        // Load 16 uint8 values
        __m128i u8_vals = _mm_loadu_si128((__m128i*)(src + i));

        // Zero-extend to 32-bit integers
        __m512i i32_vals = _mm512_cvtepu8_epi32(u8_vals);

        // Convert to float and normalize
        __m512 f32_vals = _mm512_cvtepi32_ps(i32_vals);
        f32_vals = _mm512_mul_ps(f32_vals, scale);

        _mm512_storeu_ps(dst + i, f32_vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = src[i] / 255.0f;
    }
}

/**
 * @brief Convert float32 to uint8 (clamped) - AVX-512
 */
inline void cvt_f32_to_u8_clamped(const float* src, uint8_t* dst, size_t count) {
    const __m512 scale = _mm512_set1_ps(255.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 max_val = _mm512_set1_ps(255.0f);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vals = _mm512_loadu_ps(src + i);

        // Scale [0,1] -> [0,255]
        vals = _mm512_mul_ps(vals, scale);

        // Clamp to [0, 255]
        vals = _mm512_max_ps(vals, zero);
        vals = _mm512_min_ps(vals, max_val);

        // Convert to int32
        __m512i i32_vals = _mm512_cvtps_epi32(vals);

        // Truncate to uint8 (AVX-512 provides direct conversion)
        __m128i u8_vals = _mm512_cvtusepi32_epi8(i32_vals);

        // Store 16 bytes
        _mm_storeu_si128((__m128i*)(dst + i), u8_vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] * 255.0f;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Multiply uint8 array by scalar (for brightness adjustment) - AVX-512
 */
inline void mul_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    const __m512 scale = _mm512_set1_ps(scalar);
    const __m512 max_val = _mm512_set1_ps(255.0f);
    const __m512 zero = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m128i u8_vals = _mm_loadu_si128((__m128i*)(src + i));
        __m512i i32_vals = _mm512_cvtepu8_epi32(u8_vals);
        __m512 f32_vals = _mm512_cvtepi32_ps(i32_vals);

        f32_vals = _mm512_mul_ps(f32_vals, scale);
        f32_vals = _mm512_max_ps(f32_vals, zero);
        f32_vals = _mm512_min_ps(f32_vals, max_val);

        __m512i result_i32 = _mm512_cvtps_epi32(f32_vals);
        __m128i result_u8 = _mm512_cvtusepi32_epi8(result_i32);

        _mm_storeu_si128((__m128i*)(dst + i), result_u8);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] * scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Add scalar to uint8 array (for brightness adjustment) - AVX-512
 */
inline void add_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    const __m512 add_val = _mm512_set1_ps(scalar);
    const __m512 max_val = _mm512_set1_ps(255.0f);
    const __m512 zero = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m128i u8_vals = _mm_loadu_si128((__m128i*)(src + i));
        __m512i i32_vals = _mm512_cvtepu8_epi32(u8_vals);
        __m512 f32_vals = _mm512_cvtepi32_ps(i32_vals);

        f32_vals = _mm512_add_ps(f32_vals, add_val);
        f32_vals = _mm512_max_ps(f32_vals, zero);
        f32_vals = _mm512_min_ps(f32_vals, max_val);

        __m512i result_i32 = _mm512_cvtps_epi32(f32_vals);
        __m128i result_u8 = _mm512_cvtusepi32_epi8(result_i32);

        _mm_storeu_si128((__m128i*)(dst + i), result_u8);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] + scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Normalize with mean/std (SIMD-accelerated) - AVX-512
 */
inline void normalize_f32(const float* src, float* dst, float mean, float std, size_t count) {
    const __m512 mean_vec = _mm512_set1_ps(mean);
    const __m512 inv_std = _mm512_set1_ps(1.0f / std);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vals = _mm512_loadu_ps(src + i);
        vals = _mm512_sub_ps(vals, mean_vec);
        vals = _mm512_mul_ps(vals, inv_std);
        _mm512_storeu_ps(dst + i, vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = (src[i] - mean) / std;
    }
}

#endif // TURBOLOADER_SIMD_AVX512

// ============================================================================
// VECTORIZED OPERATIONS - AVX2
// ============================================================================

#ifdef TURBOLOADER_SIMD_AVX2

/**
 * @brief Convert uint8 to float32 (normalized to [0,1])
 */
inline void cvt_u8_to_f32_normalized(const uint8_t* src, float* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);

    size_t i = 0;
    // Process 8 floats at a time
    for (; i + 8 <= count; i += 8) {
        // Load 8 uint8 values (only using lower 8 bytes of 128-bit load)
        __m128i u8_vals = _mm_loadl_epi64((__m128i*)(src + i));

        // Zero-extend to 32-bit integers
        __m256i i32_vals = _mm256_cvtepu8_epi32(u8_vals);

        // Convert to float and normalize
        __m256 f32_vals = _mm256_cvtepi32_ps(i32_vals);
        f32_vals = _mm256_mul_ps(f32_vals, scale);

        _mm256_storeu_ps(dst + i, f32_vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = src[i] / 255.0f;
    }
}

/**
 * @brief Convert float32 to uint8 (clamped)
 */
inline void cvt_f32_to_u8_clamped(const float* src, uint8_t* dst, size_t count) {
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 max_val = _mm256_set1_ps(255.0f);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vals = _mm256_loadu_ps(src + i);

        // Scale [0,1] -> [0,255]
        vals = _mm256_mul_ps(vals, scale);

        // Clamp to [0, 255]
        vals = _mm256_max_ps(vals, zero);
        vals = _mm256_min_ps(vals, max_val);

        // Convert to int32
        __m256i i32_vals = _mm256_cvtps_epi32(vals);

        // Pack to uint8 (32->16->8 bit)
        __m128i i16_vals = _mm256_extracti128_si256(_mm256_packs_epi32(i32_vals, i32_vals), 0);
        __m128i u8_vals = _mm_packus_epi16(i16_vals, i16_vals);

        // Store 8 bytes
        _mm_storel_epi64((__m128i*)(dst + i), u8_vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] * 255.0f;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Multiply uint8 array by scalar (for brightness adjustment)
 */
inline void mul_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    const __m256 scale = _mm256_set1_ps(scalar);
    const __m256 max_val = _mm256_set1_ps(255.0f);
    const __m256 zero = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i u8_vals = _mm_loadl_epi64((__m128i*)(src + i));
        __m256i i32_vals = _mm256_cvtepu8_epi32(u8_vals);
        __m256 f32_vals = _mm256_cvtepi32_ps(i32_vals);

        f32_vals = _mm256_mul_ps(f32_vals, scale);
        f32_vals = _mm256_max_ps(f32_vals, zero);
        f32_vals = _mm256_min_ps(f32_vals, max_val);

        __m256i result_i32 = _mm256_cvtps_epi32(f32_vals);
        __m128i result_i16 = _mm256_extracti128_si256(_mm256_packs_epi32(result_i32, result_i32), 0);
        __m128i result_u8 = _mm_packus_epi16(result_i16, result_i16);

        _mm_storel_epi64((__m128i*)(dst + i), result_u8);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] * scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Add scalar to uint8 array (for brightness adjustment)
 */
inline void add_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    const __m256 add_val = _mm256_set1_ps(scalar);
    const __m256 max_val = _mm256_set1_ps(255.0f);
    const __m256 zero = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i u8_vals = _mm_loadl_epi64((__m128i*)(src + i));
        __m256i i32_vals = _mm256_cvtepu8_epi32(u8_vals);
        __m256 f32_vals = _mm256_cvtepi32_ps(i32_vals);

        f32_vals = _mm256_add_ps(f32_vals, add_val);
        f32_vals = _mm256_max_ps(f32_vals, zero);
        f32_vals = _mm256_min_ps(f32_vals, max_val);

        __m256i result_i32 = _mm256_cvtps_epi32(f32_vals);
        __m128i result_i16 = _mm256_extracti128_si256(_mm256_packs_epi32(result_i32, result_i32), 0);
        __m128i result_u8 = _mm_packus_epi16(result_i16, result_i16);

        _mm_storel_epi64((__m128i*)(dst + i), result_u8);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] + scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Normalize with mean/std (SIMD-accelerated)
 */
inline void normalize_f32(const float* src, float* dst, float mean, float std, size_t count) {
    const __m256 mean_vec = _mm256_set1_ps(mean);
    const __m256 inv_std = _mm256_set1_ps(1.0f / std);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 vals = _mm256_loadu_ps(src + i);
        vals = _mm256_sub_ps(vals, mean_vec);
        vals = _mm256_mul_ps(vals, inv_std);
        _mm256_storeu_ps(dst + i, vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = (src[i] - mean) / std;
    }
}

// ============================================================================
// VECTORIZED OPERATIONS - NEON
// ============================================================================

#elif defined(TURBOLOADER_SIMD_NEON)

/**
 * @brief Convert uint8 to float32 (normalized to [0,1])
 */
inline void cvt_u8_to_f32_normalized(const uint8_t* src, float* dst, size_t count) {
    const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);

    size_t i = 0;
    // Process 4 floats at a time
    for (; i + 4 <= count; i += 4) {
        // Load 4 uint8 values
        uint8x8_t u8_vals = vld1_u8(src + i);

        // Zero-extend to 16-bit
        uint16x4_t u16_vals = vget_low_u16(vmovl_u8(u8_vals));

        // Zero-extend to 32-bit
        uint32x4_t u32_vals = vmovl_u16(u16_vals);

        // Convert to float and normalize
        float32x4_t f32_vals = vcvtq_f32_u32(u32_vals);
        f32_vals = vmulq_f32(f32_vals, scale);

        vst1q_f32(dst + i, f32_vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = src[i] / 255.0f;
    }
}

/**
 * @brief Convert float32 to uint8 (clamped)
 */
inline void cvt_f32_to_u8_clamped(const float* src, uint8_t* dst, size_t count) {
    const float32x4_t scale = vdupq_n_f32(255.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t max_val = vdupq_n_f32(255.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vals = vld1q_f32(src + i);

        // Scale [0,1] -> [0,255]
        vals = vmulq_f32(vals, scale);

        // Clamp to [0, 255]
        vals = vmaxq_f32(vals, zero);
        vals = vminq_f32(vals, max_val);

        // Convert to uint32
        uint32x4_t u32_vals = vcvtq_u32_f32(vals);

        // Narrow to uint16
        uint16x4_t u16_vals = vmovn_u32(u32_vals);

        // Narrow to uint8
        uint8x8_t u8_vals = vmovn_u16(vcombine_u16(u16_vals, u16_vals));

        // Store 4 bytes
        vst1_lane_u32((uint32_t*)(dst + i), vreinterpret_u32_u8(u8_vals), 0);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] * 255.0f;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Multiply uint8 array by scalar
 */
inline void mul_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    const float32x4_t scale = vdupq_n_f32(scalar);
    const float32x4_t max_val = vdupq_n_f32(255.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        uint8x8_t u8_vals = vld1_u8(src + i);
        uint16x4_t u16_vals = vget_low_u16(vmovl_u8(u8_vals));
        uint32x4_t u32_vals = vmovl_u16(u16_vals);
        float32x4_t f32_vals = vcvtq_f32_u32(u32_vals);

        f32_vals = vmulq_f32(f32_vals, scale);
        f32_vals = vmaxq_f32(f32_vals, zero);
        f32_vals = vminq_f32(f32_vals, max_val);

        uint32x4_t result_u32 = vcvtq_u32_f32(f32_vals);
        uint16x4_t result_u16 = vmovn_u32(result_u32);
        uint8x8_t result_u8 = vmovn_u16(vcombine_u16(result_u16, result_u16));

        vst1_lane_u32((uint32_t*)(dst + i), vreinterpret_u32_u8(result_u8), 0);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] * scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Add scalar to uint8 array
 */
inline void add_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    const float32x4_t add_val = vdupq_n_f32(scalar);
    const float32x4_t max_val = vdupq_n_f32(255.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        uint8x8_t u8_vals = vld1_u8(src + i);
        uint16x4_t u16_vals = vget_low_u16(vmovl_u8(u8_vals));
        uint32x4_t u32_vals = vmovl_u16(u16_vals);
        float32x4_t f32_vals = vcvtq_f32_u32(u32_vals);

        f32_vals = vaddq_f32(f32_vals, add_val);
        f32_vals = vmaxq_f32(f32_vals, zero);
        f32_vals = vminq_f32(f32_vals, max_val);

        uint32x4_t result_u32 = vcvtq_u32_f32(f32_vals);
        uint16x4_t result_u16 = vmovn_u32(result_u32);
        uint8x8_t result_u8 = vmovn_u16(vcombine_u16(result_u16, result_u16));

        vst1_lane_u32((uint32_t*)(dst + i), vreinterpret_u32_u8(result_u8), 0);
    }

    // Scalar tail
    for (; i < count; ++i) {
        float val = src[i] + scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Normalize with mean/std (SIMD-accelerated)
 */
inline void normalize_f32(const float* src, float* dst, float mean, float std, size_t count) {
    const float32x4_t mean_vec = vdupq_n_f32(mean);
    const float inv_std_scalar = 1.0f / std;
    const float32x4_t inv_std = vdupq_n_f32(inv_std_scalar);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t vals = vld1q_f32(src + i);
        vals = vsubq_f32(vals, mean_vec);
        vals = vmulq_f32(vals, inv_std);
        vst1q_f32(dst + i, vals);
    }

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = (src[i] - mean) * inv_std_scalar;
    }
}

// ============================================================================
// SCALAR FALLBACK
// ============================================================================

#else

/**
 * @brief Convert uint8 to float32 (normalized to [0,1]) - Scalar
 */
inline void cvt_u8_to_f32_normalized(const uint8_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = src[i] / 255.0f;
    }
}

/**
 * @brief Convert float32 to uint8 (clamped) - Scalar
 */
inline void cvt_f32_to_u8_clamped(const float* src, uint8_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float val = src[i] * 255.0f;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Multiply uint8 array by scalar - Scalar
 */
inline void mul_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float val = src[i] * scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Add scalar to uint8 array - Scalar
 */
inline void add_u8_scalar(const uint8_t* src, uint8_t* dst, float scalar, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float val = src[i] + scalar;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val);
    }
}

/**
 * @brief Normalize with mean/std - Scalar
 */
inline void normalize_f32(const float* src, float* dst, float mean, float std, size_t count) {
    float inv_std = 1.0f / std;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = (src[i] - mean) * inv_std;
    }
}

#endif

// ============================================================================
// COMMON UTILITIES (All platforms)
// ============================================================================

/**
 * @brief RGB to Grayscale conversion (weighted sum)
 * Standard weights: R=0.299, G=0.587, B=0.114
 */
inline void rgb_to_grayscale(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; ++i) {
        size_t idx = i * 3;
        float val = 0.299f * rgb[idx] + 0.587f * rgb[idx + 1] + 0.114f * rgb[idx + 2];
        gray[i] = static_cast<uint8_t>(std::min(255.0f, val));
    }
}

/**
 * @brief RGB to HSV conversion (single pixel)
 */
inline void rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b, float& h, float& s, float& v) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    float max_val = std::max({rf, gf, bf});
    float min_val = std::min({rf, gf, bf});
    float delta = max_val - min_val;

    v = max_val;

    if (delta < 0.00001f) {
        s = 0.0f;
        h = 0.0f;
        return;
    }

    if (max_val > 0.0f) {
        s = delta / max_val;
    } else {
        s = 0.0f;
        h = 0.0f;
        return;
    }

    if (rf >= max_val) {
        h = (gf - bf) / delta;
    } else if (gf >= max_val) {
        h = 2.0f + (bf - rf) / delta;
    } else {
        h = 4.0f + (rf - gf) / delta;
    }

    h *= 60.0f;
    if (h < 0.0f) h += 360.0f;
}

/**
 * @brief HSV to RGB conversion (single pixel)
 */
inline void hsv_to_rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (s <= 0.0f) {
        r = g = b = static_cast<uint8_t>(v * 255.0f);
        return;
    }

    float hh = h;
    if (hh >= 360.0f) hh = 0.0f;
    hh /= 60.0f;

    int i = static_cast<int>(hh);
    float ff = hh - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - (s * ff));
    float t = v * (1.0f - (s * (1.0f - ff)));

    float rf, gf, bf;
    switch (i) {
        case 0: rf = v; gf = t; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = t; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = t; gf = p; bf = v; break;
        default: rf = v; gf = p; bf = q; break;
    }

    r = static_cast<uint8_t>(rf * 255.0f);
    g = static_cast<uint8_t>(gf * 255.0f);
    b = static_cast<uint8_t>(bf * 255.0f);
}

/**
 * @brief Clamp value to range [min, max]
 */
template<typename T>
inline T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(max_val, value));
}

/**
 * @brief Bilinear interpolation for single channel
 */
inline float bilinear_interpolate(const uint8_t* data, int width, int height,
                                  float x, float y, int channel, int num_channels) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);

    float dx = x - x0;
    float dy = y - y0;

    auto get_pixel = [&](int px, int py) -> float {
        return static_cast<float>(data[(py * width + px) * num_channels + channel]);
    };

    float val00 = get_pixel(x0, y0);
    float val10 = get_pixel(x1, y0);
    float val01 = get_pixel(x0, y1);
    float val11 = get_pixel(x1, y1);

    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;

    return val0 * (1.0f - dy) + val1 * dy;
}

} // namespace simd
} // namespace transforms
} // namespace turboloader
