/**
 * @file posterize_transform.hpp
 * @brief SIMD-accelerated posterization (reduce bit depth)
 *
 * Features:
 * - Reduce color bits per channel (e.g., 8-bit → 3-bit)
 * - SIMD batch processing of pixels
 * - Extremely fast (bitwise operations)
 *
 * Reference: torchvision.transforms.RandomPosterize
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Random posterize transform (reduce bit depth)
 */
class RandomPosterizeTransform : public RandomTransform {
public:
    /**
     * @param bits Number of bits to keep (1-8)
     * @param probability Probability of applying transform
     * @param seed Random seed
     */
    RandomPosterizeTransform(int bits, float probability = 0.5f, unsigned seed = std::random_device{}())
        : RandomTransform(probability, seed), bits_(bits) {
        if (bits < 1 || bits > 8) {
            throw std::invalid_argument("bits must be in range [1, 8]");
        }
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (!should_apply()) {
            // Return copy without modification
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        return apply_posterize(input);
    }

    const char* name() const override { return "RandomPosterize"; }

private:
    int bits_;

    /**
     * @brief Apply posterization using SIMD bitwise operations
     */
    std::unique_ptr<ImageData> apply_posterize(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Bit mask: clear lower (8 - bits) bits
        // Example: bits=3 -> mask = 11100000 (binary) = 224 (decimal)
        uint8_t mask = static_cast<uint8_t>(~((1 << (8 - bits_)) - 1));

        size_t total_pixels = input.width * input.height * input.channels;

#ifdef TURBOLOADER_SIMD_AVX2
        // AVX2: Process 32 bytes at a time
        __m256i mask_vec = _mm256_set1_epi8(mask);
        size_t i = 0;

        for (; i + 32 <= total_pixels; i += 32) {
            __m256i pixels = _mm256_loadu_si256((__m256i*)(input.data + i));
            __m256i result = _mm256_and_si256(pixels, mask_vec);
            _mm256_storeu_si256((__m256i*)(output->data + i), result);
        }

        // Scalar tail
        for (; i < total_pixels; ++i) {
            output->data[i] = input.data[i] & mask;
        }

#elif defined(TURBOLOADER_SIMD_NEON)
        // NEON: Process 16 bytes at a time
        uint8x16_t mask_vec = vdupq_n_u8(mask);
        size_t i = 0;

        for (; i + 16 <= total_pixels; i += 16) {
            uint8x16_t pixels = vld1q_u8(input.data + i);
            uint8x16_t result = vandq_u8(pixels, mask_vec);
            vst1q_u8(output->data + i, result);
        }

        // Scalar tail
        for (; i < total_pixels; ++i) {
            output->data[i] = input.data[i] & mask;
        }

#else
        // Scalar fallback
        for (size_t i = 0; i < total_pixels; ++i) {
            output->data[i] = input.data[i] & mask;
        }
#endif

        return output;
    }
};

} // namespace transforms
} // namespace turboloader
