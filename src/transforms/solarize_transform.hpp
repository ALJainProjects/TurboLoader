/**
 * @file solarize_transform.hpp
 * @brief SIMD-accelerated solarization (invert pixels above threshold)
 *
 * Features:
 * - Invert pixel values above threshold
 * - SIMD vectorized comparison and inversion
 * - Configurable threshold
 *
 * Reference: torchvision.transforms.RandomSolarize
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Random solarize transform (invert pixels above threshold)
 */
class RandomSolarizeTransform : public RandomTransform {
public:
    /**
     * @param threshold Threshold for solarization (0-255)
     * @param probability Probability of applying transform
     * @param seed Random seed
     */
    RandomSolarizeTransform(uint8_t threshold, float probability = 0.5f, unsigned seed = std::random_device{}())
        : RandomTransform(probability, seed), threshold_(threshold) {}

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

        return apply_solarize(input);
    }

    const char* name() const override { return "RandomSolarize"; }

private:
    uint8_t threshold_;

    /**
     * @brief Apply solarization using SIMD
     * For each pixel: if pixel > threshold, pixel = 255 - pixel
     */
    std::unique_ptr<ImageData> apply_solarize(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        size_t total_pixels = input.width * input.height * input.channels;

#ifdef TURBOLOADER_SIMD_AVX2
        // AVX2: Process 32 bytes at a time
        __m256i threshold_vec = _mm256_set1_epi8(threshold_);
        __m256i max_val = _mm256_set1_epi8(255);
        size_t i = 0;

        for (; i + 32 <= total_pixels; i += 32) {
            __m256i pixels = _mm256_loadu_si256((__m256i*)(input.data + i));

            // Create mask: pixels > threshold
            __m256i mask = _mm256_cmpgt_epi8(pixels, threshold_vec);

            // Invert pixels: 255 - pixels
            __m256i inverted = _mm256_sub_epi8(max_val, pixels);

            // Select inverted or original based on mask
            __m256i result = _mm256_blendv_epi8(pixels, inverted, mask);

            _mm256_storeu_si256((__m256i*)(output->data + i), result);
        }

        // Scalar tail
        for (; i < total_pixels; ++i) {
            output->data[i] = (input.data[i] > threshold_) ? (255 - input.data[i]) : input.data[i];
        }

#elif defined(TURBOLOADER_SIMD_NEON)
        // NEON: Process 16 bytes at a time
        uint8x16_t threshold_vec = vdupq_n_u8(threshold_);
        uint8x16_t max_val = vdupq_n_u8(255);
        size_t i = 0;

        for (; i + 16 <= total_pixels; i += 16) {
            uint8x16_t pixels = vld1q_u8(input.data + i);

            // Create mask: pixels > threshold
            uint8x16_t mask = vcgtq_u8(pixels, threshold_vec);

            // Invert pixels: 255 - pixels
            uint8x16_t inverted = vsubq_u8(max_val, pixels);

            // Select inverted or original based on mask
            uint8x16_t result = vbslq_u8(mask, inverted, pixels);

            vst1q_u8(output->data + i, result);
        }

        // Scalar tail
        for (; i < total_pixels; ++i) {
            output->data[i] = (input.data[i] > threshold_) ? (255 - input.data[i]) : input.data[i];
        }

#else
        // Scalar fallback
        for (size_t i = 0; i < total_pixels; ++i) {
            output->data[i] = (input.data[i] > threshold_) ? (255 - input.data[i]) : input.data[i];
        }
#endif

        return output;
    }
};

} // namespace transforms
} // namespace turboloader
