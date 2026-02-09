/**
 * @file color_jitter_transform.hpp
 * @brief Color jitter transform (brightness, contrast, saturation, hue)
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <algorithm>

namespace turboloader {
namespace transforms {

/**
 * @brief Color jitter transform with SIMD acceleration
 */
class ColorJitterTransform : public RandomTransform {
public:
    ColorJitterTransform(float brightness = 0.0f,
                        float contrast = 0.0f,
                        float saturation = 0.0f,
                        float hue = 0.0f,
                        unsigned seed = std::random_device{}())
        : RandomTransform(1.0f, seed),
          brightness_(brightness),
          contrast_(contrast),
          saturation_(saturation),
          hue_(hue) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Copy input to output
        std::memcpy(output->data, input.data, input.size_bytes());

        // Apply jitters in random order (like PyTorch)
        std::vector<int> order = {0, 1, 2, 3};
        std::shuffle(order.begin(), order.end(), rng_);

        for (int op : order) {
            switch (op) {
                case 0: if (brightness_ > 0.0f) apply_brightness(*output); break;
                case 1: if (contrast_ > 0.0f) apply_contrast(*output); break;
                case 2: if (saturation_ > 0.0f && input.channels == 3) apply_saturation(*output); break;
                case 3: if (hue_ > 0.0f && input.channels == 3) apply_hue(*output); break;
            }
        }

        return output;
    }

    const char* name() const override { return "ColorJitter"; }

private:
    void apply_brightness(ImageData& image) {
        std::uniform_real_distribution<float> dist(
            std::max(0.0f, 1.0f - brightness_),
            1.0f + brightness_
        );
        float factor = dist(rng_);

        size_t count = image.width * image.height * image.channels;
        simd::mul_u8_scalar(image.data, image.data, factor, count);
    }

    void apply_contrast(ImageData& image) {
        std::uniform_real_distribution<float> dist(
            std::max(0.0f, 1.0f - contrast_),
            1.0f + contrast_
        );
        float factor = dist(rng_);

        // Calculate per-channel mean (matches torchvision behavior)
        size_t num_pixels = image.width * image.height;
        int ch = image.channels;
        float channel_mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < num_pixels; ++i) {
            for (int c = 0; c < ch; ++c) {
                channel_mean[c] += image.data[i * ch + c];
            }
        }
        for (int c = 0; c < ch; ++c) {
            channel_mean[c] /= num_pixels;
        }

        // Apply contrast per channel: pixel = mean[c] + factor * (pixel - mean[c])
        for (size_t i = 0; i < num_pixels; ++i) {
            for (int c = 0; c < ch; ++c) {
                float val = channel_mean[c] + factor * (image.data[i * ch + c] - channel_mean[c]);
                image.data[i * ch + c] = static_cast<uint8_t>(simd::clamp(val, 0.0f, 255.0f));
            }
        }
    }

    void apply_saturation(ImageData& image) {
        std::uniform_real_distribution<float> dist(
            std::max(0.0f, 1.0f - saturation_),
            1.0f + saturation_
        );
        float factor = dist(rng_);
        size_t num_pixels = image.width * image.height;

#ifdef TURBOLOADER_SIMD_NEON
        // Use NEON-optimized batch processing
        simd::adjust_saturation_neon(image.data, num_pixels, factor);
#else
        // Scalar fallback
        for (size_t i = 0; i < num_pixels; ++i) {
            uint8_t r = image.data[i * 3];
            uint8_t g = image.data[i * 3 + 1];
            uint8_t b = image.data[i * 3 + 2];

            float h, s, v;
            simd::rgb_to_hsv(r, g, b, h, s, v);
            s = simd::clamp(s * factor, 0.0f, 1.0f);
            simd::hsv_to_rgb(h, s, v, r, g, b);

            image.data[i * 3] = r;
            image.data[i * 3 + 1] = g;
            image.data[i * 3 + 2] = b;
        }
#endif
    }

    void apply_hue(ImageData& image) {
        std::uniform_real_distribution<float> dist(-hue_, hue_);
        float hue_shift = dist(rng_) * 360.0f;  // Convert to degrees
        size_t num_pixels = image.width * image.height;

#ifdef TURBOLOADER_SIMD_NEON
        // Use NEON-optimized batch processing
        simd::adjust_hue_neon(image.data, num_pixels, hue_shift);
#else
        // Scalar fallback
        for (size_t i = 0; i < num_pixels; ++i) {
            uint8_t r = image.data[i * 3];
            uint8_t g = image.data[i * 3 + 1];
            uint8_t b = image.data[i * 3 + 2];

            float h, s, v;
            simd::rgb_to_hsv(r, g, b, h, s, v);

            h += hue_shift;
            if (h < 0.0f) h += 360.0f;
            if (h >= 360.0f) h -= 360.0f;

            simd::hsv_to_rgb(h, s, v, r, g, b);

            image.data[i * 3] = r;
            image.data[i * 3 + 1] = g;
            image.data[i * 3 + 2] = b;
        }
#endif
    }

    float brightness_;
    float contrast_;
    float saturation_;
    float hue_;
};

} // namespace transforms
} // namespace turboloader
