/**
 * @file erasing_transform.hpp
 * @brief Random erasing (cutout) augmentation
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Random erasing transform (cutout augmentation)
 */
class RandomErasingTransform : public RandomTransform {
public:
    /**
     * @param p Probability of applying erasing
     * @param scale Range of proportion of erased area (min, max)
     * @param ratio Range of aspect ratio of erased area (min, max)
     * @param value Erasing value (single value or per-channel)
     */
    RandomErasingTransform(float p = 0.5f,
                          float scale_min = 0.02f,
                          float scale_max = 0.33f,
                          float ratio_min = 0.3f,
                          float ratio_max = 3.33f,
                          uint8_t value = 0,
                          unsigned seed = std::random_device{}())
        : RandomTransform(p, seed),
          scale_min_(scale_min),
          scale_max_(scale_max),
          ratio_min_(ratio_min),
          ratio_max_(ratio_max),
          value_(value) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Copy input
        std::memcpy(output->data, input.data, input.size_bytes());

        if (!should_apply()) {
            return output;
        }

        // Random erasing parameters
        std::uniform_real_distribution<float> scale_dist(scale_min_, scale_max_);
        std::uniform_real_distribution<float> ratio_dist(ratio_min_, ratio_max_);

        float area = input.width * input.height;
        float target_area = area * scale_dist(rng_);
        float aspect_ratio = ratio_dist(rng_);

        int h = static_cast<int>(std::sqrt(target_area / aspect_ratio));
        int w = static_cast<int>(std::sqrt(target_area * aspect_ratio));

        if (w >= input.width || h >= input.height) {
            // Can't fit, return unmodified
            return output;
        }

        // Random position
        std::uniform_int_distribution<int> x_dist(0, input.width - w);
        std::uniform_int_distribution<int> y_dist(0, input.height - h);

        int x1 = x_dist(rng_);
        int y1 = y_dist(rng_);

        // Erase the rectangle
        for (int y = y1; y < y1 + h && y < input.height; ++y) {
            for (int x = x1; x < x1 + w && x < input.width; ++x) {
                size_t idx = (y * input.width + x) * input.channels;
                for (int c = 0; c < input.channels; ++c) {
                    output->data[idx + c] = value_;
                }
            }
        }

        return output;
    }

    const char* name() const override { return "RandomErasing"; }

private:
    float scale_min_;
    float scale_max_;
    float ratio_min_;
    float ratio_max_;
    uint8_t value_;
};

} // namespace transforms
} // namespace turboloader
