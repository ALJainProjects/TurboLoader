/**
 * @file rotation_transform.hpp
 * @brief Rotation transform with SIMD-accelerated interpolation
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <cmath>

namespace turboloader {
namespace transforms {

/**
 * @brief Random rotation transform
 */
class RandomRotationTransform : public RandomTransform {
public:
    /**
     * @param degrees Maximum rotation angle in degrees (will rotate in [-degrees, +degrees])
     * @param expand If true, expand output to fit rotated image
     * @param fill Fill value for empty pixels
     */
    RandomRotationTransform(float degrees,
                           bool expand = false,
                           uint8_t fill = 0,
                           unsigned seed = std::random_device{}())
        : RandomTransform(1.0f, seed),
          degrees_(degrees),
          expand_(expand),
          fill_(fill) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        std::uniform_real_distribution<float> dist(-degrees_, degrees_);
        float angle = dist(rng_);

        return rotate_image(input, angle);
    }

    const char* name() const override { return "RandomRotation"; }

private:
    std::unique_ptr<ImageData> rotate_image(const ImageData& input, float angle_deg) {
        float angle_rad = angle_deg * 3.14159265f / 180.0f;
        float cos_a = std::cos(angle_rad);
        float sin_a = std::sin(angle_rad);

        int out_width = input.width;
        int out_height = input.height;

        if (expand_) {
            // Calculate expanded size
            float corners_x[] = {0, float(input.width), 0, float(input.width)};
            float corners_y[] = {0, 0, float(input.height), float(input.height)};

            float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
            for (int i = 0; i < 4; ++i) {
                float x = corners_x[i] * cos_a - corners_y[i] * sin_a;
                float y = corners_x[i] * sin_a + corners_y[i] * cos_a;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }

            out_width = static_cast<int>(std::ceil(max_x - min_x));
            out_height = static_cast<int>(std::ceil(max_y - min_y));
        }

        size_t out_size = out_width * out_height * input.channels;
        auto output = std::make_unique<ImageData>(
            new uint8_t[out_size],
            out_width, out_height, input.channels,
            out_width * input.channels, true
        );

        // Fill with background
        std::memset(output->data, fill_, out_size);

        // Rotation center
        float cx = input.width / 2.0f;
        float cy = input.height / 2.0f;
        float out_cx = out_width / 2.0f;
        float out_cy = out_height / 2.0f;

        // Inverse rotation
        for (int y = 0; y < out_height; ++y) {
            for (int x = 0; x < out_width; ++x) {
                // Map output pixel to input coordinates
                float dx = x - out_cx;
                float dy = y - out_cy;

                float src_x = dx * cos_a + dy * sin_a + cx;
                float src_y = -dx * sin_a + dy * cos_a + cy;

                if (src_x >= 0 && src_x < input.width - 1 &&
                    src_y >= 0 && src_y < input.height - 1) {

                    size_t dst_idx = (y * out_width + x) * input.channels;

                    for (int c = 0; c < input.channels; ++c) {
                        float val = simd::bilinear_interpolate(
                            input.data, input.width, input.height,
                            src_x, src_y, c, input.channels
                        );
                        output->data[dst_idx + c] = static_cast<uint8_t>(
                            simd::clamp(val, 0.0f, 255.0f)
                        );
                    }
                }
            }
        }

        return output;
    }

    float degrees_;
    bool expand_;
    uint8_t fill_;
};

} // namespace transforms
} // namespace turboloader
