/**
 * @file affine_transform.hpp
 * @brief Affine transformation (rotation, translation, scale, shear)
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <cmath>

namespace turboloader {
namespace transforms {

/**
 * @brief Random affine transform
 */
class RandomAffineTransform : public RandomTransform {
public:
    RandomAffineTransform(float degrees = 0.0f,
                         float translate_x = 0.0f,
                         float translate_y = 0.0f,
                         float scale_min = 1.0f,
                         float scale_max = 1.0f,
                         float shear = 0.0f,
                         uint8_t fill = 0,
                         unsigned seed = std::random_device{}())
        : RandomTransform(1.0f, seed),
          degrees_(degrees),
          translate_x_(translate_x),
          translate_y_(translate_y),
          scale_min_(scale_min),
          scale_max_(scale_max),
          shear_(shear),
          fill_(fill) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // Sample random parameters
        std::uniform_real_distribution<float> angle_dist(-degrees_, degrees_);
        float angle = angle_dist(rng_);

        std::uniform_real_distribution<float> tx_dist(-translate_x_, translate_x_);
        std::uniform_real_distribution<float> ty_dist(-translate_y_, translate_y_);
        float tx = tx_dist(rng_) * input.width;
        float ty = ty_dist(rng_) * input.height;

        std::uniform_real_distribution<float> scale_dist(scale_min_, scale_max_);
        float scale = scale_dist(rng_);

        std::uniform_real_distribution<float> shear_dist(-shear_, shear_);
        float shear = shear_dist(rng_);

        return apply_affine(input, angle, tx, ty, scale, shear);
    }

    const char* name() const override { return "RandomAffine"; }

private:
    std::unique_ptr<ImageData> apply_affine(const ImageData& input,
                                           float angle, float tx, float ty,
                                           float scale, float shear) {
        size_t out_size = input.width * input.height * input.channels;
        auto output = std::make_unique<ImageData>(
            new uint8_t[out_size],
            input.width, input.height, input.channels,
            input.width * input.channels, true
        );

        std::memset(output->data, fill_, out_size);

        // Build affine matrix
        float angle_rad = angle * 3.14159265f / 180.0f;
        float shear_rad = shear * 3.14159265f / 180.0f;
        float cos_a = std::cos(angle_rad);
        float sin_a = std::sin(angle_rad);
        float tan_shear = std::tan(shear_rad);

        // Center of image
        float cx = input.width / 2.0f;
        float cy = input.height / 2.0f;

        // Combined transformation matrix
        // M = T(cx,cy) * R(angle) * S(scale) * Sh(shear) * T(-cx,-cy)
        float m00 = scale * (cos_a - tan_shear * sin_a);
        float m01 = scale * (-sin_a - tan_shear * cos_a);
        float m10 = scale * sin_a;
        float m11 = scale * cos_a;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                // Transform output -> input coordinates
                float dx = x - cx;
                float dy = y - cy;

                // Apply inverse transformation
                float det = m00 * m11 - m01 * m10;
                if (std::abs(det) < 1e-6f) continue;

                float inv_det = 1.0f / det;
                float inv_m00 = m11 * inv_det;
                float inv_m01 = -m01 * inv_det;
                float inv_m10 = -m10 * inv_det;
                float inv_m11 = m00 * inv_det;

                float src_x = inv_m00 * dx + inv_m01 * dy + cx - tx;
                float src_y = inv_m10 * dx + inv_m11 * dy + cy - ty;

                if (src_x >= 0 && src_x < input.width - 1 &&
                    src_y >= 0 && src_y < input.height - 1) {

                    size_t dst_idx = (y * input.width + x) * input.channels;

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
    float translate_x_;
    float translate_y_;
    float scale_min_;
    float scale_max_;
    float shear_;
    uint8_t fill_;
};

} // namespace transforms
} // namespace turboloader
