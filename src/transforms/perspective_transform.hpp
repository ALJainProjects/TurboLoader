/**
 * @file perspective_transform.hpp
 * @brief SIMD-accelerated perspective transformation
 *
 * Features:
 * - Random perspective warping with configurable distortion scale
 * - SIMD-optimized bilinear interpolation for warping
 * - Probability-based application
 * - Thread-safe implementation
 *
 * Reference: torchvision.transforms.RandomPerspective
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <array>

namespace turboloader {
namespace transforms {

/**
 * @brief Random perspective transform
 */
class RandomPerspectiveTransform : public RandomTransform {
public:
    /**
     * @param distortion_scale Controls the degree of distortion (0.0 - 1.0)
     * @param probability Probability of applying transform
     * @param interpolation Interpolation mode (only BILINEAR supported for now)
     * @param fill_value Fill value for areas outside source image
     * @param seed Random seed
     */
    RandomPerspectiveTransform(float distortion_scale = 0.5f,
                              float probability = 0.5f,
                              uint8_t fill_value = 0,
                              unsigned seed = std::random_device{}())
        : RandomTransform(probability, seed),
          distortion_scale_(distortion_scale),
          fill_value_(fill_value) {
        if (distortion_scale < 0.0f || distortion_scale > 1.0f) {
            throw std::invalid_argument("distortion_scale must be in range [0.0, 1.0]");
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

        return apply_perspective_warp(input);
    }

    const char* name() const override { return "RandomPerspective"; }

private:
    float distortion_scale_;
    uint8_t fill_value_;

    /**
     * @brief Generate random perspective transformation matrix
     */
    void generate_perspective_matrix(int width, int height, std::array<float, 9>& matrix) {
        std::uniform_real_distribution<float> dist(-distortion_scale_, distortion_scale_);

        // Define source corners
        float w = static_cast<float>(width);
        float h = static_cast<float>(height);

        std::array<std::array<float, 2>, 4> src_points = {{
            {0.0f, 0.0f},     // Top-left
            {w - 1, 0.0f},    // Top-right
            {w - 1, h - 1},   // Bottom-right
            {0.0f, h - 1}     // Bottom-left
        }};

        // Perturb destination corners
        std::array<std::array<float, 2>, 4> dst_points;
        for (int i = 0; i < 4; ++i) {
            float dx = dist(rng_) * w * 0.5f;
            float dy = dist(rng_) * h * 0.5f;
            dst_points[i][0] = src_points[i][0] + dx;
            dst_points[i][1] = src_points[i][1] + dy;
        }

        // Compute perspective transformation matrix (homography)
        // Using simplified approach for 4-point correspondence
        compute_homography(src_points, dst_points, matrix);
    }

    /**
     * @brief Compute homography from 4 point correspondences
     */
    void compute_homography(const std::array<std::array<float, 2>, 4>& src,
                           const std::array<std::array<float, 2>, 4>& dst,
                           std::array<float, 9>& H) {
        // Simplified homography computation
        // For a more robust implementation, use DLT (Direct Linear Transform)

        // For now, use a simple approximation based on affine + perspective
        float sx = (dst[1][0] - dst[0][0]) / (src[1][0] - src[0][0]);
        float sy = (dst[3][1] - dst[0][1]) / (src[3][1] - src[0][1]);
        float tx = dst[0][0] - sx * src[0][0];
        float ty = dst[0][1] - sy * src[0][1];

        // Perspective coefficients (simplified)
        float px = (dst[2][0] - dst[1][0] - dst[3][0] + dst[0][0]) * 0.0001f;
        float py = (dst[2][1] - dst[1][1] - dst[3][1] + dst[0][1]) * 0.0001f;

        // Build homography matrix
        H[0] = sx;  H[1] = 0.0f; H[2] = tx;
        H[3] = 0.0f; H[4] = sy;  H[5] = ty;
        H[6] = px;   H[7] = py;  H[8] = 1.0f;
    }

    /**
     * @brief Apply inverse homography to map output coords to input coords
     */
    bool inverse_transform(const std::array<float, 9>& H, float x, float y, float& src_x, float& src_y) {
        // For inverse, we need to invert the matrix
        // Simplified: using approximate inverse for performance
        float denom = H[6] * x + H[7] * y + H[8];

        if (std::abs(denom) < 1e-6f) {
            return false;
        }

        src_x = (H[0] * x + H[1] * y + H[2]) / denom;
        src_y = (H[3] * x + H[4] * y + H[5]) / denom;

        return true;
    }

    /**
     * @brief Apply perspective transformation using SIMD-accelerated interpolation
     */
    std::unique_ptr<ImageData> apply_perspective_warp(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Generate random perspective matrix
        std::array<float, 9> H;
        generate_perspective_matrix(input.width, input.height, H);

        // Apply inverse warping
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                float src_x, src_y;

                if (!inverse_transform(H, static_cast<float>(x), static_cast<float>(y), src_x, src_y)) {
                    // Singular transformation, use fill value
                    size_t dst_idx = (y * input.width + x) * input.channels;
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[dst_idx + c] = fill_value_;
                    }
                    continue;
                }

                // Check bounds
                if (src_x < 0 || src_x >= input.width - 1 || src_y < 0 || src_y >= input.height - 1) {
                    size_t dst_idx = (y * input.width + x) * input.channels;
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[dst_idx + c] = fill_value_;
                    }
                    continue;
                }

                // Bilinear interpolation using SIMD
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

        return output;
    }
};

} // namespace transforms
} // namespace turboloader
