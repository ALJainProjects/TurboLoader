/**
 * @file resize_transform.hpp
 * @brief Resize transform with SIMD-accelerated interpolation
 *
 * Supports:
 * - Bilinear interpolation (SIMD-accelerated)
 * - Nearest neighbor interpolation
 * - Bicubic interpolation
 * - Arbitrary target dimensions
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <cmath>

namespace turboloader {
namespace transforms {

/**
 * @brief Interpolation mode for resizing
 */
enum class InterpolationMode {
    NEAREST,
    BILINEAR,
    BICUBIC,
    LANCZOS  // High-quality downsampling (windowed sinc, a=3)
};

/**
 * @brief Resize transform
 */
class ResizeTransform : public Transform {
public:
    ResizeTransform(int target_width, int target_height,
                   InterpolationMode mode = InterpolationMode::BILINEAR)
        : target_width_(target_width),
          target_height_(target_height),
          mode_(mode) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (input.width == target_width_ && input.height == target_height_) {
            // No resize needed, return copy
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        size_t output_size = target_width_ * target_height_ * input.channels;
        auto output = std::make_unique<ImageData>(
            new uint8_t[output_size],
            target_width_, target_height_, input.channels,
            target_width_ * input.channels, true
        );

        switch (mode_) {
            case InterpolationMode::NEAREST:
                resize_nearest(input, *output);
                break;
            case InterpolationMode::BILINEAR:
                resize_bilinear(input, *output);
                break;
            case InterpolationMode::BICUBIC:
                resize_bicubic(input, *output);
                break;
            case InterpolationMode::LANCZOS:
                resize_lanczos(input, *output);
                break;
        }

        return output;
    }

    const char* name() const override { return "Resize"; }

private:
    /**
     * @brief Nearest neighbor interpolation
     */
    void resize_nearest(const ImageData& input, ImageData& output) {
        float x_ratio = static_cast<float>(input.width) / target_width_;
        float y_ratio = static_cast<float>(input.height) / target_height_;

        for (int y = 0; y < target_height_; ++y) {
            int src_y = static_cast<int>(y * y_ratio);
            src_y = std::min(src_y, input.height - 1);

            for (int x = 0; x < target_width_; ++x) {
                int src_x = static_cast<int>(x * x_ratio);
                src_x = std::min(src_x, input.width - 1);

                size_t src_idx = (src_y * input.width + src_x) * input.channels;
                size_t dst_idx = (y * target_width_ + x) * output.channels;

                for (int c = 0; c < input.channels; ++c) {
                    output.data[dst_idx + c] = input.data[src_idx + c];
                }
            }
        }
    }

    /**
     * @brief Bilinear interpolation (SIMD-accelerated)
     */
    void resize_bilinear(const ImageData& input, ImageData& output) {
        float x_ratio = static_cast<float>(input.width - 1) / (target_width_ - 1);
        float y_ratio = static_cast<float>(input.height - 1) / (target_height_ - 1);

        for (int y = 0; y < target_height_; ++y) {
            float src_y = y * y_ratio;

            for (int x = 0; x < target_width_; ++x) {
                float src_x = x * x_ratio;

                size_t dst_idx = (y * target_width_ + x) * output.channels;

                for (int c = 0; c < input.channels; ++c) {
                    float val = simd::bilinear_interpolate(
                        input.data, input.width, input.height,
                        src_x, src_y, c, input.channels
                    );
                    output.data[dst_idx + c] = static_cast<uint8_t>(
                        simd::clamp(val, 0.0f, 255.0f)
                    );
                }
            }
        }
    }

    /**
     * @brief Bicubic interpolation weight (Catmull-Rom)
     */
    static float cubic_weight(float x) {
        x = std::abs(x);
        if (x <= 1.0f) {
            return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
        } else if (x < 2.0f) {
            return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
        }
        return 0.0f;
    }

    /**
     * @brief Bicubic interpolation
     */
    void resize_bicubic(const ImageData& input, ImageData& output) {
        float x_ratio = static_cast<float>(input.width) / target_width_;
        float y_ratio = static_cast<float>(input.height) / target_height_;

        for (int y = 0; y < target_height_; ++y) {
            float src_y = (y + 0.5f) * y_ratio - 0.5f;
            int y0 = static_cast<int>(std::floor(src_y));

            for (int x = 0; x < target_width_; ++x) {
                float src_x = (x + 0.5f) * x_ratio - 0.5f;
                int x0 = static_cast<int>(std::floor(src_x));

                float dx = src_x - x0;
                float dy = src_y - y0;

                size_t dst_idx = (y * target_width_ + x) * output.channels;

                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;
                    float weight_sum = 0.0f;

                    // 4x4 kernel
                    for (int ky = -1; ky <= 2; ++ky) {
                        int py = y0 + ky;
                        if (py < 0 || py >= input.height) continue;

                        float wy = cubic_weight(ky - dy);

                        for (int kx = -1; kx <= 2; ++kx) {
                            int px = x0 + kx;
                            if (px < 0 || px >= input.width) continue;

                            float wx = cubic_weight(kx - dx);
                            float w = wx * wy;

                            size_t src_idx = (py * input.width + px) * input.channels + c;
                            sum += input.data[src_idx] * w;
                            weight_sum += w;
                        }
                    }

                    if (weight_sum > 0.0f) {
                        sum /= weight_sum;
                    }

                    output.data[dst_idx + c] = static_cast<uint8_t>(
                        simd::clamp(sum, 0.0f, 255.0f)
                    );
                }
            }
        }
    }

    /**
     * @brief Lanczos kernel (windowed sinc filter, a=3)
     */
    static float lanczos_kernel(float x, int a = 3) {
        if (x == 0.0f) return 1.0f;
        if (std::abs(x) >= a) return 0.0f;

        constexpr float PI = 3.14159265358979323846f;
        float px = PI * x;
        return (a * std::sin(px) * std::sin(px / a)) / (px * px);
    }

    /**
     * @brief Lanczos interpolation (high-quality resampling)
     */
    void resize_lanczos(const ImageData& input, ImageData& output) {
        constexpr int a = 3;  // Lanczos window size
        float x_ratio = static_cast<float>(input.width) / target_width_;
        float y_ratio = static_cast<float>(input.height) / target_height_;

        for (int y = 0; y < target_height_; ++y) {
            float src_y = (y + 0.5f) * y_ratio - 0.5f;
            int y0 = static_cast<int>(std::floor(src_y));

            for (int x = 0; x < target_width_; ++x) {
                float src_x = (x + 0.5f) * x_ratio - 0.5f;
                int x0 = static_cast<int>(std::floor(src_x));

                float dx = src_x - x0;
                float dy = src_y - y0;

                size_t dst_idx = (y * target_width_ + x) * output.channels;

                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;
                    float weight_sum = 0.0f;

                    // Lanczos kernel: 6x6 window (a=3)
                    for (int ky = -a + 1; ky <= a; ++ky) {
                        int py = y0 + ky;
                        if (py < 0 || py >= input.height) continue;

                        float wy = lanczos_kernel(ky - dy, a);

                        for (int kx = -a + 1; kx <= a; ++kx) {
                            int px = x0 + kx;
                            if (px < 0 || px >= input.width) continue;

                            float wx = lanczos_kernel(kx - dx, a);
                            float w = wx * wy;

                            size_t src_idx = (py * input.width + px) * input.channels + c;
                            sum += input.data[src_idx] * w;
                            weight_sum += w;
                        }
                    }

                    if (weight_sum > 0.0f) {
                        sum /= weight_sum;
                    }

                    output.data[dst_idx + c] = static_cast<uint8_t>(
                        simd::clamp(sum, 0.0f, 255.0f)
                    );
                }
            }
        }
    }

    int target_width_;
    int target_height_;
    InterpolationMode mode_;
};

} // namespace transforms
} // namespace turboloader
