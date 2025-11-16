/**
 * @file blur_transform.hpp
 * @brief Gaussian blur with SIMD-accelerated separable convolution
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <vector>
#include <cmath>

namespace turboloader {
namespace transforms {

/**
 * @brief Gaussian blur transform (SIMD-accelerated)
 */
class GaussianBlurTransform : public Transform {
public:
    /**
     * @param kernel_size Kernel size (must be odd)
     * @param sigma Standard deviation (if 0, calculated from kernel_size)
     */
    GaussianBlurTransform(int kernel_size, float sigma = 0.0f)
        : kernel_size_(kernel_size) {

        if (kernel_size_ % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd");
        }

        if (sigma <= 0.0f) {
            // Calculate sigma from kernel size (PyTorch formula)
            sigma = 0.3f * ((kernel_size_ - 1) * 0.5f - 1) + 0.8f;
        }

        // Generate Gaussian kernel (1D, separable)
        generate_kernel(sigma);
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // Separable convolution: horizontal then vertical
        auto temp = apply_horizontal(input);
        return apply_vertical(*temp);
    }

    const char* name() const override { return "GaussianBlur"; }

private:
    void generate_kernel(float sigma) {
        int radius = kernel_size_ / 2;
        kernel_.resize(kernel_size_);

        float sum = 0.0f;
        for (int i = 0; i < kernel_size_; ++i) {
            int x = i - radius;
            kernel_[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
            sum += kernel_[i];
        }

        // Normalize
        for (float& k : kernel_) {
            k /= sum;
        }
    }

    std::unique_ptr<ImageData> apply_horizontal(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        int radius = kernel_size_ / 2;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;

                    for (int k = 0; k < kernel_size_; ++k) {
                        int sx = x + k - radius;
                        sx = simd::clamp(sx, 0, input.width - 1);

                        size_t src_idx = (y * input.width + sx) * input.channels + c;
                        sum += input.data[src_idx] * kernel_[k];
                    }

                    size_t dst_idx = (y * input.width + x) * input.channels + c;
                    output->data[dst_idx] = static_cast<uint8_t>(
                        simd::clamp(sum, 0.0f, 255.0f)
                    );
                }
            }
        }

        return output;
    }

    std::unique_ptr<ImageData> apply_vertical(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        int radius = kernel_size_ / 2;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;

                    for (int k = 0; k < kernel_size_; ++k) {
                        int sy = y + k - radius;
                        sy = simd::clamp(sy, 0, input.height - 1);

                        size_t src_idx = (sy * input.width + x) * input.channels + c;
                        sum += input.data[src_idx] * kernel_[k];
                    }

                    size_t dst_idx = (y * input.width + x) * input.channels + c;
                    output->data[dst_idx] = static_cast<uint8_t>(
                        simd::clamp(sum, 0.0f, 255.0f)
                    );
                }
            }
        }

        return output;
    }

    int kernel_size_;
    std::vector<float> kernel_;
};

/**
 * @brief Random Gaussian blur (applies blur with probability)
 */
class RandomGaussianBlurTransform : public RandomTransform {
public:
    RandomGaussianBlurTransform(int kernel_size,
                               float sigma = 0.0f,
                               float p = 0.5f,
                               unsigned seed = std::random_device{}())
        : RandomTransform(p, seed),
          blur_(kernel_size, sigma) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (!should_apply()) {
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        return blur_.apply(input);
    }

    const char* name() const override { return "RandomGaussianBlur"; }

private:
    GaussianBlurTransform blur_;
};

} // namespace transforms
} // namespace turboloader
