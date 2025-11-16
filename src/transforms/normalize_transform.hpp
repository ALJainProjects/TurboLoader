/**
 * @file normalize_transform.hpp
 * @brief Normalize transform with SIMD acceleration
 *
 * Performs per-channel mean/std normalization with optional uint8 -> float32 conversion.
 * SIMD-vectorized for maximum performance.
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <vector>
#include <stdexcept>

namespace turboloader {
namespace transforms {

/**
 * @brief Normalize transform (per-channel mean/std normalization)
 */
class NormalizeTransform : public Transform {
public:
    /**
     * @brief Constructor
     * @param mean Per-channel mean values (must match number of channels)
     * @param std Per-channel std values (must match number of channels)
     * @param to_float Convert to float32 output (default: false, keeps uint8)
     */
    NormalizeTransform(const std::vector<float>& mean,
                      const std::vector<float>& std,
                      bool to_float = false)
        : mean_(mean), std_(std), to_float_(to_float) {

        if (mean_.size() != std_.size()) {
            throw std::invalid_argument("Mean and std must have same size");
        }

        for (float s : std_) {
            if (s <= 0.0f) {
                throw std::invalid_argument("Std values must be positive");
            }
        }
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (input.channels != static_cast<int>(mean_.size())) {
            throw std::runtime_error(
                "Channel mismatch: image has " + std::to_string(input.channels) +
                " channels but normalize expects " + std::to_string(mean_.size())
            );
        }

        if (to_float_) {
            return apply_with_float_output(input);
        } else {
            return apply_with_uint8_output(input);
        }
    }

    const char* name() const override { return "Normalize"; }

private:
    /**
     * @brief Apply normalization with float32 output
     */
    std::unique_ptr<ImageData> apply_with_float_output(const ImageData& input) {
        size_t num_pixels = input.width * input.height;
        size_t output_size = num_pixels * input.channels * sizeof(float);

        auto output_data = new uint8_t[output_size];
        float* output_float = reinterpret_cast<float*>(output_data);

        auto output = std::make_unique<ImageData>(
            output_data,
            input.width, input.height, input.channels,
            input.width * input.channels * sizeof(float), true
        );

        // Process per channel
        for (int c = 0; c < input.channels; ++c) {
            std::vector<float> temp(num_pixels);

            // Extract channel and convert to float [0,1]
            for (size_t i = 0; i < num_pixels; ++i) {
                temp[i] = input.data[i * input.channels + c] / 255.0f;
            }

            // Normalize with mean/std (SIMD-accelerated)
            std::vector<float> normalized(num_pixels);
            simd::normalize_f32(temp.data(), normalized.data(),
                               mean_[c], std_[c], num_pixels);

            // Write to output (interleaved)
            for (size_t i = 0; i < num_pixels; ++i) {
                output_float[i * input.channels + c] = normalized[i];
            }
        }

        return output;
    }

    /**
     * @brief Apply normalization with uint8 output (scaled back to [0,255])
     */
    std::unique_ptr<ImageData> apply_with_uint8_output(const ImageData& input) {
        size_t output_size = input.width * input.height * input.channels;

        auto output = std::make_unique<ImageData>(
            new uint8_t[output_size],
            input.width, input.height, input.channels,
            input.width * input.channels, true
        );

        size_t num_pixels = input.width * input.height;

        // Process per channel
        for (int c = 0; c < input.channels; ++c) {
            std::vector<float> temp(num_pixels);

            // Extract channel and convert to float [0,1]
            for (size_t i = 0; i < num_pixels; ++i) {
                temp[i] = input.data[i * input.channels + c] / 255.0f;
            }

            // Normalize with mean/std
            std::vector<float> normalized(num_pixels);
            simd::normalize_f32(temp.data(), normalized.data(),
                               mean_[c], std_[c], num_pixels);

            // Scale back to [0,255] range and convert to uint8
            // Typical normalized values are in [-3, 3], we'll map to [0,255]
            // using (val + 3) / 6 * 255 approximation
            for (size_t i = 0; i < num_pixels; ++i) {
                // Simple clamp to [0,1] after denormalization
                float val = normalized[i] * std_[c] + mean_[c];
                val = std::max(0.0f, std::min(1.0f, val));
                output->data[i * input.channels + c] = static_cast<uint8_t>(val * 255.0f);
            }
        }

        return output;
    }

    std::vector<float> mean_;
    std::vector<float> std_;
    bool to_float_;
};

/**
 * @brief ImageNet normalization (common preset)
 */
class ImageNetNormalize : public NormalizeTransform {
public:
    ImageNetNormalize(bool to_float = false)
        : NormalizeTransform(
            {0.485f, 0.456f, 0.406f},  // ImageNet mean
            {0.229f, 0.224f, 0.225f},  // ImageNet std
            to_float
        ) {}
};

} // namespace transforms
} // namespace turboloader
