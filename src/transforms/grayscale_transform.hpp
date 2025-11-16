/**
 * @file grayscale_transform.hpp
 * @brief Grayscale conversion with SIMD acceleration
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Grayscale transform (RGB -> Grayscale)
 */
class GrayscaleTransform : public Transform {
public:
    /**
     * @brief Constructor
     * @param num_output_channels 1 for grayscale, 3 to keep 3 channels with replicated values
     */
    explicit GrayscaleTransform(int num_output_channels = 1)
        : num_output_channels_(num_output_channels) {
        if (num_output_channels != 1 && num_output_channels != 3) {
            throw std::invalid_argument("num_output_channels must be 1 or 3");
        }
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (input.channels == 1) {
            // Already grayscale
            if (num_output_channels_ == 1) {
                // Just copy
                auto output = std::make_unique<ImageData>(
                    new uint8_t[input.size_bytes()],
                    input.width, input.height, input.channels, input.stride, true
                );
                std::memcpy(output->data, input.data, input.size_bytes());
                return output;
            } else {
                // Replicate to 3 channels
                return replicate_to_3channels(input);
            }
        }

        if (input.channels != 3) {
            throw std::runtime_error("Grayscale transform only supports 1 or 3 channel inputs");
        }

        size_t num_pixels = input.width * input.height;

        if (num_output_channels_ == 1) {
            // RGB -> single channel grayscale
            auto output = std::make_unique<ImageData>(
                new uint8_t[num_pixels],
                input.width, input.height, 1,
                input.width, true
            );

            simd::rgb_to_grayscale(input.data, output->data, num_pixels);
            return output;
        } else {
            // RGB -> 3 channel grayscale (replicated)
            auto output = std::make_unique<ImageData>(
                new uint8_t[num_pixels * 3],
                input.width, input.height, 3,
                input.width * 3, true
            );

            std::vector<uint8_t> temp_gray(num_pixels);
            simd::rgb_to_grayscale(input.data, temp_gray.data(), num_pixels);

            // Replicate to 3 channels
            for (size_t i = 0; i < num_pixels; ++i) {
                output->data[i * 3] = temp_gray[i];
                output->data[i * 3 + 1] = temp_gray[i];
                output->data[i * 3 + 2] = temp_gray[i];
            }

            return output;
        }
    }

    const char* name() const override { return "Grayscale"; }

private:
    std::unique_ptr<ImageData> replicate_to_3channels(const ImageData& input) {
        size_t num_pixels = input.width * input.height;
        auto output = std::make_unique<ImageData>(
            new uint8_t[num_pixels * 3],
            input.width, input.height, 3,
            input.width * 3, true
        );

        for (size_t i = 0; i < num_pixels; ++i) {
            uint8_t gray = input.data[i];
            output->data[i * 3] = gray;
            output->data[i * 3 + 1] = gray;
            output->data[i * 3 + 2] = gray;
        }

        return output;
    }

    int num_output_channels_;
};

} // namespace transforms
} // namespace turboloader
