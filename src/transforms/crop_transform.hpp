/**
 * @file crop_transform.hpp
 * @brief Crop transforms (random and center)
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Padding mode for crops
 */
enum class PaddingMode {
    CONSTANT,  // Fill with constant value
    EDGE,      // Replicate edge pixels
    REFLECT    // Reflect around edges
};

// Forward declaration
inline std::unique_ptr<ImageData> crop_region(const ImageData& input,
                                              int start_x, int start_y,
                                              int crop_width, int crop_height);

/**
 * @brief Center crop transform
 */
class CenterCropTransform : public Transform {
public:
    CenterCropTransform(int crop_width, int crop_height)
        : crop_width_(crop_width), crop_height_(crop_height) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        int start_x = (input.width - crop_width_) / 2;
        int start_y = (input.height - crop_height_) / 2;

        return crop_region(input, start_x, start_y, crop_width_, crop_height_);
    }

    const char* name() const override { return "CenterCrop"; }

private:
    int crop_width_;
    int crop_height_;
};

/**
 * @brief Random crop transform
 */
class RandomCropTransform : public RandomTransform {
public:
    RandomCropTransform(int crop_width, int crop_height,
                       int padding = 0,
                       PaddingMode pad_mode = PaddingMode::CONSTANT,
                       uint8_t pad_value = 0,
                       unsigned seed = std::random_device{}())
        : RandomTransform(1.0f, seed),
          crop_width_(crop_width),
          crop_height_(crop_height),
          padding_(padding),
          pad_mode_(pad_mode),
          pad_value_(pad_value) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // Apply padding if needed
        std::unique_ptr<ImageData> padded;
        const ImageData* work_image = &input;

        if (padding_ > 0) {
            padded = apply_padding(input);
            work_image = padded.get();
        }

        // Random crop position
        std::uniform_int_distribution<int> x_dist(0, work_image->width - crop_width_);
        std::uniform_int_distribution<int> y_dist(0, work_image->height - crop_height_);

        int start_x = x_dist(rng_);
        int start_y = y_dist(rng_);

        return crop_region(*work_image, start_x, start_y, crop_width_, crop_height_);
    }

    const char* name() const override { return "RandomCrop"; }

private:
    std::unique_ptr<ImageData> apply_padding(const ImageData& input) {
        int new_width = input.width + 2 * padding_;
        int new_height = input.height + 2 * padding_;
        size_t new_size = new_width * new_height * input.channels;

        auto output = std::make_unique<ImageData>(
            new uint8_t[new_size],
            new_width, new_height, input.channels,
            new_width * input.channels, true
        );

        // Fill with padding
        if (pad_mode_ == PaddingMode::CONSTANT) {
            std::memset(output->data, pad_value_, new_size);
        }

        // Copy original image to center
        for (int y = 0; y < input.height; ++y) {
            int dst_y = y + padding_;
            const uint8_t* src_row = input.data + y * input.stride;
            uint8_t* dst_row = output->data + dst_y * output->stride + padding_ * input.channels;

            if (pad_mode_ == PaddingMode::CONSTANT) {
                std::memcpy(dst_row, src_row, input.width * input.channels);
            } else if (pad_mode_ == PaddingMode::EDGE) {
                // Fill left padding
                for (int p = 0; p < padding_; ++p) {
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[(dst_y * new_width + p) * input.channels + c] = src_row[c];
                    }
                }
                // Copy center
                std::memcpy(dst_row, src_row, input.width * input.channels);
                // Fill right padding
                for (int p = 0; p < padding_; ++p) {
                    int src_idx = (input.width - 1) * input.channels;
                    int dst_idx = (dst_y * new_width + padding_ + input.width + p) * input.channels;
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[dst_idx + c] = src_row[src_idx + c];
                    }
                }
            } else if (pad_mode_ == PaddingMode::REFLECT) {
                // Left padding (reflect)
                for (int p = 0; p < padding_; ++p) {
                    int reflect_x = std::min(padding_ - p, input.width - 1);
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[(dst_y * new_width + p) * input.channels + c] =
                            src_row[reflect_x * input.channels + c];
                    }
                }
                // Copy center
                std::memcpy(dst_row, src_row, input.width * input.channels);
                // Right padding (reflect)
                for (int p = 0; p < padding_; ++p) {
                    int reflect_x = std::max(0, input.width - 2 - p);
                    int dst_idx = (dst_y * new_width + padding_ + input.width + p) * input.channels;
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[dst_idx + c] = src_row[reflect_x * input.channels + c];
                    }
                }
            }
        }

        // Fill top and bottom padding for EDGE and REFLECT modes
        if (pad_mode_ == PaddingMode::EDGE) {
            // Top padding
            for (int p = 0; p < padding_; ++p) {
                std::memcpy(
                    output->data + p * output->stride,
                    output->data + padding_ * output->stride,
                    output->stride
                );
            }
            // Bottom padding
            for (int p = 0; p < padding_; ++p) {
                std::memcpy(
                    output->data + (padding_ + input.height + p) * output->stride,
                    output->data + (padding_ + input.height - 1) * output->stride,
                    output->stride
                );
            }
        } else if (pad_mode_ == PaddingMode::REFLECT) {
            // Top padding
            for (int p = 0; p < padding_; ++p) {
                int reflect_y = std::min(padding_ - p, input.height - 1);
                std::memcpy(
                    output->data + p * output->stride,
                    output->data + (padding_ + reflect_y) * output->stride,
                    output->stride
                );
            }
            // Bottom padding
            for (int p = 0; p < padding_; ++p) {
                int reflect_y = std::max(0, input.height - 2 - p);
                std::memcpy(
                    output->data + (padding_ + input.height + p) * output->stride,
                    output->data + (padding_ + reflect_y) * output->stride,
                    output->stride
                );
            }
        }

        return output;
    }

    int crop_width_;
    int crop_height_;
    int padding_;
    PaddingMode pad_mode_;
    uint8_t pad_value_;
};

/**
 * @brief Helper function to crop a region from an image
 */
inline std::unique_ptr<ImageData> crop_region(const ImageData& input,
                                              int start_x, int start_y,
                                              int crop_width, int crop_height) {
    // Clamp to valid range
    start_x = std::max(0, std::min(start_x, input.width - crop_width));
    start_y = std::max(0, std::min(start_y, input.height - crop_height));
    crop_width = std::min(crop_width, input.width - start_x);
    crop_height = std::min(crop_height, input.height - start_y);

    size_t output_size = crop_width * crop_height * input.channels;
    auto output = std::make_unique<ImageData>(
        new uint8_t[output_size],
        crop_width, crop_height, input.channels,
        crop_width * input.channels, true
    );

    // Copy rows
    for (int y = 0; y < crop_height; ++y) {
        const uint8_t* src_row = input.data + (start_y + y) * input.stride +
                                 start_x * input.channels;
        uint8_t* dst_row = output->data + y * output->stride;
        std::memcpy(dst_row, src_row, crop_width * input.channels);
    }

    return output;
}

} // namespace transforms
} // namespace turboloader
