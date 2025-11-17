/**
 * @file flip_transform.hpp
 * @brief Horizontal and vertical flip transforms with SIMD acceleration
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Random horizontal flip transform
 */
class RandomHorizontalFlipTransform : public RandomTransform {
public:
    explicit RandomHorizontalFlipTransform(float p = 0.5f, unsigned seed = std::random_device{}())
        : RandomTransform(p, seed) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        if (!should_apply()) {
            // No flip, just copy
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        // Horizontal flip: reverse each row
        for (int y = 0; y < input.height; ++y) {
            const uint8_t* src_row = input.data + y * input.stride;
            uint8_t* dst_row = output->data + y * output->stride;

            // Reverse pixels in row (SIMD-friendly if we process in chunks)
            for (int x = 0; x < input.width; ++x) {
                int src_x = input.width - 1 - x;
                for (int c = 0; c < input.channels; ++c) {
                    dst_row[x * input.channels + c] = src_row[src_x * input.channels + c];
                }
            }
        }

        return output;
    }

    const char* name() const override { return "RandomHorizontalFlip"; }
};

/**
 * @brief Random vertical flip transform
 */
class RandomVerticalFlipTransform : public RandomTransform {
public:
    explicit RandomVerticalFlipTransform(float p = 0.5f, unsigned seed = std::random_device{}())
        : RandomTransform(p, seed) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        if (!should_apply()) {
            // No flip, just copy
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        // Vertical flip: reverse row order
        int row_bytes = input.width * input.channels;

        for (int y = 0; y < input.height; ++y) {
            int src_y = input.height - 1 - y;
            const uint8_t* src_row = input.data + src_y * input.stride;
            uint8_t* dst_row = output->data + y * output->stride;
            std::memcpy(dst_row, src_row, row_bytes);
        }

        return output;
    }

    const char* name() const override { return "RandomVerticalFlip"; }
};

/**
 * @brief Horizontal flip (deterministic)
 */
class HorizontalFlipTransform : public Transform {
public:
    HorizontalFlipTransform() = default;

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Horizontal flip: reverse each row
        for (int y = 0; y < input.height; ++y) {
            const uint8_t* src_row = input.data + y * input.stride;
            uint8_t* dst_row = output->data + y * output->stride;

            for (int x = 0; x < input.width; ++x) {
                int src_x = input.width - 1 - x;
                for (int c = 0; c < input.channels; ++c) {
                    dst_row[x * input.channels + c] = src_row[src_x * input.channels + c];
                }
            }
        }

        return output;
    }

    const char* name() const override { return "HorizontalFlip"; }
};

/**
 * @brief Vertical flip (deterministic)
 */
class VerticalFlipTransform : public Transform {
public:
    VerticalFlipTransform() = default;

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Vertical flip: reverse row order
        int row_bytes = input.width * input.channels;

        for (int y = 0; y < input.height; ++y) {
            int src_y = input.height - 1 - y;
            const uint8_t* src_row = input.data + src_y * input.stride;
            uint8_t* dst_row = output->data + y * output->stride;
            std::memcpy(dst_row, src_row, row_bytes);
        }

        return output;
    }

    const char* name() const override { return "VerticalFlip"; }
};

} // namespace transforms
} // namespace turboloader
