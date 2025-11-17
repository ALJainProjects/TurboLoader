/**
 * @file pad_transform.hpp
 * @brief Padding transform with different modes
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Pad transform
 */
class PadTransform : public Transform {
public:
    PadTransform(int padding,
                PaddingMode mode = PaddingMode::CONSTANT,
                uint8_t value = 0)
        : padding_(padding), mode_(mode), value_(value) {}

    PadTransform(int pad_left, int pad_top, int pad_right, int pad_bottom,
                PaddingMode mode = PaddingMode::CONSTANT,
                uint8_t value = 0)
        : pad_left_(pad_left), pad_top_(pad_top),
          pad_right_(pad_right), pad_bottom_(pad_bottom),
          padding_(0), mode_(mode), value_(value) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        int left = (padding_ > 0) ? padding_ : pad_left_;
        int top = (padding_ > 0) ? padding_ : pad_top_;
        int right = (padding_ > 0) ? padding_ : pad_right_;
        int bottom = (padding_ > 0) ? padding_ : pad_bottom_;

        int new_width = input.width + left + right;
        int new_height = input.height + top + bottom;
        size_t new_size = new_width * new_height * input.channels;

        auto output = std::make_unique<ImageData>(
            new uint8_t[new_size],
            new_width, new_height, input.channels,
            new_width * input.channels, true
        );

        // Initialize with constant value if needed
        if (mode_ == PaddingMode::CONSTANT) {
            std::memset(output->data, value_, new_size);
        }

        // Copy center region
        for (int y = 0; y < input.height; ++y) {
            const uint8_t* src_row = input.data + y * input.stride;
            uint8_t* dst_row = output->data + (y + top) * output->stride +
                              left * input.channels;
            std::memcpy(dst_row, src_row, input.width * input.channels);
        }

        // Apply edge or reflect padding
        if (mode_ == PaddingMode::EDGE) {
            apply_edge_padding(*output, input, left, top, right, bottom);
        } else if (mode_ == PaddingMode::REFLECT) {
            apply_reflect_padding(*output, input, left, top, right, bottom);
        }

        return output;
    }

    const char* name() const override { return "Pad"; }

private:
    void apply_edge_padding(ImageData& output, const ImageData& input,
                           int left, int top, int right, int bottom) {
        // Top and bottom edges
        for (int p = 0; p < top; ++p) {
            std::memcpy(
                output.data + p * output.stride,
                output.data + top * output.stride,
                output.stride
            );
        }
        for (int p = 0; p < bottom; ++p) {
            std::memcpy(
                output.data + (top + input.height + p) * output.stride,
                output.data + (top + input.height - 1) * output.stride,
                output.stride
            );
        }

        // Left and right edges
        for (int y = 0; y < output.height; ++y) {
            uint8_t* row = output.data + y * output.stride;
            // Left edge
            for (int p = 0; p < left; ++p) {
                for (int c = 0; c < input.channels; ++c) {
                    row[p * input.channels + c] = row[left * input.channels + c];
                }
            }
            // Right edge
            for (int p = 0; p < right; ++p) {
                int src_x = left + input.width - 1;
                int dst_x = left + input.width + p;
                for (int c = 0; c < input.channels; ++c) {
                    row[dst_x * input.channels + c] = row[src_x * input.channels + c];
                }
            }
        }
    }

    void apply_reflect_padding(ImageData& output, const ImageData& input,
                              int left, int top, int right, int bottom) {
        // Similar to edge but with reflection
        // Top
        for (int p = 0; p < top; ++p) {
            int reflect_y = std::min(top - p, input.height - 1) + top;
            std::memcpy(
                output.data + p * output.stride,
                output.data + reflect_y * output.stride,
                output.stride
            );
        }
        // Bottom
        for (int p = 0; p < bottom; ++p) {
            int reflect_y = std::max(0, input.height - 2 - p) + top;
            std::memcpy(
                output.data + (top + input.height + p) * output.stride,
                output.data + reflect_y * output.stride,
                output.stride
            );
        }

        // Left and right with reflection
        for (int y = 0; y < output.height; ++y) {
            uint8_t* row = output.data + y * output.stride;
            for (int p = 0; p < left; ++p) {
                int reflect_x = std::min(left - p, input.width - 1) + left;
                for (int c = 0; c < input.channels; ++c) {
                    row[p * input.channels + c] = row[reflect_x * input.channels + c];
                }
            }
            for (int p = 0; p < right; ++p) {
                int reflect_x = std::max(0, input.width - 2 - p) + left;
                int dst_x = left + input.width + p;
                for (int c = 0; c < input.channels; ++c) {
                    row[dst_x * input.channels + c] = row[reflect_x * input.channels + c];
                }
            }
        }
    }

    int padding_ = 0;
    int pad_left_ = 0;
    int pad_top_ = 0;
    int pad_right_ = 0;
    int pad_bottom_ = 0;
    PaddingMode mode_;
    uint8_t value_;
};

} // namespace transforms
} // namespace turboloader
