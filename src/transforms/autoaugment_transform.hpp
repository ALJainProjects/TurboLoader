/**
 * @file autoaugment_transform.hpp
 * @brief AutoAugment learned augmentation policies
 *
 * Features:
 * - ImageNet, CIFAR10, SVHN policy sets
 * - Composite transforms (randomly select from policy list)
 * - Magnitude and probability parameters
 * - Reuses existing SIMD transforms
 *
 * Reference: torchvision.transforms.AutoAugment
 * Paper: "AutoAugment: Learning Augmentation Policies from Data" (Cubuk et al., 2019)
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include "color_jitter_transform.hpp"
#include "rotation_transform.hpp"
#include "affine_transform.hpp"
#include "posterize_transform.hpp"
#include "solarize_transform.hpp"
#include <vector>
#include <utility>

namespace turboloader {
namespace transforms {

/**
 * @brief AutoAugment policy type
 */
enum class AutoAugmentPolicy {
    IMAGENET,
    CIFAR10,
    SVHN
};

/**
 * @brief AutoAugment transform
 */
class AutoAugmentTransform : public RandomTransform {
public:
    /**
     * @param policy AutoAugment policy set
     * @param seed Random seed
     */
    AutoAugmentTransform(AutoAugmentPolicy policy = AutoAugmentPolicy::IMAGENET,
                         unsigned seed = std::random_device{}())
        : RandomTransform(1.0f, seed), policy_(policy) {
        initialize_policy();
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // Select a random sub-policy
        std::uniform_int_distribution<size_t> dist(0, sub_policies_.size() - 1);
        size_t policy_idx = dist(rng_);

        // Apply the selected sub-policy
        return apply_sub_policy(input, policy_idx);
    }

    const char* name() const override { return "AutoAugment"; }

private:
    AutoAugmentPolicy policy_;

    // Each sub-policy is a list of (operation_name, probability, magnitude)
    struct Operation {
        std::string name;
        float probability;
        float magnitude;
    };

    std::vector<std::vector<Operation>> sub_policies_;

    /**
     * @brief Initialize policy-specific sub-policies
     */
    void initialize_policy() {
        switch (policy_) {
            case AutoAugmentPolicy::IMAGENET:
                initialize_imagenet_policy();
                break;
            case AutoAugmentPolicy::CIFAR10:
                initialize_cifar10_policy();
                break;
            case AutoAugmentPolicy::SVHN:
                initialize_svhn_policy();
                break;
        }
    }

    /**
     * @brief ImageNet AutoAugment policy
     */
    void initialize_imagenet_policy() {
        sub_policies_ = {
            {{"Posterize", 0.4f, 8.0f}, {"Rotate", 0.6f, 9.0f}},
            {{"Solarize", 0.6f, 5.0f}, {"AutoContrast", 0.6f, 0.0f}},
            {{"Equalize", 0.8f, 0.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"Posterize", 0.6f, 7.0f}, {"Posterize", 0.6f, 6.0f}},
            {{"Equalize", 0.4f, 0.0f}, {"Solarize", 0.2f, 4.0f}},
            {{"Equalize", 0.4f, 0.0f}, {"Rotate", 0.8f, 8.0f}},
            {{"Solarize", 0.6f, 3.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"Posterize", 0.8f, 5.0f}, {"Equalize", 1.0f, 0.0f}},
            {{"Rotate", 0.2f, 3.0f}, {"Solarize", 0.6f, 8.0f}},
            {{"Equalize", 0.6f, 0.0f}, {"Posterize", 0.4f, 6.0f}},
            {{"Rotate", 0.8f, 8.0f}, {"Color", 0.4f, 0.0f}},
            {{"Rotate", 0.4f, 9.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"Equalize", 0.0f, 0.0f}, {"Equalize", 0.8f, 0.0f}},
            {{"Invert", 0.6f, 0.0f}, {"Equalize", 1.0f, 0.0f}},
            {{"Color", 0.6f, 4.0f}, {"Contrast", 1.0f, 8.0f}},
            {{"Rotate", 0.8f, 8.0f}, {"Color", 1.0f, 2.0f}},
            {{"Color", 0.8f, 8.0f}, {"Solarize", 0.8f, 7.0f}},
            {{"Sharpness", 0.4f, 7.0f}, {"Invert", 0.6f, 0.0f}},
            {{"ShearX", 0.6f, 5.0f}, {"Equalize", 1.0f, 0.0f}},
            {{"Color", 0.4f, 0.0f}, {"Equalize", 0.6f, 0.0f}},
        };
    }

    /**
     * @brief CIFAR10 AutoAugment policy
     */
    void initialize_cifar10_policy() {
        sub_policies_ = {
            {{"Invert", 0.1f, 0.0f}, {"Contrast", 0.2f, 6.0f}},
            {{"Rotate", 0.7f, 2.0f}, {"TranslateX", 0.3f, 9.0f}},
            {{"Sharpness", 0.8f, 1.0f}, {"Sharpness", 0.9f, 3.0f}},
            {{"ShearY", 0.5f, 8.0f}, {"TranslateY", 0.7f, 9.0f}},
            {{"AutoContrast", 0.5f, 0.0f}, {"Equalize", 0.9f, 0.0f}},
            {{"ShearY", 0.2f, 7.0f}, {"Posterize", 0.3f, 7.0f}},
            {{"Color", 0.4f, 3.0f}, {"Brightness", 0.6f, 7.0f}},
            {{"Sharpness", 0.3f, 9.0f}, {"Brightness", 0.7f, 9.0f}},
            {{"Equalize", 0.6f, 0.0f}, {"Equalize", 0.5f, 0.0f}},
            {{"Contrast", 0.6f, 7.0f}, {"Sharpness", 0.6f, 5.0f}},
            {{"Color", 0.7f, 7.0f}, {"TranslateX", 0.5f, 8.0f}},
            {{"Equalize", 0.3f, 0.0f}, {"AutoContrast", 0.4f, 0.0f}},
            {{"TranslateY", 0.4f, 3.0f}, {"Sharpness", 0.2f, 6.0f}},
            {{"Brightness", 0.9f, 6.0f}, {"Color", 0.2f, 8.0f}},
            {{"Solarize", 0.5f, 2.0f}, {"Invert", 0.0f, 0.0f}},
            {{"Equalize", 0.2f, 0.0f}, {"AutoContrast", 0.6f, 0.0f}},
            {{"Equalize", 0.2f, 0.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"Color", 0.9f, 9.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"AutoContrast", 0.8f, 0.0f}, {"Solarize", 0.2f, 8.0f}},
            {{"Brightness", 0.1f, 3.0f}, {"Color", 0.7f, 0.0f}},
        };
    }

    /**
     * @brief SVHN AutoAugment policy
     */
    void initialize_svhn_policy() {
        sub_policies_ = {
            {{"ShearX", 0.9f, 4.0f}, {"Invert", 0.2f, 0.0f}},
            {{"ShearY", 0.9f, 8.0f}, {"Invert", 0.7f, 0.0f}},
            {{"Equalize", 0.6f, 0.0f}, {"Solarize", 0.6f, 6.0f}},
            {{"Invert", 0.9f, 0.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"Equalize", 0.6f, 0.0f}, {"Rotate", 0.9f, 3.0f}},
            {{"ShearX", 0.9f, 4.0f}, {"AutoContrast", 0.8f, 0.0f}},
            {{"ShearY", 0.9f, 8.0f}, {"Invert", 0.4f, 0.0f}},
            {{"ShearY", 0.9f, 5.0f}, {"Solarize", 0.2f, 6.0f}},
            {{"Invert", 0.9f, 0.0f}, {"AutoContrast", 0.8f, 0.0f}},
            {{"Equalize", 0.6f, 0.0f}, {"Rotate", 0.9f, 3.0f}},
            {{"ShearX", 0.9f, 4.0f}, {"Solarize", 0.3f, 3.0f}},
            {{"ShearY", 0.8f, 8.0f}, {"Invert", 0.7f, 0.0f}},
            {{"Equalize", 0.9f, 0.0f}, {"TranslateY", 0.6f, 6.0f}},
            {{"Invert", 0.9f, 0.0f}, {"Equalize", 0.6f, 0.0f}},
            {{"Contrast", 0.3f, 3.0f}, {"Rotate", 0.8f, 4.0f}},
        };
    }

    /**
     * @brief Apply a sub-policy
     */
    std::unique_ptr<ImageData> apply_sub_policy(const ImageData& input, size_t policy_idx) {
        auto current = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );
        std::memcpy(current->data, input.data, input.size_bytes());

        const auto& operations = sub_policies_[policy_idx];

        for (const auto& op : operations) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if (dist(rng_) < op.probability) {
                current = apply_operation(*current, op.name, op.magnitude);
            }
        }

        return current;
    }

public:
    /**
     * @brief Apply a single operation (public for testing)
     */
    std::unique_ptr<ImageData> apply_operation(const ImageData& input,
                                               const std::string& op_name,
                                               float magnitude) {
        // Map operation names to actual transforms
        if (op_name == "Posterize") {
            int bits = static_cast<int>(std::max(1.0f, 8.0f - magnitude / 2.0f));
            RandomPosterizeTransform transform(bits, 1.0f, rng_());
            return transform.apply(input);
        }
        else if (op_name == "Solarize") {
            uint8_t threshold = static_cast<uint8_t>(256.0f - magnitude * 25.6f);
            RandomSolarizeTransform transform(threshold, 1.0f, rng_());
            return transform.apply(input);
        }
        else if (op_name == "Rotate") {
            float degrees = magnitude * 3.0f;  // Scale magnitude to degrees
            RandomRotationTransform transform(degrees, false, 0, rng_());
            return transform.apply(input);
        }
        else if (op_name == "Invert") {
            return apply_invert(input);
        }
        else if (op_name == "AutoContrast") {
            return apply_autocontrast(input);
        }
        else if (op_name == "Equalize") {
            return apply_equalize(input);
        }
        else if (op_name == "Color") {
            // Color = saturation adjustment, magnitude 0-10 maps to 0.1-1.9
            float factor = 1.0f + (magnitude / 10.0f) * 0.9f;
            return apply_color(input, factor);
        }
        else if (op_name == "Brightness") {
            // Brightness adjustment, magnitude 0-10 maps to 0.1-1.9
            float factor = 1.0f + (magnitude / 10.0f) * 0.9f;
            return apply_brightness(input, factor);
        }
        else if (op_name == "Contrast") {
            // Contrast adjustment, magnitude 0-10 maps to 0.1-1.9
            float factor = 1.0f + (magnitude / 10.0f) * 0.9f;
            return apply_contrast(input, factor);
        }
        else if (op_name == "Sharpness") {
            // Sharpness, magnitude 0-10 maps to 0.1-1.9
            float factor = 1.0f + (magnitude / 10.0f) * 0.9f;
            return apply_sharpness(input, factor);
        }
        else if (op_name == "ShearX") {
            // ShearX, magnitude 0-10 maps to 0-0.3 radians
            float shear = (magnitude / 10.0f) * 0.3f;
            return apply_shear_x(input, shear);
        }
        else if (op_name == "ShearY") {
            // ShearY, magnitude 0-10 maps to 0-0.3 radians
            float shear = (magnitude / 10.0f) * 0.3f;
            return apply_shear_y(input, shear);
        }
        else if (op_name == "TranslateX") {
            // TranslateX, magnitude 0-10 maps to 0-0.45 of width
            float translate = (magnitude / 10.0f) * 0.45f;
            return apply_translate_x(input, translate);
        }
        else if (op_name == "TranslateY") {
            // TranslateY, magnitude 0-10 maps to 0-0.45 of height
            float translate = (magnitude / 10.0f) * 0.45f;
            return apply_translate_y(input, translate);
        }

        // Unknown operation, return copy
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );
        std::memcpy(output->data, input.data, input.size_bytes());
        return output;
    }

private:
    // =========================================================================
    // Individual operation implementations (private helpers)
    // =========================================================================

    /**
     * @brief Invert all pixels (255 - pixel)
     */
    std::unique_ptr<ImageData> apply_invert(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        size_t total = input.size_bytes();
        for (size_t i = 0; i < total; ++i) {
            output->data[i] = 255 - input.data[i];
        }
        return output;
    }

    /**
     * @brief AutoContrast: stretch histogram to full range per channel
     */
    std::unique_ptr<ImageData> apply_autocontrast(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        size_t num_pixels = input.width * input.height;

        // Find min/max per channel
        for (int c = 0; c < input.channels; ++c) {
            uint8_t min_val = 255;
            uint8_t max_val = 0;

            for (size_t i = 0; i < num_pixels; ++i) {
                uint8_t val = input.data[i * input.channels + c];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }

            // Apply linear stretch
            float scale = (max_val > min_val) ? 255.0f / (max_val - min_val) : 1.0f;

            for (size_t i = 0; i < num_pixels; ++i) {
                uint8_t val = input.data[i * input.channels + c];
                float stretched = (val - min_val) * scale;
                output->data[i * input.channels + c] = static_cast<uint8_t>(
                    simd::clamp(stretched, 0.0f, 255.0f)
                );
            }
        }
        return output;
    }

    /**
     * @brief Equalize: histogram equalization per channel
     */
    std::unique_ptr<ImageData> apply_equalize(const ImageData& input) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        size_t num_pixels = input.width * input.height;

        // Process each channel independently
        for (int c = 0; c < input.channels; ++c) {
            // Build histogram
            int histogram[256] = {0};
            for (size_t i = 0; i < num_pixels; ++i) {
                histogram[input.data[i * input.channels + c]]++;
            }

            // Build cumulative distribution function (CDF)
            int cdf[256];
            cdf[0] = histogram[0];
            for (int i = 1; i < 256; ++i) {
                cdf[i] = cdf[i-1] + histogram[i];
            }

            // Find minimum non-zero CDF value
            int cdf_min = 0;
            for (int i = 0; i < 256; ++i) {
                if (cdf[i] > 0) {
                    cdf_min = cdf[i];
                    break;
                }
            }

            // Build lookup table
            uint8_t lut[256];
            float scale = (num_pixels > cdf_min) ? 255.0f / (num_pixels - cdf_min) : 0.0f;
            for (int i = 0; i < 256; ++i) {
                if (cdf[i] > 0) {
                    lut[i] = static_cast<uint8_t>(
                        simd::clamp((cdf[i] - cdf_min) * scale, 0.0f, 255.0f)
                    );
                } else {
                    lut[i] = 0;
                }
            }

            // Apply lookup table
            for (size_t i = 0; i < num_pixels; ++i) {
                output->data[i * input.channels + c] = lut[input.data[i * input.channels + c]];
            }
        }
        return output;
    }

    /**
     * @brief Color (saturation) adjustment
     */
    std::unique_ptr<ImageData> apply_color(const ImageData& input, float factor) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        if (input.channels != 3) {
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        size_t num_pixels = input.width * input.height;

        for (size_t i = 0; i < num_pixels; ++i) {
            uint8_t r = input.data[i * 3];
            uint8_t g = input.data[i * 3 + 1];
            uint8_t b = input.data[i * 3 + 2];

            // Convert to grayscale (luminance)
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            // Blend between gray and color based on factor
            output->data[i * 3] = static_cast<uint8_t>(
                simd::clamp(gray + factor * (r - gray), 0.0f, 255.0f)
            );
            output->data[i * 3 + 1] = static_cast<uint8_t>(
                simd::clamp(gray + factor * (g - gray), 0.0f, 255.0f)
            );
            output->data[i * 3 + 2] = static_cast<uint8_t>(
                simd::clamp(gray + factor * (b - gray), 0.0f, 255.0f)
            );
        }
        return output;
    }

    /**
     * @brief Brightness adjustment
     */
    std::unique_ptr<ImageData> apply_brightness(const ImageData& input, float factor) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        size_t total = input.size_bytes();
        for (size_t i = 0; i < total; ++i) {
            float val = input.data[i] * factor;
            output->data[i] = static_cast<uint8_t>(simd::clamp(val, 0.0f, 255.0f));
        }
        return output;
    }

    /**
     * @brief Contrast adjustment
     */
    std::unique_ptr<ImageData> apply_contrast(const ImageData& input, float factor) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Calculate mean brightness
        size_t total = input.size_bytes();
        float mean = 0.0f;
        for (size_t i = 0; i < total; ++i) {
            mean += input.data[i];
        }
        mean /= total;

        // Apply contrast: pixel = mean + factor * (pixel - mean)
        for (size_t i = 0; i < total; ++i) {
            float val = mean + factor * (input.data[i] - mean);
            output->data[i] = static_cast<uint8_t>(simd::clamp(val, 0.0f, 255.0f));
        }
        return output;
    }

    /**
     * @brief Sharpness adjustment using unsharp mask
     */
    std::unique_ptr<ImageData> apply_sharpness(const ImageData& input, float factor) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Create a blurred version using 3x3 box filter
        std::vector<float> blurred(input.size_bytes());

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int c = 0; c < input.channels; ++c) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < input.height && nx >= 0 && nx < input.width) {
                                sum += input.data[(ny * input.width + nx) * input.channels + c];
                                count++;
                            }
                        }
                    }
                    blurred[(y * input.width + x) * input.channels + c] = sum / count;
                }
            }
        }

        // Blend: output = blurred + factor * (original - blurred)
        for (size_t i = 0; i < input.size_bytes(); ++i) {
            float val = blurred[i] + factor * (input.data[i] - blurred[i]);
            output->data[i] = static_cast<uint8_t>(simd::clamp(val, 0.0f, 255.0f));
        }
        return output;
    }

    /**
     * @brief Shear in X direction
     */
    std::unique_ptr<ImageData> apply_shear_x(const ImageData& input, float shear) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels,
            input.width * input.channels, true
        );
        std::memset(output->data, 0, input.size_bytes());

        float cx = input.width / 2.0f;
        float cy = input.height / 2.0f;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                // Inverse transform: source = dest + shear * (y - cy)
                float src_x = x - shear * (y - cy);
                float src_y = static_cast<float>(y);

                if (src_x >= 0 && src_x < input.width - 1) {
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

    /**
     * @brief Shear in Y direction
     */
    std::unique_ptr<ImageData> apply_shear_y(const ImageData& input, float shear) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels,
            input.width * input.channels, true
        );
        std::memset(output->data, 0, input.size_bytes());

        float cx = input.width / 2.0f;
        float cy = input.height / 2.0f;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                // Inverse transform: source = dest + shear * (x - cx)
                float src_x = static_cast<float>(x);
                float src_y = y - shear * (x - cx);

                if (src_y >= 0 && src_y < input.height - 1) {
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

    /**
     * @brief Translate in X direction
     */
    std::unique_ptr<ImageData> apply_translate_x(const ImageData& input, float translate) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels,
            input.width * input.channels, true
        );
        std::memset(output->data, 0, input.size_bytes());

        float offset = translate * input.width;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                float src_x = x - offset;
                float src_y = static_cast<float>(y);

                if (src_x >= 0 && src_x < input.width - 1) {
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

    /**
     * @brief Translate in Y direction
     */
    std::unique_ptr<ImageData> apply_translate_y(const ImageData& input, float translate) {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels,
            input.width * input.channels, true
        );
        std::memset(output->data, 0, input.size_bytes());

        float offset = translate * input.height;

        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                float src_x = static_cast<float>(x);
                float src_y = y - offset;

                if (src_y >= 0 && src_y < input.height - 1) {
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
};

} // namespace transforms
} // namespace turboloader
