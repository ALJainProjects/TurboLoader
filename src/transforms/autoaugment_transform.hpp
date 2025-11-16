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

    /**
     * @brief Apply a single operation
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
        else if (op_name == "Equalize" || op_name == "AutoContrast" ||
                 op_name == "Invert" || op_name == "Sharpness" ||
                 op_name == "Color" || op_name == "Contrast" ||
                 op_name == "Brightness" || op_name == "ShearX" ||
                 op_name == "ShearY" || op_name == "TranslateX" ||
                 op_name == "TranslateY") {
            // These operations are not yet implemented, return copy
            // In a full implementation, these would use the corresponding transforms
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        // Unknown operation, return copy
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );
        std::memcpy(output->data, input.data, input.size_bytes());
        return output;
    }
};

} // namespace transforms
} // namespace turboloader
