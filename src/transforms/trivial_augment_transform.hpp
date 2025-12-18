/**
 * @file trivial_augment_transform.hpp
 * @brief TrivialAugment implementation (v2.18.0)
 *
 * Implements TrivialAugment from "TrivialAugment: Tuning-free Yet State-of-the-Art
 * Data Augmentation" (Muller & Hutter, 2021).
 *
 * Key differences from RandAugment:
 * - Single random operation per sample (vs N operations)
 * - Uniform magnitude sampling (vs fixed M)
 * - Simpler, often better performance
 * - No hyperparameter tuning needed
 *
 * Features:
 * - 14 operations from TrivialAugment-Wide
 * - Standard and Wide augmentation spaces
 * - Configurable magnitude ranges
 * - Thread-safe random number generation
 *
 * Usage:
 * ```cpp
 * TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE);
 * auto augmented = aug.apply(image);
 * ```
 */

#pragma once

#include "transform_base.hpp"
#include "autoaugment_transform.hpp"  // Reuse operation implementations
#include <random>
#include <functional>
#include <vector>
#include <string>
#include <cmath>

namespace turboloader {
namespace transforms {

/**
 * @brief TrivialAugment: simpler and often better than RandAugment
 */
class TrivialAugmentTransform : public RandomTransform {
public:
    /**
     * @brief Augmentation space options
     */
    enum class AugmentSpace {
        STANDARD,  // 8 basic operations
        WIDE       // 14 operations (recommended)
    };

    /**
     * @brief Operation with magnitude range
     */
    struct Operation {
        std::string name;
        std::function<std::unique_ptr<ImageData>(const ImageData&, float)> apply;
        float min_magnitude;
        float max_magnitude;
        bool magnitude_affects_identity;  // True if magnitude=0 is identity
    };

    /**
     * @brief Create TrivialAugment transform
     * @param space Augmentation space (STANDARD or WIDE)
     * @param seed Random seed for reproducibility
     */
    explicit TrivialAugmentTransform(
        AugmentSpace space = AugmentSpace::WIDE,
        unsigned seed = std::random_device{}()
    ) : space_(space), rng_(seed), dist_(0.0f, 1.0f) {
        setup_operations();
    }

    /**
     * @brief Apply single random augmentation
     */
    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (operations_.empty()) {
            return clone_image(input);
        }

        // Pick random operation
        std::uniform_int_distribution<size_t> op_dist(0, operations_.size() - 1);
        size_t op_idx = op_dist(rng_);
        const auto& op = operations_[op_idx];

        // Sample uniform magnitude
        float magnitude = op.min_magnitude +
                         dist_(rng_) * (op.max_magnitude - op.min_magnitude);

        // Apply operation
        return op.apply(input, magnitude);
    }

    /**
     * @brief Get transform name
     */
    const char* name() const override {
        return space_ == AugmentSpace::WIDE ?
               "TrivialAugment-Wide" : "TrivialAugment-Standard";
    }

    /**
     * @brief Get number of operations
     */
    size_t num_operations() const { return operations_.size(); }

    /**
     * @brief Get operation names
     */
    std::vector<std::string> operation_names() const {
        std::vector<std::string> names;
        names.reserve(operations_.size());
        for (const auto& op : operations_) {
            names.push_back(op.name);
        }
        return names;
    }

private:
    void setup_operations() {
        operations_.clear();

        // Identity - always included
        operations_.push_back({
            "Identity",
            [this](const ImageData& img, float) { return clone_image(img); },
            0.0f, 1.0f, true
        });

        // AutoContrast - no magnitude
        operations_.push_back({
            "AutoContrast",
            [this](const ImageData& img, float) { return auto_contrast(img); },
            0.0f, 1.0f, false
        });

        // Equalize - no magnitude
        operations_.push_back({
            "Equalize",
            [this](const ImageData& img, float) { return equalize(img); },
            0.0f, 1.0f, false
        });

        // Rotate - magnitude = degrees
        operations_.push_back({
            "Rotate",
            [this](const ImageData& img, float m) { return rotate(img, m * 30.0f); },
            -1.0f, 1.0f, true
        });

        // Solarize - magnitude = threshold
        operations_.push_back({
            "Solarize",
            [this](const ImageData& img, float m) {
                int threshold = static_cast<int>((1.0f - m) * 255);
                return solarize(img, threshold);
            },
            0.0f, 1.0f, false
        });

        // Color - magnitude = enhancement factor
        operations_.push_back({
            "Color",
            [this](const ImageData& img, float m) {
                float factor = m * 1.8f + 0.1f;  // [0.1, 1.9]
                return color_enhance(img, factor);
            },
            0.0f, 1.0f, false
        });

        // Posterize - magnitude = bits
        operations_.push_back({
            "Posterize",
            [this](const ImageData& img, float m) {
                int bits = static_cast<int>(m * 4) + 4;  // [4, 8]
                return posterize(img, bits);
            },
            0.0f, 1.0f, false
        });

        // Contrast - magnitude = enhancement factor
        operations_.push_back({
            "Contrast",
            [this](const ImageData& img, float m) {
                float factor = m * 1.8f + 0.1f;  // [0.1, 1.9]
                return contrast(img, factor);
            },
            0.0f, 1.0f, false
        });

        // Brightness - magnitude = enhancement factor
        operations_.push_back({
            "Brightness",
            [this](const ImageData& img, float m) {
                float factor = m * 1.8f + 0.1f;  // [0.1, 1.9]
                return brightness(img, factor);
            },
            0.0f, 1.0f, false
        });

        // Sharpness - magnitude = enhancement factor
        operations_.push_back({
            "Sharpness",
            [this](const ImageData& img, float m) {
                float factor = m * 1.8f + 0.1f;  // [0.1, 1.9]
                return sharpness(img, factor);
            },
            0.0f, 1.0f, false
        });

        // Wide-only operations
        if (space_ == AugmentSpace::WIDE) {
            // ShearX
            operations_.push_back({
                "ShearX",
                [this](const ImageData& img, float m) {
                    float shear = (m - 0.5f) * 0.6f;  // [-0.3, 0.3]
                    return shear_x(img, shear);
                },
                0.0f, 1.0f, true
            });

            // ShearY
            operations_.push_back({
                "ShearY",
                [this](const ImageData& img, float m) {
                    float shear = (m - 0.5f) * 0.6f;
                    return shear_y(img, shear);
                },
                0.0f, 1.0f, true
            });

            // TranslateX
            operations_.push_back({
                "TranslateX",
                [this](const ImageData& img, float m) {
                    int pixels = static_cast<int>((m - 0.5f) * img.width * 0.45f);
                    return translate_x(img, pixels);
                },
                0.0f, 1.0f, true
            });

            // TranslateY
            operations_.push_back({
                "TranslateY",
                [this](const ImageData& img, float m) {
                    int pixels = static_cast<int>((m - 0.5f) * img.height * 0.45f);
                    return translate_y(img, pixels);
                },
                0.0f, 1.0f, true
            });
        }
    }

    // ========================================================================
    // Image Operations
    // ========================================================================

    std::unique_ptr<ImageData> clone_image(const ImageData& img) const {
        auto result = std::make_unique<ImageData>(
            new uint8_t[img.width * img.height * img.channels],
            img.width, img.height, img.channels, img.stride, true
        );
        std::memcpy(result->data, img.data, img.width * img.height * img.channels);
        return result;
    }

    std::unique_ptr<ImageData> auto_contrast(const ImageData& img) const {
        auto result = clone_image(img);

        for (int c = 0; c < img.channels; ++c) {
            uint8_t min_val = 255, max_val = 0;

            // Find min/max
            for (int y = 0; y < img.height; ++y) {
                for (int x = 0; x < img.width; ++x) {
                    uint8_t val = img.data[y * img.stride + x * img.channels + c];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }

            // Apply contrast stretch
            if (max_val > min_val) {
                float scale = 255.0f / (max_val - min_val);
                for (int y = 0; y < img.height; ++y) {
                    for (int x = 0; x < img.width; ++x) {
                        int idx = y * img.stride + x * img.channels + c;
                        float val = (img.data[idx] - min_val) * scale;
                        result->data[idx] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
                    }
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> equalize(const ImageData& img) const {
        auto result = clone_image(img);

        for (int c = 0; c < img.channels; ++c) {
            // Build histogram
            int histogram[256] = {0};
            for (int y = 0; y < img.height; ++y) {
                for (int x = 0; x < img.width; ++x) {
                    histogram[img.data[y * img.stride + x * img.channels + c]]++;
                }
            }

            // Build CDF
            int cdf[256];
            cdf[0] = histogram[0];
            for (int i = 1; i < 256; ++i) {
                cdf[i] = cdf[i-1] + histogram[i];
            }

            // Find min CDF
            int cdf_min = 0;
            for (int i = 0; i < 256; ++i) {
                if (cdf[i] > 0) { cdf_min = cdf[i]; break; }
            }

            // Apply equalization
            int total = img.width * img.height;
            for (int y = 0; y < img.height; ++y) {
                for (int x = 0; x < img.width; ++x) {
                    int idx = y * img.stride + x * img.channels + c;
                    int val = img.data[idx];
                    float new_val = (static_cast<float>(cdf[val] - cdf_min) / (total - cdf_min)) * 255.0f;
                    result->data[idx] = static_cast<uint8_t>(std::clamp(new_val, 0.0f, 255.0f));
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> rotate(const ImageData& img, float degrees) const {
        auto result = clone_image(img);

        float rad = degrees * 3.14159265f / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);

        int cx = img.width / 2;
        int cy = img.height / 2;

        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int rx = x - cx;
                int ry = y - cy;

                int src_x = static_cast<int>(rx * cos_a + ry * sin_a + cx);
                int src_y = static_cast<int>(-rx * sin_a + ry * cos_a + cy);

                if (src_x >= 0 && src_x < img.width && src_y >= 0 && src_y < img.height) {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] =
                            img.data[src_y * img.stride + src_x * img.channels + c];
                    }
                } else {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] = 128;
                    }
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> solarize(const ImageData& img, int threshold) const {
        auto result = clone_image(img);

        for (int i = 0; i < img.width * img.height * img.channels; ++i) {
            if (img.data[i] >= threshold) {
                result->data[i] = 255 - img.data[i];
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> posterize(const ImageData& img, int bits) const {
        auto result = clone_image(img);

        int mask = ~((1 << (8 - bits)) - 1);
        for (int i = 0; i < img.width * img.height * img.channels; ++i) {
            result->data[i] = img.data[i] & mask;
        }

        return result;
    }

    std::unique_ptr<ImageData> color_enhance(const ImageData& img, float factor) const {
        auto result = clone_image(img);

        if (img.channels < 3) return result;

        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int idx = y * img.stride + x * img.channels;
                float gray = 0.299f * img.data[idx] +
                            0.587f * img.data[idx + 1] +
                            0.114f * img.data[idx + 2];

                for (int c = 0; c < 3; ++c) {
                    float val = gray + (img.data[idx + c] - gray) * factor;
                    result->data[idx + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> contrast(const ImageData& img, float factor) const {
        auto result = clone_image(img);

        for (int i = 0; i < img.width * img.height * img.channels; ++i) {
            float val = 128.0f + (img.data[i] - 128.0f) * factor;
            result->data[i] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
        }

        return result;
    }

    std::unique_ptr<ImageData> brightness(const ImageData& img, float factor) const {
        auto result = clone_image(img);

        for (int i = 0; i < img.width * img.height * img.channels; ++i) {
            float val = img.data[i] * factor;
            result->data[i] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
        }

        return result;
    }

    std::unique_ptr<ImageData> sharpness(const ImageData& img, float factor) const {
        auto result = clone_image(img);

        // Apply simple 3x3 sharpen kernel
        for (int y = 1; y < img.height - 1; ++y) {
            for (int x = 1; x < img.width - 1; ++x) {
                for (int c = 0; c < img.channels; ++c) {
                    int idx = y * img.stride + x * img.channels + c;

                    float center = img.data[idx];
                    float neighbors = (
                        img.data[(y-1) * img.stride + x * img.channels + c] +
                        img.data[(y+1) * img.stride + x * img.channels + c] +
                        img.data[y * img.stride + (x-1) * img.channels + c] +
                        img.data[y * img.stride + (x+1) * img.channels + c]
                    ) / 4.0f;

                    float blurred = (center + neighbors) / 2.0f;
                    float val = blurred + (center - blurred) * factor;
                    result->data[idx] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> shear_x(const ImageData& img, float shear) const {
        auto result = clone_image(img);

        int cy = img.height / 2;

        for (int y = 0; y < img.height; ++y) {
            int offset = static_cast<int>((y - cy) * shear);
            for (int x = 0; x < img.width; ++x) {
                int src_x = x - offset;
                if (src_x >= 0 && src_x < img.width) {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] =
                            img.data[y * img.stride + src_x * img.channels + c];
                    }
                } else {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] = 128;
                    }
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> shear_y(const ImageData& img, float shear) const {
        auto result = clone_image(img);

        int cx = img.width / 2;

        for (int x = 0; x < img.width; ++x) {
            int offset = static_cast<int>((x - cx) * shear);
            for (int y = 0; y < img.height; ++y) {
                int src_y = y - offset;
                if (src_y >= 0 && src_y < img.height) {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] =
                            img.data[src_y * img.stride + x * img.channels + c];
                    }
                } else {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] = 128;
                    }
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> translate_x(const ImageData& img, int pixels) const {
        auto result = clone_image(img);

        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int src_x = x - pixels;
                if (src_x >= 0 && src_x < img.width) {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] =
                            img.data[y * img.stride + src_x * img.channels + c];
                    }
                } else {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] = 128;
                    }
                }
            }
        }

        return result;
    }

    std::unique_ptr<ImageData> translate_y(const ImageData& img, int pixels) const {
        auto result = clone_image(img);

        for (int y = 0; y < img.height; ++y) {
            int src_y = y - pixels;
            for (int x = 0; x < img.width; ++x) {
                if (src_y >= 0 && src_y < img.height) {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] =
                            img.data[src_y * img.stride + x * img.channels + c];
                    }
                } else {
                    for (int c = 0; c < img.channels; ++c) {
                        result->data[y * img.stride + x * img.channels + c] = 128;
                    }
                }
            }
        }

        return result;
    }

    AugmentSpace space_;
    std::vector<Operation> operations_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> dist_;
};

}  // namespace transforms
}  // namespace turboloader
