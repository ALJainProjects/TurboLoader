/**
 * @file modern_augment_transform.hpp
 * @brief Modern data augmentation transforms for TurboLoader v1.8.0
 *
 * Implements state-of-the-art augmentation techniques:
 * - MixUp: Linear interpolation between two images
 * - CutMix: Rectangular patch mixing
 * - Mosaic: 4-image grid composition
 * - RandAugment: Automated augmentation with magnitude control
 *
 * These augmentations are essential for modern training pipelines.
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <random>
#include <cmath>
#include <algorithm>

namespace turboloader {
namespace transforms {

/**
 * @brief MixUp augmentation
 *
 * Blends two images using linear interpolation:
 * output = lambda * image1 + (1 - lambda) * image2
 *
 * Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2017)
 */
class MixUpTransform : public Transform {
public:
    /**
     * @param alpha Beta distribution parameter (default: 0.4)
     * @param seed Random seed
     */
    MixUpTransform(float alpha = 0.4f, unsigned seed = std::random_device{}())
        : alpha_(alpha), rng_(seed), beta_dist_(alpha, alpha) {}

    /**
     * @brief Set the second image for mixing
     */
    void set_mix_image(const ImageData& mix_image) {
        mix_image_ = &mix_image;
    }

    /**
     * @brief Get the lambda value used for the last mix
     */
    float get_lambda() const { return last_lambda_; }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (!mix_image_) {
            // No mix image set, return copy
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        // Sample lambda from Beta distribution
        last_lambda_ = static_cast<float>(beta_dist_(rng_));

        // Create output image
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Blend images: output = lambda * input + (1 - lambda) * mix_image
        size_t total_pixels = input.width * input.height * input.channels;

#ifdef TURBOLOADER_SIMD_NEON
        mixup_blend_neon(input.data, mix_image_->data, output->data,
                         last_lambda_, total_pixels);
#else
        for (size_t i = 0; i < total_pixels; ++i) {
            float val = last_lambda_ * input.data[i] +
                       (1.0f - last_lambda_) * mix_image_->data[i];
            output->data[i] = static_cast<uint8_t>(simd::clamp(val, 0.0f, 255.0f));
        }
#endif

        return output;
    }

    const char* name() const override { return "MixUp"; }

private:
#ifdef TURBOLOADER_SIMD_NEON
    void mixup_blend_neon(const uint8_t* src1, const uint8_t* src2, uint8_t* dst,
                          float lambda, size_t count) {
        float32x4_t lambda_vec = vdupq_n_f32(lambda);
        float32x4_t inv_lambda_vec = vdupq_n_f32(1.0f - lambda);
        float32x4_t max_val = vdupq_n_f32(255.0f);
        float32x4_t zero = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            // Load and process first 4 pixels
            uint8x8_t u8_1 = vld1_u8(src1 + i);
            uint8x8_t u8_2 = vld1_u8(src2 + i);

            uint16x4_t u16_1_lo = vget_low_u16(vmovl_u8(u8_1));
            uint16x4_t u16_2_lo = vget_low_u16(vmovl_u8(u8_2));

            float32x4_t f1_lo = vcvtq_f32_u32(vmovl_u16(u16_1_lo));
            float32x4_t f2_lo = vcvtq_f32_u32(vmovl_u16(u16_2_lo));

            float32x4_t result_lo = vaddq_f32(
                vmulq_f32(f1_lo, lambda_vec),
                vmulq_f32(f2_lo, inv_lambda_vec)
            );
            result_lo = vmaxq_f32(vminq_f32(result_lo, max_val), zero);

            // Process second 4 pixels
            uint16x4_t u16_1_hi = vget_high_u16(vmovl_u8(u8_1));
            uint16x4_t u16_2_hi = vget_high_u16(vmovl_u8(u8_2));

            float32x4_t f1_hi = vcvtq_f32_u32(vmovl_u16(u16_1_hi));
            float32x4_t f2_hi = vcvtq_f32_u32(vmovl_u16(u16_2_hi));

            float32x4_t result_hi = vaddq_f32(
                vmulq_f32(f1_hi, lambda_vec),
                vmulq_f32(f2_hi, inv_lambda_vec)
            );
            result_hi = vmaxq_f32(vminq_f32(result_hi, max_val), zero);

            // Convert back to uint8
            uint32x4_t r_lo = vcvtq_u32_f32(result_lo);
            uint32x4_t r_hi = vcvtq_u32_f32(result_hi);
            uint16x4_t r16_lo = vmovn_u32(r_lo);
            uint16x4_t r16_hi = vmovn_u32(r_hi);
            uint8x8_t result = vmovn_u16(vcombine_u16(r16_lo, r16_hi));

            vst1_u8(dst + i, result);
        }

        // Scalar tail
        for (; i < count; ++i) {
            float val = lambda * src1[i] + (1.0f - lambda) * src2[i];
            dst[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
    }
#endif

    float alpha_;
    float last_lambda_ = 1.0f;
    std::mt19937 rng_;
    std::gamma_distribution<double> beta_dist_;
    const ImageData* mix_image_ = nullptr;
};

/**
 * @brief CutMix augmentation
 *
 * Cuts a rectangular patch from one image and pastes it onto another.
 * Labels are mixed proportionally to the area.
 *
 * Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong
 *            Classifiers with Localizable Features" (2019)
 */
class CutMixTransform : public Transform {
public:
    /**
     * @param alpha Beta distribution parameter (default: 1.0)
     * @param seed Random seed
     */
    CutMixTransform(float alpha = 1.0f, unsigned seed = std::random_device{}())
        : alpha_(alpha), rng_(seed), uniform_(0.0, 1.0) {}

    /**
     * @brief Set the source image for the cut patch
     */
    void set_source_image(const ImageData& source) {
        source_image_ = &source;
    }

    /**
     * @brief Get the lambda value (ratio of mixed area)
     */
    float get_lambda() const { return last_lambda_; }

    /**
     * @brief Get the bounding box of the cut region
     */
    void get_bbox(int& x1, int& y1, int& x2, int& y2) const {
        x1 = bbox_x1_; y1 = bbox_y1_;
        x2 = bbox_x2_; y2 = bbox_y2_;
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        if (!source_image_) {
            // No source image, return copy
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        // Sample lambda from Beta distribution
        std::gamma_distribution<double> gamma1(alpha_, 1.0);
        std::gamma_distribution<double> gamma2(alpha_, 1.0);
        double g1 = gamma1(rng_);
        double g2 = gamma2(rng_);
        double lambda = g1 / (g1 + g2);

        // Calculate cut size
        int W = input.width;
        int H = input.height;
        double cut_ratio = std::sqrt(1.0 - lambda);
        int cut_w = static_cast<int>(W * cut_ratio);
        int cut_h = static_cast<int>(H * cut_ratio);

        // Random center position
        int cx = static_cast<int>(uniform_(rng_) * W);
        int cy = static_cast<int>(uniform_(rng_) * H);

        // Calculate bounding box
        bbox_x1_ = std::max(0, cx - cut_w / 2);
        bbox_y1_ = std::max(0, cy - cut_h / 2);
        bbox_x2_ = std::min(W, cx + cut_w / 2);
        bbox_y2_ = std::min(H, cy + cut_h / 2);

        // Adjust lambda based on actual box area
        last_lambda_ = 1.0f - static_cast<float>((bbox_x2_ - bbox_x1_) *
                       (bbox_y2_ - bbox_y1_)) / (W * H);

        // Create output (start with copy of input)
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );
        std::memcpy(output->data, input.data, input.size_bytes());

        // Paste the cut region from source image
        for (int y = bbox_y1_; y < bbox_y2_; ++y) {
            for (int x = bbox_x1_; x < bbox_x2_; ++x) {
                size_t src_idx = (y * source_image_->width + x) * input.channels;
                size_t dst_idx = (y * W + x) * input.channels;

                for (int c = 0; c < input.channels; ++c) {
                    output->data[dst_idx + c] = source_image_->data[src_idx + c];
                }
            }
        }

        return output;
    }

    const char* name() const override { return "CutMix"; }

private:
    float alpha_;
    float last_lambda_ = 1.0f;
    int bbox_x1_ = 0, bbox_y1_ = 0, bbox_x2_ = 0, bbox_y2_ = 0;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_;
    const ImageData* source_image_ = nullptr;
};

/**
 * @brief Mosaic augmentation
 *
 * Creates a 2x2 grid from 4 images, commonly used in YOLO training.
 *
 * Reference: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy
 *            of Object Detection" (2020)
 */
class MosaicTransform : public Transform {
public:
    /**
     * @param output_size Size of the output mosaic (square)
     * @param seed Random seed
     */
    MosaicTransform(int output_size = 640, unsigned seed = std::random_device{}())
        : output_size_(output_size), rng_(seed), uniform_(0.4, 0.6) {}

    /**
     * @brief Set the 4 images for mosaic (indices 0-3)
     */
    void set_images(const ImageData* img0, const ImageData* img1,
                   const ImageData* img2, const ImageData* img3) {
        images_[0] = img0;
        images_[1] = img1;
        images_[2] = img2;
        images_[3] = img3;
    }

    /**
     * @brief Get the center point of the mosaic
     */
    void get_center(int& cx, int& cy) const {
        cx = center_x_; cy = center_y_;
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // Set first image if not already set
        if (!images_[0]) images_[0] = &input;

        // Use input for any missing images
        for (int i = 0; i < 4; ++i) {
            if (!images_[i]) images_[i] = &input;
        }

        int channels = input.channels;

        // Random center point
        center_x_ = static_cast<int>(uniform_(rng_) * output_size_);
        center_y_ = static_cast<int>(uniform_(rng_) * output_size_);

        // Create output image
        size_t output_bytes = output_size_ * output_size_ * channels;
        auto output = std::make_unique<ImageData>(
            new uint8_t[output_bytes],
            output_size_, output_size_, channels,
            output_size_ * channels, true
        );

        // Fill with gray background
        std::memset(output->data, 114, output_bytes);

        // Place each image in its quadrant
        // Top-left: images_[0]
        place_image(*images_[0], *output, 0, 0, center_x_, center_y_);

        // Top-right: images_[1]
        place_image(*images_[1], *output, center_x_, 0,
                   output_size_ - center_x_, center_y_);

        // Bottom-left: images_[2]
        place_image(*images_[2], *output, 0, center_y_,
                   center_x_, output_size_ - center_y_);

        // Bottom-right: images_[3]
        place_image(*images_[3], *output, center_x_, center_y_,
                   output_size_ - center_x_, output_size_ - center_y_);

        // Reset image pointers
        for (int i = 0; i < 4; ++i) images_[i] = nullptr;

        return output;
    }

    const char* name() const override { return "Mosaic"; }

private:
    void place_image(const ImageData& src, ImageData& dst,
                     int dst_x, int dst_y, int region_w, int region_h) {
        // Scale source to fit region
        float scale_x = static_cast<float>(src.width) / region_w;
        float scale_y = static_cast<float>(src.height) / region_h;
        float scale = std::max(scale_x, scale_y);

        int new_w = static_cast<int>(src.width / scale);
        int new_h = static_cast<int>(src.height / scale);

        // Center in region
        int offset_x = (region_w - new_w) / 2;
        int offset_y = (region_h - new_h) / 2;

        // Copy scaled image to destination
        for (int y = 0; y < new_h && (dst_y + offset_y + y) < dst.height; ++y) {
            int src_y = static_cast<int>(y * scale);
            if (src_y >= src.height) continue;

            for (int x = 0; x < new_w && (dst_x + offset_x + x) < dst.width; ++x) {
                int src_x = static_cast<int>(x * scale);
                if (src_x >= src.width) continue;

                int dest_px = dst_x + offset_x + x;
                int dest_py = dst_y + offset_y + y;

                if (dest_px < 0 || dest_py < 0) continue;

                size_t src_idx = (src_y * src.width + src_x) * src.channels;
                size_t dst_idx = (dest_py * dst.width + dest_px) * dst.channels;

                for (int c = 0; c < src.channels; ++c) {
                    dst.data[dst_idx + c] = src.data[src_idx + c];
                }
            }
        }
    }

    int output_size_;
    int center_x_ = 0, center_y_ = 0;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_;
    const ImageData* images_[4] = {nullptr, nullptr, nullptr, nullptr};
};

/**
 * @brief RandAugment implementation
 *
 * Applies N random augmentations with magnitude M.
 *
 * Reference: Cubuk et al., "RandAugment: Practical automated data
 *            augmentation with a reduced search space" (2020)
 */
class RandAugmentTransform : public Transform {
public:
    /**
     * @param num_ops Number of augmentation operations to apply (N)
     * @param magnitude Magnitude of augmentations (M, 0-30 scale)
     * @param seed Random seed
     */
    RandAugmentTransform(int num_ops = 2, int magnitude = 9,
                         unsigned seed = std::random_device{}())
        : num_ops_(num_ops), magnitude_(magnitude), rng_(seed) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // List of available augmentations
        enum class AugmentOp {
            IDENTITY,
            AUTO_CONTRAST,
            EQUALIZE,
            ROTATE,
            SOLARIZE,
            COLOR,
            POSTERIZE,
            CONTRAST,
            BRIGHTNESS,
            SHARPNESS,
            SHEAR_X,
            SHEAR_Y,
            TRANSLATE_X,
            TRANSLATE_Y,
            NUM_OPS
        };

        std::uniform_int_distribution<int> op_dist(0, 13);

        // Start with copy of input
        size_t size = input.size_bytes();
        std::vector<uint8_t> current_data(input.data, input.data + size);
        int width = input.width;
        int height = input.height;
        int channels = input.channels;

        for (int i = 0; i < num_ops_; ++i) {
            AugmentOp op = static_cast<AugmentOp>(op_dist(rng_));
            std::vector<uint8_t> temp_data(size);

            switch (op) {
                case AugmentOp::IDENTITY:
                    // No change
                    break;

                case AugmentOp::BRIGHTNESS:
                    apply_brightness(current_data.data(), temp_data.data(),
                                    width * height * channels);
                    std::swap(current_data, temp_data);
                    break;

                case AugmentOp::CONTRAST:
                    apply_contrast(current_data.data(), temp_data.data(),
                                  width * height * channels);
                    std::swap(current_data, temp_data);
                    break;

                case AugmentOp::SOLARIZE:
                    apply_solarize(current_data.data(), temp_data.data(),
                                  width * height * channels);
                    std::swap(current_data, temp_data);
                    break;

                case AugmentOp::POSTERIZE:
                    apply_posterize(current_data.data(), temp_data.data(),
                                   width * height * channels);
                    std::swap(current_data, temp_data);
                    break;

                case AugmentOp::AUTO_CONTRAST:
                    apply_auto_contrast(current_data.data(), temp_data.data(),
                                       width, height, channels);
                    std::swap(current_data, temp_data);
                    break;

                default:
                    // For unimplemented ops, apply brightness as fallback
                    apply_brightness(current_data.data(), temp_data.data(),
                                    width * height * channels);
                    std::swap(current_data, temp_data);
                    break;
            }
        }

        // Create output
        auto output = std::make_unique<ImageData>(
            new uint8_t[size],
            width, height, channels, width * channels, true
        );
        std::memcpy(output->data, current_data.data(), size);

        return output;
    }

    const char* name() const override { return "RandAugment"; }

private:
    float magnitude_to_factor() const {
        // Convert magnitude (0-30) to factor (0.0-1.0)
        return magnitude_ / 30.0f;
    }

    void apply_brightness(const uint8_t* src, uint8_t* dst, size_t count) {
        float factor = 1.0f + (magnitude_to_factor() - 0.5f) * 0.8f;
        simd::mul_u8_scalar(src, dst, factor, count);
    }

    void apply_contrast(const uint8_t* src, uint8_t* dst, size_t count) {
        float factor = 1.0f + (magnitude_to_factor() - 0.5f) * 0.8f;
        for (size_t i = 0; i < count; ++i) {
            float val = (src[i] - 128.0f) * factor + 128.0f;
            dst[i] = static_cast<uint8_t>(simd::clamp(val, 0.0f, 255.0f));
        }
    }

    void apply_solarize(const uint8_t* src, uint8_t* dst, size_t count) {
        int threshold = static_cast<int>(256 - magnitude_to_factor() * 256);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = (src[i] >= threshold) ? (255 - src[i]) : src[i];
        }
    }

    void apply_posterize(const uint8_t* src, uint8_t* dst, size_t count) {
        int bits = 8 - static_cast<int>(magnitude_to_factor() * 4);
        bits = std::max(1, std::min(8, bits));
        uint8_t mask = static_cast<uint8_t>(0xFF << (8 - bits));

        for (size_t i = 0; i < count; ++i) {
            dst[i] = src[i] & mask;
        }
    }

    void apply_auto_contrast(const uint8_t* src, uint8_t* dst,
                             int width, int height, int channels) {
        for (int c = 0; c < channels; ++c) {
            // Find min/max for this channel
            uint8_t min_val = 255, max_val = 0;
            for (int i = c; i < width * height * channels; i += channels) {
                min_val = std::min(min_val, src[i]);
                max_val = std::max(max_val, src[i]);
            }

            // Apply contrast stretching
            if (max_val > min_val) {
                float scale = 255.0f / (max_val - min_val);
                for (int i = c; i < width * height * channels; i += channels) {
                    dst[i] = static_cast<uint8_t>((src[i] - min_val) * scale);
                }
            } else {
                for (int i = c; i < width * height * channels; i += channels) {
                    dst[i] = src[i];
                }
            }
        }
    }

    int num_ops_;
    int magnitude_;
    std::mt19937 rng_;
};

/**
 * @brief GridMask augmentation
 *
 * Applies a grid-based mask to the image for regularization.
 *
 * Reference: Chen et al., "GridMask Data Augmentation" (2020)
 */
class GridMaskTransform : public Transform {
public:
    /**
     * @param d Grid cell size ratio
     * @param ratio Mask ratio within cell
     * @param p Probability of applying
     * @param seed Random seed
     */
    GridMaskTransform(float d = 0.5f, float ratio = 0.6f, float p = 0.5f,
                      unsigned seed = std::random_device{}())
        : d_(d), ratio_(ratio), probability_(p), rng_(seed), uniform_(0.0f, 1.0f) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );

        // Check probability
        if (uniform_(rng_) >= probability_) {
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        std::memcpy(output->data, input.data, input.size_bytes());

        // Calculate grid parameters
        int d = static_cast<int>(std::min(input.width, input.height) * d_);
        d = std::max(d, 2);
        int l = static_cast<int>(d * ratio_);

        // Random offset
        std::uniform_int_distribution<int> offset_dist(0, d - 1);
        int offset_x = offset_dist(rng_);
        int offset_y = offset_dist(rng_);

        // Apply grid mask
        for (int y = 0; y < input.height; ++y) {
            int grid_y = (y + offset_y) % d;

            for (int x = 0; x < input.width; ++x) {
                int grid_x = (x + offset_x) % d;

                // Mask if within the masked region of the grid cell
                if (grid_x < l && grid_y < l) {
                    size_t idx = (y * input.width + x) * input.channels;
                    for (int c = 0; c < input.channels; ++c) {
                        output->data[idx + c] = 0;  // Black mask
                    }
                }
            }
        }

        return output;
    }

    const char* name() const override { return "GridMask"; }

private:
    float d_;
    float ratio_;
    float probability_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> uniform_;
};

} // namespace transforms
} // namespace turboloader
