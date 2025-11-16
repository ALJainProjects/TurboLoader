#pragma once

#include <cstdint>
#include <random>
#include <memory>
#include <vector>

namespace turboloader {
namespace transforms {

/**
 * Base class for all augmentation transforms
 *
 * Augmentation transforms modify images with random parameters for data augmentation.
 * All transforms are SIMD-optimized where possible.
 */
class AugmentationTransform {
public:
    virtual ~AugmentationTransform() = default;

    /**
     * Apply the transform in-place to image data
     *
     * @param data Pointer to image data (will be modified)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (typically 3 for RGB)
     * @param rng Thread-local random number generator
     */
    virtual void apply(
        uint8_t* data,
        int width,
        int height,
        int channels,
        std::mt19937& rng
    ) = 0;

    /**
     * Check if this transform should be applied (based on probability)
     */
    virtual bool should_apply(std::mt19937& rng) const {
        if (probability_ >= 1.0f) return true;
        if (probability_ <= 0.0f) return false;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng) < probability_;
    }

    void set_probability(float p) { probability_ = p; }
    float get_probability() const { return probability_; }

protected:
    float probability_ = 1.0f;  // Probability of applying transform
};

/**
 * Horizontal flip augmentation (SIMD optimized)
 */
class RandomHorizontalFlip : public AugmentationTransform {
public:
    explicit RandomHorizontalFlip(float probability = 0.5f) {
        probability_ = probability;
    }

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

private:
    void flip_horizontal_simd(uint8_t* data, int width, int height, int channels);
    void flip_horizontal_avx2(uint8_t* data, int width, int height, int channels);
    void flip_horizontal_neon(uint8_t* data, int width, int height, int channels);
};

/**
 * Vertical flip augmentation (SIMD optimized)
 */
class RandomVerticalFlip : public AugmentationTransform {
public:
    explicit RandomVerticalFlip(float probability = 0.5f) {
        probability_ = probability;
    }

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

private:
    void flip_vertical_simd(uint8_t* data, int width, int height, int channels);
};

/**
 * Color jitter: random brightness, contrast, saturation adjustments (SIMD optimized)
 */
class ColorJitter : public AugmentationTransform {
public:
    ColorJitter(
        float brightness = 0.0f,
        float contrast = 0.0f,
        float saturation = 0.0f,
        float hue = 0.0f
    );

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

    void set_brightness_range(float range) { brightness_range_ = range; }
    void set_contrast_range(float range) { contrast_range_ = range; }
    void set_saturation_range(float range) { saturation_range_ = range; }
    void set_hue_range(float range) { hue_range_ = range; }

private:
    void adjust_brightness_simd(uint8_t* data, size_t n, float factor);
    void adjust_contrast_simd(uint8_t* data, size_t n, float factor);
    void adjust_saturation_simd(uint8_t* data, size_t n, float factor);

    void adjust_brightness_avx2(uint8_t* data, size_t n, float factor);
    void adjust_contrast_avx2(uint8_t* data, size_t n, float factor);
    void adjust_saturation_avx2(uint8_t* data, size_t n, float factor);

    void adjust_brightness_neon(uint8_t* data, size_t n, float factor);
    void adjust_contrast_neon(uint8_t* data, size_t n, float factor);
    void adjust_saturation_neon(uint8_t* data, size_t n, float factor);

    float brightness_range_ = 0.0f;
    float contrast_range_ = 0.0f;
    float saturation_range_ = 0.0f;
    float hue_range_ = 0.0f;
};

/**
 * Random rotation (SIMD bilinear interpolation)
 */
class RandomRotation : public AugmentationTransform {
public:
    explicit RandomRotation(float degrees = 0.0f);

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

    void set_degrees(float degrees) { degrees_ = degrees; }
    void set_fill_color(uint8_t r, uint8_t g, uint8_t b) {
        fill_color_[0] = r;
        fill_color_[1] = g;
        fill_color_[2] = b;
    }

private:
    void rotate_bilinear(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels,
        float angle_rad
    );

    void rotate_bilinear_avx2(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels,
        float angle_rad
    );

    float degrees_ = 0.0f;
    uint8_t fill_color_[3] = {0, 0, 0};  // Black fill
};

/**
 * Random crop
 */
class RandomCrop : public AugmentationTransform {
public:
    RandomCrop(int crop_width, int crop_height);

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

private:
    int crop_width_;
    int crop_height_;
};

/**
 * Random erasing (Cutout augmentation)
 */
class RandomErasing : public AugmentationTransform {
public:
    RandomErasing(
        float probability = 0.5f,
        float scale_min = 0.02f,
        float scale_max = 0.33f,
        float ratio_min = 0.3f,
        float ratio_max = 3.3f
    );

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

private:
    float scale_min_, scale_max_;
    float ratio_min_, ratio_max_;
};

/**
 * Gaussian blur (separable filter, SIMD optimized)
 */
class GaussianBlur : public AugmentationTransform {
public:
    explicit GaussianBlur(float sigma = 1.0f, int kernel_size = 5);

    void apply(uint8_t* data, int width, int height, int channels, std::mt19937& rng) override;

    void set_sigma_range(float min_sigma, float max_sigma) {
        sigma_min_ = min_sigma;
        sigma_max_ = max_sigma;
    }

private:
    void create_gaussian_kernel(float sigma, std::vector<float>& kernel);

    void gaussian_blur_separable_simd(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels,
        const std::vector<float>& kernel
    );

    void gaussian_blur_horizontal_avx2(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels,
        const std::vector<float>& kernel
    );

    void gaussian_blur_vertical_avx2(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        int channels,
        const std::vector<float>& kernel
    );

    float sigma_min_ = 0.1f;
    float sigma_max_ = 2.0f;
    int kernel_size_ = 5;
    std::vector<float> kernel_cache_;
};

/**
 * Composable augmentation pipeline
 *
 * Example usage:
 *   AugmentationPipeline pipeline;
 *   pipeline.add_transform(std::make_unique<RandomHorizontalFlip>(0.5));
 *   pipeline.add_transform(std::make_unique<ColorJitter>(0.2, 0.2, 0.2, 0.0));
 *   pipeline.add_transform(std::make_unique<RandomRotation>(15.0));
 *   pipeline.apply_all(data, width, height, channels);
 */
class AugmentationPipeline {
public:
    AugmentationPipeline() : rng_(std::random_device{}()) {}

    explicit AugmentationPipeline(uint64_t seed) : rng_(seed) {}

    void add_transform(std::unique_ptr<AugmentationTransform> transform) {
        transforms_.push_back(std::move(transform));
    }

    void apply_all(uint8_t* data, int width, int height, int channels) {
        for (auto& transform : transforms_) {
            if (transform->should_apply(rng_)) {
                transform->apply(data, width, height, channels, rng_);
            }
        }
    }

    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

    size_t num_transforms() const {
        return transforms_.size();
    }

    void clear() {
        transforms_.clear();
    }

private:
    std::vector<std::unique_ptr<AugmentationTransform>> transforms_;
    std::mt19937 rng_;
};

}  // namespace transforms
}  // namespace turboloader
