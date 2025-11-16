#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace turboloader {

/**
 * Image transform interface
 *
 * All transforms operate on RGB images (HWC format, uint8)
 */
class ImageTransform {
public:
    virtual ~ImageTransform() = default;

    /**
     * Apply transform to image
     * @param src Source image data (RGB, HWC)
     * @param width Image width
     * @param height Image height
     * @return Transformed image (may be different size)
     */
    struct TransformResult {
        std::vector<uint8_t> data;
        int width;
        int height;
        int channels{3};
    };

    virtual TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) = 0;
};

/**
 * Resize transform using bilinear interpolation
 *
 * Features:
 * - SIMD optimized (AVX2 on x86, NEON on ARM)
 * - Bilinear interpolation for quality
 * - Fast path for common sizes (224x224, 256x256)
 */
class ResizeTransform : public ImageTransform {
public:
    ResizeTransform(int target_width, int target_height)
        : target_width_(target_width)
        , target_height_(target_height) {}

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;

private:
    int target_width_;
    int target_height_;

    // SIMD-optimized resize implementations
    void resize_bilinear_simd(
        const uint8_t* src, int src_w, int src_h,
        uint8_t* dst, int dst_w, int dst_h
    );
};

/**
 * Normalize transform (RGB to float, normalize, convert back)
 *
 * Features:
 * - SIMD vectorized operations
 * - Configurable mean/std per channel
 * - In-place when possible
 */
class NormalizeTransform : public ImageTransform {
public:
    NormalizeTransform(
        float mean_r = 0.485f, float mean_g = 0.456f, float mean_b = 0.406f,
        float std_r = 0.229f, float std_g = 0.224f, float std_b = 0.225f
    ) : mean_{mean_r, mean_g, mean_b}
      , std_{std_r, std_g, std_b} {}

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;

private:
    [[maybe_unused]] float mean_[3];
    [[maybe_unused]] float std_[3];

    // SIMD-optimized normalize
    void normalize_simd(
        const uint8_t* src, uint8_t* dst, size_t pixels
    );
};

/**
 * Center crop transform
 *
 * Crops image to target size from center
 */
class CenterCropTransform : public ImageTransform {
public:
    CenterCropTransform(int target_width, int target_height)
        : target_width_(target_width)
        , target_height_(target_height) {}

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;

private:
    int target_width_;
    int target_height_;
};

/**
 * Random crop transform
 *
 * Crops image to target size from random location
 * Thread-safe with per-thread RNG state
 */
class RandomCropTransform : public ImageTransform {
public:
    RandomCropTransform(int target_width, int target_height)
        : target_width_(target_width)
        , target_height_(target_height) {}

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;

private:
    int target_width_;
    int target_height_;
};

/**
 * Horizontal flip transform
 */
class HorizontalFlipTransform : public ImageTransform {
public:
    HorizontalFlipTransform() = default;

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;
};

/**
 * Vertical flip transform
 */
class VerticalFlipTransform : public ImageTransform {
public:
    VerticalFlipTransform() = default;

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;
};

/**
 * Random horizontal flip transform (50% probability)
 */
class RandomHorizontalFlipTransform : public ImageTransform {
public:
    RandomHorizontalFlipTransform() = default;

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;
};

/**
 * Compose multiple transforms into a pipeline
 */
class ComposedTransform : public ImageTransform {
public:
    void add(std::unique_ptr<ImageTransform> transform) {
        transforms_.push_back(std::move(transform));
    }

    TransformResult apply(
        std::span<const uint8_t> src,
        int width,
        int height
    ) override;

private:
    std::vector<std::unique_ptr<ImageTransform>> transforms_;
};

}  // namespace turboloader
