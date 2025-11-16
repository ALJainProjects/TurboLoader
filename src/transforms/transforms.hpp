/**
 * @file transforms.hpp
 * @brief All-in-one header for TurboLoader transforms
 *
 * Includes all available transforms:
 * - SIMD-accelerated operations
 * - PyTorch/TensorFlow tensor conversion
 * - Full augmentation pipeline
 */

#pragma once

// Base infrastructure
#include "transform_base.hpp"
#include "simd_utils.hpp"

// Core transforms
#include "resize_transform.hpp"
#include "normalize_transform.hpp"
#include "crop_transform.hpp"
#include "flip_transform.hpp"
#include "pad_transform.hpp"
#include "grayscale_transform.hpp"

// Augmentation transforms
#include "color_jitter_transform.hpp"
#include "rotation_transform.hpp"
#include "affine_transform.hpp"
#include "blur_transform.hpp"
#include "erasing_transform.hpp"

// Advanced transforms (v0.7.0)
#include "posterize_transform.hpp"
#include "solarize_transform.hpp"
#include "perspective_transform.hpp"
#include "autoaugment_transform.hpp"

// Tensor conversion
#include "tensor_conversion.hpp"

namespace turboloader {
namespace transforms {

/**
 * @brief Create a standard ImageNet training pipeline
 */
inline std::unique_ptr<TransformPipeline> create_imagenet_train_pipeline(
    int target_size = 224,
    bool to_tensor = true,
    TensorFormat format = TensorFormat::PYTORCH_CHW) {

    auto pipeline = std::make_unique<TransformPipeline>();

    // Random resized crop
    pipeline->add(std::make_unique<RandomCropTransform>(target_size, target_size, 32));

    // Random horizontal flip
    pipeline->add(std::make_unique<RandomHorizontalFlipTransform>(0.5f));

    // Color jitter
    pipeline->add(std::make_unique<ColorJitterTransform>(0.4f, 0.4f, 0.4f, 0.1f));

    // Normalize to [0,1] and optionally convert to tensor
    if (to_tensor) {
        pipeline->add(std::make_unique<ToTensorTransform>(format, true));
        pipeline->add(std::make_unique<ImageNetNormalize>(true));
    } else {
        pipeline->add(std::make_unique<ImageNetNormalize>(false));
    }

    return pipeline;
}

/**
 * @brief Create a standard ImageNet validation pipeline
 */
inline std::unique_ptr<TransformPipeline> create_imagenet_val_pipeline(
    int target_size = 224,
    bool to_tensor = true,
    TensorFormat format = TensorFormat::PYTORCH_CHW) {

    auto pipeline = std::make_unique<TransformPipeline>();

    // Resize to 256
    pipeline->add(std::make_unique<ResizeTransform>(256, 256));

    // Center crop to 224
    pipeline->add(std::make_unique<CenterCropTransform>(target_size, target_size));

    // Normalize
    if (to_tensor) {
        pipeline->add(std::make_unique<ToTensorTransform>(format, true));
        pipeline->add(std::make_unique<ImageNetNormalize>(true));
    } else {
        pipeline->add(std::make_unique<ImageNetNormalize>(false));
    }

    return pipeline;
}

} // namespace transforms
} // namespace turboloader
