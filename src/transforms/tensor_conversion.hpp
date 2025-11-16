/**
 * @file tensor_conversion.hpp
 * @brief Zero-copy tensor conversion for PyTorch and TensorFlow
 *
 * Provides efficient conversion from ImageData to:
 * - PyTorch tensors (CHW format, float32)
 * - TensorFlow tensors (HWC format, float32)
 *
 * Zero-copy optimization when memory layout allows.
 */

#pragma once

#include "transform_base.hpp"
#include "simd_utils.hpp"
#include <vector>
#include <memory>
#include <cstring>

namespace turboloader {
namespace transforms {

/**
 * @brief Tensor format (layout)
 */
enum class TensorFormat {
    NONE,          // No conversion
    PYTORCH_CHW,   // PyTorch: (C, H, W) float32
    TENSORFLOW_HWC // TensorFlow: (H, W, C) float32
};

/**
 * @brief Tensor data holder (for zero-copy support)
 */
struct TensorData {
    float* data;              // Float32 data pointer
    std::vector<int> shape;   // Tensor shape
    size_t size_bytes;        // Total size in bytes
    bool owns_data;           // Whether this struct owns the data

    TensorData(float* data_, std::vector<int> shape_, bool owns = false)
        : data(data_), shape(std::move(shape_)), owns_data(owns) {
        size_bytes = sizeof(float);
        for (int dim : shape) {
            size_bytes *= dim;
        }
    }

    ~TensorData() {
        if (owns_data && data) {
            delete[] data;
        }
    }

    // Prevent copying
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;

    // Allow moving
    TensorData(TensorData&& other) noexcept
        : data(other.data), shape(std::move(other.shape)),
          size_bytes(other.size_bytes), owns_data(other.owns_data) {
        other.data = nullptr;
        other.owns_data = false;
    }

    TensorData& operator=(TensorData&& other) noexcept {
        if (this != &other) {
            if (owns_data && data) {
                delete[] data;
            }
            data = other.data;
            shape = std::move(other.shape);
            size_bytes = other.size_bytes;
            owns_data = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }
};

/**
 * @brief Convert ImageData to PyTorch tensor (CHW format)
 *
 * PyTorch format: (C, H, W) float32, normalized to [0, 1]
 * Memory layout: all channel 0, then all channel 1, then all channel 2
 */
inline std::unique_ptr<TensorData> to_pytorch_tensor(const ImageData& image,
                                                      bool normalize = true) {
    size_t num_pixels = image.width * image.height;
    size_t num_elements = num_pixels * image.channels;

    auto data = new float[num_elements];
    auto tensor = std::make_unique<TensorData>(
        data,
        std::vector<int>{image.channels, image.height, image.width},
        true
    );

    if (normalize) {
        // Convert uint8 [0,255] to float32 [0,1] and transpose to CHW
        for (int c = 0; c < image.channels; ++c) {
            std::vector<uint8_t> channel_data(num_pixels);

            // Extract channel
            for (size_t i = 0; i < num_pixels; ++i) {
                channel_data[i] = image.data[i * image.channels + c];
            }

            // Convert to float and normalize (SIMD-accelerated)
            float* channel_output = data + c * num_pixels;
            simd::cvt_u8_to_f32_normalized(channel_data.data(), channel_output, num_pixels);
        }
    } else {
        // Just transpose to CHW without normalization
        for (int c = 0; c < image.channels; ++c) {
            for (size_t i = 0; i < num_pixels; ++i) {
                data[c * num_pixels + i] = static_cast<float>(
                    image.data[i * image.channels + c]
                );
            }
        }
    }

    return tensor;
}

/**
 * @brief Convert ImageData to TensorFlow tensor (HWC format)
 *
 * TensorFlow format: (H, W, C) float32, normalized to [0, 1]
 * Memory layout: same as ImageData (interleaved channels)
 */
inline std::unique_ptr<TensorData> to_tensorflow_tensor(const ImageData& image,
                                                         bool normalize = true) {
    size_t num_elements = image.width * image.height * image.channels;

    auto data = new float[num_elements];
    auto tensor = std::make_unique<TensorData>(
        data,
        std::vector<int>{image.height, image.width, image.channels},
        true
    );

    if (normalize) {
        // Convert uint8 [0,255] to float32 [0,1] (SIMD-accelerated)
        simd::cvt_u8_to_f32_normalized(image.data, data, num_elements);
    } else {
        // Just convert to float without normalization
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = static_cast<float>(image.data[i]);
        }
    }

    return tensor;
}

/**
 * @brief Convert PyTorch tensor back to ImageData
 */
inline std::unique_ptr<ImageData> from_pytorch_tensor(const TensorData& tensor) {
    if (tensor.shape.size() != 3) {
        throw std::runtime_error("Expected 3D tensor (C, H, W)");
    }

    int channels = tensor.shape[0];
    int height = tensor.shape[1];
    int width = tensor.shape[2];

    size_t num_pixels = width * height;
    size_t output_size = num_pixels * channels;

    auto output = std::make_unique<ImageData>(
        new uint8_t[output_size],
        width, height, channels, width * channels, true
    );

    // Convert from CHW to HWC and float to uint8
    for (int c = 0; c < channels; ++c) {
        const float* channel_data = tensor.data + c * num_pixels;
        std::vector<uint8_t> channel_u8(num_pixels);

        simd::cvt_f32_to_u8_clamped(channel_data, channel_u8.data(), num_pixels);

        // Interleave
        for (size_t i = 0; i < num_pixels; ++i) {
            output->data[i * channels + c] = channel_u8[i];
        }
    }

    return output;
}

/**
 * @brief Convert TensorFlow tensor back to ImageData
 */
inline std::unique_ptr<ImageData> from_tensorflow_tensor(const TensorData& tensor) {
    if (tensor.shape.size() != 3) {
        throw std::runtime_error("Expected 3D tensor (H, W, C)");
    }

    int height = tensor.shape[0];
    int width = tensor.shape[1];
    int channels = tensor.shape[2];

    size_t num_elements = width * height * channels;

    auto output = std::make_unique<ImageData>(
        new uint8_t[num_elements],
        width, height, channels, width * channels, true
    );

    // Convert float to uint8 (SIMD-accelerated)
    simd::cvt_f32_to_u8_clamped(tensor.data, output->data, num_elements);

    return output;
}

/**
 * @brief Batch tensor converter
 */
class BatchTensorConverter {
public:
    /**
     * @brief Convert batch of images to tensor
     * @param images Vector of ImageData pointers
     * @param format Target tensor format
     * @param normalize Normalize to [0,1]
     * @return Batched tensor with shape (N, ...) where N is batch size
     */
    static std::unique_ptr<TensorData> convert_batch(
        const std::vector<const ImageData*>& images,
        TensorFormat format,
        bool normalize = true) {

        if (images.empty()) {
            throw std::runtime_error("Empty batch");
        }

        // Verify all images have same dimensions
        int width = images[0]->width;
        int height = images[0]->height;
        int channels = images[0]->channels;

        for (const auto* img : images) {
            if (img->width != width || img->height != height || img->channels != channels) {
                throw std::runtime_error("All images in batch must have same dimensions");
            }
        }

        int batch_size = images.size();

        if (format == TensorFormat::PYTORCH_CHW) {
            // PyTorch: (N, C, H, W)
            size_t single_size = width * height * channels;
            size_t total_size = batch_size * single_size;
            auto data = new float[total_size];

            auto tensor = std::make_unique<TensorData>(
                data,
                std::vector<int>{batch_size, channels, height, width},
                true
            );

            for (int n = 0; n < batch_size; ++n) {
                auto single = to_pytorch_tensor(*images[n], normalize);
                std::memcpy(data + n * single_size, single->data, single_size * sizeof(float));
            }

            return tensor;

        } else if (format == TensorFormat::TENSORFLOW_HWC) {
            // TensorFlow: (N, H, W, C)
            size_t single_size = width * height * channels;
            size_t total_size = batch_size * single_size;
            auto data = new float[total_size];

            auto tensor = std::make_unique<TensorData>(
                data,
                std::vector<int>{batch_size, height, width, channels},
                true
            );

            for (int n = 0; n < batch_size; ++n) {
                auto single = to_tensorflow_tensor(*images[n], normalize);
                std::memcpy(data + n * single_size, single->data, single_size * sizeof(float));
            }

            return tensor;
        }

        throw std::runtime_error("Unsupported tensor format");
    }
};

/**
 * @brief Transform that converts to tensor format
 */
class ToTensorTransform : public Transform {
public:
    explicit ToTensorTransform(TensorFormat format = TensorFormat::PYTORCH_CHW,
                              bool normalize = true)
        : format_(format), normalize_(normalize) {}

    std::unique_ptr<ImageData> apply(const ImageData& input) override {
        // Note: This returns ImageData but the data is actually float32
        // For proper tensor output, use to_pytorch_tensor or to_tensorflow_tensor directly

        if (format_ == TensorFormat::PYTORCH_CHW) {
            auto tensor = to_pytorch_tensor(input, normalize_);

            // Wrap tensor data as ImageData (for pipeline compatibility)
            auto output = std::make_unique<ImageData>(
                reinterpret_cast<uint8_t*>(tensor->data),
                input.width, input.height, input.channels,
                input.width * input.channels * sizeof(float), true
            );

            // Transfer ownership
            tensor->owns_data = false;
            return output;

        } else if (format_ == TensorFormat::TENSORFLOW_HWC) {
            auto tensor = to_tensorflow_tensor(input, normalize_);

            auto output = std::make_unique<ImageData>(
                reinterpret_cast<uint8_t*>(tensor->data),
                input.width, input.height, input.channels,
                input.width * input.channels * sizeof(float), true
            );

            tensor->owns_data = false;
            return output;
        }

        // No conversion
        auto output = std::make_unique<ImageData>(
            new uint8_t[input.size_bytes()],
            input.width, input.height, input.channels, input.stride, true
        );
        std::memcpy(output->data, input.data, input.size_bytes());
        return output;
    }

    const char* name() const override { return "ToTensor"; }

private:
    TensorFormat format_;
    bool normalize_;
};

} // namespace transforms
} // namespace turboloader
