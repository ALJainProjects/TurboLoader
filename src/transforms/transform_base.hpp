#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace turboloader {
namespace transforms {

/**
 * @brief Image data format for transforms
 */
struct ImageData {
    uint8_t* data;           // Raw pixel data
    int width;               // Image width
    int height;              // Image height
    int channels;            // Number of channels (1, 3, or 4)
    int stride;              // Row stride in bytes
    bool owns_data;          // Whether this struct owns the data

    ImageData(uint8_t* data_, int w, int h, int c, int stride_ = 0, bool owns = false)
        : data(data_), width(w), height(h), channels(c),
          stride(stride_ > 0 ? stride_ : w * c), owns_data(owns) {}

    ~ImageData() {
        if (owns_data && data) {
            delete[] data;
        }
    }

    // Prevent copying to avoid double-free
    ImageData(const ImageData&) = delete;
    ImageData& operator=(const ImageData&) = delete;

    // Allow moving
    ImageData(ImageData&& other) noexcept
        : data(other.data), width(other.width), height(other.height),
          channels(other.channels), stride(other.stride), owns_data(other.owns_data) {
        other.data = nullptr;
        other.owns_data = false;
    }

    ImageData& operator=(ImageData&& other) noexcept {
        if (this != &other) {
            if (owns_data && data) {
                delete[] data;
            }
            data = other.data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            stride = other.stride;
            owns_data = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    size_t size_bytes() const {
        return stride * height;
    }
};

/**
 * @brief Base class for all image transforms
 */
class Transform {
public:
    virtual ~Transform() = default;

    /**
     * @brief Apply transform to image
     * @param input Input image data
     * @return Transformed image data (may be same as input for in-place transforms)
     */
    virtual std::unique_ptr<ImageData> apply(const ImageData& input) = 0;

    /**
     * @brief Whether this transform is deterministic
     */
    virtual bool is_deterministic() const { return true; }

    /**
     * @brief Get transform name for debugging
     */
    virtual const char* name() const = 0;
};

/**
 * @brief Base class for random transforms
 */
class RandomTransform : public Transform {
protected:
    std::mt19937 rng_;
    float probability_;

public:
    RandomTransform(float prob = 1.0f, unsigned seed = std::random_device{}())
        : rng_(seed), probability_(prob) {}

    bool is_deterministic() const override { return false; }

    /**
     * @brief Check if transform should be applied based on probability
     */
    bool should_apply() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng_) < probability_;
    }

    void set_seed(unsigned seed) {
        rng_.seed(seed);
    }
};

/**
 * @brief Transform pipeline that chains multiple transforms
 */
class TransformPipeline {
private:
    std::vector<std::unique_ptr<Transform>> transforms_;

public:
    TransformPipeline() = default;

    void add(std::unique_ptr<Transform> transform) {
        transforms_.push_back(std::move(transform));
    }

    std::unique_ptr<ImageData> apply(const ImageData& input) {
        if (transforms_.empty()) {
            // No transforms, return copy
            auto output = std::make_unique<ImageData>(
                new uint8_t[input.size_bytes()],
                input.width, input.height, input.channels, input.stride, true
            );
            std::memcpy(output->data, input.data, input.size_bytes());
            return output;
        }

        // Apply first transform
        auto current = transforms_[0]->apply(input);

        // Apply remaining transforms
        for (size_t i = 1; i < transforms_.size(); ++i) {
            current = transforms_[i]->apply(*current);
        }

        return current;
    }

    size_t size() const { return transforms_.size(); }
    bool empty() const { return transforms_.empty(); }
};

} // namespace transforms
} // namespace turboloader
