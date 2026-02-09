/**
 * @file test_transforms.cpp
 * @brief Unit tests for TurboLoader transforms
 */

#include <gtest/gtest.h>
#include "../src/transforms/transforms.hpp"
#include <cmath>
#include <vector>

using namespace turboloader::transforms;

// Helper to create a test image
std::unique_ptr<ImageData> create_test_image(int width, int height, int channels = 3) {
    size_t size = width * height * channels;
    auto data = new uint8_t[size];

    // Create gradient pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                data[(y * width + x) * channels + c] = (x + y + c * 50) % 256;
            }
        }
    }

    return std::make_unique<ImageData>(data, width, height, channels,
                                       width * channels, true);
}

// Test Resize
TEST(TransformTest, ResizeNearest) {
    auto input = create_test_image(100, 100, 3);
    ResizeTransform resize(50, 50, InterpolationMode::NEAREST);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

TEST(TransformTest, ResizeBilinear) {
    auto input = create_test_image(100, 100, 3);
    ResizeTransform resize(200, 200, InterpolationMode::BILINEAR);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 200);
    EXPECT_EQ(output->height, 200);
    EXPECT_EQ(output->channels, 3);
}

TEST(TransformTest, ResizeBicubic) {
    auto input = create_test_image(100, 100, 3);
    ResizeTransform resize(150, 150, InterpolationMode::BICUBIC);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 150);
    EXPECT_EQ(output->height, 150);
    EXPECT_EQ(output->channels, 3);
}

// Test Normalize
TEST(TransformTest, Normalize) {
    auto input = create_test_image(10, 10, 3);
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    NormalizeTransform normalize(mean, std, false);
    auto output = normalize.apply(*input);

    EXPECT_EQ(output->width, 10);
    EXPECT_EQ(output->height, 10);
    EXPECT_EQ(output->channels, 3);
}

TEST(TransformTest, ImageNetNormalize) {
    auto input = create_test_image(10, 10, 3);
    ImageNetNormalize normalize(false);

    auto output = normalize.apply(*input);

    EXPECT_EQ(output->width, 10);
    EXPECT_EQ(output->height, 10);
    EXPECT_EQ(output->channels, 3);
}

// Test Normalize uint8 roundtrip (denormalization must include *255)
TEST(TransformTest, NormalizeUint8Roundtrip) {
    // Create a uniform image with known pixel value
    int width = 4, height = 4, channels = 3;
    size_t size = width * height * channels;
    auto data = new uint8_t[size];
    // Set all pixels to 128
    std::memset(data, 128, size);
    auto input = std::make_unique<ImageData>(data, width, height, channels,
                                              width * channels, true);

    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> std_vals = {0.5f, 0.5f, 0.5f};

    // With these values: normalize = (128/255 - 0.5) / 0.5 ≈ 0.00392
    // Denormalize: (0.00392 * 0.5 + 0.5) * 255 ≈ 128
    NormalizeTransform normalize(mean, std_vals, false);
    auto output = normalize.apply(*input);

    EXPECT_EQ(output->width, width);
    EXPECT_EQ(output->height, height);

    // After normalize+denormalize roundtrip, pixel values should be close to original
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(output->data[i], 128, 2)
            << "Pixel " << i << " diverged after normalize roundtrip";
    }
}

// Test Normalize denorm produces values in valid [0,255] range
TEST(TransformTest, NormalizeDenormRange) {
    auto input = create_test_image(10, 10, 3);
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_vals = {0.229f, 0.224f, 0.225f};

    NormalizeTransform normalize(mean, std_vals, false);
    auto output = normalize.apply(*input);

    size_t size = 10 * 10 * 3;
    for (size_t i = 0; i < size; ++i) {
        // uint8 values are always in [0,255] by type, but we verify
        // the denorm formula doesn't produce garbage (e.g., all zeros or all 255)
        // by checking that at least some variation exists
    }
    // Verify we have pixel variation (not all clamped to 0 or 255)
    int min_val = 255, max_val = 0;
    for (size_t i = 0; i < size; ++i) {
        min_val = std::min(min_val, (int)output->data[i]);
        max_val = std::max(max_val, (int)output->data[i]);
    }
    EXPECT_LT(min_val, 200) << "All pixels clamped high - denorm likely broken";
    EXPECT_GT(max_val, 50) << "All pixels clamped low - denorm likely broken";
}

// Test Flips
TEST(TransformTest, HorizontalFlip) {
    auto input = create_test_image(10, 10, 3);
    uint8_t first_pixel = input->data[0];
    uint8_t last_pixel_in_row = input->data[(10 - 1) * 3];

    RandomHorizontalFlipTransform flip(1.0f, 42);  // Always flip
    auto output = flip.apply(*input);

    EXPECT_EQ(output->width, 10);
    EXPECT_EQ(output->height, 10);

    // Check if flipped (first pixel should become last pixel in row)
    EXPECT_EQ(output->data[(10 - 1) * 3], first_pixel);
}

TEST(TransformTest, VerticalFlip) {
    auto input = create_test_image(10, 10, 3);
    uint8_t first_pixel = input->data[0];
    uint8_t last_row_first_pixel = input->data[(10 - 1) * 10 * 3];

    RandomVerticalFlipTransform flip(1.0f, 42);  // Always flip
    auto output = flip.apply(*input);

    EXPECT_EQ(output->width, 10);
    EXPECT_EQ(output->height, 10);
}

// Test Crops
TEST(TransformTest, CenterCrop) {
    auto input = create_test_image(100, 100, 3);
    CenterCropTransform crop(50, 50);

    auto output = crop.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

TEST(TransformTest, RandomCrop) {
    auto input = create_test_image(100, 100, 3);
    RandomCropTransform crop(50, 50, 10, PaddingMode::CONSTANT, 0, 42);

    auto output = crop.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test ColorJitter
TEST(TransformTest, ColorJitter) {
    auto input = create_test_image(50, 50, 3);
    ColorJitterTransform jitter(0.5f, 0.5f, 0.5f, 0.1f, 42);

    auto output = jitter.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test Grayscale
TEST(TransformTest, GrayscaleToSingleChannel) {
    auto input = create_test_image(50, 50, 3);
    GrayscaleTransform gray(1);

    auto output = gray.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 1);
}

TEST(TransformTest, GrayscaleToThreeChannels) {
    auto input = create_test_image(50, 50, 3);
    GrayscaleTransform gray(3);

    auto output = gray.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);

    // Check that all channels are equal
    EXPECT_EQ(output->data[0], output->data[1]);
    EXPECT_EQ(output->data[1], output->data[2]);
}

// Test Pad
TEST(TransformTest, PadConstant) {
    auto input = create_test_image(50, 50, 3);
    PadTransform pad(10, PaddingMode::CONSTANT, 0);

    auto output = pad.apply(*input);

    EXPECT_EQ(output->width, 70);
    EXPECT_EQ(output->height, 70);
    EXPECT_EQ(output->channels, 3);
}

TEST(TransformTest, PadEdge) {
    auto input = create_test_image(50, 50, 3);
    PadTransform pad(10, PaddingMode::EDGE, 0);

    auto output = pad.apply(*input);

    EXPECT_EQ(output->width, 70);
    EXPECT_EQ(output->height, 70);
    EXPECT_EQ(output->channels, 3);
}

// Test Rotation
TEST(TransformTest, RandomRotation) {
    auto input = create_test_image(50, 50, 3);
    RandomRotationTransform rotate(45.0f, false, 0, 42);

    auto output = rotate.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test Affine
TEST(TransformTest, RandomAffine) {
    auto input = create_test_image(50, 50, 3);
    RandomAffineTransform affine(15.0f, 0.1f, 0.1f, 0.9f, 1.1f, 10.0f, 0, 42);

    auto output = affine.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test Gaussian Blur
TEST(TransformTest, GaussianBlur) {
    auto input = create_test_image(50, 50, 3);
    GaussianBlurTransform blur(5, 1.5f);

    auto output = blur.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test Random Erasing
TEST(TransformTest, RandomErasing) {
    auto input = create_test_image(50, 50, 3);
    RandomErasingTransform erase(1.0f, 0.02f, 0.33f, 0.3f, 3.33f, 0, 42);

    auto output = erase.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test Tensor Conversion
TEST(TransformTest, ToPyTorchTensor) {
    auto input = create_test_image(10, 10, 3);
    auto tensor = to_pytorch_tensor(*input, true);

    EXPECT_EQ(tensor->shape.size(), 3);
    EXPECT_EQ(tensor->shape[0], 3);   // C
    EXPECT_EQ(tensor->shape[1], 10);  // H
    EXPECT_EQ(tensor->shape[2], 10);  // W

    // Check normalization (should be in [0, 1])
    for (size_t i = 0; i < 10 * 10 * 3; ++i) {
        EXPECT_GE(tensor->data[i], 0.0f);
        EXPECT_LE(tensor->data[i], 1.0f);
    }
}

TEST(TransformTest, ToTensorFlowTensor) {
    auto input = create_test_image(10, 10, 3);
    auto tensor = to_tensorflow_tensor(*input, true);

    EXPECT_EQ(tensor->shape.size(), 3);
    EXPECT_EQ(tensor->shape[0], 10);  // H
    EXPECT_EQ(tensor->shape[1], 10);  // W
    EXPECT_EQ(tensor->shape[2], 3);   // C

    // Check normalization
    for (size_t i = 0; i < 10 * 10 * 3; ++i) {
        EXPECT_GE(tensor->data[i], 0.0f);
        EXPECT_LE(tensor->data[i], 1.0f);
    }
}

TEST(TransformTest, TensorRoundTrip) {
    auto input = create_test_image(10, 10, 3);

    // PyTorch: Image -> Tensor -> Image
    auto pt_tensor = to_pytorch_tensor(*input, true);
    auto pt_output = from_pytorch_tensor(*pt_tensor);

    EXPECT_EQ(pt_output->width, 10);
    EXPECT_EQ(pt_output->height, 10);
    EXPECT_EQ(pt_output->channels, 3);

    // TensorFlow: Image -> Tensor -> Image
    auto tf_tensor = to_tensorflow_tensor(*input, true);
    auto tf_output = from_tensorflow_tensor(*tf_tensor);

    EXPECT_EQ(tf_output->width, 10);
    EXPECT_EQ(tf_output->height, 10);
    EXPECT_EQ(tf_output->channels, 3);
}

// Test Pipeline
TEST(TransformTest, Pipeline) {
    auto input = create_test_image(100, 100, 3);

    TransformPipeline pipeline;
    pipeline.add(std::make_unique<ResizeTransform>(50, 50));
    pipeline.add(std::make_unique<RandomHorizontalFlipTransform>(1.0f, 42));
    pipeline.add(std::make_unique<GrayscaleTransform>(3));

    auto output = pipeline.apply(*input);

    EXPECT_EQ(output->width, 50);
    EXPECT_EQ(output->height, 50);
    EXPECT_EQ(output->channels, 3);
}

// Test SIMD utilities
TEST(SIMDTest, Uint8ToFloat32Normalized) {
    std::vector<uint8_t> input = {0, 127, 255};
    std::vector<float> output(3);

    simd::cvt_u8_to_f32_normalized(input.data(), output.data(), 3);

    EXPECT_NEAR(output[0], 0.0f, 0.01f);
    EXPECT_NEAR(output[1], 0.498f, 0.01f);
    EXPECT_NEAR(output[2], 1.0f, 0.01f);
}

TEST(SIMDTest, Float32ToUint8Clamped) {
    std::vector<float> input = {0.0f, 0.5f, 1.0f, 1.5f, -0.5f};
    std::vector<uint8_t> output(5);

    simd::cvt_f32_to_u8_clamped(input.data(), output.data(), 5);

    EXPECT_EQ(output[0], 0);
    EXPECT_NEAR(output[1], 127, 1);
    EXPECT_EQ(output[2], 255);
    EXPECT_EQ(output[3], 255);  // Clamped
    EXPECT_EQ(output[4], 0);    // Clamped
}

TEST(SIMDTest, RGBToGrayscale) {
    std::vector<uint8_t> rgb = {255, 0, 0};  // Pure red
    std::vector<uint8_t> gray(1);

    simd::rgb_to_grayscale(rgb.data(), gray.data(), 1);

    // R=255 * 0.299 = 76.245
    EXPECT_NEAR(gray[0], 76, 2);
}

TEST(SIMDTest, HSVConversion) {
    uint8_t r = 255, g = 0, b = 0;  // Pure red
    float h, s, v;

    simd::rgb_to_hsv(r, g, b, h, s, v);

    EXPECT_NEAR(h, 0.0f, 1.0f);     // Red is at 0 degrees
    EXPECT_NEAR(s, 1.0f, 0.01f);    // Full saturation
    EXPECT_NEAR(v, 1.0f, 0.01f);    // Full value

    // Convert back
    uint8_t r2, g2, b2;
    simd::hsv_to_rgb(h, s, v, r2, g2, b2);

    EXPECT_NEAR(r2, r, 1);
    EXPECT_NEAR(g2, g, 1);
    EXPECT_NEAR(b2, b, 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
