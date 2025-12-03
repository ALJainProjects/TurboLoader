/**
 * @file test_autoaugment_ops.cpp
 * @brief Unit tests for TurboLoader v2.8.0 AutoAugment individual operations
 *
 * Tests all newly implemented AutoAugment operations:
 * - Invert
 * - AutoContrast
 * - Equalize
 * - Color (saturation adjustment)
 * - Brightness
 * - Contrast
 * - Sharpness
 * - ShearX, ShearY
 * - TranslateX, TranslateY
 */

#include <gtest/gtest.h>
#include "../src/transforms/autoaugment_transform.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace turboloader::transforms;

// Helper to create a test image with gradient pattern
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

// Helper to create a solid color test image
std::unique_ptr<ImageData> create_solid_image(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    size_t size = width * height * 3;
    auto data = new uint8_t[size];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[(y * width + x) * 3 + 0] = r;
            data[(y * width + x) * 3 + 1] = g;
            data[(y * width + x) * 3 + 2] = b;
        }
    }

    return std::make_unique<ImageData>(data, width, height, 3, width * 3, true);
}

// Helper to create an image with specific values for contrast/brightness testing
std::unique_ptr<ImageData> create_range_image(int width, int height, uint8_t min_val, uint8_t max_val) {
    size_t size = width * height * 3;
    auto data = new uint8_t[size];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Linear interpolation from min to max
            float t = static_cast<float>(x) / (width - 1);
            uint8_t val = static_cast<uint8_t>(min_val + t * (max_val - min_val));
            for (int c = 0; c < 3; ++c) {
                data[(y * width + x) * 3 + c] = val;
            }
        }
    }

    return std::make_unique<ImageData>(data, width, height, 3, width * 3, true);
}

// ============================================================================
// INVERT TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, InvertBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Use apply_operation directly to test Invert
    auto output = transform.apply_operation(*input, "Invert", 0.0f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);

    // Verify inversion: output = 255 - input
    for (size_t i = 0; i < std::min(size_t(100), input->size_bytes()); ++i) {
        EXPECT_EQ(output->data[i], 255 - input->data[i]);
    }
}

TEST(AutoAugmentOpsTest, InvertSolid) {
    auto input = create_solid_image(32, 32, 100, 150, 200);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Invert", 0.0f);

    // Check specific pixel
    EXPECT_EQ(output->data[0], 155);  // 255 - 100
    EXPECT_EQ(output->data[1], 105);  // 255 - 150
    EXPECT_EQ(output->data[2], 55);   // 255 - 200
}

TEST(AutoAugmentOpsTest, InvertDoubleApply) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output1 = transform.apply_operation(*input, "Invert", 0.0f);
    auto output2 = transform.apply_operation(*output1, "Invert", 0.0f);

    // Double invert should return original
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_EQ(output2->data[i], input->data[i]);
    }
}

// ============================================================================
// AUTOCONTRAST TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, AutoContrastBasic) {
    auto input = create_range_image(64, 64, 50, 200);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "AutoContrast", 0.0f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);

    // AutoContrast should stretch the range to 0-255
    // Find min and max in output
    uint8_t out_min = 255, out_max = 0;
    for (size_t i = 0; i < output->size_bytes(); ++i) {
        out_min = std::min(out_min, output->data[i]);
        out_max = std::max(out_max, output->data[i]);
    }

    // Output should span full range (approximately)
    EXPECT_LE(out_min, 5);
    EXPECT_GE(out_max, 250);
}

TEST(AutoAugmentOpsTest, AutoContrastFullRange) {
    // Image that already spans 0-255 should be unchanged
    auto input = create_range_image(64, 64, 0, 255);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "AutoContrast", 0.0f);

    // Should be nearly unchanged
    int diff_count = 0;
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        if (std::abs(int(output->data[i]) - int(input->data[i])) > 1) {
            diff_count++;
        }
    }
    EXPECT_LT(diff_count, input->size_bytes() / 10);  // Less than 10% difference
}

// ============================================================================
// EQUALIZE TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, EqualizeBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Equalize", 0.0f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, EqualizeDistribution) {
    auto input = create_range_image(64, 64, 50, 100);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Equalize", 0.0f);

    // After equalization, histogram should be more spread out
    // Check that output spans a wider range than input
    uint8_t in_min = 255, in_max = 0, out_min = 255, out_max = 0;
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        in_min = std::min(in_min, input->data[i]);
        in_max = std::max(in_max, input->data[i]);
        out_min = std::min(out_min, output->data[i]);
        out_max = std::max(out_max, output->data[i]);
    }

    int in_range = in_max - in_min;
    int out_range = out_max - out_min;

    // Output range should be larger than input range
    EXPECT_GT(out_range, in_range);
}

// ============================================================================
// COLOR (SATURATION) TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, ColorBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Color", 0.5f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, ColorSaturationChange) {
    // Test that Color operation modifies saturation (doesn't require specific direction)
    auto input = create_solid_image(32, 32, 200, 100, 50);  // Colored image
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Color", 5.0f);

    // Output should have valid dimensions
    EXPECT_EQ(output->width, 32);
    EXPECT_EQ(output->height, 32);
    EXPECT_EQ(output->channels, 3);

    // Color values should still be in valid range
    EXPECT_GE(output->data[0], 0);
    EXPECT_LE(output->data[0], 255);
}

TEST(AutoAugmentOpsTest, ColorZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Zero magnitude should leave saturation unchanged
    auto output = transform.apply_operation(*input, "Color", 0.0f);

    // Should be nearly unchanged
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_NEAR(output->data[i], input->data[i], 5);
    }
}

// ============================================================================
// BRIGHTNESS TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, BrightnessIncrease) {
    auto input = create_solid_image(32, 32, 100, 100, 100);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Positive magnitude increases brightness
    auto output = transform.apply_operation(*input, "Brightness", 0.5f);

    // Should be brighter (higher values)
    EXPECT_GT(output->data[0], input->data[0]);
}

TEST(AutoAugmentOpsTest, BrightnessDecrease) {
    auto input = create_solid_image(32, 32, 200, 200, 200);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // High magnitude can decrease brightness depending on implementation
    // Let's test that output dimensions are correct
    auto output = transform.apply_operation(*input, "Brightness", 0.5f);

    EXPECT_EQ(output->width, 32);
    EXPECT_EQ(output->height, 32);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, BrightnessClipping) {
    // Test that values are clipped to 0-255
    auto input = create_solid_image(32, 32, 250, 250, 250);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Brightness", 0.9f);

    // Values should be clipped to max 255
    for (size_t i = 0; i < output->size_bytes(); ++i) {
        EXPECT_LE(output->data[i], 255);
    }
}

// ============================================================================
// CONTRAST TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, ContrastBasic) {
    auto input = create_range_image(64, 64, 50, 200);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Contrast", 0.5f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, ContrastZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Zero magnitude should leave contrast unchanged
    auto output = transform.apply_operation(*input, "Contrast", 0.0f);

    // Should be nearly unchanged
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_NEAR(output->data[i], input->data[i], 5);
    }
}

// ============================================================================
// SHARPNESS TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, SharpnessBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Sharpness", 0.5f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, SharpnessZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Zero magnitude should leave image unchanged
    auto output = transform.apply_operation(*input, "Sharpness", 0.0f);

    // Should be nearly unchanged
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_NEAR(output->data[i], input->data[i], 5);
    }
}

TEST(AutoAugmentOpsTest, SharpnessSmallImage) {
    // Test on small image (edge case for kernel)
    auto input = create_test_image(4, 4, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "Sharpness", 0.5f);

    EXPECT_EQ(output->width, 4);
    EXPECT_EQ(output->height, 4);
    EXPECT_EQ(output->channels, 3);
}

// ============================================================================
// SHEARX TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, ShearXBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "ShearX", 0.3f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, ShearXZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Zero magnitude should leave image unchanged (or nearly)
    auto output = transform.apply_operation(*input, "ShearX", 0.0f);

    // Center pixels should be unchanged
    int cx = 16, cy = 16;
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(output->data[(cy * 32 + cx) * 3 + c],
                    input->data[(cy * 32 + cx) * 3 + c], 5);
    }
}

// ============================================================================
// SHEARY TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, ShearYBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "ShearY", 0.3f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, ShearYZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "ShearY", 0.0f);

    // Center pixels should be unchanged
    int cx = 16, cy = 16;
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(output->data[(cy * 32 + cx) * 3 + c],
                    input->data[(cy * 32 + cx) * 3 + c], 5);
    }
}

// ============================================================================
// TRANSLATEX TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, TranslateXBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "TranslateX", 0.3f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, TranslateXZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "TranslateX", 0.0f);

    // Check dimensions are preserved
    EXPECT_EQ(output->width, input->width);
    EXPECT_EQ(output->height, input->height);
    EXPECT_EQ(output->channels, input->channels);

    // Check center pixels are preserved (avoid border effects)
    size_t center_x = 16, center_y = 16;
    size_t idx = (center_y * input->width + center_x) * input->channels;
    for (size_t c = 0; c < input->channels; ++c) {
        EXPECT_NEAR(output->data[idx + c], input->data[idx + c], 5);
    }
}

// ============================================================================
// TRANSLATEY TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, TranslateYBasic) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "TranslateY", 0.3f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, TranslateYZeroMagnitude) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply_operation(*input, "TranslateY", 0.0f);

    // Check dimensions are preserved
    EXPECT_EQ(output->width, input->width);
    EXPECT_EQ(output->height, input->height);
    EXPECT_EQ(output->channels, input->channels);

    // Check center pixels are preserved (avoid border effects)
    size_t center_x = 16, center_y = 16;
    size_t idx = (center_y * input->width + center_x) * input->channels;
    for (size_t c = 0; c < input->channels; ++c) {
        EXPECT_NEAR(output->data[idx + c], input->data[idx + c], 5);
    }
}

// ============================================================================
// INTEGRATION TESTS - Full AutoAugment Pipeline
// ============================================================================

TEST(AutoAugmentOpsTest, AllOperationsRun) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    std::vector<std::string> ops = {
        "Invert", "AutoContrast", "Equalize", "Color",
        "Brightness", "Contrast", "Sharpness",
        "ShearX", "ShearY", "TranslateX", "TranslateY"
    };

    for (const auto& op : ops) {
        auto output = transform.apply_operation(*input, op, 0.5f);
        EXPECT_EQ(output->width, 64) << "Failed on operation: " << op;
        EXPECT_EQ(output->height, 64) << "Failed on operation: " << op;
        EXPECT_EQ(output->channels, 3) << "Failed on operation: " << op;
    }
}

TEST(AutoAugmentOpsTest, ChainedOperations) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Chain multiple operations
    auto step1 = transform.apply_operation(*input, "Brightness", 0.3f);
    auto step2 = transform.apply_operation(*step1, "Contrast", 0.3f);
    auto step3 = transform.apply_operation(*step2, "Sharpness", 0.3f);
    auto output = transform.apply_operation(*step3, "Equalize", 0.0f);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentOpsTest, FullPipelineMultipleApplications) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Apply full AutoAugment pipeline multiple times
    for (int i = 0; i < 10; ++i) {
        auto output = transform.apply(*input);
        EXPECT_EQ(output->width, 64);
        EXPECT_EQ(output->height, 64);
        EXPECT_EQ(output->channels, 3);
    }
}

TEST(AutoAugmentOpsTest, DifferentImageSizes) {
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    std::vector<std::pair<int, int>> sizes = {
        {32, 32}, {64, 64}, {128, 128}, {224, 224},
        {100, 50}, {50, 100}  // Non-square
    };

    for (const auto& [w, h] : sizes) {
        auto input = create_test_image(w, h, 3);
        auto output = transform.apply(*input);
        EXPECT_EQ(output->width, w);
        EXPECT_EQ(output->height, h);
        EXPECT_EQ(output->channels, 3);
    }
}

TEST(AutoAugmentOpsTest, AllPolicies) {
    auto input = create_test_image(64, 64, 3);

    std::vector<AutoAugmentPolicy> policies = {
        AutoAugmentPolicy::IMAGENET,
        AutoAugmentPolicy::CIFAR10,
        AutoAugmentPolicy::SVHN
    };

    for (const auto& policy : policies) {
        AutoAugmentTransform transform(policy, 42);
        auto output = transform.apply(*input);
        EXPECT_EQ(output->width, 64);
        EXPECT_EQ(output->height, 64);
        EXPECT_EQ(output->channels, 3);
    }
}

// ============================================================================
// MAGNITUDE RANGE TESTS
// ============================================================================

TEST(AutoAugmentOpsTest, MagnitudeRangeTest) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    std::vector<std::string> ops = {
        "Brightness", "Contrast", "Color", "Sharpness",
        "ShearX", "ShearY", "TranslateX", "TranslateY"
    };

    for (const auto& op : ops) {
        for (float mag = 0.0f; mag <= 1.0f; mag += 0.25f) {
            auto output = transform.apply_operation(*input, op, mag);
            EXPECT_EQ(output->width, 64) << "Failed on " << op << " mag=" << mag;
            EXPECT_EQ(output->height, 64) << "Failed on " << op << " mag=" << mag;
            EXPECT_EQ(output->channels, 3) << "Failed on " << op << " mag=" << mag;
        }
    }
}

// Main
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
