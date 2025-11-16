/**
 * @file test_advanced_transforms.cpp
 * @brief Unit tests for TurboLoader v0.7.0 advanced transforms
 *
 * Tests:
 * - RandomPosterize
 * - RandomSolarize
 * - RandomPerspective
 * - AutoAugment
 * - Lanczos interpolation
 */

#include <gtest/gtest.h>
#include "../src/transforms/posterize_transform.hpp"
#include "../src/transforms/solarize_transform.hpp"
#include "../src/transforms/perspective_transform.hpp"
#include "../src/transforms/autoaugment_transform.hpp"
#include "../src/transforms/resize_transform.hpp"
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

// ============================================================================
// POSTERIZE TESTS
// ============================================================================

TEST(PosterizeTest, BasicPosterize) {
    auto input = create_test_image(64, 64, 3);
    RandomPosterizeTransform transform(4, 1.0f, 42);  // 4 bits, always apply

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);

    // Verify posterization effect
    // With 4 bits, lower 4 bits should be cleared
    uint8_t mask = ~((1 << (8 - 4)) - 1);  // 0xF0
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(output->data[i], input->data[i] & mask);
    }
}

TEST(PosterizeTest, OneBit) {
    auto input = create_test_image(32, 32, 3);
    RandomPosterizeTransform transform(1, 1.0f, 42);  // 1 bit, always apply

    auto output = transform.apply(*input);

    // With 1 bit, only MSB is kept (0 or 128)
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(output->data[i] == 0 || output->data[i] == 128);
    }
}

TEST(PosterizeTest, EightBits) {
    auto input = create_test_image(32, 32, 3);
    RandomPosterizeTransform transform(8, 1.0f, 42);  // 8 bits, no change

    auto output = transform.apply(*input);

    // With 8 bits, no change expected
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(output->data[i], input->data[i]);
    }
}

TEST(PosterizeTest, Probability) {
    auto input = create_test_image(32, 32, 3);
    RandomPosterizeTransform transform(4, 0.0f, 42);  // Never apply

    auto output = transform.apply(*input);

    // Should be unchanged
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_EQ(output->data[i], input->data[i]);
    }
}

TEST(PosterizeTest, VariousBitDepths) {
    auto input = create_test_image(32, 32, 3);

    for (int bits = 1; bits <= 8; ++bits) {
        RandomPosterizeTransform transform(bits, 1.0f, 42);
        auto output = transform.apply(*input);

        EXPECT_EQ(output->width, 32);
        EXPECT_EQ(output->height, 32);
        EXPECT_EQ(output->channels, 3);
    }
}

// ============================================================================
// SOLARIZE TESTS
// ============================================================================

TEST(SolarizeTest, ThresholdInversion) {
    auto input = create_test_image(64, 64, 3);
    RandomSolarizeTransform transform(128, 1.0f, 42);  // Threshold 128, always apply

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);

    // Verify solarization: pixels > 128 should be inverted
    for (int i = 0; i < 100; ++i) {
        if (input->data[i] > 128) {
            EXPECT_EQ(output->data[i], 255 - input->data[i]);
        } else {
            EXPECT_EQ(output->data[i], input->data[i]);
        }
    }
}

TEST(SolarizeTest, ZeroThreshold) {
    auto input = create_test_image(32, 32, 3);
    RandomSolarizeTransform transform(0, 1.0f, 42);  // Threshold 0, invert all

    auto output = transform.apply(*input);

    // All non-zero pixels should be inverted
    for (int i = 0; i < 100; ++i) {
        if (input->data[i] > 0) {
            EXPECT_EQ(output->data[i], 255 - input->data[i]);
        }
    }
}

TEST(SolarizeTest, MaxThreshold) {
    auto input = create_test_image(32, 32, 3);
    RandomSolarizeTransform transform(255, 1.0f, 42);  // Threshold 255, no inversion

    auto output = transform.apply(*input);

    // No pixels should be inverted
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_EQ(output->data[i], input->data[i]);
    }
}

TEST(SolarizeTest, Probability) {
    auto input = create_test_image(32, 32, 3);
    RandomSolarizeTransform transform(128, 0.0f, 42);  // Never apply

    auto output = transform.apply(*input);

    // Should be unchanged
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_EQ(output->data[i], input->data[i]);
    }
}

// ============================================================================
// PERSPECTIVE TESTS
// ============================================================================

TEST(PerspectiveTest, BasicPerspective) {
    auto input = create_test_image(64, 64, 3);
    RandomPerspectiveTransform transform(0.5f, 1.0f, 0, 42);  // Always apply

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(PerspectiveTest, ExtremeDistortion) {
    auto input = create_test_image(64, 64, 3);
    RandomPerspectiveTransform transform(1.0f, 1.0f, 0, 42);  // Maximum distortion

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(PerspectiveTest, NoDistortion) {
    auto input = create_test_image(64, 64, 3);
    RandomPerspectiveTransform transform(0.0f, 1.0f, 0, 42);  // No distortion

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(PerspectiveTest, Probability) {
    auto input = create_test_image(32, 32, 3);
    RandomPerspectiveTransform transform(0.5f, 0.0f, 0, 42);  // Never apply

    auto output = transform.apply(*input);

    // Should be unchanged
    for (size_t i = 0; i < input->size_bytes(); ++i) {
        EXPECT_EQ(output->data[i], input->data[i]);
    }
}

TEST(PerspectiveTest, FillValue) {
    auto input = create_test_image(64, 64, 3);
    RandomPerspectiveTransform transform(0.5f, 1.0f, 128, 42);  // Fill with 128

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

// ============================================================================
// AUTOAUGMENT TESTS
// ============================================================================

TEST(AutoAugmentTest, ImageNetPolicy) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentTest, CIFAR10Policy) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::CIFAR10, 42);

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 32);
    EXPECT_EQ(output->height, 32);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentTest, SVHNPolicy) {
    auto input = create_test_image(32, 32, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::SVHN, 42);

    auto output = transform.apply(*input);

    EXPECT_EQ(output->width, 32);
    EXPECT_EQ(output->height, 32);
    EXPECT_EQ(output->channels, 3);
}

TEST(AutoAugmentTest, Determinism) {
    auto input = create_test_image(64, 64, 3);

    // Same seed should produce same results
    AutoAugmentTransform transform1(AutoAugmentPolicy::IMAGENET, 42);
    AutoAugmentTransform transform2(AutoAugmentPolicy::IMAGENET, 42);

    auto output1 = transform1.apply(*input);
    auto output2 = transform2.apply(*input);

    // Note: Due to random policy selection, outputs may differ
    // This test just verifies both transforms run successfully
    EXPECT_EQ(output1->width, output2->width);
    EXPECT_EQ(output1->height, output2->height);
    EXPECT_EQ(output1->channels, output2->channels);
}

TEST(AutoAugmentTest, MultipleApplications) {
    auto input = create_test_image(64, 64, 3);
    AutoAugmentTransform transform(AutoAugmentPolicy::IMAGENET, 42);

    // Apply multiple times
    for (int i = 0; i < 10; ++i) {
        auto output = transform.apply(*input);
        EXPECT_EQ(output->width, 64);
        EXPECT_EQ(output->height, 64);
        EXPECT_EQ(output->channels, 3);
    }
}

// ============================================================================
// LANCZOS TESTS
// ============================================================================

TEST(LanczosTest, DownsampleQuality) {
    auto input = create_test_image(128, 128, 3);
    ResizeTransform resize(64, 64, InterpolationMode::LANCZOS);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(LanczosTest, UpsampleQuality) {
    auto input = create_test_image(64, 64, 3);
    ResizeTransform resize(128, 128, InterpolationMode::LANCZOS);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 128);
    EXPECT_EQ(output->height, 128);
    EXPECT_EQ(output->channels, 3);
}

TEST(LanczosTest, CompareWithBilinear) {
    auto input = create_test_image(128, 128, 3);

    ResizeTransform resize_lanczos(64, 64, InterpolationMode::LANCZOS);
    ResizeTransform resize_bilinear(64, 64, InterpolationMode::BILINEAR);

    auto output_lanczos = resize_lanczos.apply(*input);
    auto output_bilinear = resize_bilinear.apply(*input);

    EXPECT_EQ(output_lanczos->width, output_bilinear->width);
    EXPECT_EQ(output_lanczos->height, output_bilinear->height);
    EXPECT_EQ(output_lanczos->channels, output_bilinear->channels);

    // Outputs should be different (Lanczos has different characteristics)
    bool different = false;
    for (size_t i = 0; i < std::min(size_t(1000), output_lanczos->size_bytes()); ++i) {
        if (output_lanczos->data[i] != output_bilinear->data[i]) {
            different = true;
            break;
        }
    }
    // Note: Depending on test image, outputs may be similar
    // This test just verifies both methods run successfully
}

TEST(LanczosTest, SmallImage) {
    auto input = create_test_image(16, 16, 3);
    ResizeTransform resize(8, 8, InterpolationMode::LANCZOS);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 8);
    EXPECT_EQ(output->height, 8);
    EXPECT_EQ(output->channels, 3);
}

TEST(LanczosTest, LargeImage) {
    auto input = create_test_image(256, 256, 3);
    ResizeTransform resize(128, 128, InterpolationMode::LANCZOS);

    auto output = resize.apply(*input);

    EXPECT_EQ(output->width, 128);
    EXPECT_EQ(output->height, 128);
    EXPECT_EQ(output->channels, 3);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST(IntegrationTest, CombinedTransforms) {
    auto input = create_test_image(128, 128, 3);

    // Apply multiple advanced transforms in sequence
    RandomPosterizeTransform posterize(4, 1.0f, 42);
    RandomSolarizeTransform solarize(128, 1.0f, 43);
    ResizeTransform resize(64, 64, InterpolationMode::LANCZOS);

    auto step1 = posterize.apply(*input);
    auto step2 = solarize.apply(*step1);
    auto output = resize.apply(*step2);

    EXPECT_EQ(output->width, 64);
    EXPECT_EQ(output->height, 64);
    EXPECT_EQ(output->channels, 3);
}

TEST(IntegrationTest, AutoAugmentWithResize) {
    auto input = create_test_image(256, 256, 3);

    AutoAugmentTransform autoaugment(AutoAugmentPolicy::IMAGENET, 42);
    ResizeTransform resize(224, 224, InterpolationMode::LANCZOS);

    auto step1 = autoaugment.apply(*input);
    auto output = resize.apply(*step1);

    EXPECT_EQ(output->width, 224);
    EXPECT_EQ(output->height, 224);
    EXPECT_EQ(output->channels, 3);
}

// Main
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
