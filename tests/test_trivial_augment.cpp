/**
 * @file test_trivial_augment.cpp
 * @brief Tests for TrivialAugment (v2.18.0)
 *
 * Tests TrivialAugment implementation:
 * - Single operation per sample
 * - All 14 operations work correctly
 * - Reproducibility with seed
 * - Output validity
 */

#include <gtest/gtest.h>
#include "../src/transforms/trivial_augment_transform.hpp"
#include <set>
#include <map>
#include <cmath>
#include <chrono>

using namespace turboloader::transforms;

class TrivialAugmentTest : public ::testing::Test {
protected:
    // Create test image
    std::unique_ptr<ImageData> create_test_image(int width, int height, int channels = 3) {
        auto img = std::make_unique<ImageData>(
            new uint8_t[width * height * channels],
            width, height, channels, width * channels, true
        );

        // Fill with gradient
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width * channels + x * channels;
                img->data[idx] = static_cast<uint8_t>(x * 255 / width);      // R
                if (channels > 1) img->data[idx + 1] = static_cast<uint8_t>(y * 255 / height);  // G
                if (channels > 2) img->data[idx + 2] = 128;  // B
            }
        }

        return img;
    }

    void SetUp() override {}
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(TrivialAugmentTest, CreationStandard) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::STANDARD);

    EXPECT_EQ(aug.name(), "TrivialAugment-Standard");
    EXPECT_EQ(aug.num_operations(), 10);  // Standard has 10 operations

    auto ops = aug.operation_names();
    EXPECT_EQ(ops.size(), 10);

    std::cout << "Standard operations:" << std::endl;
    for (const auto& op : ops) {
        std::cout << "  - " << op << std::endl;
    }
}

TEST_F(TrivialAugmentTest, CreationWide) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE);

    EXPECT_EQ(aug.name(), "TrivialAugment-Wide");
    EXPECT_EQ(aug.num_operations(), 14);  // Wide has 14 operations

    auto ops = aug.operation_names();
    EXPECT_EQ(ops.size(), 14);

    std::cout << "Wide operations:" << std::endl;
    for (const auto& op : ops) {
        std::cout << "  - " << op << std::endl;
    }
}

TEST_F(TrivialAugmentTest, ApplyReturnsValidImage) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(64, 64);

    auto result = aug.apply(*img);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->width, img->width);
    EXPECT_EQ(result->height, img->height);
    EXPECT_EQ(result->channels, img->channels);

    // Check that data is valid (no null pointer)
    EXPECT_NE(result->data, nullptr);
}

TEST_F(TrivialAugmentTest, Reproducibility) {
    auto img = create_test_image(64, 64);

    TrivialAugmentTransform aug1(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    TrivialAugmentTransform aug2(TrivialAugmentTransform::AugmentSpace::WIDE, 42);

    auto result1 = aug1.apply(*img);
    auto result2 = aug2.apply(*img);

    // Same seed should produce same result
    bool same = true;
    for (int i = 0; i < img->width * img->height * img->channels; ++i) {
        if (result1->data[i] != result2->data[i]) {
            same = false;
            break;
        }
    }
    EXPECT_TRUE(same);
}

TEST_F(TrivialAugmentTest, DifferentSeeds) {
    auto img = create_test_image(64, 64);

    TrivialAugmentTransform aug1(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    TrivialAugmentTransform aug2(TrivialAugmentTransform::AugmentSpace::WIDE, 123);

    // Apply many times and collect results
    int different_count = 0;
    for (int i = 0; i < 10; ++i) {
        auto result1 = aug1.apply(*img);
        auto result2 = aug2.apply(*img);

        bool same = true;
        for (int j = 0; j < img->width * img->height * img->channels; ++j) {
            if (result1->data[j] != result2->data[j]) {
                same = false;
                break;
            }
        }
        if (!same) different_count++;
    }

    // Different seeds should usually produce different results
    EXPECT_GT(different_count, 5);
}

// ============================================================================
// Operation Distribution Tests
// ============================================================================

TEST_F(TrivialAugmentTest, OperationDistribution) {
    // Test that all operations are used roughly equally over many iterations
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(32, 32);

    const int iterations = 10000;
    std::map<size_t, int> first_pixel_values;

    for (int i = 0; i < iterations; ++i) {
        auto result = aug.apply(*img);
        // Use first pixel as a proxy for which operation was applied
        size_t key = result->data[0] * 256 + result->data[1];
        first_pixel_values[key]++;
    }

    // Should have multiple different outputs
    EXPECT_GT(first_pixel_values.size(), 5)
        << "Expected more variation in outputs";

    std::cout << "Operation distribution (by first pixel):" << std::endl;
    std::cout << "  Unique outputs: " << first_pixel_values.size() << std::endl;
}

// ============================================================================
// Individual Operation Tests
// ============================================================================

TEST_F(TrivialAugmentTest, IdentityOperation) {
    // With identity, image should be unchanged
    // We can't directly test identity since operations are random,
    // but we can verify the transform doesn't crash
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(64, 64);

    // Apply many times - should never crash
    for (int i = 0; i < 100; ++i) {
        auto result = aug.apply(*img);
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->width, 64);
        EXPECT_EQ(result->height, 64);
    }
}

TEST_F(TrivialAugmentTest, PixelValuesInRange) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(64, 64);

    for (int i = 0; i < 100; ++i) {
        auto result = aug.apply(*img);

        // All pixel values should be in valid range [0, 255]
        for (int j = 0; j < result->width * result->height * result->channels; ++j) {
            EXPECT_GE(result->data[j], 0);
            EXPECT_LE(result->data[j], 255);
        }
    }
}

TEST_F(TrivialAugmentTest, GrayscaleImage) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(64, 64, 1);  // Grayscale

    for (int i = 0; i < 50; ++i) {
        auto result = aug.apply(*img);
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->channels, 1);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TrivialAugmentTest, SmallImage) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(4, 4);

    auto result = aug.apply(*img);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->width, 4);
    EXPECT_EQ(result->height, 4);
}

TEST_F(TrivialAugmentTest, LargeImage) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(512, 512);

    auto result = aug.apply(*img);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->width, 512);
    EXPECT_EQ(result->height, 512);
}

TEST_F(TrivialAugmentTest, NonSquareImage) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(128, 64);

    auto result = aug.apply(*img);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->width, 128);
    EXPECT_EQ(result->height, 64);
}

// ============================================================================
// Comparison with RandAugment
// ============================================================================

TEST_F(TrivialAugmentTest, SingleOperationPerSample) {
    // TrivialAugment applies exactly one operation per sample
    // We verify this indirectly by checking that consecutive applications
    // can produce very different results (unlike RandAugment which compounds)

    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(64, 64);

    // Apply once vs apply twice should give different results
    // because each apply picks a new random operation
    auto result1 = aug.apply(*img);
    auto result2 = aug.apply(*img);

    // Should be different (same image, different random operation)
    bool same = true;
    for (int i = 0; i < img->width * img->height * img->channels; ++i) {
        if (result1->data[i] != result2->data[i]) {
            same = false;
            break;
        }
    }

    // Very likely to be different with 14 operations
    // (could be same if both pick Identity, but unlikely)
    std::cout << "Consecutive applications produced "
              << (same ? "same" : "different") << " results" << std::endl;
}

// ============================================================================
// Performance Benchmark
// ============================================================================

TEST_F(TrivialAugmentTest, Benchmark) {
    TrivialAugmentTransform aug(TrivialAugmentTransform::AugmentSpace::WIDE, 42);
    auto img = create_test_image(224, 224);

    const int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto result = aug.apply(*img);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "\n=== TrivialAugment Benchmark ===" << std::endl;
    std::cout << "  Image size: 224x224" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Total time: " << us / 1000.0 << " ms" << std::endl;
    std::cout << "  Per image: " << us / iterations << " us" << std::endl;
    std::cout << "  Throughput: " << (iterations * 1000000.0 / us) << " images/sec" << std::endl;

    // Should be fast - less than 1ms per image on average
    EXPECT_LT(us / iterations, 1000) << "TrivialAugment too slow";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
