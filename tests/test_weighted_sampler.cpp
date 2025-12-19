/**
 * @file test_weighted_sampler.cpp
 * @brief Tests for weighted sampling (v2.21.0)
 *
 * Tests:
 * - WeightedRandomSampler (with/without replacement)
 * - ClassBalancedSampler
 * - OverSampler / UnderSampler
 * - ImportanceSampler (temperature)
 * - StratifiedSampler
 * - Edge cases and error handling
 */

#include <gtest/gtest.h>
#include "../src/sampling/weighted_sampler.hpp"
#include <cmath>
#include <set>
#include <numeric>

using namespace turboloader;

// ============================================================================
// WeightedRandomSampler Tests
// ============================================================================

TEST(WeightedRandomSampler, BasicConstruction) {
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
    WeightedRandomSampler sampler(weights);

    EXPECT_EQ(sampler.dataset_size(), 4u);
    EXPECT_EQ(sampler.num_samples(), 4u);
    EXPECT_TRUE(sampler.replacement());
}

TEST(WeightedRandomSampler, CustomNumSamples) {
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
    WeightedRandomSampler sampler(weights, 100);

    auto indices = sampler.sample();
    EXPECT_EQ(indices.size(), 100u);
}

TEST(WeightedRandomSampler, WithReplacement) {
    std::vector<double> weights = {0.0, 0.0, 0.0, 1.0};  // Only index 3 has weight
    WeightedRandomSampler sampler(weights, 100, true, 42);

    auto indices = sampler.sample();
    EXPECT_EQ(indices.size(), 100u);

    // All samples should be index 3
    for (size_t idx : indices) {
        EXPECT_EQ(idx, 3u);
    }
}

TEST(WeightedRandomSampler, WithoutReplacement) {
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0};
    WeightedRandomSampler sampler(weights, 5, false, 42);

    auto indices = sampler.sample();
    EXPECT_EQ(indices.size(), 5u);

    // All indices should be unique
    std::set<size_t> unique_indices(indices.begin(), indices.end());
    EXPECT_EQ(unique_indices.size(), 5u);
}

TEST(WeightedRandomSampler, WeightDistribution) {
    // Index 3 has 10x weight of others
    std::vector<double> weights = {1.0, 1.0, 1.0, 10.0};
    WeightedRandomSampler sampler(weights, 10000, true, 42);

    auto indices = sampler.sample();

    // Count occurrences
    std::vector<size_t> counts(4, 0);
    for (size_t idx : indices) {
        counts[idx]++;
    }

    // Index 3 should appear roughly 10x more often
    // With weight ratio 10:1:1:1, expected ratio is about 10:1
    double ratio = static_cast<double>(counts[3]) / counts[0];
    EXPECT_GT(ratio, 5.0);  // Should be close to 10
}

TEST(WeightedRandomSampler, Reproducibility) {
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};

    WeightedRandomSampler sampler1(weights, 100, true, 12345);
    WeightedRandomSampler sampler2(weights, 100, true, 12345);

    auto indices1 = sampler1.sample();
    auto indices2 = sampler2.sample();

    EXPECT_EQ(indices1, indices2);
}

TEST(WeightedRandomSampler, SetSeed) {
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
    WeightedRandomSampler sampler(weights, 100, true, 0);

    sampler.set_seed(12345);
    auto indices1 = sampler.sample();

    sampler.set_seed(12345);
    auto indices2 = sampler.sample();

    EXPECT_EQ(indices1, indices2);
}

TEST(WeightedRandomSampler, SampleOne) {
    std::vector<double> weights = {0.0, 1.0, 0.0, 0.0};  // Only index 1
    WeightedRandomSampler sampler(weights, 1, true, 42);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(sampler.sample_one(), 1u);
    }
}

TEST(WeightedRandomSampler, UpdateWeights) {
    std::vector<double> weights = {1.0, 0.0, 0.0, 0.0};
    WeightedRandomSampler sampler(weights, 10, true, 42);

    // Initially only index 0
    auto indices = sampler.sample();
    for (size_t idx : indices) {
        EXPECT_EQ(idx, 0u);
    }

    // Update to only index 3
    sampler.set_weights({0.0, 0.0, 0.0, 1.0});
    indices = sampler.sample();
    for (size_t idx : indices) {
        EXPECT_EQ(idx, 3u);
    }
}

TEST(WeightedRandomSampler, EmptyWeights) {
    std::vector<double> empty_weights;
    EXPECT_THROW(WeightedRandomSampler{empty_weights}, std::invalid_argument);
}

TEST(WeightedRandomSampler, NegativeWeight) {
    std::vector<double> negative_weights = {1.0, -1.0, 1.0};
    EXPECT_THROW(WeightedRandomSampler{negative_weights}, std::invalid_argument);
}

TEST(WeightedRandomSampler, ZeroTotalWeight) {
    std::vector<double> zero_weights = {0.0, 0.0, 0.0};
    EXPECT_THROW(WeightedRandomSampler{zero_weights}, std::invalid_argument);
}

// ============================================================================
// ClassBalancedSampler Tests
// ============================================================================

TEST(ClassBalancedSampler, BasicConstruction) {
    std::vector<size_t> labels = {0, 0, 0, 1, 1, 2};
    ClassBalancedSampler sampler(labels);

    EXPECT_EQ(sampler.num_classes(), 3u);
    EXPECT_EQ(sampler.dataset_size(), 6u);
}

TEST(ClassBalancedSampler, BalancedSampling) {
    // Very imbalanced: 90 class 0, 10 class 1
    std::vector<size_t> labels(100);
    std::fill(labels.begin(), labels.begin() + 90, 0);
    std::fill(labels.begin() + 90, labels.end(), 1);

    ClassBalancedSampler sampler(labels, 42);

    // Sample many times
    std::vector<size_t> class_counts(2, 0);
    auto indices = sampler.sample(10000);

    for (size_t idx : indices) {
        class_counts[labels[idx]]++;
    }

    // Should be roughly balanced (50/50)
    double ratio = static_cast<double>(class_counts[0]) / class_counts[1];
    EXPECT_GT(ratio, 0.7);
    EXPECT_LT(ratio, 1.3);
}

TEST(ClassBalancedSampler, ImbalanceRatio) {
    std::vector<size_t> labels = {0, 0, 0, 0, 0, 1};  // 5:1 ratio
    ClassBalancedSampler sampler(labels);

    EXPECT_DOUBLE_EQ(sampler.imbalance_ratio(), 5.0);
}

TEST(ClassBalancedSampler, ClassCounts) {
    std::vector<size_t> labels = {0, 0, 1, 1, 1, 2};
    ClassBalancedSampler sampler(labels);

    auto counts = sampler.class_counts();
    EXPECT_EQ(counts.at(0), 2u);
    EXPECT_EQ(counts.at(1), 3u);
    EXPECT_EQ(counts.at(2), 1u);
}

TEST(ClassBalancedSampler, Weights) {
    std::vector<size_t> labels = {0, 0, 1};  // 2 class 0, 1 class 1
    ClassBalancedSampler sampler(labels);

    auto weights = sampler.weights();
    EXPECT_EQ(weights.size(), 3u);

    // Class 1 (minority) should have higher weight
    EXPECT_GT(weights[2], weights[0]);
    EXPECT_GT(weights[2], weights[1]);
}

TEST(ClassBalancedSampler, SampleEpoch) {
    std::vector<size_t> labels = {0, 0, 1, 1, 2, 2};
    ClassBalancedSampler sampler(labels, 42);

    auto epoch = sampler.sample_epoch();
    EXPECT_EQ(epoch.size(), labels.size());
}

TEST(ClassBalancedSampler, EmptyLabels) {
    std::vector<size_t> empty_labels;
    EXPECT_THROW(ClassBalancedSampler{empty_labels}, std::invalid_argument);
}

// ============================================================================
// OverSampler Tests
// ============================================================================

TEST(OverSampler, BasicOversampling) {
    // 3 class 0, 1 class 1
    std::vector<size_t> labels = {0, 0, 0, 1};
    OverSampler sampler(labels);

    // Should oversample class 1 to match class 0
    EXPECT_EQ(sampler.total_samples(), 6u);  // 3 + 3
    EXPECT_EQ(sampler.original_size(), 4u);
}

TEST(OverSampler, SampleEpoch) {
    std::vector<size_t> labels = {0, 0, 0, 1};
    OverSampler sampler(labels, 42);

    auto epoch = sampler.sample_epoch();
    EXPECT_EQ(epoch.size(), 6u);

    // Count occurrences of index 3 (only class 1 sample)
    size_t class1_count = std::count(epoch.begin(), epoch.end(), 3);
    EXPECT_EQ(class1_count, 3u);  // Repeated 3 times
}

TEST(OverSampler, Reproducibility) {
    std::vector<size_t> labels = {0, 0, 1, 2, 2, 2};

    OverSampler sampler1(labels, 12345);
    OverSampler sampler2(labels, 12345);

    EXPECT_EQ(sampler1.sample_epoch(), sampler2.sample_epoch());
}

// ============================================================================
// UnderSampler Tests
// ============================================================================

TEST(UnderSampler, BasicUndersampling) {
    // 3 class 0, 1 class 1
    std::vector<size_t> labels = {0, 0, 0, 1};
    UnderSampler sampler(labels);

    // Should undersample class 0 to match class 1
    EXPECT_EQ(sampler.total_samples(), 2u);  // 1 + 1
    EXPECT_EQ(sampler.original_size(), 4u);
}

TEST(UnderSampler, SampleEpoch) {
    std::vector<size_t> labels = {0, 0, 0, 1};
    UnderSampler sampler(labels, 42);

    auto epoch = sampler.sample_epoch();
    EXPECT_EQ(epoch.size(), 2u);

    // Should have one from each class
    std::set<size_t> sampled_labels;
    for (size_t idx : epoch) {
        sampled_labels.insert(labels[idx]);
    }
    EXPECT_EQ(sampled_labels.size(), 2u);
}

TEST(UnderSampler, MultipleClasses) {
    // 5 class 0, 3 class 1, 2 class 2
    std::vector<size_t> labels = {0, 0, 0, 0, 0, 1, 1, 1, 2, 2};
    UnderSampler sampler(labels);

    // Undersample to smallest class (2 samples)
    EXPECT_EQ(sampler.total_samples(), 6u);  // 2 * 3 classes
}

// ============================================================================
// ImportanceSampler Tests
// ============================================================================

TEST(ImportanceSampler, Temperature1) {
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
    ImportanceSampler sampler(weights, 1.0);

    // Temperature 1.0 should keep weights unchanged
    auto adjusted = sampler.adjusted_weights();
    EXPECT_EQ(adjusted.size(), weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        EXPECT_DOUBLE_EQ(adjusted[i], weights[i]);
    }
}

TEST(ImportanceSampler, HighTemperature) {
    std::vector<double> weights = {1.0, 100.0};  // Large difference
    ImportanceSampler sampler(weights, 10.0, 42);

    // High temperature should soften distribution
    auto adjusted = sampler.adjusted_weights();

    // Ratio should be reduced
    double original_ratio = weights[1] / weights[0];
    double adjusted_ratio = adjusted[1] / adjusted[0];
    EXPECT_LT(adjusted_ratio, original_ratio);
}

TEST(ImportanceSampler, LowTemperature) {
    std::vector<double> weights = {1.0, 2.0};
    ImportanceSampler sampler(weights, 0.5, 42);

    // Low temperature should sharpen distribution
    auto adjusted = sampler.adjusted_weights();

    double original_ratio = weights[1] / weights[0];
    double adjusted_ratio = adjusted[1] / adjusted[0];
    EXPECT_GT(adjusted_ratio, original_ratio);
}

TEST(ImportanceSampler, SetTemperature) {
    std::vector<double> weights = {1.0, 2.0};
    ImportanceSampler sampler(weights, 1.0);

    sampler.set_temperature(2.0);
    EXPECT_DOUBLE_EQ(sampler.temperature(), 2.0);
}

TEST(ImportanceSampler, Sample) {
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};
    ImportanceSampler sampler(weights, 1.0, 42);

    auto indices = sampler.sample(100);
    EXPECT_EQ(indices.size(), 100u);
}

// ============================================================================
// StratifiedSampler Tests
// ============================================================================

TEST(StratifiedSampler, BasicConstruction) {
    std::vector<size_t> labels = {0, 0, 1, 1, 2, 2};
    StratifiedSampler sampler(labels);

    EXPECT_EQ(sampler.num_classes(), 3u);
    EXPECT_EQ(sampler.dataset_size(), 6u);
}

TEST(StratifiedSampler, SampleEpoch) {
    std::vector<size_t> labels = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    StratifiedSampler sampler(labels, 42);

    auto epoch = sampler.sample_epoch();
    EXPECT_EQ(epoch.size(), labels.size());

    // All indices should appear exactly once
    std::set<size_t> unique(epoch.begin(), epoch.end());
    EXPECT_EQ(unique.size(), labels.size());
}

TEST(StratifiedSampler, Interleaving) {
    // 3 per class
    std::vector<size_t> labels = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    StratifiedSampler sampler(labels, 42);

    auto epoch = sampler.sample_epoch();

    // Check that no two consecutive samples are from same class often
    // (perfect interleaving: 0,1,2,0,1,2,...)
    int same_class_consecutive = 0;
    for (size_t i = 1; i < epoch.size(); ++i) {
        if (labels[epoch[i]] == labels[epoch[i-1]]) {
            same_class_consecutive++;
        }
    }

    // With 3 classes and 9 samples, consecutive same-class should be rare
    EXPECT_LT(same_class_consecutive, 4);
}

TEST(StratifiedSampler, SampleBatch) {
    std::vector<size_t> labels = {0, 0, 1, 1, 2, 2};
    StratifiedSampler sampler(labels, 42);

    auto batch = sampler.sample_batch(4);
    EXPECT_EQ(batch.size(), 4u);
}

TEST(StratifiedSampler, EmptyLabels) {
    std::vector<size_t> empty_labels;
    EXPECT_THROW(StratifiedSampler{empty_labels}, std::invalid_argument);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(Integration, LargeDataset) {
    // Simulate large imbalanced dataset
    std::vector<size_t> labels(10000);
    std::fill(labels.begin(), labels.begin() + 9000, 0);  // 90%
    std::fill(labels.begin() + 9000, labels.begin() + 9900, 1);  // 9%
    std::fill(labels.begin() + 9900, labels.end(), 2);  // 1%

    ClassBalancedSampler sampler(labels, 42);

    // Sample one epoch
    auto indices = sampler.sample(10000);

    // Count per class
    std::vector<size_t> counts(3, 0);
    for (size_t idx : indices) {
        counts[labels[idx]]++;
    }

    // Should be roughly balanced
    for (size_t c = 0; c < 3; ++c) {
        double expected = 10000.0 / 3;
        double actual = counts[c];
        EXPECT_GT(actual, expected * 0.8);
        EXPECT_LT(actual, expected * 1.2);
    }
}

TEST(Integration, EpochReproducibility) {
    std::vector<size_t> labels(100);
    std::iota(labels.begin(), labels.end(), 0);  // 100 unique classes

    StratifiedSampler sampler1(labels, 12345);
    StratifiedSampler sampler2(labels, 12345);

    EXPECT_EQ(sampler1.sample_epoch(), sampler2.sample_epoch());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
