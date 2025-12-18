/**
 * @file test_quasi_random_sampler.cpp
 * @brief Tests for QuasiRandomSampler (v2.17.0)
 *
 * Tests FFCV-style quasi-random sampling:
 * - Page-based shuffling
 * - Constant memory usage
 * - Reproducibility
 * - Distributed training support
 * - Entropy/randomness quality
 */

#include <gtest/gtest.h>
#include "../src/sampling/quasi_random_sampler.hpp"
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace turboloader::sampling;

class QuasiRandomSamplerTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(QuasiRandomSamplerTest, Creation) {
    QuasiRandomSampler sampler(10000, 1024 * 1024, 10 * 1024);

    EXPECT_EQ(sampler.dataset_size(), 10000);
    EXPECT_GT(sampler.num_pages(), 0);
    EXPECT_GT(sampler.samples_per_page(), 0);
    EXPECT_GT(sampler.buffer_pages(), 0);

    std::cout << sampler.describe() << std::endl;
}

TEST_F(QuasiRandomSamplerTest, AllIndicesReturned) {
    const size_t dataset_size = 1000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 1024);

    auto indices = sampler.get_indices_for_epoch(0, 42);

    // All indices should be returned exactly once
    EXPECT_EQ(indices.size(), dataset_size);

    std::set<size_t> unique_indices(indices.begin(), indices.end());
    EXPECT_EQ(unique_indices.size(), dataset_size);

    // All indices should be in range [0, dataset_size)
    for (size_t idx : indices) {
        EXPECT_LT(idx, dataset_size);
    }
}

TEST_F(QuasiRandomSamplerTest, Reproducibility) {
    const size_t dataset_size = 1000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 1024);

    // Same epoch and seed should give same order
    auto indices1 = sampler.get_indices_for_epoch(0, 42);
    auto indices2 = sampler.get_indices_for_epoch(0, 42);

    EXPECT_EQ(indices1, indices2);

    // Different epoch should give different order
    auto indices3 = sampler.get_indices_for_epoch(1, 42);
    EXPECT_NE(indices1, indices3);

    // Different seed should give different order
    auto indices4 = sampler.get_indices_for_epoch(0, 123);
    EXPECT_NE(indices1, indices4);
}

TEST_F(QuasiRandomSamplerTest, DifferentEpochs) {
    const size_t dataset_size = 1000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 1024);

    std::vector<std::vector<size_t>> epoch_indices;
    for (size_t epoch = 0; epoch < 5; ++epoch) {
        epoch_indices.push_back(sampler.get_indices_for_epoch(epoch, 42));
    }

    // Each epoch should have all indices
    for (const auto& indices : epoch_indices) {
        EXPECT_EQ(indices.size(), dataset_size);
        std::set<size_t> unique(indices.begin(), indices.end());
        EXPECT_EQ(unique.size(), dataset_size);
    }

    // Epochs should be different from each other
    for (size_t i = 0; i < epoch_indices.size(); ++i) {
        for (size_t j = i + 1; j < epoch_indices.size(); ++j) {
            EXPECT_NE(epoch_indices[i], epoch_indices[j]);
        }
    }
}

// ============================================================================
// Shuffling Quality Tests
// ============================================================================

TEST_F(QuasiRandomSamplerTest, NotSequential) {
    const size_t dataset_size = 1000;
    QuasiRandomSampler sampler(dataset_size, 50 * 1024, 1024);

    auto indices = sampler.get_indices_for_epoch(0, 42);

    // Count how many indices are in their original position
    size_t in_place = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] == i) in_place++;
    }

    // Should have very few indices in original position (expect ~1/1000)
    double in_place_ratio = static_cast<double>(in_place) / dataset_size;
    EXPECT_LT(in_place_ratio, 0.05);  // Less than 5% in place

    // Count sequential pairs
    size_t sequential_pairs = 0;
    for (size_t i = 0; i < indices.size() - 1; ++i) {
        if (indices[i] + 1 == indices[i + 1]) sequential_pairs++;
    }

    // Should have relatively few sequential pairs
    double sequential_ratio = static_cast<double>(sequential_pairs) / (dataset_size - 1);
    EXPECT_LT(sequential_ratio, 0.1);  // Less than 10% sequential
}

TEST_F(QuasiRandomSamplerTest, DistributionQuality) {
    // Test that samples from different parts of dataset are mixed
    const size_t dataset_size = 10000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 10 * 1024);

    auto indices = sampler.get_indices_for_epoch(0, 42);

    // Divide dataset into quartiles and check mixing
    // For first 1000 samples, count how many come from each quartile
    std::map<int, int> quartile_counts;
    for (size_t i = 0; i < 1000; ++i) {
        int quartile = indices[i] / (dataset_size / 4);
        quartile_counts[quartile]++;
    }

    // Each quartile should have a reasonable representation (at least 15%)
    for (int q = 0; q < 4; ++q) {
        double ratio = static_cast<double>(quartile_counts[q]) / 1000;
        EXPECT_GT(ratio, 0.10) << "Quartile " << q << " underrepresented";
        EXPECT_LT(ratio, 0.40) << "Quartile " << q << " overrepresented";
    }
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

TEST_F(QuasiRandomSamplerTest, ConstantMemoryEstimate) {
    // Memory should be constant regardless of dataset size
    QuasiRandomSampler small_sampler(1000, 100 * 1024, 1024);
    QuasiRandomSampler large_sampler(1000000, 100 * 1024, 1024);

    // Both should have similar buffer pages
    small_sampler.set_buffer_pages(4);
    large_sampler.set_buffer_pages(4);

    // Memory estimate should be similar (within 10x)
    size_t small_mem = small_sampler.estimated_memory_usage();
    size_t large_mem = large_sampler.estimated_memory_usage();

    EXPECT_LT(large_mem, small_mem * 10);

    std::cout << "Small dataset memory: " << small_mem / 1024 << " KB" << std::endl;
    std::cout << "Large dataset memory: " << large_mem / 1024 << " KB" << std::endl;
}

TEST_F(QuasiRandomSamplerTest, BufferPageConfiguration) {
    const size_t dataset_size = 10000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 1024);

    // Test different buffer sizes
    sampler.set_buffer_pages(1);
    EXPECT_EQ(sampler.buffer_pages(), 1);

    sampler.set_buffer_pages(8);
    EXPECT_LE(sampler.buffer_pages(), sampler.num_pages());

    // Memory should scale with buffer pages
    sampler.set_buffer_pages(2);
    size_t mem2 = sampler.estimated_memory_usage();

    sampler.set_buffer_pages(4);
    size_t mem4 = sampler.estimated_memory_usage();

    EXPECT_GT(mem4, mem2);
}

// ============================================================================
// Distributed Training Tests
// ============================================================================

TEST_F(QuasiRandomSamplerTest, DistributedSampling) {
    const size_t dataset_size = 1000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 1024);

    const int world_size = 4;
    std::vector<std::vector<size_t>> rank_indices;

    for (int rank = 0; rank < world_size; ++rank) {
        rank_indices.push_back(sampler.get_indices_for_rank(0, 42, rank, world_size));
    }

    // Collect all indices from all ranks
    std::set<size_t> all_indices;
    for (const auto& indices : rank_indices) {
        all_indices.insert(indices.begin(), indices.end());
    }

    // All indices should be covered
    EXPECT_EQ(all_indices.size(), dataset_size);

    // Print distribution
    std::cout << "Distributed sampling across " << world_size << " ranks:" << std::endl;
    for (int rank = 0; rank < world_size; ++rank) {
        std::cout << "  Rank " << rank << ": " << rank_indices[rank].size() << " samples" << std::endl;
    }
}

TEST_F(QuasiRandomSamplerTest, DistributedDisjoint) {
    const size_t dataset_size = 1000;
    QuasiRandomSampler sampler(dataset_size, 100 * 1024, 1024);

    const int world_size = 4;
    std::vector<std::set<size_t>> rank_sets;

    for (int rank = 0; rank < world_size; ++rank) {
        auto indices = sampler.get_indices_for_rank(0, 42, rank, world_size);
        rank_sets.emplace_back(indices.begin(), indices.end());
    }

    // Check that ranks don't overlap (much - some overlap is okay due to page boundaries)
    for (int i = 0; i < world_size; ++i) {
        for (int j = i + 1; j < world_size; ++j) {
            std::vector<size_t> intersection;
            std::set_intersection(
                rank_sets[i].begin(), rank_sets[i].end(),
                rank_sets[j].begin(), rank_sets[j].end(),
                std::back_inserter(intersection)
            );

            // No overlap expected with page-based distribution
            EXPECT_EQ(intersection.size(), 0)
                << "Ranks " << i << " and " << j << " have " << intersection.size() << " overlapping samples";
        }
    }
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_F(QuasiRandomSamplerTest, IteratorBasic) {
    const size_t dataset_size = 100;
    QuasiRandomSampler sampler(dataset_size, 10 * 1024, 1024);

    std::vector<size_t> from_iterator;
    for (auto it = sampler.begin(0, 42); it != sampler.end(); ++it) {
        from_iterator.push_back(*it);
    }

    auto from_method = sampler.get_indices_for_epoch(0, 42);

    EXPECT_EQ(from_iterator, from_method);
}

TEST_F(QuasiRandomSamplerTest, RangeBasedFor) {
    const size_t dataset_size = 100;
    QuasiRandomSampler sampler(dataset_size, 10 * 1024, 1024);

    // This would work with proper begin()/end() that don't need parameters
    // For now, we test the explicit iterator version
    size_t count = 0;
    for (auto it = sampler.begin(0, 42); it != sampler.end(); ++it) {
        count++;
    }

    EXPECT_EQ(count, dataset_size);
}

// ============================================================================
// Other Sampler Tests
// ============================================================================

TEST_F(QuasiRandomSamplerTest, SequentialSampler) {
    const size_t dataset_size = 100;
    SequentialSampler sampler(dataset_size);

    auto indices = sampler.get_indices();

    EXPECT_EQ(indices.size(), dataset_size);

    // Should be in order
    for (size_t i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], i);
    }
}

TEST_F(QuasiRandomSamplerTest, RandomSampler) {
    const size_t dataset_size = 100;
    RandomSampler sampler(dataset_size);

    auto indices = sampler.get_indices_for_epoch(0, 42);

    EXPECT_EQ(indices.size(), dataset_size);

    // All indices should be present
    std::set<size_t> unique(indices.begin(), indices.end());
    EXPECT_EQ(unique.size(), dataset_size);

    // Should be reproducible
    auto indices2 = sampler.get_indices_for_epoch(0, 42);
    EXPECT_EQ(indices, indices2);
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(QuasiRandomSamplerTest, BenchmarkLargeDataset) {
    const size_t dataset_size = 1000000;  // 1M samples
    QuasiRandomSampler sampler(dataset_size, 8 * 1024 * 1024, 100 * 1024);

    std::cout << "\n=== Benchmark: 1M samples ===" << std::endl;
    std::cout << sampler.describe() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto indices = sampler.get_indices_for_epoch(0, 42);

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(indices.size(), dataset_size);

    std::cout << "Time to generate indices: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (dataset_size * 1000.0 / ms) << " indices/sec" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
