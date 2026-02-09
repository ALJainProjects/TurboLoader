/**
 * @file test_sharding_strategies.cpp
 * @brief Tests for advanced sharding strategies
 */

#include <gtest/gtest.h>
#include "../src/distributed/sharding_strategies.hpp"
#include "../src/distributed/distributed_dataloader.hpp"
#include <algorithm>
#include <numeric>
#include <set>

using namespace turboloader;

// ============================================================================
// ContiguousSharding Tests
// ============================================================================

TEST(ContiguousShardingTest, BasicDistribution) {
    // 100 samples, 4 workers
    ContiguousSharding shard(100, 4, 0);
    EXPECT_EQ(shard.size(), 25);
    EXPECT_EQ(shard.start(), 0);
    EXPECT_EQ(shard.end(), 25);
}

TEST(ContiguousShardingTest, AllRanks) {
    // Test all 4 ranks
    for (size_t rank = 0; rank < 4; ++rank) {
        ContiguousSharding shard(100, 4, rank);
        EXPECT_EQ(shard.size(), 25);
        EXPECT_EQ(shard.start(), rank * 25);
        EXPECT_EQ(shard.end(), (rank + 1) * 25);
    }
}

TEST(ContiguousShardingTest, UnevenDistribution) {
    // 103 samples, 4 workers - first 3 get 26, last gets 25
    ContiguousSharding shard0(103, 4, 0);
    ContiguousSharding shard1(103, 4, 1);
    ContiguousSharding shard2(103, 4, 2);
    ContiguousSharding shard3(103, 4, 3);

    EXPECT_EQ(shard0.size(), 26);
    EXPECT_EQ(shard1.size(), 26);
    EXPECT_EQ(shard2.size(), 26);
    EXPECT_EQ(shard3.size(), 25);

    // Total should be 103
    EXPECT_EQ(shard0.size() + shard1.size() + shard2.size() + shard3.size(), 103);
}

TEST(ContiguousShardingTest, SingleWorker) {
    ContiguousSharding shard(100, 1, 0);
    EXPECT_EQ(shard.size(), 100);
    EXPECT_EQ(shard.start(), 0);
    EXPECT_EQ(shard.end(), 100);
}

TEST(ContiguousShardingTest, MoreWorkersThanSamples) {
    // 3 samples, 10 workers
    for (size_t rank = 0; rank < 10; ++rank) {
        ContiguousSharding shard(3, 10, rank);
        if (rank < 3) {
            EXPECT_EQ(shard.size(), 1);
        } else {
            EXPECT_EQ(shard.size(), 0);
        }
    }
}

TEST(ContiguousShardingTest, GetIndices) {
    ContiguousSharding shard(100, 4, 1);
    auto indices = shard.get_indices();

    EXPECT_EQ(indices.size(), 25);
    for (size_t i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], 25 + i);
    }
}

TEST(ContiguousShardingTest, GlobalIndex) {
    ContiguousSharding shard(100, 4, 2);

    EXPECT_EQ(shard.global_index(0), 50);
    EXPECT_EQ(shard.global_index(10), 60);
    EXPECT_EQ(shard.global_index(24), 74);

    EXPECT_THROW(shard.global_index(25), std::out_of_range);
}

TEST(ContiguousShardingTest, OwnsIndex) {
    ContiguousSharding shard(100, 4, 2);

    EXPECT_FALSE(shard.owns_index(0));
    EXPECT_FALSE(shard.owns_index(49));
    EXPECT_TRUE(shard.owns_index(50));
    EXPECT_TRUE(shard.owns_index(74));
    EXPECT_FALSE(shard.owns_index(75));
    EXPECT_FALSE(shard.owns_index(99));
}

// ============================================================================
// InterleavedSharding Tests
// ============================================================================

TEST(InterleavedShardingTest, BasicDistribution) {
    InterleavedSharding shard(100, 4, 0);
    EXPECT_EQ(shard.size(), 25);

    auto indices = shard.get_indices();
    for (size_t i = 0; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], i * 4);  // 0, 4, 8, 12, ...
    }
}

TEST(InterleavedShardingTest, AllRanksGetEveryNth) {
    for (size_t rank = 0; rank < 4; ++rank) {
        InterleavedSharding shard(100, 4, rank);
        auto indices = shard.get_indices();

        for (size_t i = 0; i < indices.size(); ++i) {
            EXPECT_EQ(indices[i], rank + i * 4);
        }
    }
}

TEST(InterleavedShardingTest, UnevenDistribution) {
    // 103 samples, 4 workers
    InterleavedSharding shard0(103, 4, 0);
    InterleavedSharding shard1(103, 4, 1);
    InterleavedSharding shard2(103, 4, 2);
    InterleavedSharding shard3(103, 4, 3);

    // Rank 0, 1, 2 get 26 each (indices 0,4,8...100 and 1,5,9...101 and 2,6,10...102)
    // Rank 3 gets 25 (indices 3,7,11...99)
    EXPECT_EQ(shard0.size(), 26);
    EXPECT_EQ(shard1.size(), 26);
    EXPECT_EQ(shard2.size(), 26);
    EXPECT_EQ(shard3.size(), 25);
}

TEST(InterleavedShardingTest, GlobalIndex) {
    InterleavedSharding shard(100, 4, 1);

    EXPECT_EQ(shard.global_index(0), 1);   // 1
    EXPECT_EQ(shard.global_index(1), 5);   // 1 + 4
    EXPECT_EQ(shard.global_index(2), 9);   // 1 + 8
}

TEST(InterleavedShardingTest, OwnsIndex) {
    InterleavedSharding shard(100, 4, 2);

    EXPECT_FALSE(shard.owns_index(0));  // rank 0
    EXPECT_FALSE(shard.owns_index(1));  // rank 1
    EXPECT_TRUE(shard.owns_index(2));   // rank 2
    EXPECT_FALSE(shard.owns_index(3));  // rank 3
    EXPECT_TRUE(shard.owns_index(6));   // rank 2
    EXPECT_TRUE(shard.owns_index(98));  // rank 2
}

// ============================================================================
// HashBasedSharding Tests
// ============================================================================

TEST(HashBasedShardingTest, DeterministicAssignment) {
    HashBasedSharding shard1(100, 4, 0, 42);
    HashBasedSharding shard2(100, 4, 0, 42);

    auto indices1 = shard1.get_indices();
    auto indices2 = shard2.get_indices();

    EXPECT_EQ(indices1, indices2);
}

TEST(HashBasedShardingTest, DifferentSeedsGiveDifferentAssignments) {
    HashBasedSharding shard1(100, 4, 0, 42);
    HashBasedSharding shard2(100, 4, 0, 123);

    auto indices1 = shard1.get_indices();
    auto indices2 = shard2.get_indices();

    // Very unlikely to be exactly the same with different seeds
    EXPECT_NE(indices1, indices2);
}

TEST(HashBasedShardingTest, AllSamplesDistributed) {
    std::set<size_t> all_indices;

    for (size_t rank = 0; rank < 4; ++rank) {
        HashBasedSharding shard(100, 4, rank);
        auto indices = shard.get_indices();
        for (size_t idx : indices) {
            EXPECT_TRUE(all_indices.find(idx) == all_indices.end())
                << "Duplicate index " << idx;
            all_indices.insert(idx);
        }
    }

    EXPECT_EQ(all_indices.size(), 100);
}

TEST(HashBasedShardingTest, OwnsIndex) {
    HashBasedSharding shard(100, 4, 0);

    // Count how many indices this shard owns
    size_t count = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (shard.owns_index(i)) {
            count++;
        }
    }

    EXPECT_EQ(count, shard.size());
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(ShardingFactoryTest, CreateContiguous) {
    auto shard = ShardingStrategy::create(ShardingType::CONTIGUOUS, 100, 4, 0);
    EXPECT_NE(shard, nullptr);
    EXPECT_EQ(shard->size(), 25);
}

TEST(ShardingFactoryTest, CreateInterleaved) {
    auto shard = ShardingStrategy::create(ShardingType::INTERLEAVED, 100, 4, 1);
    EXPECT_NE(shard, nullptr);
    auto indices = shard->get_indices();
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 5);
}

TEST(ShardingFactoryTest, CreateHashBased) {
    auto shard = ShardingStrategy::create(ShardingType::HASH_BASED, 100, 4, 2);
    EXPECT_NE(shard, nullptr);
    EXPECT_GT(shard->size(), 0);
}

TEST(ShardingFactoryTest, InvalidRank) {
    EXPECT_THROW(
        ShardingStrategy::create(ShardingType::CONTIGUOUS, 100, 4, 5),
        std::invalid_argument
    );
}

TEST(ShardingFactoryTest, InvalidWorldSize) {
    EXPECT_THROW(
        ShardingStrategy::create(ShardingType::CONTIGUOUS, 100, 0, 0),
        std::invalid_argument
    );
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST(ShardingIteratorTest, RangeBasedFor) {
    auto shard = ShardingStrategy::create(ShardingType::CONTIGUOUS, 100, 4, 0);

    size_t count = 0;
    for (size_t idx : *shard) {
        EXPECT_LT(idx, 25);
        count++;
    }
    EXPECT_EQ(count, 25);
}

TEST(ShardingIteratorTest, ExplicitIteration) {
    InterleavedSharding shard(100, 4, 1);

    auto it = shard.begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 5);

    auto it2 = it++;
    EXPECT_EQ(*it2, 5);
    EXPECT_EQ(*it, 9);
}

// ============================================================================
// ShardingCoordinator Tests
// ============================================================================

TEST(ShardingCoordinatorTest, VerifyCoverage) {
    ShardingCoordinator coord(100, 4, ShardingType::CONTIGUOUS);
    EXPECT_TRUE(coord.verify_coverage());
}

TEST(ShardingCoordinatorTest, VerifyCoverageInterleaved) {
    ShardingCoordinator coord(100, 4, ShardingType::INTERLEAVED);
    EXPECT_TRUE(coord.verify_coverage());
}

TEST(ShardingCoordinatorTest, VerifyCoverageHashBased) {
    ShardingCoordinator coord(100, 4, ShardingType::HASH_BASED);
    EXPECT_TRUE(coord.verify_coverage());
}

TEST(ShardingCoordinatorTest, OwnerRank) {
    ShardingCoordinator coord(100, 4, ShardingType::CONTIGUOUS);

    EXPECT_EQ(coord.owner_rank(0), 0);
    EXPECT_EQ(coord.owner_rank(24), 0);
    EXPECT_EQ(coord.owner_rank(25), 1);
    EXPECT_EQ(coord.owner_rank(50), 2);
    EXPECT_EQ(coord.owner_rank(75), 3);
    EXPECT_EQ(coord.owner_rank(99), 3);
}

TEST(ShardingCoordinatorTest, GetShardSizes) {
    ShardingCoordinator coord(100, 4, ShardingType::CONTIGUOUS);

    auto sizes = coord.get_shard_sizes();
    EXPECT_EQ(sizes.size(), 4);
    for (size_t size : sizes) {
        EXPECT_EQ(size, 25);
    }
}

TEST(ShardingCoordinatorTest, ImbalanceRatioBalanced) {
    ShardingCoordinator coord(100, 4, ShardingType::CONTIGUOUS);
    EXPECT_DOUBLE_EQ(coord.imbalance_ratio(), 1.0);
}

TEST(ShardingCoordinatorTest, ImbalanceRatioUnbalanced) {
    ShardingCoordinator coord(103, 4, ShardingType::CONTIGUOUS);
    double ratio = coord.imbalance_ratio();
    EXPECT_GT(ratio, 1.0);
    EXPECT_LT(ratio, 1.1);  // 26/25 = 1.04
}

// ============================================================================
// String Conversion Tests
// ============================================================================

TEST(ShardingStringTest, TypeToString) {
    EXPECT_EQ(sharding_type_to_string(ShardingType::CONTIGUOUS), "CONTIGUOUS");
    EXPECT_EQ(sharding_type_to_string(ShardingType::INTERLEAVED), "INTERLEAVED");
    EXPECT_EQ(sharding_type_to_string(ShardingType::HASH_BASED), "HASH_BASED");
}

TEST(ShardingStringTest, StringToType) {
    EXPECT_EQ(string_to_sharding_type("CONTIGUOUS"), ShardingType::CONTIGUOUS);
    EXPECT_EQ(string_to_sharding_type("contiguous"), ShardingType::CONTIGUOUS);
    EXPECT_EQ(string_to_sharding_type("INTERLEAVED"), ShardingType::INTERLEAVED);
    EXPECT_EQ(string_to_sharding_type("HASH_BASED"), ShardingType::HASH_BASED);
}

TEST(ShardingStringTest, InvalidString) {
    EXPECT_THROW(string_to_sharding_type("INVALID"), std::invalid_argument);
}

// ============================================================================
// Hash Distribution Balance Tests
// ============================================================================

TEST(HashBalanceTest, EvenDistribution) {
    // With splitmix64 hash, distribution across ranks should be near-uniform
    size_t total = 10000;
    size_t world_size = 8;

    std::vector<size_t> counts(world_size, 0);
    for (size_t rank = 0; rank < world_size; ++rank) {
        HashBasedSharding shard(total, world_size, rank);
        counts[rank] = shard.size();
    }

    // Each rank should get roughly total/world_size = 1250 samples
    // Allow ±5% deviation (1187 to 1312)
    size_t expected = total / world_size;
    for (size_t rank = 0; rank < world_size; ++rank) {
        EXPECT_GT(counts[rank], expected * 95 / 100)
            << "Rank " << rank << " has too few samples: " << counts[rank];
        EXPECT_LT(counts[rank], expected * 105 / 100)
            << "Rank " << rank << " has too many samples: " << counts[rank];
    }
}

// ============================================================================
// Distributed Sampler Shuffle Tests
// ============================================================================

TEST(DistributedShuffleTest, FisherYatesProducesPermutation) {
    // Verify the shuffle produces a valid permutation (no duplicates, all present)
    turboloader::distributed::DistributedConfig config;
    config.world_size = 1;
    config.rank = 0;
    config.shuffle = true;
    config.seed = 42;

    turboloader::distributed::DistributedSampler sampler(100, config);
    auto indices = sampler.get_indices(0);

    // Should have all 100 indices
    EXPECT_EQ(indices.size(), 100);

    std::set<size_t> unique_indices(indices.begin(), indices.end());
    EXPECT_EQ(unique_indices.size(), 100) << "Shuffle produced duplicates!";
}

TEST(DistributedShuffleTest, DifferentEpochsDifferentOrder) {
    turboloader::distributed::DistributedConfig config;
    config.world_size = 1;
    config.rank = 0;
    config.shuffle = true;
    config.seed = 42;

    turboloader::distributed::DistributedSampler sampler(100, config);
    auto epoch0 = sampler.get_indices(0);
    auto epoch1 = sampler.get_indices(1);

    EXPECT_NE(epoch0, epoch1) << "Same order for different epochs";
}

TEST(DistributedShuffleTest, SameEpochSameOrder) {
    turboloader::distributed::DistributedConfig config;
    config.world_size = 1;
    config.rank = 0;
    config.shuffle = true;
    config.seed = 42;

    turboloader::distributed::DistributedSampler sampler(100, config);
    auto run1 = sampler.get_indices(5);
    auto run2 = sampler.get_indices(5);

    EXPECT_EQ(run1, run2) << "Same epoch+seed should produce same order";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ShardingEdgeCaseTest, SingleSample) {
    auto shard = ShardingStrategy::create(ShardingType::CONTIGUOUS, 1, 1, 0);
    EXPECT_EQ(shard->size(), 1);
    EXPECT_EQ(shard->global_index(0), 0);
}

TEST(ShardingEdgeCaseTest, EmptyShard) {
    // 2 samples, 4 workers - ranks 2 and 3 get 0 samples
    ContiguousSharding shard(2, 4, 3);
    EXPECT_EQ(shard.size(), 0);
    EXPECT_TRUE(shard.get_indices().empty());
}

TEST(ShardingEdgeCaseTest, LargeDataset) {
    size_t total = 1000000;
    size_t world_size = 128;

    for (size_t rank = 0; rank < world_size; ++rank) {
        auto shard = ShardingStrategy::create(ShardingType::CONTIGUOUS, total, world_size, rank);
        EXPECT_GT(shard->size(), 0);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
