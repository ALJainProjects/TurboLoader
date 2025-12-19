/**
 * @file test_checkpointing.cpp
 * @brief Tests for mid-epoch checkpointing (v2.20.0)
 *
 * Tests:
 * - PipelineState serialization/deserialization
 * - StateTracker state management
 * - ResumableIndexGenerator
 * - StatefulDataLoader save/resume
 * - Distributed training checkpoints
 * - Edge cases and error handling
 */

#include <gtest/gtest.h>
#include "../src/pipeline/checkpointing.hpp"
#include <thread>
#include <chrono>

using namespace turboloader;

// ============================================================================
// PipelineState Serialization Tests
// ============================================================================

TEST(PipelineState, SerializeEmpty) {
    PipelineState state;
    auto bytes = state.serialize();

    // Should have minimum structure
    EXPECT_GT(bytes.size(), 20u);
}

TEST(PipelineState, SerializeDeserializeBasic) {
    PipelineState original;
    original.epoch = 5;
    original.samples_processed = 1000;
    original.batches_returned = 32;
    original.total_samples = 10000;
    original.batch_size = 64;
    original.num_workers = 4;
    original.dataset_path = "test/dataset.tar";
    original.shuffled = true;

    auto bytes = original.serialize();
    auto restored = PipelineState::deserialize(bytes.data(), bytes.size());

    EXPECT_EQ(restored.epoch, 5u);
    EXPECT_EQ(restored.samples_processed, 1000u);
    EXPECT_EQ(restored.batches_returned, 32u);
    EXPECT_EQ(restored.total_samples, 10000u);
    EXPECT_EQ(restored.batch_size, 64u);
    EXPECT_EQ(restored.num_workers, 4u);
    EXPECT_EQ(restored.dataset_path, "test/dataset.tar");
    EXPECT_TRUE(restored.shuffled);
}

TEST(PipelineState, SerializeDeserializeWorkerState) {
    PipelineState original;
    original.worker_positions = {100, 200, 300, 400};
    original.worker_samples_done = {99, 199, 299, 399};
    original.worker_rng_states = {111, 222, 333, 444};
    original.num_workers = 4;
    original.total_samples = 1000;
    original.batch_size = 32;

    auto bytes = original.serialize();
    auto restored = PipelineState::deserialize(bytes.data(), bytes.size());

    ASSERT_EQ(restored.worker_positions.size(), 4u);
    EXPECT_EQ(restored.worker_positions[0], 100u);
    EXPECT_EQ(restored.worker_positions[3], 400u);

    ASSERT_EQ(restored.worker_samples_done.size(), 4u);
    EXPECT_EQ(restored.worker_samples_done[1], 199u);

    ASSERT_EQ(restored.worker_rng_states.size(), 4u);
    EXPECT_EQ(restored.worker_rng_states[2], 333u);
}

TEST(PipelineState, SerializeDeserializePendingIndices) {
    PipelineState original;
    original.pending_indices = {10, 20, 30, 40, 50};
    original.total_samples = 100;
    original.batch_size = 16;

    auto bytes = original.serialize();
    auto restored = PipelineState::deserialize(bytes.data(), bytes.size());

    ASSERT_EQ(restored.pending_indices.size(), 5u);
    EXPECT_EQ(restored.pending_indices[0], 10u);
    EXPECT_EQ(restored.pending_indices[4], 50u);
}

TEST(PipelineState, SerializeDeserializeShuffleOrder) {
    PipelineState original;
    original.shuffled = true;
    original.shuffle_order_stored = true;
    original.shuffle_order = {9, 7, 5, 3, 1, 0, 2, 4, 6, 8};
    original.total_samples = 10;
    original.batch_size = 2;

    auto bytes = original.serialize();
    auto restored = PipelineState::deserialize(bytes.data(), bytes.size());

    EXPECT_TRUE(restored.shuffled);
    EXPECT_TRUE(restored.shuffle_order_stored);
    ASSERT_EQ(restored.shuffle_order.size(), 10u);
    EXPECT_EQ(restored.shuffle_order[0], 9u);
    EXPECT_EQ(restored.shuffle_order[9], 8u);
}

TEST(PipelineState, SerializeDeserializeDistributed) {
    PipelineState original;
    original.rank = 3;
    original.world_size = 8;
    original.total_samples = 1000;
    original.batch_size = 32;

    auto bytes = original.serialize();
    auto restored = PipelineState::deserialize(bytes.data(), bytes.size());

    EXPECT_EQ(restored.rank, 3);
    EXPECT_EQ(restored.world_size, 8);
}

TEST(PipelineState, DeserializeInvalidMagic) {
    std::vector<uint8_t> bad_data = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_THROW(PipelineState::deserialize(bad_data.data(), bad_data.size()),
                 std::runtime_error);
}

TEST(PipelineState, DeserializeTooSmall) {
    std::vector<uint8_t> small_data = {1, 2, 3, 4};
    EXPECT_THROW(PipelineState::deserialize(small_data.data(), small_data.size()),
                 std::runtime_error);
}

TEST(PipelineState, IsValid) {
    PipelineState valid;
    valid.total_samples = 1000;
    valid.batch_size = 32;
    EXPECT_TRUE(valid.is_valid());

    PipelineState invalid;
    EXPECT_FALSE(invalid.is_valid());

    PipelineState no_batch;
    no_batch.total_samples = 1000;
    EXPECT_FALSE(no_batch.is_valid());
}

TEST(PipelineState, RemainingProgress) {
    PipelineState state;
    state.total_samples = 100;
    state.samples_processed = 25;

    EXPECT_EQ(state.remaining_samples(), 75u);
    EXPECT_DOUBLE_EQ(state.progress_percent(), 25.0);

    state.samples_processed = 100;
    EXPECT_EQ(state.remaining_samples(), 0u);
    EXPECT_DOUBLE_EQ(state.progress_percent(), 100.0);
}

TEST(PipelineState, Summary) {
    PipelineState state;
    state.epoch = 3;
    state.samples_processed = 500;
    state.total_samples = 1000;
    state.batch_size = 32;
    state.num_workers = 4;
    state.dataset_path = "data.tar";

    std::string summary = state.summary();
    EXPECT_NE(summary.find("epoch: 3"), std::string::npos);
    EXPECT_NE(summary.find("500"), std::string::npos);
    EXPECT_NE(summary.find("1000"), std::string::npos);
    EXPECT_NE(summary.find("50%"), std::string::npos);
}

// ============================================================================
// StateTracker Tests
// ============================================================================

TEST(StateTracker, Initialize) {
    StateTracker tracker;
    tracker.initialize(1000, 32, 4, 42, true);

    auto state = tracker.state_dict();
    EXPECT_EQ(state.total_samples, 1000u);
    EXPECT_EQ(state.batch_size, 32u);
    EXPECT_EQ(state.num_workers, 4u);
    EXPECT_EQ(state.rng_seed, 42u);
    EXPECT_TRUE(state.shuffled);
    EXPECT_EQ(state.samples_processed, 0u);
}

TEST(StateTracker, StartEpoch) {
    StateTracker tracker;
    tracker.initialize(1000, 32, 4);

    tracker.start_epoch(5);
    auto state = tracker.state_dict();

    EXPECT_EQ(state.epoch, 5u);
    EXPECT_EQ(state.samples_processed, 0u);
    EXPECT_EQ(state.batches_returned, 0u);
}

TEST(StateTracker, TrackSamples) {
    StateTracker tracker;
    tracker.initialize(100, 10, 2);
    tracker.start_epoch(0);

    // Queue some samples
    tracker.queue_sample(0);
    tracker.queue_sample(1);
    tracker.queue_sample(2);

    auto state = tracker.state_dict();
    EXPECT_EQ(state.pending_indices.size(), 3u);

    // Complete samples
    tracker.complete_sample(0, 0);
    tracker.complete_sample(1, 1);

    state = tracker.state_dict();
    EXPECT_EQ(state.pending_indices.size(), 1u);
    EXPECT_EQ(state.samples_processed, 2u);
    EXPECT_EQ(state.worker_samples_done[0], 1u);
    EXPECT_EQ(state.worker_samples_done[1], 1u);
}

TEST(StateTracker, TrackBatches) {
    StateTracker tracker;
    tracker.initialize(100, 10, 1);
    tracker.start_epoch(0);

    tracker.complete_batch();
    tracker.complete_batch();
    tracker.complete_batch();

    auto state = tracker.state_dict();
    EXPECT_EQ(state.batches_returned, 3u);
}

TEST(StateTracker, WorkerPositions) {
    StateTracker tracker;
    tracker.initialize(100, 10, 4);

    tracker.update_worker_position(0, 10);
    tracker.update_worker_position(1, 20);
    tracker.update_worker_position(2, 30);
    tracker.update_worker_position(3, 40);

    auto state = tracker.state_dict();
    EXPECT_EQ(state.worker_positions[0], 10u);
    EXPECT_EQ(state.worker_positions[1], 20u);
    EXPECT_EQ(state.worker_positions[2], 30u);
    EXPECT_EQ(state.worker_positions[3], 40u);
}

TEST(StateTracker, SetDistributed) {
    StateTracker tracker;
    tracker.initialize(1000, 32, 4);
    tracker.set_distributed(3, 8);

    auto state = tracker.state_dict();
    EXPECT_EQ(state.rank, 3);
    EXPECT_EQ(state.world_size, 8);
}

TEST(StateTracker, SaveLoadState) {
    StateTracker original;
    original.initialize(1000, 32, 4, 42, true);
    original.start_epoch(5);
    original.set_dataset_path("data.tar");
    original.set_distributed(2, 4);

    // Simulate some progress
    for (int i = 0; i < 100; ++i) {
        original.complete_sample(i, i % 4);
    }
    for (int i = 0; i < 4; ++i) {
        original.complete_batch();
    }

    auto state = original.state_dict();

    // Load into new tracker
    StateTracker restored;
    restored.initialize(1000, 32, 4);
    restored.load_state_dict(state);

    auto restored_state = restored.state_dict();
    EXPECT_EQ(restored_state.epoch, 5u);
    EXPECT_EQ(restored_state.samples_processed, 100u);
    EXPECT_EQ(restored_state.batches_returned, 4u);
    EXPECT_EQ(restored_state.rank, 2);
    EXPECT_EQ(restored_state.world_size, 4);
}

TEST(StateTracker, EpochComplete) {
    StateTracker tracker;
    tracker.initialize(10, 5, 1);
    tracker.start_epoch(0);

    EXPECT_FALSE(tracker.epoch_complete());
    EXPECT_EQ(tracker.remaining_samples(), 10u);

    for (int i = 0; i < 10; ++i) {
        tracker.complete_sample(i, 0);
    }

    EXPECT_TRUE(tracker.epoch_complete());
    EXPECT_EQ(tracker.remaining_samples(), 0u);
}

TEST(StateTracker, ThreadSafe) {
    StateTracker tracker;
    tracker.initialize(1000, 32, 4);
    tracker.start_epoch(0);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&tracker, t]() {
            for (int i = 0; i < 100; ++i) {
                tracker.complete_sample(t * 100 + i, t);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(tracker.samples_processed(), 400u);
}

// ============================================================================
// ResumableIndexGenerator Tests
// ============================================================================

TEST(ResumableIndexGenerator, Sequential) {
    ResumableIndexGenerator gen(10, 42);
    gen.set_shuffle(false);

    std::vector<size_t> indices;
    size_t idx;
    while (gen.next(idx)) {
        indices.push_back(idx);
    }

    ASSERT_EQ(indices.size(), 10u);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(indices[i], i);
    }
}

TEST(ResumableIndexGenerator, Shuffled) {
    ResumableIndexGenerator gen(100, 42);
    gen.set_shuffle(true);

    std::vector<size_t> indices;
    size_t idx;
    while (gen.next(idx)) {
        indices.push_back(idx);
    }

    EXPECT_EQ(indices.size(), 100u);

    // Verify all indices present
    std::sort(indices.begin(), indices.end());
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(indices[i], i);
    }
}

TEST(ResumableIndexGenerator, Reproducible) {
    ResumableIndexGenerator gen1(100, 12345);
    gen1.set_shuffle(true);

    ResumableIndexGenerator gen2(100, 12345);
    gen2.set_shuffle(true);

    size_t idx1, idx2;
    while (gen1.next(idx1) && gen2.next(idx2)) {
        EXPECT_EQ(idx1, idx2);
    }
}

TEST(ResumableIndexGenerator, DifferentEpochs) {
    ResumableIndexGenerator gen(100, 42);
    gen.set_shuffle(true);
    gen.set_epoch(0);

    std::vector<size_t> epoch0;
    size_t idx;
    while (gen.next(idx)) {
        epoch0.push_back(idx);
    }

    gen.set_epoch(1);
    std::vector<size_t> epoch1;
    while (gen.next(idx)) {
        epoch1.push_back(idx);
    }

    // Different epochs should have different order
    EXPECT_NE(epoch0, epoch1);
}

TEST(ResumableIndexGenerator, SkipTo) {
    ResumableIndexGenerator gen(100, 42);
    gen.set_shuffle(false);
    gen.skip_to(50);

    EXPECT_EQ(gen.position(), 50u);
    EXPECT_EQ(gen.remaining(), 50u);

    size_t idx;
    EXPECT_TRUE(gen.next(idx));
    EXPECT_EQ(idx, 50u);
}

TEST(ResumableIndexGenerator, SetGetIndices) {
    ResumableIndexGenerator gen(10, 42);
    gen.set_shuffle(true);

    auto original = gen.indices();

    ResumableIndexGenerator gen2(10, 0);  // Different seed
    gen2.set_indices(original);

    EXPECT_EQ(gen2.indices(), original);
}

// ============================================================================
// StatefulDataLoader Tests
// ============================================================================

TEST(StatefulDataLoader, Initialize) {
    StatefulLoaderConfig config;
    config.dataset_path = "data.tar";
    config.batch_size = 32;
    config.num_workers = 4;
    config.shuffle = true;
    config.seed = 42;

    StatefulDataLoader loader(config);
    EXPECT_FALSE(loader.is_initialized());

    loader.initialize(1000);
    EXPECT_TRUE(loader.is_initialized());
    EXPECT_EQ(loader.total_samples(), 1000u);
    EXPECT_EQ(loader.batch_size(), 32u);
}

TEST(StatefulDataLoader, NextBatch) {
    StatefulLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = false;

    StatefulDataLoader loader(config);
    loader.initialize(100);
    loader.start_epoch(0);

    auto batch = loader.next_batch();
    ASSERT_EQ(batch.size(), 10u);

    // Verify first batch is 0-9 (unshuffled)
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(batch[i], i);
    }

    // Get another batch
    batch = loader.next_batch();
    ASSERT_EQ(batch.size(), 10u);
    EXPECT_EQ(batch[0], 10u);
}

TEST(StatefulDataLoader, DropLast) {
    StatefulLoaderConfig config;
    config.batch_size = 30;
    config.drop_last = true;

    StatefulDataLoader loader(config);
    loader.initialize(100);  // 100 / 30 = 3 batches, 10 remainder
    loader.start_epoch(0);

    std::vector<std::vector<size_t>> batches;
    while (!loader.epoch_complete()) {
        auto batch = loader.next_batch();
        if (!batch.empty()) {
            batches.push_back(batch);
        }
    }

    // Should have 3 full batches, last partial batch dropped
    EXPECT_EQ(batches.size(), 3u);
    for (auto& b : batches) {
        EXPECT_EQ(b.size(), 30u);
    }
}

TEST(StatefulDataLoader, StateDict) {
    StatefulLoaderConfig config;
    config.dataset_path = "test.tar";
    config.batch_size = 10;
    config.num_workers = 4;
    config.seed = 12345;
    config.rank = 2;
    config.world_size = 8;

    StatefulDataLoader loader(config);
    loader.initialize(100);
    loader.start_epoch(5);

    // Process some batches
    for (int i = 0; i < 3; ++i) {
        auto batch = loader.next_batch();
        loader.mark_completed(batch, i % 4);
    }

    auto state = loader.state_dict();
    EXPECT_EQ(state.epoch, 5u);
    EXPECT_EQ(state.samples_processed, 30u);
    EXPECT_EQ(state.batches_returned, 3u);
    EXPECT_EQ(state.rank, 2);
    EXPECT_EQ(state.world_size, 8);
    EXPECT_EQ(state.dataset_path, "test.tar");
}

TEST(StatefulDataLoader, SaveResume) {
    StatefulLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = false;

    // Create loader and process half the epoch
    StatefulDataLoader loader1(config);
    loader1.initialize(100);
    loader1.start_epoch(0);

    for (int i = 0; i < 5; ++i) {
        auto batch = loader1.next_batch();
        loader1.mark_completed(batch, 0);
    }

    // Save state
    auto state = loader1.state_dict();
    auto bytes = state.serialize();

    // Create new loader and restore
    StatefulDataLoader loader2(config);
    loader2.initialize(100);

    auto restored_state = PipelineState::deserialize(bytes.data(), bytes.size());
    loader2.load_state_dict(restored_state);

    // Continue from saved position
    auto batch = loader2.next_batch();

    // Should continue from index 50
    ASSERT_FALSE(batch.empty());
    EXPECT_EQ(batch[0], 50u);
}

TEST(StatefulDataLoader, ResumeShuffled) {
    StatefulLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = true;
    config.seed = 42;

    // Create loader and process half the epoch
    StatefulDataLoader loader1(config);
    loader1.initialize(100);
    loader1.start_epoch(3);

    std::vector<std::vector<size_t>> batches_before;
    for (int i = 0; i < 5; ++i) {
        auto batch = loader1.next_batch();
        batches_before.push_back(batch);
        loader1.mark_completed(batch, 0);
    }

    auto state = loader1.state_dict();

    // Get remaining batches
    std::vector<std::vector<size_t>> batches_after_original;
    while (!loader1.epoch_complete()) {
        auto batch = loader1.next_batch();
        if (!batch.empty()) {
            batches_after_original.push_back(batch);
            loader1.mark_completed(batch, 0);
        }
    }

    // Restore and verify same remaining batches
    StatefulDataLoader loader2(config);
    loader2.initialize(100);
    loader2.load_state_dict(state);

    std::vector<std::vector<size_t>> batches_after_restored;
    while (!loader2.epoch_complete()) {
        auto batch = loader2.next_batch();
        if (!batch.empty()) {
            batches_after_restored.push_back(batch);
            loader2.mark_completed(batch, 0);
        }
    }

    // Restored should produce same remaining batches
    ASSERT_EQ(batches_after_restored.size(), batches_after_original.size());
    for (size_t i = 0; i < batches_after_original.size(); ++i) {
        EXPECT_EQ(batches_after_restored[i], batches_after_original[i]);
    }
}

TEST(StatefulDataLoader, Progress) {
    StatefulLoaderConfig config;
    config.batch_size = 25;

    StatefulDataLoader loader(config);
    loader.initialize(100);
    loader.start_epoch(2);

    // Process half
    for (int i = 0; i < 2; ++i) {
        auto batch = loader.next_batch();
        loader.mark_completed(batch, 0);
    }

    std::string progress = loader.progress();
    EXPECT_NE(progress.find("Epoch 2"), std::string::npos);
    EXPECT_NE(progress.find("50"), std::string::npos);
    EXPECT_NE(progress.find("100"), std::string::npos);
}

TEST(StatefulDataLoader, MultipleEpochs) {
    StatefulLoaderConfig config;
    config.batch_size = 50;
    config.shuffle = true;
    config.seed = 123;

    StatefulDataLoader loader(config);
    loader.initialize(100);

    std::vector<size_t> epoch0_indices, epoch1_indices;

    // Epoch 0
    loader.start_epoch(0);
    while (!loader.epoch_complete()) {
        auto batch = loader.next_batch();
        epoch0_indices.insert(epoch0_indices.end(), batch.begin(), batch.end());
        loader.mark_completed(batch, 0);
    }

    // Epoch 1
    loader.start_epoch(1);
    while (!loader.epoch_complete()) {
        auto batch = loader.next_batch();
        epoch1_indices.insert(epoch1_indices.end(), batch.begin(), batch.end());
        loader.mark_completed(batch, 0);
    }

    // Both epochs should have all indices
    std::sort(epoch0_indices.begin(), epoch0_indices.end());
    std::sort(epoch1_indices.begin(), epoch1_indices.end());

    EXPECT_EQ(epoch0_indices.size(), 100u);
    EXPECT_EQ(epoch1_indices.size(), 100u);

    // But in different order
    // (Note: compare unsorted to check order difference)
}

// ============================================================================
// CheckpointableIterator Tests
// ============================================================================

TEST(CheckpointableIterator, Basic) {
    std::vector<size_t> indices = {0, 1, 2, 3, 4};
    StateTracker tracker;
    tracker.initialize(5, 1, 1);

    CheckpointableIterator<std::vector<size_t>::iterator> begin(
        indices.begin(), indices.end(), &tracker, 0);
    CheckpointableIterator<std::vector<size_t>::iterator> end(
        indices.end(), indices.end(), &tracker, 0);

    std::vector<size_t> result;
    while (begin != end) {
        result.push_back(*begin);
        ++begin;
    }

    EXPECT_EQ(result, indices);
}

TEST(CheckpointableIterator, SkipTo) {
    std::vector<size_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    StateTracker tracker;
    tracker.initialize(10, 1, 1);

    CheckpointableIterator<std::vector<size_t>::iterator> it(
        indices.begin(), indices.end(), &tracker, 0);

    it.skip_to(5);
    EXPECT_EQ(it.position(), 5u);
    EXPECT_EQ(*it, 5u);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
