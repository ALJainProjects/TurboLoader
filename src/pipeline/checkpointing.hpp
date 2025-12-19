/**
 * @file checkpointing.hpp
 * @brief Mid-epoch checkpointing for resumable training (v2.20.0)
 *
 * Provides save/resume functionality for exact dataset position recovery,
 * matching TorchData StatefulDataLoader capabilities.
 *
 * Features:
 * - Save exact position in dataset (epoch, sample, batch)
 * - Resume from checkpoint with exact reproducibility
 * - Track per-worker iteration state
 * - RNG state preservation for deterministic resume
 * - Binary serialization for efficient storage
 * - Distributed training support
 *
 * Usage:
 * ```cpp
 * StatefulDataLoader loader(config);
 *
 * // Save checkpoint
 * PipelineState state = loader.state_dict();
 * auto bytes = state.serialize();
 *
 * // Resume from checkpoint
 * PipelineState loaded = PipelineState::deserialize(bytes.data(), bytes.size());
 * loader.load_state_dict(loaded);
 * ```
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <random>
#include <atomic>
#include <mutex>

namespace turboloader {

/**
 * @brief Complete pipeline state for checkpointing
 *
 * Captures all information needed to resume training from exact position.
 */
struct PipelineState {
    // Version for format compatibility
    static constexpr uint32_t FORMAT_VERSION = 1;

    // Core iteration state
    size_t epoch = 0;                          // Current epoch number
    size_t samples_processed = 0;              // Total samples processed this epoch
    size_t batches_returned = 0;               // Batches returned to user this epoch
    size_t total_samples = 0;                  // Total samples in dataset

    // Per-worker state (for multi-worker dataloaders)
    std::vector<size_t> worker_positions;       // Current position per worker
    std::vector<size_t> worker_samples_done;    // Samples completed per worker

    // Samples currently in-flight (queued but not yet returned)
    std::vector<size_t> pending_indices;

    // RNG state for reproducibility
    uint64_t rng_seed = 0;
    uint64_t rng_state = 0;                    // Current RNG counter state
    std::vector<uint64_t> worker_rng_states;   // Per-worker RNG states

    // Shuffle state
    bool shuffled = false;
    std::vector<size_t> shuffle_order;         // Current epoch's shuffle order (if small enough)
    bool shuffle_order_stored = false;         // Whether shuffle order is stored

    // Distributed training state
    int rank = 0;
    int world_size = 1;

    // Metadata
    std::string dataset_path;
    size_t batch_size = 0;
    size_t num_workers = 0;

    /**
     * @brief Check if state is valid for resumption
     */
    bool is_valid() const {
        return total_samples > 0 && batch_size > 0;
    }

    /**
     * @brief Get remaining samples in current epoch
     */
    size_t remaining_samples() const {
        return total_samples > samples_processed ?
               total_samples - samples_processed : 0;
    }

    /**
     * @brief Get completion percentage
     */
    double progress_percent() const {
        if (total_samples == 0) return 0.0;
        return 100.0 * samples_processed / total_samples;
    }

    /**
     * @brief Serialize state to binary format
     */
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buffer;
        buffer.reserve(1024);  // Initial estimate

        // Helper to write primitive types
        auto write_value = [&buffer](const auto& val) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&val);
            buffer.insert(buffer.end(), ptr, ptr + sizeof(val));
        };

        // Helper to write vectors
        auto write_vector = [&buffer, &write_value](const auto& vec) {
            write_value(vec.size());
            for (const auto& v : vec) {
                write_value(v);
            }
        };

        // Helper to write strings
        auto write_string = [&buffer, &write_value](const std::string& str) {
            write_value(str.size());
            buffer.insert(buffer.end(), str.begin(), str.end());
        };

        // Magic number for format identification
        write_value(uint32_t(0x54425343));  // "TBSC" - TurBoloader State Checkpoint
        write_value(FORMAT_VERSION);

        // Core state
        write_value(epoch);
        write_value(samples_processed);
        write_value(batches_returned);
        write_value(total_samples);

        // Worker states
        write_vector(worker_positions);
        write_vector(worker_samples_done);
        write_vector(pending_indices);

        // RNG state
        write_value(rng_seed);
        write_value(rng_state);
        write_vector(worker_rng_states);

        // Shuffle state
        write_value(uint8_t(shuffled ? 1 : 0));
        write_value(uint8_t(shuffle_order_stored ? 1 : 0));
        if (shuffle_order_stored) {
            write_vector(shuffle_order);
        }

        // Distributed state
        write_value(rank);
        write_value(world_size);

        // Metadata
        write_string(dataset_path);
        write_value(batch_size);
        write_value(num_workers);

        // Checksum (simple XOR of all bytes)
        uint32_t checksum = 0;
        for (uint8_t b : buffer) {
            checksum ^= (checksum << 5) + b + (checksum >> 2);
        }
        write_value(checksum);

        return buffer;
    }

    /**
     * @brief Deserialize state from binary format
     */
    static PipelineState deserialize(const uint8_t* data, size_t size) {
        if (size < 16) {
            throw std::runtime_error("Checkpoint data too small");
        }

        PipelineState state;
        size_t pos = 0;

        // Helper to read primitive types
        auto read_value = [data, size, &pos](auto& val) {
            if (pos + sizeof(val) > size) {
                throw std::runtime_error("Checkpoint data truncated");
            }
            std::memcpy(&val, data + pos, sizeof(val));
            pos += sizeof(val);
        };

        // Helper to read vectors
        auto read_vector = [&read_value](auto& vec) {
            size_t count;
            read_value(count);
            vec.resize(count);
            for (auto& v : vec) {
                read_value(v);
            }
        };

        // Helper to read strings
        auto read_string = [data, size, &pos, &read_value](std::string& str) {
            size_t len;
            read_value(len);
            if (pos + len > size) {
                throw std::runtime_error("Checkpoint data truncated");
            }
            str.assign(reinterpret_cast<const char*>(data + pos), len);
            pos += len;
        };

        // Verify magic number
        uint32_t magic;
        read_value(magic);
        if (magic != 0x54425343) {
            throw std::runtime_error("Invalid checkpoint format (bad magic number)");
        }

        // Check version
        uint32_t version;
        read_value(version);
        if (version != FORMAT_VERSION) {
            throw std::runtime_error("Incompatible checkpoint version");
        }

        // Core state
        read_value(state.epoch);
        read_value(state.samples_processed);
        read_value(state.batches_returned);
        read_value(state.total_samples);

        // Worker states
        read_vector(state.worker_positions);
        read_vector(state.worker_samples_done);
        read_vector(state.pending_indices);

        // RNG state
        read_value(state.rng_seed);
        read_value(state.rng_state);
        read_vector(state.worker_rng_states);

        // Shuffle state
        uint8_t shuffled_byte, stored_byte;
        read_value(shuffled_byte);
        read_value(stored_byte);
        state.shuffled = (shuffled_byte != 0);
        state.shuffle_order_stored = (stored_byte != 0);
        if (state.shuffle_order_stored) {
            read_vector(state.shuffle_order);
        }

        // Distributed state
        read_value(state.rank);
        read_value(state.world_size);

        // Metadata
        read_string(state.dataset_path);
        read_value(state.batch_size);
        read_value(state.num_workers);

        // Verify checksum
        uint32_t stored_checksum;
        read_value(stored_checksum);

        uint32_t computed_checksum = 0;
        for (size_t i = 0; i < pos - sizeof(uint32_t); ++i) {
            computed_checksum ^= (computed_checksum << 5) + data[i] + (computed_checksum >> 2);
        }

        if (stored_checksum != computed_checksum) {
            throw std::runtime_error("Checkpoint data corrupted (checksum mismatch)");
        }

        return state;
    }

    /**
     * @brief Create human-readable summary
     */
    std::string summary() const {
        std::string result;
        result += "PipelineState {\n";
        result += "  epoch: " + std::to_string(epoch) + "\n";
        result += "  samples_processed: " + std::to_string(samples_processed) + "/" +
                  std::to_string(total_samples) + " (" +
                  std::to_string(static_cast<int>(progress_percent())) + "%)\n";
        result += "  batches_returned: " + std::to_string(batches_returned) + "\n";
        result += "  batch_size: " + std::to_string(batch_size) + "\n";
        result += "  num_workers: " + std::to_string(num_workers) + "\n";
        if (world_size > 1) {
            result += "  rank: " + std::to_string(rank) + "/" +
                      std::to_string(world_size) + "\n";
        }
        result += "  shuffled: " + std::string(shuffled ? "true" : "false") + "\n";
        result += "  dataset: " + dataset_path + "\n";
        result += "}\n";
        return result;
    }
};

/**
 * @brief State tracker for pipeline iteration
 *
 * Thread-safe state tracking for checkpointing support.
 */
class StateTracker {
public:
    StateTracker() = default;

    /**
     * @brief Initialize tracker for new dataset
     */
    void initialize(size_t total_samples, size_t batch_size, size_t num_workers,
                   uint64_t seed = 42, bool shuffle = false) {
        std::lock_guard<std::mutex> lock(mutex_);

        state_.total_samples = total_samples;
        state_.batch_size = batch_size;
        state_.num_workers = num_workers;
        state_.rng_seed = seed;
        state_.shuffled = shuffle;

        // Initialize worker states
        state_.worker_positions.resize(num_workers, 0);
        state_.worker_samples_done.resize(num_workers, 0);
        state_.worker_rng_states.resize(num_workers);

        // Initialize per-worker RNG states
        std::mt19937_64 seed_gen(seed);
        for (size_t i = 0; i < num_workers; ++i) {
            state_.worker_rng_states[i] = seed_gen();
        }

        reset_epoch_internal();
    }

    /**
     * @brief Reset for new epoch
     */
    void reset_epoch() {
        std::lock_guard<std::mutex> lock(mutex_);
        reset_epoch_internal();
    }

    /**
     * @brief Start new epoch
     */
    void start_epoch(size_t epoch_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.epoch = epoch_num;
        reset_epoch_internal();

        // Update RNG state for new epoch
        state_.rng_state = state_.rng_seed ^ (epoch_num * 0x9e3779b97f4a7c15ULL);
    }

    /**
     * @brief Record sample being queued for processing
     */
    void queue_sample(size_t sample_idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.pending_indices.push_back(sample_idx);
    }

    /**
     * @brief Record sample processing completion
     */
    void complete_sample(size_t sample_idx, size_t worker_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Remove from pending
        auto it = std::find(state_.pending_indices.begin(),
                           state_.pending_indices.end(), sample_idx);
        if (it != state_.pending_indices.end()) {
            state_.pending_indices.erase(it);
        }

        state_.samples_processed++;
        if (worker_id < state_.worker_samples_done.size()) {
            state_.worker_samples_done[worker_id]++;
        }
    }

    /**
     * @brief Record batch being returned to user
     */
    void complete_batch() {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.batches_returned++;
    }

    /**
     * @brief Update worker position
     */
    void update_worker_position(size_t worker_id, size_t position) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_id < state_.worker_positions.size()) {
            state_.worker_positions[worker_id] = position;
        }
    }

    /**
     * @brief Set shuffle order for current epoch
     */
    void set_shuffle_order(const std::vector<size_t>& order) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Only store if reasonably small (< 10M samples)
        if (order.size() < 10000000) {
            state_.shuffle_order = order;
            state_.shuffle_order_stored = true;
        } else {
            state_.shuffle_order.clear();
            state_.shuffle_order_stored = false;
        }
    }

    /**
     * @brief Set distributed training info
     */
    void set_distributed(int rank, int world_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.rank = rank;
        state_.world_size = world_size;
    }

    /**
     * @brief Set dataset path
     */
    void set_dataset_path(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.dataset_path = path;
    }

    /**
     * @brief Get current state (thread-safe copy)
     */
    PipelineState state_dict() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

    /**
     * @brief Load state from checkpoint
     */
    void load_state_dict(const PipelineState& state) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = state;
    }

    /**
     * @brief Get current epoch
     */
    size_t current_epoch() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_.epoch;
    }

    /**
     * @brief Get samples processed this epoch
     */
    size_t samples_processed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_.samples_processed;
    }

    /**
     * @brief Get remaining samples
     */
    size_t remaining_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_.remaining_samples();
    }

    /**
     * @brief Check if epoch is complete
     */
    bool epoch_complete() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_.samples_processed >= state_.total_samples;
    }

private:
    mutable std::mutex mutex_;
    PipelineState state_;

    // Internal reset without locking (called from locked methods)
    void reset_epoch_internal() {
        state_.samples_processed = 0;
        state_.batches_returned = 0;
        state_.pending_indices.clear();

        for (size_t i = 0; i < state_.num_workers; ++i) {
            state_.worker_positions[i] = 0;
            state_.worker_samples_done[i] = 0;
        }
    }
};

/**
 * @brief Checkpointable iterator wrapper
 *
 * Wraps an index iterator to track position for checkpointing.
 */
template<typename IndexIterator>
class CheckpointableIterator {
public:
    using value_type = typename std::iterator_traits<IndexIterator>::value_type;
    using difference_type = typename std::iterator_traits<IndexIterator>::difference_type;
    using pointer = typename std::iterator_traits<IndexIterator>::pointer;
    using reference = typename std::iterator_traits<IndexIterator>::reference;
    using iterator_category = std::forward_iterator_tag;

    CheckpointableIterator(IndexIterator it, IndexIterator end,
                           StateTracker* tracker, size_t worker_id = 0)
        : current_(it), end_(end), tracker_(tracker),
          worker_id_(worker_id), position_(0) {}

    value_type operator*() const {
        return *current_;
    }

    CheckpointableIterator& operator++() {
        if (current_ != end_) {
            if (tracker_) {
                tracker_->complete_sample(*current_, worker_id_);
                tracker_->update_worker_position(worker_id_, ++position_);
            }
            ++current_;
        }
        return *this;
    }

    CheckpointableIterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const CheckpointableIterator& other) const {
        return current_ == other.current_;
    }

    bool operator!=(const CheckpointableIterator& other) const {
        return !(*this == other);
    }

    /**
     * @brief Skip to position (for resume)
     */
    void skip_to(size_t position) {
        while (position_ < position && current_ != end_) {
            ++current_;
            ++position_;
        }
        if (tracker_) {
            tracker_->update_worker_position(worker_id_, position_);
        }
    }

    size_t position() const { return position_; }

private:
    IndexIterator current_;
    IndexIterator end_;
    StateTracker* tracker_;
    size_t worker_id_;
    size_t position_;
};

/**
 * @brief Resumable index generator
 *
 * Generates indices with checkpoint/resume capability.
 */
class ResumableIndexGenerator {
public:
    ResumableIndexGenerator(size_t total_samples, uint64_t seed = 42)
        : total_samples_(total_samples), seed_(seed),
          current_epoch_(0), current_position_(0) {
        regenerate_indices();
    }

    /**
     * @brief Set up for new epoch
     */
    void set_epoch(size_t epoch) {
        current_epoch_ = epoch;
        current_position_ = 0;
        regenerate_indices();
    }

    /**
     * @brief Enable/disable shuffling
     */
    void set_shuffle(bool shuffle) {
        shuffle_ = shuffle;
        regenerate_indices();
    }

    /**
     * @brief Skip to position (for resume)
     */
    void skip_to(size_t position) {
        current_position_ = std::min(position, total_samples_);
    }

    /**
     * @brief Get next index
     */
    bool next(size_t& idx) {
        if (current_position_ >= total_samples_) {
            return false;
        }
        idx = indices_[current_position_++];
        return true;
    }

    /**
     * @brief Check if more indices available
     */
    bool has_next() const {
        return current_position_ < total_samples_;
    }

    /**
     * @brief Get remaining count
     */
    size_t remaining() const {
        return total_samples_ > current_position_ ?
               total_samples_ - current_position_ : 0;
    }

    /**
     * @brief Get current position
     */
    size_t position() const {
        return current_position_;
    }

    /**
     * @brief Get current epoch
     */
    size_t epoch() const {
        return current_epoch_;
    }

    /**
     * @brief Get indices (for state saving)
     */
    const std::vector<size_t>& indices() const {
        return indices_;
    }

    /**
     * @brief Set indices (for state loading)
     */
    void set_indices(const std::vector<size_t>& indices) {
        indices_ = indices;
    }

private:
    void regenerate_indices() {
        indices_.resize(total_samples_);
        for (size_t i = 0; i < total_samples_; ++i) {
            indices_[i] = i;
        }

        if (shuffle_) {
            // Deterministic shuffle based on seed and epoch
            uint64_t epoch_seed = seed_ ^ (current_epoch_ * 0x9e3779b97f4a7c15ULL);
            std::mt19937_64 rng(epoch_seed);
            std::shuffle(indices_.begin(), indices_.end(), rng);
        }
    }

    size_t total_samples_;
    uint64_t seed_;
    size_t current_epoch_;
    size_t current_position_;
    bool shuffle_ = false;
    std::vector<size_t> indices_;
};

/**
 * @brief Configuration for stateful data loading
 */
struct StatefulLoaderConfig {
    std::string dataset_path;
    size_t batch_size = 32;
    size_t num_workers = 4;
    bool shuffle = true;
    uint64_t seed = 42;
    bool drop_last = false;
    size_t prefetch_factor = 2;

    // Distributed settings
    int rank = 0;
    int world_size = 1;
};

/**
 * @brief Stateful data loader with checkpoint/resume support
 *
 * TorchData-compatible interface for resumable training.
 */
class StatefulDataLoader {
public:
    explicit StatefulDataLoader(const StatefulLoaderConfig& config)
        : config_(config) {
        tracker_.set_dataset_path(config.dataset_path);
        tracker_.set_distributed(config.rank, config.world_size);
    }

    /**
     * @brief Initialize with dataset size
     */
    void initialize(size_t total_samples) {
        total_samples_ = total_samples;
        tracker_.initialize(total_samples, config_.batch_size,
                           config_.num_workers, config_.seed, config_.shuffle);
        index_gen_ = std::make_unique<ResumableIndexGenerator>(total_samples, config_.seed);
        index_gen_->set_shuffle(config_.shuffle);
        initialized_ = true;
    }

    /**
     * @brief Get state dict for checkpointing
     */
    PipelineState state_dict() const {
        auto state = tracker_.state_dict();

        // Add index generator state
        if (index_gen_) {
            if (config_.shuffle && index_gen_->indices().size() < 10000000) {
                state.shuffle_order = index_gen_->indices();
                state.shuffle_order_stored = true;
            }
        }

        return state;
    }

    /**
     * @brief Load state dict for resumption
     */
    void load_state_dict(const PipelineState& state) {
        tracker_.load_state_dict(state);

        // Restore index generator
        if (index_gen_) {
            index_gen_->set_epoch(state.epoch);
            if (state.shuffle_order_stored && !state.shuffle_order.empty()) {
                index_gen_->set_indices(state.shuffle_order);
            }
            index_gen_->skip_to(state.samples_processed);
        }

        resumed_ = true;
    }

    /**
     * @brief Start epoch (or resume if state loaded)
     */
    void start_epoch(size_t epoch) {
        if (!resumed_) {
            tracker_.start_epoch(epoch);
            if (index_gen_) {
                index_gen_->set_epoch(epoch);
            }
        }
        resumed_ = false;  // Clear for next epoch
    }

    /**
     * @brief Get next batch of indices
     */
    std::vector<size_t> next_batch() {
        if (!index_gen_) {
            return {};
        }

        std::vector<size_t> batch;
        batch.reserve(config_.batch_size);

        size_t idx;
        while (batch.size() < config_.batch_size && index_gen_->next(idx)) {
            batch.push_back(idx);
            tracker_.queue_sample(idx);
        }

        // Handle drop_last
        if (config_.drop_last && batch.size() < config_.batch_size) {
            batch.clear();
        }

        if (!batch.empty()) {
            tracker_.complete_batch();
        }

        return batch;
    }

    /**
     * @brief Mark samples as completed
     */
    void mark_completed(const std::vector<size_t>& indices, size_t worker_id = 0) {
        for (size_t idx : indices) {
            tracker_.complete_sample(idx, worker_id);
        }
    }

    /**
     * @brief Check if epoch is complete
     *
     * Returns true when either:
     * - All indices have been returned as batches (no more to fetch)
     * - All samples have been marked as processed
     */
    bool epoch_complete() const {
        // Check if index generator has no more indices
        if (index_gen_ && !index_gen_->has_next()) {
            return true;
        }
        // Also check tracker (for cases where mark_completed is used)
        return tracker_.epoch_complete();
    }

    /**
     * @brief Get progress info
     */
    std::string progress() const {
        auto state = tracker_.state_dict();
        return "Epoch " + std::to_string(state.epoch) + ": " +
               std::to_string(state.samples_processed) + "/" +
               std::to_string(state.total_samples) + " samples (" +
               std::to_string(static_cast<int>(state.progress_percent())) + "%)";
    }

    /**
     * @brief Get total samples
     */
    size_t total_samples() const { return total_samples_; }

    /**
     * @brief Get batch size
     */
    size_t batch_size() const { return config_.batch_size; }

    /**
     * @brief Check if initialized
     */
    bool is_initialized() const { return initialized_; }

private:
    StatefulLoaderConfig config_;
    StateTracker tracker_;
    std::unique_ptr<ResumableIndexGenerator> index_gen_;
    size_t total_samples_ = 0;
    bool initialized_ = false;
    bool resumed_ = false;
};

}  // namespace turboloader
