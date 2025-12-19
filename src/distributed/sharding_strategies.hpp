/**
 * @file sharding_strategies.hpp
 * @brief Advanced sharding strategies for distributed training (v2.23.0)
 *
 * Provides multiple strategies for distributing data across workers:
 * - CONTIGUOUS: Traditional approach, each rank gets [start, end) range
 * - INTERLEAVED: Each rank gets every N-th sample (better for heterogeneous data)
 * - HASH_BASED: Hash(sample_id) % world_size == rank (deterministic shuffling)
 *
 * Usage:
 * ```cpp
 * // Create sharding strategy
 * auto sharder = ShardingStrategy::create(
 *     ShardingType::INTERLEAVED,
 *     total_samples,
 *     world_size,
 *     rank
 * );
 *
 * // Get indices for this rank
 * auto indices = sharder->get_indices();
 *
 * // Or iterate directly
 * for (size_t idx : *sharder) {
 *     // Process sample at idx
 * }
 * ```
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <functional>
#include <algorithm>

namespace turboloader {

/**
 * @brief Types of sharding strategies
 */
enum class ShardingType {
    CONTIGUOUS,   // Each rank gets a contiguous range [start, end)
    INTERLEAVED,  // Each rank gets every N-th sample
    HASH_BASED    // Hash(sample_id) % world_size == rank
};

/**
 * @brief Base class for sharding strategies
 */
class ShardingStrategy {
public:
    virtual ~ShardingStrategy() = default;

    /**
     * @brief Get total number of samples in this shard
     */
    virtual size_t size() const = 0;

    /**
     * @brief Get all indices for this shard
     */
    virtual std::vector<size_t> get_indices() const = 0;

    /**
     * @brief Get global index for local index in this shard
     */
    virtual size_t global_index(size_t local_idx) const = 0;

    /**
     * @brief Check if global index belongs to this shard
     */
    virtual bool owns_index(size_t global_idx) const = 0;

    /**
     * @brief Get shard configuration info
     */
    size_t total_samples() const { return total_samples_; }
    size_t world_size() const { return world_size_; }
    size_t rank() const { return rank_; }

    /**
     * @brief Factory method to create sharding strategy
     */
    static std::unique_ptr<ShardingStrategy> create(
        ShardingType type,
        size_t total_samples,
        size_t world_size,
        size_t rank
    );

    /**
     * @brief Iterator for shard indices
     */
    class Iterator {
    public:
        using value_type = size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const size_t*;
        using reference = size_t;
        using iterator_category = std::forward_iterator_tag;

        Iterator(const ShardingStrategy* strategy, size_t pos)
            : strategy_(strategy), pos_(pos) {}

        reference operator*() const { return strategy_->global_index(pos_); }

        Iterator& operator++() { ++pos_; return *this; }
        Iterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }

        bool operator==(const Iterator& other) const { return pos_ == other.pos_; }
        bool operator!=(const Iterator& other) const { return pos_ != other.pos_; }

    private:
        const ShardingStrategy* strategy_;
        size_t pos_;
    };

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end() const { return Iterator(this, size()); }

protected:
    ShardingStrategy(size_t total_samples, size_t world_size, size_t rank)
        : total_samples_(total_samples), world_size_(world_size), rank_(rank) {
        if (world_size == 0) {
            throw std::invalid_argument("world_size must be > 0");
        }
        if (rank >= world_size) {
            throw std::invalid_argument("rank must be < world_size");
        }
    }

    size_t total_samples_;
    size_t world_size_;
    size_t rank_;
};

/**
 * @brief Contiguous sharding - each rank gets a contiguous range
 *
 * For total=100, world_size=4:
 * - Rank 0: [0, 25)
 * - Rank 1: [25, 50)
 * - Rank 2: [50, 75)
 * - Rank 3: [75, 100)
 */
class ContiguousSharding : public ShardingStrategy {
public:
    ContiguousSharding(size_t total_samples, size_t world_size, size_t rank)
        : ShardingStrategy(total_samples, world_size, rank) {
        // Compute start and end for this rank
        size_t base = total_samples / world_size;
        size_t remainder = total_samples % world_size;

        // Distribute remainder across first 'remainder' ranks
        if (rank < remainder) {
            start_ = rank * (base + 1);
            end_ = start_ + base + 1;
        } else {
            start_ = rank * base + remainder;
            end_ = start_ + base;
        }
    }

    size_t size() const override {
        return end_ - start_;
    }

    std::vector<size_t> get_indices() const override {
        std::vector<size_t> indices;
        indices.reserve(size());
        for (size_t i = start_; i < end_; ++i) {
            indices.push_back(i);
        }
        return indices;
    }

    size_t global_index(size_t local_idx) const override {
        if (local_idx >= size()) {
            throw std::out_of_range("local index out of range");
        }
        return start_ + local_idx;
    }

    bool owns_index(size_t global_idx) const override {
        return global_idx >= start_ && global_idx < end_;
    }

    size_t start() const { return start_; }
    size_t end() const { return end_; }

private:
    size_t start_;
    size_t end_;
};

/**
 * @brief Interleaved sharding - each rank gets every N-th sample
 *
 * For total=100, world_size=4:
 * - Rank 0: [0, 4, 8, 12, ...]
 * - Rank 1: [1, 5, 9, 13, ...]
 * - Rank 2: [2, 6, 10, 14, ...]
 * - Rank 3: [3, 7, 11, 15, ...]
 *
 * Benefits:
 * - Better load balancing for heterogeneous data (varying sample sizes)
 * - Each rank sees similar data distribution
 */
class InterleavedSharding : public ShardingStrategy {
public:
    InterleavedSharding(size_t total_samples, size_t world_size, size_t rank)
        : ShardingStrategy(total_samples, world_size, rank) {
        // Count how many samples this rank owns
        count_ = total_samples / world_size;
        if (rank < (total_samples % world_size)) {
            count_++;
        }
    }

    size_t size() const override {
        return count_;
    }

    std::vector<size_t> get_indices() const override {
        std::vector<size_t> indices;
        indices.reserve(count_);
        for (size_t i = rank_; i < total_samples_; i += world_size_) {
            indices.push_back(i);
        }
        return indices;
    }

    size_t global_index(size_t local_idx) const override {
        if (local_idx >= count_) {
            throw std::out_of_range("local index out of range");
        }
        return rank_ + local_idx * world_size_;
    }

    bool owns_index(size_t global_idx) const override {
        return global_idx < total_samples_ && (global_idx % world_size_) == rank_;
    }

private:
    size_t count_;
};

/**
 * @brief Hash-based sharding - deterministic assignment via hashing
 *
 * Uses a simple hash function: hash(sample_id) % world_size == rank
 *
 * Benefits:
 * - Deterministic (same assignment for same sample across runs)
 * - Good for sample-level deduplication across workers
 * - Pseudorandom distribution
 */
class HashBasedSharding : public ShardingStrategy {
public:
    HashBasedSharding(size_t total_samples, size_t world_size, size_t rank,
                       uint64_t seed = 0x517cc1b727220a95ULL)
        : ShardingStrategy(total_samples, world_size, rank), seed_(seed) {
        // Pre-compute indices for this rank
        indices_.reserve(total_samples / world_size + 1);
        for (size_t i = 0; i < total_samples; ++i) {
            if (hash_index(i) == rank) {
                indices_.push_back(i);
            }
        }
    }

    size_t size() const override {
        return indices_.size();
    }

    std::vector<size_t> get_indices() const override {
        return indices_;
    }

    size_t global_index(size_t local_idx) const override {
        if (local_idx >= indices_.size()) {
            throw std::out_of_range("local index out of range");
        }
        return indices_[local_idx];
    }

    bool owns_index(size_t global_idx) const override {
        return global_idx < total_samples_ && hash_index(global_idx) == rank_;
    }

    /**
     * @brief Get the hash value for a sample index
     */
    size_t hash_index(size_t idx) const {
        // FNV-1a hash variant
        uint64_t hash = seed_;
        hash ^= idx;
        hash *= 0x100000001b3ULL;
        hash ^= (idx >> 32);
        hash *= 0x100000001b3ULL;
        return hash % world_size_;
    }

private:
    uint64_t seed_;
    std::vector<size_t> indices_;
};

// Factory implementation
inline std::unique_ptr<ShardingStrategy> ShardingStrategy::create(
    ShardingType type,
    size_t total_samples,
    size_t world_size,
    size_t rank
) {
    switch (type) {
        case ShardingType::CONTIGUOUS:
            return std::make_unique<ContiguousSharding>(total_samples, world_size, rank);
        case ShardingType::INTERLEAVED:
            return std::make_unique<InterleavedSharding>(total_samples, world_size, rank);
        case ShardingType::HASH_BASED:
            return std::make_unique<HashBasedSharding>(total_samples, world_size, rank);
        default:
            throw std::invalid_argument("Unknown sharding type");
    }
}

/**
 * @brief Helper to convert ShardingType to string
 */
inline std::string sharding_type_to_string(ShardingType type) {
    switch (type) {
        case ShardingType::CONTIGUOUS: return "CONTIGUOUS";
        case ShardingType::INTERLEAVED: return "INTERLEAVED";
        case ShardingType::HASH_BASED: return "HASH_BASED";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Helper to parse ShardingType from string
 */
inline ShardingType string_to_sharding_type(const std::string& s) {
    if (s == "CONTIGUOUS" || s == "contiguous") return ShardingType::CONTIGUOUS;
    if (s == "INTERLEAVED" || s == "interleaved") return ShardingType::INTERLEAVED;
    if (s == "HASH_BASED" || s == "hash_based") return ShardingType::HASH_BASED;
    throw std::invalid_argument("Unknown sharding type: " + s);
}

/**
 * @brief Sharding coordinator for multi-worker setups
 *
 * Manages sharding across all workers and provides utilities for
 * distributed training coordination.
 */
class ShardingCoordinator {
public:
    ShardingCoordinator(size_t total_samples, size_t world_size,
                        ShardingType type = ShardingType::CONTIGUOUS)
        : total_samples_(total_samples),
          world_size_(world_size),
          type_(type) {
        // Pre-create strategies for all ranks
        strategies_.reserve(world_size);
        for (size_t r = 0; r < world_size; ++r) {
            strategies_.push_back(ShardingStrategy::create(type, total_samples, world_size, r));
        }
    }

    /**
     * @brief Get strategy for specific rank
     */
    const ShardingStrategy& get_strategy(size_t rank) const {
        if (rank >= world_size_) {
            throw std::out_of_range("rank out of range");
        }
        return *strategies_[rank];
    }

    /**
     * @brief Find which rank owns a global index
     */
    size_t owner_rank(size_t global_idx) const {
        for (size_t r = 0; r < world_size_; ++r) {
            if (strategies_[r]->owns_index(global_idx)) {
                return r;
            }
        }
        throw std::out_of_range("global index not owned by any rank");
    }

    /**
     * @brief Verify all samples are covered exactly once
     */
    bool verify_coverage() const {
        std::vector<bool> covered(total_samples_, false);
        for (size_t r = 0; r < world_size_; ++r) {
            for (size_t idx : strategies_[r]->get_indices()) {
                if (idx >= total_samples_ || covered[idx]) {
                    return false;  // Out of range or duplicate
                }
                covered[idx] = true;
            }
        }
        return std::all_of(covered.begin(), covered.end(), [](bool b) { return b; });
    }

    /**
     * @brief Get sizes for all ranks
     */
    std::vector<size_t> get_shard_sizes() const {
        std::vector<size_t> sizes;
        sizes.reserve(world_size_);
        for (const auto& s : strategies_) {
            sizes.push_back(s->size());
        }
        return sizes;
    }

    /**
     * @brief Get maximum size imbalance ratio (max/min)
     */
    double imbalance_ratio() const {
        auto sizes = get_shard_sizes();
        auto [min_it, max_it] = std::minmax_element(sizes.begin(), sizes.end());
        if (*min_it == 0) return std::numeric_limits<double>::infinity();
        return static_cast<double>(*max_it) / *min_it;
    }

    size_t total_samples() const { return total_samples_; }
    size_t world_size() const { return world_size_; }
    ShardingType type() const { return type_; }

private:
    size_t total_samples_;
    size_t world_size_;
    ShardingType type_;
    std::vector<std::unique_ptr<ShardingStrategy>> strategies_;
};

}  // namespace turboloader
