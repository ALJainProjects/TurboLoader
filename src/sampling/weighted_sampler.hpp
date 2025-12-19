/**
 * @file weighted_sampler.hpp
 * @brief Weighted sampling for imbalanced datasets (v2.21.0)
 *
 * Provides sampling strategies for handling class imbalance:
 * - Per-sample weighted sampling
 * - Class-balanced sampling
 * - Oversampling minority classes
 * - Undersampling majority classes
 *
 * Usage:
 * ```cpp
 * // Per-sample weights
 * WeightedRandomSampler sampler(weights, num_samples);
 * auto indices = sampler.sample();
 *
 * // Class-balanced sampling
 * ClassBalancedSampler balanced(labels);
 * auto indices = balanced.sample(batch_size);
 * ```
 */

#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <cmath>

namespace turboloader {

/**
 * @brief Weighted random sampler with per-sample weights
 *
 * Samples indices according to provided weights using discrete distribution.
 */
class WeightedRandomSampler {
public:
    /**
     * @brief Construct with sample weights
     * @param weights Per-sample weights (must be non-negative)
     * @param num_samples Number of samples to draw (0 = same as weights.size())
     * @param replacement Whether to sample with replacement (default: true)
     * @param seed Random seed for reproducibility
     */
    WeightedRandomSampler(const std::vector<double>& weights,
                          size_t num_samples = 0,
                          bool replacement = true,
                          uint64_t seed = std::random_device{}())
        : weights_(weights),
          num_samples_(num_samples > 0 ? num_samples : weights.size()),
          replacement_(replacement),
          rng_(seed) {

        if (weights.empty()) {
            throw std::invalid_argument("Weights cannot be empty");
        }

        // Validate all weights are non-negative
        for (double w : weights) {
            if (w < 0) {
                throw std::invalid_argument("Weights must be non-negative");
            }
        }

        // Check if total weight is positive
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (total <= 0) {
            throw std::invalid_argument("Total weight must be positive");
        }

        // Create distribution
        distribution_ = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

    /**
     * @brief Sample indices according to weights
     * @return Vector of sampled indices
     */
    std::vector<size_t> sample() {
        std::vector<size_t> indices;
        indices.reserve(num_samples_);

        if (replacement_) {
            for (size_t i = 0; i < num_samples_; ++i) {
                indices.push_back(distribution_(rng_));
            }
        } else {
            // Without replacement - use reservoir-like approach
            // Create normalized CDF and use weighted shuffle
            std::vector<std::pair<double, size_t>> weighted_indices;
            weighted_indices.reserve(weights_.size());

            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            for (size_t i = 0; i < weights_.size(); ++i) {
                if (weights_[i] > 0) {
                    // Use -log(u) / w to get weighted random order
                    double key = -std::log(uniform(rng_)) / weights_[i];
                    weighted_indices.emplace_back(key, i);
                }
            }

            // Sort by key (ascending) and take first num_samples
            std::partial_sort(weighted_indices.begin(),
                             weighted_indices.begin() + std::min(num_samples_, weighted_indices.size()),
                             weighted_indices.end());

            size_t n = std::min(num_samples_, weighted_indices.size());
            for (size_t i = 0; i < n; ++i) {
                indices.push_back(weighted_indices[i].second);
            }
        }

        return indices;
    }

    /**
     * @brief Sample a single index
     */
    size_t sample_one() {
        return distribution_(rng_);
    }

    /**
     * @brief Reset RNG with new seed
     */
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

    /**
     * @brief Get number of samples to draw
     */
    size_t num_samples() const { return num_samples_; }

    /**
     * @brief Get dataset size
     */
    size_t dataset_size() const { return weights_.size(); }

    /**
     * @brief Check if sampling with replacement
     */
    bool replacement() const { return replacement_; }

    /**
     * @brief Update weights
     */
    void set_weights(const std::vector<double>& weights) {
        if (weights.empty()) {
            throw std::invalid_argument("Weights cannot be empty");
        }
        weights_ = weights;
        distribution_ = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

private:
    std::vector<double> weights_;
    size_t num_samples_;
    bool replacement_;
    std::mt19937_64 rng_;
    std::discrete_distribution<size_t> distribution_;
};

/**
 * @brief Class-balanced sampler for imbalanced datasets
 *
 * Automatically computes weights to balance class frequencies.
 */
class ClassBalancedSampler {
public:
    /**
     * @brief Construct from class labels
     * @param labels Class label for each sample (integers)
     * @param seed Random seed
     */
    explicit ClassBalancedSampler(const std::vector<size_t>& labels,
                                  uint64_t seed = std::random_device{}())
        : labels_(labels), rng_(seed) {

        if (labels.empty()) {
            throw std::invalid_argument("Labels cannot be empty");
        }

        compute_class_weights();
    }

    /**
     * @brief Sample batch_size indices with class balancing
     */
    std::vector<size_t> sample(size_t batch_size) {
        std::vector<size_t> indices;
        indices.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            indices.push_back(distribution_(rng_));
        }

        return indices;
    }

    /**
     * @brief Sample all indices for one epoch (balanced)
     */
    std::vector<size_t> sample_epoch() {
        return sample(labels_.size());
    }

    /**
     * @brief Get computed sample weights
     */
    const std::vector<double>& weights() const { return weights_; }

    /**
     * @brief Get class counts
     */
    const std::unordered_map<size_t, size_t>& class_counts() const {
        return class_counts_;
    }

    /**
     * @brief Get number of classes
     */
    size_t num_classes() const { return class_counts_.size(); }

    /**
     * @brief Get dataset size
     */
    size_t dataset_size() const { return labels_.size(); }

    /**
     * @brief Reset RNG with new seed
     */
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

    /**
     * @brief Get imbalance ratio (max_count / min_count)
     */
    double imbalance_ratio() const {
        if (class_counts_.empty()) return 1.0;

        size_t min_count = std::numeric_limits<size_t>::max();
        size_t max_count = 0;

        for (const auto& [cls, count] : class_counts_) {
            min_count = std::min(min_count, count);
            max_count = std::max(max_count, count);
        }

        return min_count > 0 ? static_cast<double>(max_count) / min_count : 0.0;
    }

private:
    void compute_class_weights() {
        // Count samples per class
        class_counts_.clear();
        for (size_t label : labels_) {
            class_counts_[label]++;
        }

        // Compute weight as inverse of class frequency
        // w_i = N / (C * n_i) where N = total, C = num_classes, n_i = class count
        double total = static_cast<double>(labels_.size());
        double num_classes = static_cast<double>(class_counts_.size());

        weights_.resize(labels_.size());
        for (size_t i = 0; i < labels_.size(); ++i) {
            size_t label = labels_[i];
            size_t count = class_counts_[label];
            weights_[i] = total / (num_classes * count);
        }

        // Create distribution
        distribution_ = std::discrete_distribution<size_t>(weights_.begin(), weights_.end());
    }

    std::vector<size_t> labels_;
    std::vector<double> weights_;
    std::unordered_map<size_t, size_t> class_counts_;
    std::mt19937_64 rng_;
    std::discrete_distribution<size_t> distribution_;
};

/**
 * @brief Oversampler for minority classes
 *
 * Repeats minority class samples to match majority class count.
 */
class OverSampler {
public:
    /**
     * @brief Construct from class labels
     * @param labels Class label for each sample
     */
    explicit OverSampler(const std::vector<size_t>& labels,
                         uint64_t seed = std::random_device{}())
        : labels_(labels), rng_(seed) {

        if (labels.empty()) {
            throw std::invalid_argument("Labels cannot be empty");
        }

        compute_oversampling();
    }

    /**
     * @brief Get indices for one epoch (all samples including repeats)
     */
    std::vector<size_t> sample_epoch() {
        // Shuffle the oversampled indices
        std::vector<size_t> result = oversampled_indices_;
        std::shuffle(result.begin(), result.end(), rng_);
        return result;
    }

    /**
     * @brief Get total samples after oversampling
     */
    size_t total_samples() const { return oversampled_indices_.size(); }

    /**
     * @brief Get original dataset size
     */
    size_t original_size() const { return labels_.size(); }

    /**
     * @brief Reset RNG with new seed
     */
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

private:
    void compute_oversampling() {
        // Count per class and find max
        std::unordered_map<size_t, std::vector<size_t>> class_indices;
        for (size_t i = 0; i < labels_.size(); ++i) {
            class_indices[labels_[i]].push_back(i);
        }

        size_t max_count = 0;
        for (const auto& [cls, indices] : class_indices) {
            max_count = std::max(max_count, indices.size());
        }

        // Create oversampled indices - repeat minority classes
        oversampled_indices_.clear();
        oversampled_indices_.reserve(max_count * class_indices.size());

        for (const auto& [cls, indices] : class_indices) {
            // Add all original indices
            for (size_t idx : indices) {
                oversampled_indices_.push_back(idx);
            }

            // Repeat to match max_count
            size_t repeats_needed = max_count - indices.size();
            for (size_t i = 0; i < repeats_needed; ++i) {
                oversampled_indices_.push_back(indices[i % indices.size()]);
            }
        }
    }

    std::vector<size_t> labels_;
    std::vector<size_t> oversampled_indices_;
    std::mt19937_64 rng_;
};

/**
 * @brief Undersampler for majority classes
 *
 * Randomly samples from majority classes to match minority class count.
 */
class UnderSampler {
public:
    /**
     * @brief Construct from class labels
     * @param labels Class label for each sample
     */
    explicit UnderSampler(const std::vector<size_t>& labels,
                          uint64_t seed = std::random_device{}())
        : labels_(labels), rng_(seed) {

        if (labels.empty()) {
            throw std::invalid_argument("Labels cannot be empty");
        }

        // Count per class and find min
        for (size_t i = 0; i < labels_.size(); ++i) {
            class_indices_[labels_[i]].push_back(i);
        }

        min_count_ = std::numeric_limits<size_t>::max();
        for (const auto& [cls, indices] : class_indices_) {
            min_count_ = std::min(min_count_, indices.size());
        }
    }

    /**
     * @brief Get indices for one epoch (undersampled)
     */
    std::vector<size_t> sample_epoch() {
        std::vector<size_t> result;
        result.reserve(min_count_ * class_indices_.size());

        for (auto& [cls, indices] : class_indices_) {
            // Shuffle and take first min_count_
            std::shuffle(indices.begin(), indices.end(), rng_);
            result.insert(result.end(), indices.begin(), indices.begin() + min_count_);
        }

        // Shuffle final result
        std::shuffle(result.begin(), result.end(), rng_);
        return result;
    }

    /**
     * @brief Get total samples after undersampling
     */
    size_t total_samples() const {
        return min_count_ * class_indices_.size();
    }

    /**
     * @brief Get original dataset size
     */
    size_t original_size() const { return labels_.size(); }

    /**
     * @brief Reset RNG with new seed
     */
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

private:
    std::vector<size_t> labels_;
    std::unordered_map<size_t, std::vector<size_t>> class_indices_;
    size_t min_count_;
    std::mt19937_64 rng_;
};

/**
 * @brief Importance sampler with adjustable temperature
 *
 * Sharpens or softens the weight distribution using temperature parameter.
 */
class ImportanceSampler {
public:
    /**
     * @brief Construct with sample weights and temperature
     * @param weights Per-sample importance weights
     * @param temperature Temperature (>1 softens, <1 sharpens, 1 = unchanged)
     * @param seed Random seed
     */
    ImportanceSampler(const std::vector<double>& weights,
                      double temperature = 1.0,
                      uint64_t seed = std::random_device{}())
        : original_weights_(weights),
          temperature_(temperature),
          rng_(seed) {

        if (weights.empty()) {
            throw std::invalid_argument("Weights cannot be empty");
        }

        update_distribution();
    }

    /**
     * @brief Sample batch_size indices
     */
    std::vector<size_t> sample(size_t batch_size) {
        std::vector<size_t> indices;
        indices.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            indices.push_back(distribution_(rng_));
        }

        return indices;
    }

    /**
     * @brief Set temperature (updates distribution)
     */
    void set_temperature(double temperature) {
        temperature_ = temperature;
        update_distribution();
    }

    /**
     * @brief Get current temperature
     */
    double temperature() const { return temperature_; }

    /**
     * @brief Get adjusted weights
     */
    const std::vector<double>& adjusted_weights() const { return adjusted_weights_; }

    /**
     * @brief Reset RNG with new seed
     */
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

private:
    void update_distribution() {
        adjusted_weights_.resize(original_weights_.size());

        if (temperature_ == 1.0) {
            adjusted_weights_ = original_weights_;
        } else {
            // Apply temperature: w_adjusted = w^(1/T)
            double inv_temp = 1.0 / temperature_;
            for (size_t i = 0; i < original_weights_.size(); ++i) {
                adjusted_weights_[i] = std::pow(original_weights_[i], inv_temp);
            }
        }

        distribution_ = std::discrete_distribution<size_t>(
            adjusted_weights_.begin(), adjusted_weights_.end());
    }

    std::vector<double> original_weights_;
    std::vector<double> adjusted_weights_;
    double temperature_;
    std::mt19937_64 rng_;
    std::discrete_distribution<size_t> distribution_;
};

/**
 * @brief Stratified sampler ensuring proportional class representation
 *
 * Maintains original class proportions in each batch/epoch.
 */
class StratifiedSampler {
public:
    /**
     * @brief Construct from class labels
     * @param labels Class label for each sample
     * @param seed Random seed
     */
    explicit StratifiedSampler(const std::vector<size_t>& labels,
                               uint64_t seed = std::random_device{}())
        : labels_(labels), rng_(seed) {

        if (labels.empty()) {
            throw std::invalid_argument("Labels cannot be empty");
        }

        // Group indices by class
        for (size_t i = 0; i < labels_.size(); ++i) {
            class_indices_[labels_[i]].push_back(i);
        }
    }

    /**
     * @brief Get stratified indices for one epoch
     *
     * Interleaves samples from each class to ensure even distribution.
     */
    std::vector<size_t> sample_epoch() {
        // Shuffle each class's indices
        std::vector<std::vector<size_t>> shuffled_classes;
        for (auto& [cls, indices] : class_indices_) {
            auto shuffled = indices;
            std::shuffle(shuffled.begin(), shuffled.end(), rng_);
            shuffled_classes.push_back(std::move(shuffled));
        }

        // Interleave using round-robin
        std::vector<size_t> result;
        result.reserve(labels_.size());

        std::vector<size_t> positions(shuffled_classes.size(), 0);
        size_t remaining = labels_.size();

        while (remaining > 0) {
            for (size_t c = 0; c < shuffled_classes.size() && remaining > 0; ++c) {
                if (positions[c] < shuffled_classes[c].size()) {
                    result.push_back(shuffled_classes[c][positions[c]++]);
                    --remaining;
                }
            }
        }

        return result;
    }

    /**
     * @brief Get stratified batch of given size
     */
    std::vector<size_t> sample_batch(size_t batch_size) {
        auto epoch = sample_epoch();
        if (epoch.size() <= batch_size) {
            return epoch;
        }
        epoch.resize(batch_size);
        return epoch;
    }

    /**
     * @brief Get number of classes
     */
    size_t num_classes() const { return class_indices_.size(); }

    /**
     * @brief Get dataset size
     */
    size_t dataset_size() const { return labels_.size(); }

    /**
     * @brief Reset RNG with new seed
     */
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }

private:
    std::vector<size_t> labels_;
    std::unordered_map<size_t, std::vector<size_t>> class_indices_;
    std::mt19937_64 rng_;
};

}  // namespace turboloader
