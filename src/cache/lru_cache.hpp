/**
 * @file lru_cache.hpp
 * @brief Thread-safe LRU cache with memory-based eviction
 */

#pragma once

#include "cache_key.hpp"
#include <list>
#include <unordered_map>
#include <shared_mutex>
#include <memory>
#include <atomic>
#include <functional>

namespace turboloader {
namespace cache {

/**
 * @brief Trait to get size of cached values
 *
 * Specialize this for custom value types to enable memory-based eviction.
 */
template<typename T>
struct CacheSizeTrait {
    static size_t size(const T& value) {
        return sizeof(T);
    }
};

/**
 * @brief Thread-safe LRU cache with memory-based eviction
 *
 * Features:
 * - Shared mutex for concurrent reads (multiple workers)
 * - Memory-based eviction (not count-based)
 * - Returns shared_ptr to avoid copies
 * - Thread-safe with minimal contention
 *
 * @tparam Key Key type (must be hashable)
 * @tparam Value Value type
 */
template<typename Key, typename Value>
class LRUCache {
public:
    using ValuePtr = std::shared_ptr<Value>;
    using SizeFunc = std::function<size_t(const Value&)>;

private:
    struct CacheEntry {
        Key key;
        ValuePtr value;
        size_t size_bytes;
    };

    using ListType = std::list<CacheEntry>;
    using ListIterator = typename ListType::iterator;
    using MapType = std::unordered_map<Key, ListIterator>;

    ListType items_;
    MapType lookup_;
    mutable std::shared_mutex mutex_;

    size_t max_memory_bytes_;
    std::atomic<size_t> current_memory_bytes_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};

    SizeFunc size_func_;

public:
    /**
     * @brief Construct LRU cache with memory limit
     * @param max_memory_mb Maximum memory in megabytes
     * @param size_func Optional function to compute value size
     */
    explicit LRUCache(size_t max_memory_mb,
                      SizeFunc size_func = nullptr)
        : max_memory_bytes_(max_memory_mb * 1024 * 1024),
          size_func_(size_func ? size_func :
                     [](const Value& v) { return CacheSizeTrait<Value>::size(v); }) {}

    /**
     * @brief Get value from cache
     * @param key Cache key
     * @return Shared pointer to value, or nullptr if not found
     */
    ValuePtr get(const Key& key) {
        // Try read lock first
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto it = lookup_.find(key);
            if (it == lookup_.end()) {
                misses_++;
                return nullptr;
            }
        }

        // Upgrade to write lock to move to front
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = lookup_.find(key);
        if (it == lookup_.end()) {
            misses_++;
            return nullptr;
        }

        // Move to front (most recently used)
        items_.splice(items_.begin(), items_, it->second);
        hits_++;
        return it->second->value;
    }

    /**
     * @brief Put value in cache
     * @param key Cache key
     * @param value Shared pointer to value
     */
    void put(const Key& key, ValuePtr value) {
        if (!value) return;

        size_t value_size = size_func_(*value);

        std::unique_lock<std::shared_mutex> lock(mutex_);

        // Check if key already exists
        auto it = lookup_.find(key);
        if (it != lookup_.end()) {
            // Update existing entry
            current_memory_bytes_ -= it->second->size_bytes;
            it->second->value = value;
            it->second->size_bytes = value_size;
            current_memory_bytes_ += value_size;
            items_.splice(items_.begin(), items_, it->second);
            return;
        }

        // Evict if necessary
        while (current_memory_bytes_ + value_size > max_memory_bytes_ &&
               !items_.empty()) {
            evict_lru();
        }

        // Insert new entry at front
        items_.push_front({key, value, value_size});
        lookup_[key] = items_.begin();
        current_memory_bytes_ += value_size;
    }

    /**
     * @brief Check if key exists in cache
     * @param key Cache key
     * @return True if key exists
     */
    bool contains(const Key& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return lookup_.find(key) != lookup_.end();
    }

    /**
     * @brief Remove key from cache
     * @param key Cache key
     * @return True if key was removed
     */
    bool remove(const Key& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = lookup_.find(key);
        if (it == lookup_.end()) {
            return false;
        }

        current_memory_bytes_ -= it->second->size_bytes;
        items_.erase(it->second);
        lookup_.erase(it);
        return true;
    }

    /**
     * @brief Clear all entries from cache
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        items_.clear();
        lookup_.clear();
        current_memory_bytes_ = 0;
    }

    /**
     * @brief Get cache statistics
     */
    CacheStats stats() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        CacheStats s;
        s.hits = hits_;
        s.misses = misses_;
        s.evictions = evictions_;
        s.current_size_bytes = current_memory_bytes_;
        s.max_size_bytes = max_memory_bytes_;
        s.item_count = items_.size();
        return s;
    }

    /**
     * @brief Get current memory usage in bytes
     */
    size_t memory_usage() const {
        return current_memory_bytes_;
    }

    /**
     * @brief Get number of items in cache
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return items_.size();
    }

    /**
     * @brief Check if cache is empty
     */
    bool empty() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return items_.empty();
    }

    /**
     * @brief Get maximum memory in bytes
     */
    size_t max_memory() const {
        return max_memory_bytes_;
    }

    /**
     * @brief Reset statistics
     */
    void reset_stats() {
        hits_ = 0;
        misses_ = 0;
        evictions_ = 0;
    }

private:
    /**
     * @brief Evict least recently used entry
     * @note Must be called with write lock held
     */
    void evict_lru() {
        if (items_.empty()) return;

        auto& lru = items_.back();
        current_memory_bytes_ -= lru.size_bytes;
        lookup_.erase(lru.key);
        items_.pop_back();
        evictions_++;
    }
};

} // namespace cache
} // namespace turboloader
