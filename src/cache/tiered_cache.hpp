/**
 * @file tiered_cache.hpp
 * @brief Tiered L1 (memory) + L2 (disk) cache facade
 */

#pragma once

#include "cache_key.hpp"
#include "lru_cache.hpp"
#include "disk_cache.hpp"
#include <memory>
#include <string>

namespace turboloader {
namespace cache {

/**
 * @brief Size trait specialization for CachedImageData
 */
template<>
struct CacheSizeTrait<CachedImageData> {
    static size_t size(const CachedImageData& value) {
        return value.size_bytes() + sizeof(CachedImageData);
    }
};

/**
 * @brief Combined L1 + L2 cache statistics
 */
struct TieredCacheStats {
    CacheStats l1;
    CacheStats l2;

    uint64_t total_hits() const { return l1.hits + l2.hits; }
    uint64_t total_misses() const { return l2.misses; }  // L2 miss = total miss

    double overall_hit_rate() const {
        uint64_t total = total_hits() + total_misses();
        return total > 0 ? static_cast<double>(total_hits()) / total : 0.0;
    }

    double l1_hit_rate() const { return l1.hit_rate(); }
    double l2_hit_rate() const {
        // L2 hits are from L1 misses that found data in L2
        uint64_t l1_misses = l1.misses;
        return l1_misses > 0 ? static_cast<double>(l2.hits) / l1_misses : 0.0;
    }
};

/**
 * @brief Tiered cache with L1 memory + L2 disk
 *
 * Implements cache-aside pattern:
 * 1. Check L1 (memory) -> return if hit
 * 2. Check L2 (disk) -> promote to L1 if hit, return
 * 3. Miss -> caller decodes, then calls put() to cache
 *
 * Features:
 * - L1: Fast memory-based LRU cache with shared_mutex
 * - L2: Disk-based persistent cache with async writes
 * - Automatic L2 -> L1 promotion on hits
 * - Configurable sizes for both tiers
 */
class TieredCache {
public:
    /**
     * @brief Construct tiered cache
     * @param l1_mb L1 memory cache size in MB (default 512)
     * @param l2_gb L2 disk cache size in GB (0 = disabled, default 0)
     * @param cache_dir L2 cache directory (default /tmp/turboloader_cache)
     */
    TieredCache(size_t l1_mb = 512,
                size_t l2_gb = 0,
                const std::string& cache_dir = "/tmp/turboloader_cache")
        : l1_enabled_(l1_mb > 0),
          l2_enabled_(l2_gb > 0) {

        if (l1_enabled_) {
            l1_ = std::make_unique<LRUCache<CacheKey, CachedImageData>>(l1_mb);
        }

        if (l2_enabled_) {
            l2_ = std::make_unique<DiskCache>(cache_dir, l2_gb);
        }
    }

    ~TieredCache() {
        if (l2_) {
            l2_->flush();
        }
    }

    // Non-copyable
    TieredCache(const TieredCache&) = delete;
    TieredCache& operator=(const TieredCache&) = delete;

    // Movable
    TieredCache(TieredCache&&) = default;
    TieredCache& operator=(TieredCache&&) = default;

    /**
     * @brief Get cached data
     *
     * Lookup order: L1 -> L2 -> miss
     * On L2 hit, data is promoted to L1.
     *
     * @param key Cache key
     * @return Shared pointer to cached data, or nullptr if not found
     */
    std::shared_ptr<CachedImageData> get(const CacheKey& key) {
        // Check L1 first
        if (l1_enabled_) {
            auto result = l1_->get(key);
            if (result) {
                return result;
            }
        }

        // Check L2
        if (l2_enabled_) {
            auto result = l2_->get(key);
            if (result) {
                // Promote to L1
                if (l1_enabled_) {
                    l1_->put(key, *result);
                }
                return *result;
            }
        }

        return nullptr;
    }

    /**
     * @brief Put data in cache
     *
     * Data is written to L1 immediately and queued for async L2 write.
     *
     * @param key Cache key
     * @param data Image data to cache
     */
    void put(const CacheKey& key, std::shared_ptr<CachedImageData> data) {
        if (!data) return;

        // Write to L1
        if (l1_enabled_) {
            l1_->put(key, data);
        }

        // Queue for L2 write
        if (l2_enabled_) {
            l2_->put_async(key, data);
        }
    }

    /**
     * @brief Create CachedImageData from raw image buffer
     * @param data Raw pixel data
     * @param size Data size in bytes
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param stride Row stride in bytes
     * @return Shared pointer to cached image data
     */
    static std::shared_ptr<CachedImageData> make_cached_data(
            const uint8_t* data, size_t size,
            uint32_t width, uint32_t height,
            uint32_t channels, uint32_t stride = 0) {

        auto cached = std::make_shared<CachedImageData>();
        cached->data.assign(data, data + size);
        cached->width = width;
        cached->height = height;
        cached->channels = channels;
        cached->stride = stride > 0 ? stride : width * channels;
        return cached;
    }

    /**
     * @brief Check if key exists in any tier
     */
    bool contains(const CacheKey& key) const {
        if (l1_enabled_ && l1_->contains(key)) return true;
        if (l2_enabled_ && l2_->contains(key)) return true;
        return false;
    }

    /**
     * @brief Clear all caches
     */
    void clear() {
        if (l1_enabled_) l1_->clear();
        if (l2_enabled_) l2_->clear();
    }

    /**
     * @brief Flush pending L2 writes
     */
    void flush() {
        if (l2_enabled_) l2_->flush();
    }

    /**
     * @brief Get combined statistics
     */
    TieredCacheStats stats() const {
        TieredCacheStats s;
        if (l1_enabled_) s.l1 = l1_->stats();
        if (l2_enabled_) s.l2 = l2_->stats();
        return s;
    }

    /**
     * @brief Check if L1 is enabled
     */
    bool l1_enabled() const { return l1_enabled_; }

    /**
     * @brief Check if L2 is enabled
     */
    bool l2_enabled() const { return l2_enabled_; }

    /**
     * @brief Get L1 memory usage in bytes
     */
    size_t l1_memory_usage() const {
        return l1_enabled_ ? l1_->memory_usage() : 0;
    }

    /**
     * @brief Reset statistics
     */
    void reset_stats() {
        if (l1_enabled_) l1_->reset_stats();
        // L2 stats reset would require additional method
    }

private:
    bool l1_enabled_;
    bool l2_enabled_;
    std::unique_ptr<LRUCache<CacheKey, CachedImageData>> l1_;
    std::unique_ptr<DiskCache> l2_;
};

} // namespace cache
} // namespace turboloader
