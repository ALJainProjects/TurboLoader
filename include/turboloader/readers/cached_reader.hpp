#pragma once

#include "turboloader/readers/storage_reader.hpp"
#include <memory>
#include <string>
#include <filesystem>

namespace turboloader {

/**
 * Caching layer for cloud storage readers
 *
 * Features:
 * - LRU eviction policy
 * - Configurable cache size
 * - Thread-safe operations
 * - Prefetching support
 * - Cache warming
 */
class CachedStorageReader : public StorageReader {
public:
    struct Config {
        // Maximum cache size in bytes (default: 10GB)
        size_t max_cache_size{10ULL * 1024 * 1024 * 1024};

        // Cache directory
        std::string cache_dir{"/tmp/turboloader_cache"};

        // Enable prefetching
        bool enable_prefetch{true};

        // Number of files to prefetch ahead
        size_t prefetch_count{2};

        // Clean cache on startup
        bool clean_on_startup{false};
    };

    /**
     * Construct cached reader wrapping another storage reader
     */
    CachedStorageReader(std::unique_ptr<StorageReader> backend, const Config& config);
    ~CachedStorageReader() override;

    // Non-copyable but movable
    CachedStorageReader(const CachedStorageReader&) = delete;
    CachedStorageReader& operator=(const CachedStorageReader&) = delete;
    CachedStorageReader(CachedStorageReader&&) noexcept;
    CachedStorageReader& operator=(CachedStorageReader&&) noexcept;

    /**
     * Read file (from cache if available, otherwise download and cache)
     */
    std::vector<uint8_t> read(const std::string& path) override;

    /**
     * Read byte range (not cached, forwarded to backend)
     */
    std::vector<uint8_t> read_range(const std::string& path,
                                     size_t offset,
                                     size_t length) override;

    /**
     * Check if file exists (check cache first, then backend)
     */
    bool exists(const std::string& path) override;

    /**
     * Get file size (from cache metadata or backend)
     */
    size_t size(const std::string& path) override;

    /**
     * Get reader type
     */
    std::string type() const override { return "cached"; }

    /**
     * Prefetch files asynchronously
     */
    void prefetch(const std::vector<std::string>& paths);

    /**
     * Clear cache
     */
    void clear_cache();

    /**
     * Get cache statistics
     */
    struct CacheStats {
        size_t hits{0};
        size_t misses{0};
        size_t evictions{0};
        size_t current_size{0};
        size_t num_entries{0};

        double hit_rate() const {
            size_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

    CacheStats get_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace turboloader
