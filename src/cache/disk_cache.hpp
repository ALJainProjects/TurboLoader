/**
 * @file disk_cache.hpp
 * @brief Disk-based L2 cache with async writes
 */

#pragma once

#include "cache_key.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <vector>
#include <cstring>
#include <map>

namespace turboloader {
namespace cache {

/**
 * @brief Cache file header for disk storage
 */
struct CacheFileHeader {
    static constexpr uint32_t MAGIC = 0x54424C43;  // "TBLC" - TurboLoader Cache
    static constexpr uint32_t VERSION = 1;

    uint32_t magic = MAGIC;
    uint32_t version = VERSION;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    uint32_t stride = 0;
    uint64_t data_size = 0;
    uint64_t content_hash = 0;
    uint32_t checksum = 0;
    uint32_t reserved = 0;

    bool is_valid() const {
        return magic == MAGIC && version == VERSION;
    }

    static uint32_t compute_checksum(const void* data, size_t size) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint32_t sum = 0;
        for (size_t i = 0; i < size; i++) {
            sum = ((sum << 5) + sum) + p[i];  // djb2-like
        }
        return sum;
    }
};

/**
 * @brief Cached image data structure
 */
struct CachedImageData {
    std::vector<uint8_t> data;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    uint32_t stride = 0;

    size_t size_bytes() const {
        return data.size();
    }
};

/**
 * @brief Write request for async disk writes
 */
struct CacheWriteRequest {
    CacheKey key;
    std::shared_ptr<CachedImageData> data;
};

/**
 * @brief Disk-based L2 cache with async writes
 *
 * Features:
 * - Async writes via background thread (non-blocking)
 * - LZ4 compression option (future enhancement)
 * - LRU eviction when size limit reached
 * - File naming based on content hash
 */
class DiskCache {
public:
    /**
     * @brief Construct disk cache
     * @param cache_dir Directory for cache files
     * @param max_size_gb Maximum cache size in gigabytes (0 = unlimited)
     */
    DiskCache(const std::string& cache_dir, size_t max_size_gb = 10)
        : cache_dir_(cache_dir),
          max_size_bytes_(max_size_gb * 1024ULL * 1024ULL * 1024ULL),
          running_(true) {

        // Create cache directory if it doesn't exist
        std::filesystem::create_directories(cache_dir_);

        // Scan existing cache files
        scan_cache_directory();

        // Start background writer thread
        writer_thread_ = std::thread(&DiskCache::writer_loop, this);
    }

    ~DiskCache() {
        shutdown();
    }

    // Non-copyable
    DiskCache(const DiskCache&) = delete;
    DiskCache& operator=(const DiskCache&) = delete;

    /**
     * @brief Get cached data
     * @param key Cache key
     * @return Cached data or nullopt if not found
     */
    std::optional<std::shared_ptr<CachedImageData>> get(const CacheKey& key) {
        std::string filepath = get_filepath(key);

        if (!std::filesystem::exists(filepath)) {
            misses_++;
            return std::nullopt;
        }

        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            misses_++;
            return std::nullopt;
        }

        // Read header
        CacheFileHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file || !header.is_valid()) {
            misses_++;
            return std::nullopt;
        }

        // Read data
        auto data = std::make_shared<CachedImageData>();
        data->width = header.width;
        data->height = header.height;
        data->channels = header.channels;
        data->stride = header.stride;
        data->data.resize(header.data_size);

        file.read(reinterpret_cast<char*>(data->data.data()), header.data_size);
        if (!file) {
            misses_++;
            return std::nullopt;
        }

        // Verify checksum
        uint32_t checksum = CacheFileHeader::compute_checksum(
            data->data.data(), data->data.size());
        if (checksum != header.checksum) {
            // Corrupted file, remove it
            std::filesystem::remove(filepath);
            misses_++;
            return std::nullopt;
        }

        // Update access time for LRU
        update_access_time(filepath);

        hits_++;
        return data;
    }

    /**
     * @brief Queue data for async write
     * @param key Cache key
     * @param data Image data to cache
     */
    void put_async(const CacheKey& key, std::shared_ptr<CachedImageData> data) {
        if (!data) return;

        CacheWriteRequest request{key, data};

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            write_queue_.push(std::move(request));
        }
        queue_cv_.notify_one();
    }

    /**
     * @brief Synchronously write data to cache
     * @param key Cache key
     * @param data Image data to cache
     */
    void put(const CacheKey& key, const CachedImageData& data) {
        std::string filepath = get_filepath(key);

        // Evict if necessary
        evict_if_needed(data.size_bytes());

        // Write file
        std::ofstream file(filepath, std::ios::binary);
        if (!file) return;

        CacheFileHeader header;
        header.width = data.width;
        header.height = data.height;
        header.channels = data.channels;
        header.stride = data.stride;
        header.data_size = data.data.size();
        header.content_hash = key.content_hash;
        header.checksum = CacheFileHeader::compute_checksum(
            data.data.data(), data.data.size());

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.write(reinterpret_cast<const char*>(data.data.data()), data.data.size());

        if (file) {
            std::lock_guard<std::mutex> lock(files_mutex_);
            current_size_bytes_ += sizeof(header) + data.data.size();
            cache_files_[filepath] = std::filesystem::last_write_time(filepath);
        }
    }

    /**
     * @brief Flush pending writes
     */
    void flush() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        flush_cv_.wait(lock, [this] { return write_queue_.empty(); });
    }

    /**
     * @brief Shutdown the cache
     */
    void shutdown() {
        if (!running_) return;

        running_ = false;
        queue_cv_.notify_all();

        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
    }

    /**
     * @brief Clear all cache files
     */
    void clear() {
        shutdown();

        std::lock_guard<std::mutex> lock(files_mutex_);
        for (const auto& [filepath, _] : cache_files_) {
            std::filesystem::remove(filepath);
        }
        cache_files_.clear();
        current_size_bytes_ = 0;
    }

    /**
     * @brief Check if key exists in cache
     */
    bool contains(const CacheKey& key) const {
        return std::filesystem::exists(get_filepath(key));
    }

    /**
     * @brief Get cache statistics
     */
    CacheStats stats() const {
        std::lock_guard<std::mutex> lock(files_mutex_);
        CacheStats s;
        s.hits = hits_;
        s.misses = misses_;
        s.evictions = evictions_;
        s.current_size_bytes = current_size_bytes_;
        s.max_size_bytes = max_size_bytes_;
        s.item_count = cache_files_.size();
        return s;
    }

    /**
     * @brief Get cache directory
     */
    const std::filesystem::path& cache_dir() const {
        return cache_dir_;
    }

private:
    std::filesystem::path cache_dir_;
    size_t max_size_bytes_;
    std::atomic<bool> running_;

    // Background writer
    std::thread writer_thread_;
    std::queue<CacheWriteRequest> write_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable flush_cv_;

    // Cache file tracking
    mutable std::mutex files_mutex_;
    std::map<std::string, std::filesystem::file_time_type> cache_files_;
    size_t current_size_bytes_ = 0;

    // Stats
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};

    std::string get_filepath(const CacheKey& key) const {
        return (cache_dir_ / key.to_filename()).string();
    }

    void scan_cache_directory() {
        std::lock_guard<std::mutex> lock(files_mutex_);
        cache_files_.clear();
        current_size_bytes_ = 0;

        if (!std::filesystem::exists(cache_dir_)) return;

        for (const auto& entry : std::filesystem::directory_iterator(cache_dir_)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".cache") {
                cache_files_[entry.path().string()] = entry.last_write_time();
                current_size_bytes_ += entry.file_size();
            }
        }
    }

    void update_access_time(const std::string& filepath) {
        try {
            std::filesystem::last_write_time(filepath,
                std::filesystem::file_time_type::clock::now());

            std::lock_guard<std::mutex> lock(files_mutex_);
            cache_files_[filepath] = std::filesystem::file_time_type::clock::now();
        } catch (...) {
            // Ignore errors
        }
    }

    void evict_if_needed(size_t new_size) {
        if (max_size_bytes_ == 0) return;

        std::lock_guard<std::mutex> lock(files_mutex_);

        while (current_size_bytes_ + new_size > max_size_bytes_ &&
               !cache_files_.empty()) {
            // Find oldest file (LRU)
            auto oldest = cache_files_.begin();
            for (auto it = cache_files_.begin(); it != cache_files_.end(); ++it) {
                if (it->second < oldest->second) {
                    oldest = it;
                }
            }

            // Remove oldest file
            try {
                auto file_size = std::filesystem::file_size(oldest->first);
                std::filesystem::remove(oldest->first);
                current_size_bytes_ -= file_size;
                evictions_++;
            } catch (...) {
                // Ignore errors
            }

            cache_files_.erase(oldest);
        }
    }

    void writer_loop() {
        while (running_) {
            CacheWriteRequest request;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !write_queue_.empty() || !running_;
                });

                if (!running_ && write_queue_.empty()) break;

                if (!write_queue_.empty()) {
                    request = std::move(write_queue_.front());
                    write_queue_.pop();
                } else {
                    continue;
                }
            }

            // Write to disk
            if (request.data) {
                put(request.key, *request.data);
            }

            // Notify flush waiters if queue is empty
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (write_queue_.empty()) {
                    flush_cv_.notify_all();
                }
            }
        }
    }
};

} // namespace cache
} // namespace turboloader
