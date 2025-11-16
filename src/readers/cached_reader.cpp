#include "turboloader/readers/cached_reader.hpp"
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <list>
#include <thread>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

namespace turboloader {

// LRU cache entry
struct CacheEntry {
    std::string path;
    std::string cache_file;
    size_t file_size;
    std::chrono::system_clock::time_point last_access;
};

struct CachedStorageReader::Impl {
    Config config_;
    std::unique_ptr<StorageReader> backend_;

    // LRU cache
    std::mutex cache_mutex_;
    std::unordered_map<std::string, std::list<CacheEntry>::iterator> cache_map_;
    std::list<CacheEntry> lru_list_;  // Most recently used at front
    size_t current_cache_size_{0};

    // Statistics
    mutable std::mutex stats_mutex_;
    CacheStats stats_;

    // Prefetch queue
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::queue<std::string> prefetch_queue_;
    std::thread prefetch_thread_;
    bool prefetch_running_{false};

    Impl(std::unique_ptr<StorageReader> backend, const Config& config)
        : config_(config)
        , backend_(std::move(backend)) {

        // Create cache directory
        fs::create_directories(config_.cache_dir);

        // Clean cache if requested
        if (config_.clean_on_startup) {
            fs::remove_all(config_.cache_dir);
            fs::create_directories(config_.cache_dir);
        }

        // Start prefetch thread
        if (config_.enable_prefetch) {
            prefetch_running_ = true;
            prefetch_thread_ = std::thread([this]() { prefetch_worker(); });
        }
    }

    ~Impl() {
        // Stop prefetch thread
        if (config_.enable_prefetch) {
            {
                std::lock_guard<std::mutex> lock(prefetch_mutex_);
                prefetch_running_ = false;
                prefetch_cv_.notify_one();
            }
            if (prefetch_thread_.joinable()) {
                prefetch_thread_.join();
            }
        }
    }

    std::string get_cache_path(const std::string& path) {
        // Hash path to create cache filename
        std::hash<std::string> hasher;
        size_t hash = hasher(path);
        return config_.cache_dir + "/cache_" + std::to_string(hash);
    }

    bool is_in_cache(const std::string& path) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cache_map_.find(path) != cache_map_.end();
    }

    std::vector<uint8_t> read_from_cache(const std::string& path) {
        std::unique_lock<std::mutex> lock(cache_mutex_);

        auto it = cache_map_.find(path);
        if (it == cache_map_.end()) {
            return {};  // Not in cache
        }

        // Move to front (most recently used)
        auto list_it = it->second;
        lru_list_.splice(lru_list_.begin(), lru_list_, list_it);
        list_it->last_access = std::chrono::system_clock::now();

        std::string cache_file = list_it->cache_file;

        lock.unlock();

        // Read from cache file
        std::ifstream file(cache_file, std::ios::binary | std::ios::ate);
        if (!file) {
            return {};
        }

        size_t file_size = file.tellg();
        file.seekg(0);

        std::vector<uint8_t> data(file_size);
        file.read(reinterpret_cast<char*>(data.data()), file_size);

        // Update stats
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.hits++;
        }

        return data;
    }

    void write_to_cache(const std::string& path, const std::vector<uint8_t>& data) {
        std::string cache_file = get_cache_path(path);

        // Write data to cache file
        std::ofstream file(cache_file, std::ios::binary);
        if (!file) {
            return;  // Cache write failed, not fatal
        }

        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if already in cache
        auto it = cache_map_.find(path);
        if (it != cache_map_.end()) {
            // Update existing entry
            auto list_it = it->second;
            lru_list_.splice(lru_list_.begin(), lru_list_, list_it);
            list_it->last_access = std::chrono::system_clock::now();
            return;
        }

        // Add new entry
        CacheEntry entry{path, cache_file, data.size(), std::chrono::system_clock::now()};

        // Check if we need to evict
        while (current_cache_size_ + data.size() > config_.max_cache_size && !lru_list_.empty()) {
            evict_lru();
        }

        // Add to cache
        lru_list_.push_front(entry);
        cache_map_[path] = lru_list_.begin();
        current_cache_size_ += data.size();

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.num_entries = cache_map_.size();
            stats_.current_size = current_cache_size_;
        }
    }

    void evict_lru() {
        // Must be called with cache_mutex_ held
        if (lru_list_.empty()) {
            return;
        }

        // Remove least recently used (back of list)
        auto& entry = lru_list_.back();

        // Delete cache file
        fs::remove(entry.cache_file);

        // Update cache size
        current_cache_size_ -= entry.file_size;

        // Remove from map and list
        cache_map_.erase(entry.path);
        lru_list_.pop_back();

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.evictions++;
            stats_.num_entries = cache_map_.size();
            stats_.current_size = current_cache_size_;
        }
    }

    void prefetch_worker() {
        while (prefetch_running_) {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);

            prefetch_cv_.wait(lock, [this]() {
                return !prefetch_queue_.empty() || !prefetch_running_;
            });

            if (!prefetch_running_) {
                break;
            }

            if (prefetch_queue_.empty()) {
                continue;
            }

            std::string path = prefetch_queue_.front();
            prefetch_queue_.pop();

            lock.unlock();

            // Check if already in cache
            if (is_in_cache(path)) {
                continue;
            }

            // Download and cache
            try {
                auto data = backend_->read(path);
                write_to_cache(path, data);
            } catch (...) {
                // Prefetch failure is non-fatal
            }
        }
    }
};

CachedStorageReader::CachedStorageReader(std::unique_ptr<StorageReader> backend,
                                         const Config& config)
    : pimpl_(std::make_unique<Impl>(std::move(backend), config)) {
}

CachedStorageReader::~CachedStorageReader() = default;
CachedStorageReader::CachedStorageReader(CachedStorageReader&&) noexcept = default;
CachedStorageReader& CachedStorageReader::operator=(CachedStorageReader&&) noexcept = default;

std::vector<uint8_t> CachedStorageReader::read(const std::string& path) {
    // Try cache first
    auto data = pimpl_->read_from_cache(path);
    if (!data.empty()) {
        return data;
    }

    // Cache miss - fetch from backend
    {
        std::lock_guard<std::mutex> lock(pimpl_->stats_mutex_);
        pimpl_->stats_.misses++;
    }

    data = pimpl_->backend_->read(path);

    // Write to cache
    pimpl_->write_to_cache(path, data);

    return data;
}

std::vector<uint8_t> CachedStorageReader::read_range(const std::string& path,
                                                      size_t offset,
                                                      size_t length) {
    // Range requests bypass cache
    return pimpl_->backend_->read_range(path, offset, length);
}

bool CachedStorageReader::exists(const std::string& path) {
    // Check cache first
    if (pimpl_->is_in_cache(path)) {
        return true;
    }

    return pimpl_->backend_->exists(path);
}

size_t CachedStorageReader::size(const std::string& path) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(pimpl_->cache_mutex_);
        auto it = pimpl_->cache_map_.find(path);
        if (it != pimpl_->cache_map_.end()) {
            return it->second->file_size;
        }
    }

    return pimpl_->backend_->size(path);
}

void CachedStorageReader::prefetch(const std::vector<std::string>& paths) {
    if (!pimpl_->config_.enable_prefetch) {
        return;
    }

    std::lock_guard<std::mutex> lock(pimpl_->prefetch_mutex_);

    for (const auto& path : paths) {
        pimpl_->prefetch_queue_.push(path);
    }

    pimpl_->prefetch_cv_.notify_one();
}

void CachedStorageReader::clear_cache() {
    std::lock_guard<std::mutex> lock(pimpl_->cache_mutex_);

    // Delete all cache files
    for (const auto& [path, it] : pimpl_->cache_map_) {
        fs::remove(it->cache_file);
    }

    pimpl_->cache_map_.clear();
    pimpl_->lru_list_.clear();
    pimpl_->current_cache_size_ = 0;

    {
        std::lock_guard<std::mutex> stats_lock(pimpl_->stats_mutex_);
        pimpl_->stats_.num_entries = 0;
        pimpl_->stats_.current_size = 0;
    }
}

CachedStorageReader::CacheStats CachedStorageReader::get_stats() const {
    std::lock_guard<std::mutex> lock(pimpl_->stats_mutex_);
    return pimpl_->stats_;
}

}  // namespace turboloader
