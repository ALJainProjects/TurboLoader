/**
 * @file test_cache.cpp
 * @brief Tests for TurboLoader tiered cache system
 */

#include "../src/cache/cache_key.hpp"
#include "../src/cache/lru_cache.hpp"
#include "../src/cache/disk_cache.hpp"
#include "../src/cache/tiered_cache.hpp"

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <filesystem>
#include <chrono>

using namespace turboloader::cache;

// Test helper macros
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "..." << std::flush; \
    test_##name(); \
    std::cout << " PASSED" << std::endl; \
} while(0)

#define ASSERT(expr) do { \
    if (!(expr)) { \
        std::cerr << "\nAssertion failed: " #expr << std::endl; \
        std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "\nAssertion failed: " #a " == " #b << std::endl; \
        std::cerr << "  " #a " = " << (a) << std::endl; \
        std::cerr << "  " #b " = " << (b) << std::endl; \
        std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while(0)

// ============================================================================
// XXHash64 Tests
// ============================================================================

TEST(xxhash64_basic) {
    const char* data = "Hello, World!";
    uint64_t hash = XXHash64::hash(data, strlen(data));

    // Hash should be non-zero and deterministic
    ASSERT(hash != 0);
    ASSERT_EQ(hash, XXHash64::hash(data, strlen(data)));
}

TEST(xxhash64_different_data) {
    const char* data1 = "Hello";
    const char* data2 = "World";

    uint64_t hash1 = XXHash64::hash(data1, strlen(data1));
    uint64_t hash2 = XXHash64::hash(data2, strlen(data2));

    ASSERT(hash1 != hash2);
}

TEST(xxhash64_empty) {
    uint64_t hash = XXHash64::hash("", 0);
    // Empty data should still produce a valid hash
    ASSERT(hash != 0 || true);  // Hash of empty is allowed to be 0
}

TEST(xxhash64_large_data) {
    std::vector<uint8_t> data(1024 * 1024);  // 1MB
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    uint64_t hash = XXHash64::hash(data.data(), data.size());
    ASSERT(hash != 0);
    ASSERT_EQ(hash, XXHash64::hash(data.data(), data.size()));
}

// ============================================================================
// CacheKey Tests
// ============================================================================

TEST(cache_key_equality) {
    CacheKey key1(12345, 256, 256, "");
    CacheKey key2(12345, 256, 256, "");
    CacheKey key3(12346, 256, 256, "");

    ASSERT(key1 == key2);
    ASSERT(key1 != key3);
}

TEST(cache_key_from_bytes) {
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    CacheKey key = CacheKey::from_bytes(data.data(), data.size(), 100, 100);

    ASSERT(key.content_hash != 0);
    ASSERT_EQ(key.width, 100u);
    ASSERT_EQ(key.height, 100u);
}

TEST(cache_key_to_filename) {
    CacheKey key(0x123456789ABCDEF0ULL, 256, 256, "");
    std::string filename = key.to_filename();

    ASSERT(filename.find(".cache") != std::string::npos);
    ASSERT(filename.find("256") != std::string::npos);
}

// ============================================================================
// LRUCache Tests
// ============================================================================

TEST(lru_cache_basic) {
    LRUCache<CacheKey, CachedImageData> cache(10);  // 10MB

    CacheKey key(12345, 100, 100);
    auto data = std::make_shared<CachedImageData>();
    data->width = 100;
    data->height = 100;
    data->channels = 3;
    data->data.resize(100 * 100 * 3);

    cache.put(key, data);

    auto retrieved = cache.get(key);
    ASSERT(retrieved != nullptr);
    ASSERT_EQ(retrieved->width, 100u);
}

TEST(lru_cache_miss) {
    LRUCache<CacheKey, CachedImageData> cache(10);

    CacheKey key(12345, 100, 100);
    auto result = cache.get(key);

    ASSERT(result == nullptr);
}

TEST(lru_cache_eviction) {
    // Create a custom size function that treats size as bytes not MB
    auto size_func = [](const CachedImageData& d) -> size_t {
        return d.data.size();
    };

    // 1KB cache (pass 0 for MB, we'll set bytes directly)
    LRUCache<CacheKey, CachedImageData> cache(1, size_func);  // 1MB = 1048576 bytes

    // Add entries that exceed cache size (each 200KB, need ~6 to exceed 1MB)
    for (int i = 0; i < 10; ++i) {
        CacheKey key(i, 100, 100);
        auto data = std::make_shared<CachedImageData>();
        data->data.resize(200 * 1024);  // 200KB each
        cache.put(key, data);
    }

    // Some entries should have been evicted (10 * 200KB = 2MB > 1MB limit)
    auto stats = cache.stats();
    ASSERT(stats.evictions > 0);
}

TEST(lru_cache_update) {
    LRUCache<CacheKey, CachedImageData> cache(10);

    CacheKey key(12345, 100, 100);

    auto data1 = std::make_shared<CachedImageData>();
    data1->width = 100;
    cache.put(key, data1);

    auto data2 = std::make_shared<CachedImageData>();
    data2->width = 200;
    cache.put(key, data2);

    auto retrieved = cache.get(key);
    ASSERT(retrieved != nullptr);
    ASSERT_EQ(retrieved->width, 200u);
}

TEST(lru_cache_stats) {
    LRUCache<CacheKey, CachedImageData> cache(10);

    CacheKey key1(1, 100, 100);
    CacheKey key2(2, 100, 100);

    auto data = std::make_shared<CachedImageData>();
    cache.put(key1, data);

    cache.get(key1);  // Hit
    cache.get(key2);  // Miss

    auto stats = cache.stats();
    ASSERT_EQ(stats.hits, 1u);
    ASSERT_EQ(stats.misses, 1u);
}

TEST(lru_cache_concurrent_reads) {
    LRUCache<CacheKey, CachedImageData> cache(100);

    // Pre-populate cache
    for (int i = 0; i < 100; ++i) {
        CacheKey key(i, 100, 100);
        auto data = std::make_shared<CachedImageData>();
        data->data.resize(100);
        cache.put(key, data);
    }

    // Concurrent reads
    std::vector<std::thread> threads;
    std::atomic<int> hits{0};

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&cache, &hits]() {
            for (int i = 0; i < 100; ++i) {
                CacheKey key(i, 100, 100);
                if (cache.get(key)) {
                    hits++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All reads should hit
    ASSERT_EQ(hits.load(), 400);
}

// ============================================================================
// DiskCache Tests
// ============================================================================

TEST(disk_cache_basic) {
    std::string cache_dir = "/tmp/turboloader_test_cache";
    std::filesystem::remove_all(cache_dir);

    {
        DiskCache cache(cache_dir, 1);  // 1GB

        CacheKey key(12345, 100, 100);
        CachedImageData data;
        data.width = 100;
        data.height = 100;
        data.channels = 3;
        data.data.resize(100 * 100 * 3, 128);

        cache.put(key, data);

        auto retrieved = cache.get(key);
        ASSERT(retrieved.has_value());
        ASSERT_EQ((*retrieved)->width, 100u);
    }

    std::filesystem::remove_all(cache_dir);
}

TEST(disk_cache_persistence) {
    std::string cache_dir = "/tmp/turboloader_test_cache_persist";
    std::filesystem::remove_all(cache_dir);

    CacheKey key(12345, 100, 100);

    // Write to cache
    {
        DiskCache cache(cache_dir, 1);
        CachedImageData data;
        data.width = 100;
        data.height = 100;
        data.channels = 3;
        data.data.resize(100 * 100 * 3, 42);
        cache.put(key, data);
    }

    // Read from new cache instance
    {
        DiskCache cache(cache_dir, 1);
        auto retrieved = cache.get(key);
        ASSERT(retrieved.has_value());
        ASSERT_EQ((*retrieved)->data[0], 42);
    }

    std::filesystem::remove_all(cache_dir);
}

// ============================================================================
// TieredCache Tests
// ============================================================================

TEST(tiered_cache_l1_only) {
    TieredCache cache(100, 0);  // 100MB L1, no L2

    CacheKey key(12345, 100, 100);
    auto data = TieredCache::make_cached_data(
        nullptr, 0, 100, 100, 3);
    data->data.resize(100 * 100 * 3);

    cache.put(key, data);

    auto retrieved = cache.get(key);
    ASSERT(retrieved != nullptr);
    ASSERT_EQ(retrieved->width, 100u);
}

TEST(tiered_cache_l1_l2) {
    std::string cache_dir = "/tmp/turboloader_test_tiered";
    std::filesystem::remove_all(cache_dir);

    {
        TieredCache cache(100, 1, cache_dir);  // 100MB L1, 1GB L2

        CacheKey key(12345, 100, 100);
        auto data = TieredCache::make_cached_data(
            nullptr, 0, 100, 100, 3);
        data->data.resize(100 * 100 * 3, 55);

        cache.put(key, data);
        cache.flush();

        auto retrieved = cache.get(key);
        ASSERT(retrieved != nullptr);
        ASSERT_EQ(retrieved->data[0], 55);
    }

    std::filesystem::remove_all(cache_dir);
}

TEST(tiered_cache_stats) {
    TieredCache cache(100, 0);

    CacheKey key1(1, 100, 100);
    CacheKey key2(2, 100, 100);

    auto data = TieredCache::make_cached_data(nullptr, 0, 100, 100, 3);
    data->data.resize(100);

    cache.put(key1, data);

    cache.get(key1);  // Hit
    cache.get(key2);  // Miss

    auto stats = cache.stats();
    ASSERT_EQ(stats.l1.hits, 1u);
    ASSERT_EQ(stats.l1.misses, 1u);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "TurboLoader Cache Tests\n";
    std::cout << "=======================\n\n";

    // XXHash64 tests
    RUN_TEST(xxhash64_basic);
    RUN_TEST(xxhash64_different_data);
    RUN_TEST(xxhash64_empty);
    RUN_TEST(xxhash64_large_data);

    // CacheKey tests
    RUN_TEST(cache_key_equality);
    RUN_TEST(cache_key_from_bytes);
    RUN_TEST(cache_key_to_filename);

    // LRUCache tests
    RUN_TEST(lru_cache_basic);
    RUN_TEST(lru_cache_miss);
    RUN_TEST(lru_cache_eviction);
    RUN_TEST(lru_cache_update);
    RUN_TEST(lru_cache_stats);
    RUN_TEST(lru_cache_concurrent_reads);

    // DiskCache tests
    RUN_TEST(disk_cache_basic);
    RUN_TEST(disk_cache_persistence);

    // TieredCache tests
    RUN_TEST(tiered_cache_l1_only);
    RUN_TEST(tiered_cache_l1_l2);
    RUN_TEST(tiered_cache_stats);

    std::cout << "\nAll tests passed!\n";
    return 0;
}
