/**
 * @file cache_key.hpp
 * @brief Cache key with xxHash64 for fast content hashing
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>

namespace turboloader {
namespace cache {

/**
 * @brief xxHash64 implementation for fast content hashing
 *
 * Based on xxHash by Yann Collet - extremely fast non-cryptographic hash
 * Achieves 10+ GB/s on modern CPUs
 */
class XXHash64 {
public:
    static constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    static constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    static constexpr uint64_t PRIME3 = 0x165667B19E3779F9ULL;
    static constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
    static constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

    /**
     * @brief Compute xxHash64 of a byte buffer
     * @param data Pointer to data
     * @param length Length in bytes
     * @param seed Optional seed value (default 0)
     * @return 64-bit hash value
     */
    static uint64_t hash(const void* data, size_t length, uint64_t seed = 0) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        const uint8_t* const end = p + length;
        uint64_t h64;

        if (length >= 32) {
            const uint8_t* const limit = end - 32;
            uint64_t v1 = seed + PRIME1 + PRIME2;
            uint64_t v2 = seed + PRIME2;
            uint64_t v3 = seed + 0;
            uint64_t v4 = seed - PRIME1;

            do {
                v1 = round(v1, read64(p)); p += 8;
                v2 = round(v2, read64(p)); p += 8;
                v3 = round(v3, read64(p)); p += 8;
                v4 = round(v4, read64(p)); p += 8;
            } while (p <= limit);

            h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
            h64 = mergeRound(h64, v1);
            h64 = mergeRound(h64, v2);
            h64 = mergeRound(h64, v3);
            h64 = mergeRound(h64, v4);
        } else {
            h64 = seed + PRIME5;
        }

        h64 += static_cast<uint64_t>(length);

        // Process remaining bytes
        while (p + 8 <= end) {
            uint64_t k1 = round(0, read64(p));
            h64 ^= k1;
            h64 = rotl64(h64, 27) * PRIME1 + PRIME4;
            p += 8;
        }

        if (p + 4 <= end) {
            h64 ^= static_cast<uint64_t>(read32(p)) * PRIME1;
            h64 = rotl64(h64, 23) * PRIME2 + PRIME3;
            p += 4;
        }

        while (p < end) {
            h64 ^= static_cast<uint64_t>(*p) * PRIME5;
            h64 = rotl64(h64, 11) * PRIME1;
            p++;
        }

        // Finalize
        h64 ^= h64 >> 33;
        h64 *= PRIME2;
        h64 ^= h64 >> 29;
        h64 *= PRIME3;
        h64 ^= h64 >> 32;

        return h64;
    }

private:
    static inline uint64_t rotl64(uint64_t x, int r) {
        return (x << r) | (x >> (64 - r));
    }

    static inline uint64_t read64(const void* ptr) {
        uint64_t val;
        std::memcpy(&val, ptr, sizeof(val));
        return val;
    }

    static inline uint32_t read32(const void* ptr) {
        uint32_t val;
        std::memcpy(&val, ptr, sizeof(val));
        return val;
    }

    static inline uint64_t round(uint64_t acc, uint64_t input) {
        acc += input * PRIME2;
        acc = rotl64(acc, 31);
        acc *= PRIME1;
        return acc;
    }

    static inline uint64_t mergeRound(uint64_t acc, uint64_t val) {
        val = round(0, val);
        acc ^= val;
        acc = acc * PRIME1 + PRIME4;
        return acc;
    }
};

/**
 * @brief Cache key for decoded images
 *
 * Combines content hash with optional dimensions and transform signature
 * to uniquely identify cached items.
 */
struct CacheKey {
    uint64_t content_hash;              // xxHash64 of raw bytes
    uint32_t width;                     // Image width (0 if unknown)
    uint32_t height;                    // Image height (0 if unknown)
    std::string transform_signature;    // Optional: for caching transformed images

    CacheKey() : content_hash(0), width(0), height(0) {}

    CacheKey(uint64_t hash, uint32_t w = 0, uint32_t h = 0,
             const std::string& transform = "")
        : content_hash(hash), width(w), height(h),
          transform_signature(transform) {}

    /**
     * @brief Create cache key from raw bytes
     */
    static CacheKey from_bytes(const void* data, size_t length,
                               uint32_t w = 0, uint32_t h = 0,
                               const std::string& transform = "") {
        return CacheKey(XXHash64::hash(data, length), w, h, transform);
    }

    /**
     * @brief Equality comparison
     */
    bool operator==(const CacheKey& other) const {
        return content_hash == other.content_hash &&
               width == other.width &&
               height == other.height &&
               transform_signature == other.transform_signature;
    }

    bool operator!=(const CacheKey& other) const {
        return !(*this == other);
    }

    /**
     * @brief Hash function for use in unordered containers
     */
    size_t hash() const {
        // Combine all fields into a single hash
        size_t h = std::hash<uint64_t>{}(content_hash);
        h ^= std::hash<uint32_t>{}(width) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(height) + 0x9e3779b9 + (h << 6) + (h >> 2);
        if (!transform_signature.empty()) {
            h ^= std::hash<std::string>{}(transform_signature) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    /**
     * @brief Convert to string for disk cache filename
     */
    std::string to_filename() const {
        char buf[64];
        if (transform_signature.empty()) {
            snprintf(buf, sizeof(buf), "%016llx_%u_%u.cache",
                     static_cast<unsigned long long>(content_hash), width, height);
        } else {
            // Include transform hash in filename
            size_t transform_hash = std::hash<std::string>{}(transform_signature);
            snprintf(buf, sizeof(buf), "%016llx_%u_%u_%016zx.cache",
                     static_cast<unsigned long long>(content_hash), width, height,
                     transform_hash);
        }
        return std::string(buf);
    }
};

/**
 * @brief Cache statistics
 */
struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    size_t current_size_bytes = 0;
    size_t max_size_bytes = 0;
    size_t item_count = 0;

    double hit_rate() const {
        uint64_t total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    }

    double utilization() const {
        return max_size_bytes > 0 ?
               static_cast<double>(current_size_bytes) / max_size_bytes : 0.0;
    }
};

} // namespace cache
} // namespace turboloader

// Hash specialization for std::unordered_map
namespace std {
template<>
struct hash<turboloader::cache::CacheKey> {
    size_t operator()(const turboloader::cache::CacheKey& key) const {
        return key.hash();
    }
};
} // namespace std
