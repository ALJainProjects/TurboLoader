#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace turboloader {

/**
 * Memory-mapped file reader for zero-copy file access
 *
 * Benefits:
 * - Zero-copy reads (no memcpy overhead)
 * - OS handles paging (automatic caching)
 * - Lazy loading (only touched pages loaded)
 * - Shared across processes
 *
 * Perfect for large files (models, datasets) where random access is needed.
 */
class MmapReader {
public:
    /**
     * Open file for memory-mapped reading
     * @param path File path
     * @param advise Advice to OS about access pattern (true = sequential)
     */
    explicit MmapReader(const std::string& path, bool advise_sequential = true);

    ~MmapReader();

    // Non-copyable
    MmapReader(const MmapReader&) = delete;
    MmapReader& operator=(const MmapReader&) = delete;

    // Movable
    MmapReader(MmapReader&& other) noexcept;
    MmapReader& operator=(MmapReader&& other) noexcept;

    /**
     * Get pointer to mapped memory
     */
    const uint8_t* data() const { return data_; }

    /**
     * Get file size
     */
    size_t size() const { return size_; }

    /**
     * Read data at offset (returns view, no copy)
     * @param offset Byte offset from start
     * @param length Bytes to read (0 = to end of file)
     * @return View of data (invalidated when MmapReader is destroyed)
     */
    std::span<const uint8_t> read(size_t offset, size_t length = 0) const;

    /**
     * Read as string view (for text files)
     */
    std::string_view read_string(size_t offset, size_t length = 0) const;

    /**
     * Prefetch range into memory (hint to OS)
     */
    void prefetch(size_t offset, size_t length);

    /**
     * Check if file is successfully mapped
     */
    bool is_open() const { return data_ != nullptr; }

    /**
     * Get file path
     */
    const std::string& path() const { return path_; }

private:
    std::string path_;
    int fd_{-1};
    uint8_t* data_{nullptr};
    size_t size_{0};

    void close();
};

/**
 * Helper: List all files matching glob pattern
 * Example: glob("/data/ *.tar") returns list of matching files
 */
std::vector<std::string> glob(const std::string& pattern);

}  // namespace turboloader
