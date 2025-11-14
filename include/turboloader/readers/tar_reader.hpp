#pragma once

#include "turboloader/readers/mmap_reader.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace turboloader {

/**
 * Efficient TAR archive reader for WebDataset format
 *
 * WebDataset format:
 * - Sharded datasets stored as TAR files
 * - Each sample is a group of files with same basename:
 *   - 000000.jpg, 000000.json, 000000.txt
 *   - 000001.jpg, 000001.json, 000001.txt
 *
 * This reader:
 * - Uses mmap for zero-copy access
 * - Builds index of all files on open
 * - Groups files by sample key
 * - Enables random access to samples
 */
class TarReader {
public:
    struct TarEntry {
        std::string name;
        size_t offset;
        size_t size;
        char type;  // '0' = file, '5' = directory
    };

    struct Sample {
        std::string key;  // Basename (e.g., "000000")
        std::unordered_map<std::string, TarEntry> files;  // extension -> entry
    };

    /**
     * Open TAR file
     * @param path Path to .tar file
     */
    explicit TarReader(const std::string& path);

    /**
     * Get number of samples in TAR
     */
    size_t num_samples() const { return samples_.size(); }

    /**
     * Get sample by index
     */
    const Sample& get_sample(size_t index) const;

    /**
     * Read file data from TAR (zero-copy)
     * @param entry File entry from sample
     * @return Span pointing to mmap'd data
     */
    std::span<const uint8_t> read_file(const TarEntry& entry) const;

    /**
     * Get all samples
     */
    const std::vector<Sample>& samples() const { return samples_; }

    /**
     * Check if TAR is successfully opened
     */
    bool is_open() const { return mmap_.is_open(); }

private:
    MmapReader mmap_;
    std::vector<Sample> samples_;

    // Parse TAR header and build index
    void parse_tar();

    // Parse single TAR header (512 bytes)
    std::optional<TarEntry> parse_header(size_t offset);

    // Convert octal string to number (TAR uses octal)
    static size_t parse_octal(const char* str, size_t len);

    // Extract basename and extension from filename
    static std::pair<std::string, std::string> split_name(const std::string& name);
};

}  // namespace turboloader
