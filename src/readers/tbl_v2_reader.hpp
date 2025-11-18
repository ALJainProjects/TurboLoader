/**
 * @file tbl_v2_reader.hpp
 * @brief TBL v2 format reader with LZ4 decompression
 *
 * Features:
 * - Memory-mapped I/O for zero-copy reads
 * - Automatic LZ4 decompression
 * - Checksum verification
 * - Metadata access
 * - Dimension-based filtering
 */

#pragma once

#include "../formats/tbl_v2_format.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <lz4.h>

namespace turboloader {
namespace readers {

/**
 * @brief TBL v2 reader with decompression and verification
 */
class TblReaderV2 {
public:
    /**
     * @brief Construct a new TBL v2 reader
     *
     * @param path Path to TBL v2 file
     * @param verify_checksums Enable checksum verification (default: true)
     */
    explicit TblReaderV2(const std::string& path, bool verify_checksums = true)
        : path_(path)
        , fd_(-1)
        , mapped_data_(nullptr)
        , file_size_(0)
        , verify_checksums_(verify_checksums)
    {
        // Open file
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        // Get file size
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            throw std::runtime_error("Failed to stat file: " + path);
        }
        file_size_ = st.st_size;

        // Memory-map the file
        mapped_data_ = static_cast<const uint8_t*>(
            mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0)
        );

        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap file: " + path);
        }

        // Read and validate header
        if (file_size_ < sizeof(formats::TblHeaderV2)) {
            cleanup();
            throw std::runtime_error("File too small to be valid TBL v2: " + path);
        }

        std::memcpy(&header_, mapped_data_, sizeof(formats::TblHeaderV2));

        if (!header_.is_valid()) {
            cleanup();
            throw std::runtime_error("Invalid TBL v2 header (magic/version mismatch): " + path);
        }

        // Verify header checksum if requested
        if (verify_checksums_) {
            if (!header_.verify_checksum()) {
                cleanup();
                throw std::runtime_error("Header checksum verification failed: " + path);
            }
        }

        // Load index table
        size_t index_offset = sizeof(formats::TblHeaderV2);
        size_t index_size = header_.num_samples * sizeof(formats::TblIndexEntryV2);

        if (index_offset + index_size > file_size_) {
            cleanup();
            throw std::runtime_error("Index table extends beyond file size: " + path);
        }

        index_.resize(header_.num_samples);
        std::memcpy(index_.data(), mapped_data_ + index_offset, index_size);
    }

    /**
     * @brief Read a sample by index
     *
     * Returns decompressed data if sample is compressed.
     * Verifies checksum if verification is enabled.
     *
     * @param index Sample index
     * @return Pair of (data pointer, size). Lifetime managed by reader.
     */
    std::pair<const uint8_t*, size_t> read_sample(size_t index) {
        if (index >= header_.num_samples) {
            throw std::out_of_range("Sample index out of range");
        }

        const formats::TblIndexEntryV2& entry = index_[index];

        // Validate offset
        if (entry.offset + entry.size > file_size_) {
            throw std::runtime_error("Sample data extends beyond file size");
        }

        const uint8_t* data = mapped_data_ + entry.offset;
        size_t size = entry.size;

        // Decompress if needed
        if (entry.is_compressed()) {
            // Allocate decompression buffer (reuse if possible)
            size_t uncompressed_size = entry.uncompressed_size;
            if (uncompressed_size == 0) {
                throw std::runtime_error("Compressed sample has no uncompressed size");
            }

            // Resize decompression buffer if needed
            if (decompress_buffer_.size() < uncompressed_size) {
                decompress_buffer_.resize(uncompressed_size);
            }

            // Decompress with LZ4
            int decompressed_size = LZ4_decompress_safe(
                reinterpret_cast<const char*>(data),
                reinterpret_cast<char*>(decompress_buffer_.data()),
                size,
                uncompressed_size
            );

            if (decompressed_size < 0) {
                throw std::runtime_error("LZ4 decompression failed for sample " + std::to_string(index));
            }

            if (static_cast<size_t>(decompressed_size) != uncompressed_size) {
                throw std::runtime_error("Decompressed size mismatch for sample " + std::to_string(index));
            }

            data = decompress_buffer_.data();
            size = uncompressed_size;
        }

        // Verify checksum if enabled
        if (verify_checksums_ && (header_.flags & formats::TBL_V2_FLAG_HAS_CHECKSUMS)) {
            uint16_t calculated = formats::calculate_crc16(data, size);
            if (calculated != entry.checksum) {
                throw std::runtime_error("Checksum verification failed for sample " + std::to_string(index));
            }
        }

        return {data, size};
    }

    /**
     * @brief Read metadata for a sample
     *
     * @param index Sample index
     * @return Pair of (metadata string, metadata type). Empty if no metadata.
     */
    std::pair<std::string, formats::MetadataType> read_metadata(size_t index) {
        if (index >= header_.num_samples) {
            throw std::out_of_range("Sample index out of range");
        }

        // Check if sample has metadata
        if (!index_[index].has_metadata()) {
            return {"", formats::MetadataType::NONE};
        }

        // Check if file has metadata section
        if (header_.metadata_offset == 0 || header_.metadata_size == 0) {
            return {"", formats::MetadataType::NONE};
        }

        // Scan metadata section for this sample's metadata
        uint64_t offset = header_.metadata_offset;
        uint64_t end_offset = header_.metadata_offset + header_.metadata_size;

        while (offset < end_offset) {
            if (offset + sizeof(formats::MetadataBlockHeader) > file_size_) {
                break;
            }

            formats::MetadataBlockHeader mheader;
            std::memcpy(&mheader, mapped_data_ + offset, sizeof(mheader));
            offset += sizeof(mheader);

            if (mheader.sample_index == index) {
                // Found metadata for this sample
                if (offset + mheader.metadata_size > file_size_) {
                    throw std::runtime_error("Metadata extends beyond file size");
                }

                std::string metadata(
                    reinterpret_cast<const char*>(mapped_data_ + offset),
                    mheader.metadata_size
                );

                return {metadata, mheader.type};
            }

            // Skip to next metadata block
            offset += mheader.metadata_size;
        }

        return {"", formats::MetadataType::NONE};
    }

    /**
     * @brief Get sample information without reading data
     *
     * @param index Sample index
     * @return Index entry with format, dimensions, flags, etc.
     */
    const formats::TblIndexEntryV2& get_sample_info(size_t index) const {
        if (index >= header_.num_samples) {
            throw std::out_of_range("Sample index out of range");
        }
        return index_[index];
    }

    /**
     * @brief Get number of samples in the file
     */
    size_t num_samples() const {
        return header_.num_samples;
    }

    /**
     * @brief Get file header
     */
    const formats::TblHeaderV2& header() const {
        return header_;
    }

    /**
     * @brief Check if file uses compression
     */
    bool is_compressed() const {
        return (header_.flags & formats::TBL_V2_FLAG_COMPRESSED) != 0;
    }

    /**
     * @brief Check if file has metadata
     */
    bool has_metadata() const {
        return (header_.flags & formats::TBL_V2_FLAG_HAS_METADATA) != 0;
    }

    /**
     * @brief Get indices of samples matching dimension filter
     *
     * @param min_width Minimum width (0 = no filter)
     * @param min_height Minimum height (0 = no filter)
     * @param max_width Maximum width (0 = no filter)
     * @param max_height Maximum height (0 = no filter)
     * @return Vector of matching sample indices
     */
    std::vector<size_t> filter_by_dimensions(
        uint16_t min_width = 0, uint16_t min_height = 0,
        uint16_t max_width = 0, uint16_t max_height = 0) const {

        std::vector<size_t> result;
        result.reserve(header_.num_samples);

        for (size_t i = 0; i < header_.num_samples; ++i) {
            const auto& entry = index_[i];

            // Skip if dimensions are unknown
            if (entry.width == 0 || entry.height == 0) {
                continue;
            }

            // Apply filters
            if (min_width > 0 && entry.width < min_width) continue;
            if (min_height > 0 && entry.height < min_height) continue;
            if (max_width > 0 && entry.width > max_width) continue;
            if (max_height > 0 && entry.height > max_height) continue;

            result.push_back(i);
        }

        return result;
    }

    /**
     * @brief Get indices of samples matching format filter
     *
     * @param format Sample format to filter by
     * @return Vector of matching sample indices
     */
    std::vector<size_t> filter_by_format(formats::SampleFormat format) const {
        std::vector<size_t> result;
        result.reserve(header_.num_samples);

        for (size_t i = 0; i < header_.num_samples; ++i) {
            if (index_[i].format == format) {
                result.push_back(i);
            }
        }

        return result;
    }

    /**
     * @brief Destructor - cleanup resources
     */
    ~TblReaderV2() {
        cleanup();
    }

    // Disable copying
    TblReaderV2(const TblReaderV2&) = delete;
    TblReaderV2& operator=(const TblReaderV2&) = delete;

private:
    void cleanup() {
        if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
            munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
            mapped_data_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    std::string path_;
    int fd_;
    const uint8_t* mapped_data_;
    size_t file_size_;
    bool verify_checksums_;
    formats::TblHeaderV2 header_;
    std::vector<formats::TblIndexEntryV2> index_;
    mutable std::vector<uint8_t> decompress_buffer_;  // Reusable decompression buffer
};

} // namespace readers
} // namespace turboloader
