/**
 * @file tbl_reader.hpp
 * @brief TurboLoader Binary (.tbl) format reader
 *
 * High-performance reader for .tbl format with memory-mapped I/O support.
 */

#pragma once

#include "../formats/tbl_format.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>

namespace turboloader {
namespace readers {

using namespace formats;

/**
 * @brief TBL file reader with memory-mapped I/O
 */
class TblReader {
public:
    explicit TblReader(const std::string& filepath)
        : filepath_(filepath)
        , fd_(-1)
        , file_size_(0)
        , mapped_data_(nullptr)
        , header_()
        , index_()
    {
        open_file();
        read_header();
        read_index();
    }

    ~TblReader() {
        close_file();
    }

    // Disable copy
    TblReader(const TblReader&) = delete;
    TblReader& operator=(const TblReader&) = delete;

    /**
     * @brief Get number of samples in file
     */
    size_t num_samples() const {
        return header_.num_samples;
    }

    /**
     * @brief Read sample data by index
     *
     * @param index Sample index
     * @return Pair of (data pointer, size)
     */
    std::pair<const uint8_t*, size_t> read_sample(size_t index) const {
        if (index >= header_.num_samples) {
            throw std::out_of_range("Sample index out of range");
        }

        const TblIndexEntry& entry = index_[index];

        // Verify offset is within file bounds
        if (entry.offset + entry.size > file_size_) {
            throw std::runtime_error("Invalid sample offset/size in index");
        }

        const uint8_t* data = mapped_data_ + entry.offset;
        return {data, entry.size};
    }

    /**
     * @brief Get sample format
     */
    SampleFormat get_sample_format(size_t index) const {
        if (index >= header_.num_samples) {
            throw std::out_of_range("Sample index out of range");
        }
        return index_[index].format;
    }

    /**
     * @brief Read sample data into buffer
     */
    void read_sample_to_buffer(size_t index, std::vector<uint8_t>& buffer) const {
        auto [data, size] = read_sample(index);
        buffer.resize(size);
        std::memcpy(buffer.data(), data, size);
    }

    /**
     * @brief Get file path
     */
    const std::string& filepath() const {
        return filepath_;
    }

private:
    void open_file() {
        // Open file
        fd_ = ::open(filepath_.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file: " + filepath_);
        }

        // Get file size
        struct stat st;
        if (::fstat(fd_, &st) < 0) {
            ::close(fd_);
            throw std::runtime_error("Failed to stat file: " + filepath_);
        }
        file_size_ = st.st_size;

        // Memory-map the file
        mapped_data_ = static_cast<const uint8_t*>(
            ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0)
        );

        if (mapped_data_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("Failed to mmap file: " + filepath_);
        }

        // Advise kernel for sequential access
        ::madvise(const_cast<uint8_t*>(mapped_data_), file_size_, MADV_SEQUENTIAL);
    }

    void close_file() {
        if (mapped_data_ && mapped_data_ != MAP_FAILED) {
            ::munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
            mapped_data_ = nullptr;
        }

        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void read_header() {
        if (file_size_ < sizeof(TblHeader)) {
            throw std::runtime_error("File too small to contain TBL header");
        }

        // Read header
        std::memcpy(&header_, mapped_data_, sizeof(TblHeader));

        // Validate header
        if (!header_.is_valid()) {
            throw std::runtime_error("Invalid TBL header (magic/version mismatch)");
        }

        // Verify header size
        if (header_.header_size > file_size_) {
            throw std::runtime_error("Invalid header size");
        }
    }

    void read_index() {
        const size_t expected_index_size = header_.num_samples * sizeof(TblIndexEntry);
        const size_t expected_header_size = sizeof(TblHeader) + expected_index_size;

        if (header_.header_size != expected_header_size) {
            throw std::runtime_error("Header size mismatch");
        }

        if (file_size_ < expected_header_size) {
            throw std::runtime_error("File too small to contain index table");
        }

        // Read index table
        index_.resize(header_.num_samples);
        const uint8_t* index_data = mapped_data_ + sizeof(TblHeader);
        std::memcpy(index_.data(), index_data, expected_index_size);
    }

    std::string filepath_;
    int fd_;
    size_t file_size_;
    const uint8_t* mapped_data_;
    TblHeader header_;
    std::vector<TblIndexEntry> index_;
};

} // namespace readers
} // namespace turboloader
