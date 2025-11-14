#include "turboloader/readers/mmap_reader.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <system_error>
#include <glob.h>

namespace turboloader {

MmapReader::MmapReader(const std::string& path, bool advise_sequential)
    : path_(path) {

    // Open file
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::system_error(errno, std::system_category(),
                               "Failed to open file: " + path);
    }

    // Get file size
    struct stat st;
    if (::fstat(fd_, &st) == -1) {
        ::close(fd_);
        throw std::system_error(errno, std::system_category(),
                               "Failed to stat file: " + path);
    }

    size_ = static_cast<size_t>(st.st_size);

    if (size_ == 0) {
        // Empty file, nothing to map
        ::close(fd_);
        fd_ = -1;
        return;
    }

    // Memory map file
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd_);
        throw std::system_error(errno, std::system_category(),
                               "Failed to mmap file: " + path);
    }

    data_ = static_cast<uint8_t*>(mapped);

    // Advise OS about access pattern
    if (advise_sequential) {
#ifdef MADV_SEQUENTIAL
        ::madvise(data_, size_, MADV_SEQUENTIAL);
#endif
    }

    // Advise OS to prefetch ahead
#ifdef MADV_WILLNEED
    // Prefetch first 1MB for faster initial access
    size_t prefetch_size = std::min(size_, size_t(1024 * 1024));
    ::madvise(data_, prefetch_size, MADV_WILLNEED);
#endif
}

MmapReader::~MmapReader() {
    close();
}

MmapReader::MmapReader(MmapReader&& other) noexcept
    : path_(std::move(other.path_))
    , fd_(other.fd_)
    , data_(other.data_)
    , size_(other.size_) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
}

MmapReader& MmapReader::operator=(MmapReader&& other) noexcept {
    if (this != &other) {
        close();

        path_ = std::move(other.path_);
        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;

        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MmapReader::close() {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }

    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
}

std::span<const uint8_t> MmapReader::read(size_t offset, size_t length) const {
    if (!is_open()) {
        throw std::runtime_error("File not open: " + path_);
    }

    if (offset >= size_) {
        throw std::out_of_range("Offset beyond file size");
    }

    if (length == 0) {
        length = size_ - offset;
    }

    if (offset + length > size_) {
        throw std::out_of_range("Read beyond file size");
    }

    return std::span<const uint8_t>(data_ + offset, length);
}

std::string_view MmapReader::read_string(size_t offset, size_t length) const {
    auto bytes = read(offset, length);
    return std::string_view(reinterpret_cast<const char*>(bytes.data()),
                           bytes.size());
}

void MmapReader::prefetch(size_t offset, size_t length) {
    if (!is_open()) {
        return;
    }

    if (offset >= size_) {
        return;
    }

    if (length == 0) {
        length = size_ - offset;
    }

    length = std::min(length, size_ - offset);

#ifdef MADV_WILLNEED
    ::madvise(data_ + offset, length, MADV_WILLNEED);
#endif
}

std::vector<std::string> glob(const std::string& pattern) {
    std::vector<std::string> results;

    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    int ret = ::glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);

    if (ret == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            results.emplace_back(glob_result.gl_pathv[i]);
        }
    } else if (ret != GLOB_NOMATCH) {
        globfree(&glob_result);
        throw std::runtime_error("glob() failed for pattern: " + pattern);
    }

    globfree(&glob_result);
    return results;
}

}  // namespace turboloader
