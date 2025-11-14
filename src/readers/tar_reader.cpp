#include "turboloader/readers/tar_reader.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace turboloader {

// TAR header format (POSIX ustar)
struct TarHeader {
    char name[100];
    char mode[8];
    char uid[8];
    char gid[8];
    char size[12];
    char mtime[12];
    char checksum[8];
    char typeflag;
    char linkname[100];
    char magic[6];
    char version[2];
    char uname[32];
    char gname[32];
    char devmajor[8];
    char devminor[8];
    char prefix[155];
    char padding[12];
};

static_assert(sizeof(TarHeader) == 512, "TAR header must be 512 bytes");

TarReader::TarReader(const std::string& path)
    : mmap_(path, true) {  // Sequential access
    if (mmap_.is_open()) {
        parse_tar();
    }
}

const TarReader::Sample& TarReader::get_sample(size_t index) const {
    if (index >= samples_.size()) {
        throw std::out_of_range("Sample index out of range");
    }
    return samples_[index];
}

std::span<const uint8_t> TarReader::read_file(const TarEntry& entry) const {
    return mmap_.read(entry.offset, entry.size);
}

void TarReader::parse_tar() {
    size_t offset = 0;
    size_t file_size = mmap_.size();

    std::unordered_map<std::string, Sample> sample_map;

    while (offset + 512 <= file_size) {
        auto entry_opt = parse_header(offset);

        if (!entry_opt) {
            // End of archive (two consecutive zero blocks)
            break;
        }

        auto& entry = *entry_opt;

        // Skip directories
        if (entry.type != '0' && entry.type != '\0') {
            offset += 512;
            continue;
        }

        // Extract basename and extension
        auto [basename, ext] = split_name(entry.name);

        if (!basename.empty()) {
            // Add to sample map
            auto& sample = sample_map[basename];
            sample.key = basename;
            sample.files[ext] = entry;
        }

        // Move to next header (data is aligned to 512 bytes)
        size_t data_blocks = (entry.size + 511) / 512;
        offset += 512 + data_blocks * 512;
    }

    // Convert map to vector and sort by key
    samples_.reserve(sample_map.size());
    for (auto& [key, sample] : sample_map) {
        samples_.push_back(std::move(sample));
    }

    std::sort(samples_.begin(), samples_.end(),
              [](const Sample& a, const Sample& b) {
                  return a.key < b.key;
              });
}

std::optional<TarReader::TarEntry> TarReader::parse_header(size_t offset) {
    auto header_bytes = mmap_.read(offset, 512);
    const auto* header = reinterpret_cast<const TarHeader*>(header_bytes.data());

    // Check for end of archive (zero block)
    if (header->name[0] == '\0') {
        return std::nullopt;
    }

    // Verify magic (ustar)
    if (std::strncmp(header->magic, "ustar", 5) != 0) {
        // Not ustar format, might be old tar, try anyway
    }

    TarEntry entry;
    entry.name = std::string(header->name, strnlen(header->name, 100));
    entry.size = parse_octal(header->size, 12);
    entry.type = header->typeflag;
    entry.offset = offset + 512;  // Data starts after header

    return entry;
}

size_t TarReader::parse_octal(const char* str, size_t len) {
    size_t value = 0;

    for (size_t i = 0; i < len && str[i] != '\0' && str[i] != ' '; ++i) {
        if (str[i] >= '0' && str[i] <= '7') {
            value = value * 8 + (str[i] - '0');
        }
    }

    return value;
}

std::pair<std::string, std::string> TarReader::split_name(const std::string& name) {
    // Find last dot
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) {
        return {name, ""};
    }

    // Find last slash (directory separator)
    size_t slash = name.rfind('/');
    size_t basename_start = (slash == std::string::npos) ? 0 : slash + 1;

    std::string basename = name.substr(basename_start, dot - basename_start);
    std::string ext = name.substr(dot + 1);

    return {basename, ext};
}

}  // namespace turboloader
