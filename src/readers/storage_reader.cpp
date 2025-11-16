#include "turboloader/readers/storage_reader.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <sstream>

namespace turboloader {

// ============================================================================
// LocalStorageReader
// ============================================================================

std::vector<uint8_t> LocalStorageReader::read(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(file_size);
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    if (!file) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return data;
}

std::vector<uint8_t> LocalStorageReader::read_range(
    const std::string& path,
    size_t offset,
    size_t length
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    file.seekg(offset);
    std::vector<uint8_t> data(length);
    file.read(reinterpret_cast<char*>(data.data()), length);

    if (!file) {
        throw std::runtime_error("Failed to read file range: " + path);
    }

    return data;
}

bool LocalStorageReader::exists(const std::string& path) {
    return std::filesystem::exists(path);
}

size_t LocalStorageReader::size(const std::string& path) {
    return std::filesystem::file_size(path);
}

// ============================================================================
// S3StorageReader (Stub)
// ============================================================================

struct S3StorageReader::Impl {
    Config config;
};

S3StorageReader::S3StorageReader(const Config& config)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->config = config;
}

S3StorageReader::~S3StorageReader() = default;

std::vector<uint8_t> S3StorageReader::read(const std::string& path) {
    // S3 storage support requires AWS SDK for C++
    // Future implementation will use Aws::S3::S3Client to fetch objects
    // For now, users should pre-download data locally or use HTTP reader
    throw std::runtime_error(
        "S3 storage not yet implemented. "
        "Install AWS SDK for C++ and rebuild with -DWITH_AWS_SDK=ON. "
        "Path: " + path
    );
}

std::vector<uint8_t> S3StorageReader::read_range(
    const std::string& path,
    size_t offset,
    size_t length
) {
    (void)path; (void)offset; (void)length;  // Unused in stub
    throw std::runtime_error("S3 storage not yet implemented");
}

bool S3StorageReader::exists(const std::string& path) {
    (void)path;  // Unused in stub
    throw std::runtime_error("S3 storage not yet implemented");
}

size_t S3StorageReader::size(const std::string& path) {
    (void)path;  // Unused in stub
    throw std::runtime_error("S3 storage not yet implemented");
}

// ============================================================================
// GCSStorageReader (Stub)
// ============================================================================

struct GCSStorageReader::Impl {
    Config config;
};

GCSStorageReader::GCSStorageReader(const Config& config)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->config = config;
}

GCSStorageReader::~GCSStorageReader() = default;

std::vector<uint8_t> GCSStorageReader::read(const std::string& path) {
    throw std::runtime_error(
        "GCS storage not yet implemented. "
        "Install google-cloud-cpp and rebuild. "
        "Path: " + path
    );
}

std::vector<uint8_t> GCSStorageReader::read_range(
    const std::string& path,
    size_t offset,
    size_t length
) {
    (void)path; (void)offset; (void)length;  // Unused in stub
    throw std::runtime_error("GCS storage not yet implemented");
}

bool GCSStorageReader::exists(const std::string& path) {
    (void)path;  // Unused in stub
    throw std::runtime_error("GCS storage not yet implemented");
}

size_t GCSStorageReader::size(const std::string& path) {
    (void)path;  // Unused in stub
    throw std::runtime_error("GCS storage not yet implemented");
}

// ============================================================================
// AutoStorageReader
// ============================================================================

struct AutoStorageReader::Impl {
    std::unique_ptr<LocalStorageReader> local;
    std::unique_ptr<S3StorageReader> s3;
    std::unique_ptr<GCSStorageReader> gcs;
};

AutoStorageReader::AutoStorageReader()
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->local = std::make_unique<LocalStorageReader>();
}

AutoStorageReader::~AutoStorageReader() = default;

void AutoStorageReader::configure_s3(const S3StorageReader::Config& config) {
    pimpl_->s3 = std::make_unique<S3StorageReader>(config);
}

void AutoStorageReader::configure_gcs(const GCSStorageReader::Config& config) {
    pimpl_->gcs = std::make_unique<GCSStorageReader>(config);
}

StorageReader* AutoStorageReader::get_reader(const std::string& path) {
    auto url = StorageURL::parse(path);

    if (url.scheme == "s3") {
        if (!pimpl_->s3) {
            throw std::runtime_error("S3 not configured. Call configure_s3() first.");
        }
        return pimpl_->s3.get();
    }

    if (url.scheme == "gs") {
        if (!pimpl_->gcs) {
            throw std::runtime_error("GCS not configured. Call configure_gcs() first.");
        }
        return pimpl_->gcs.get();
    }

    // Default to local filesystem
    return pimpl_->local.get();
}

std::vector<uint8_t> AutoStorageReader::read(const std::string& path) {
    return get_reader(path)->read(path);
}

std::vector<uint8_t> AutoStorageReader::read_range(
    const std::string& path,
    size_t offset,
    size_t length
) {
    return get_reader(path)->read_range(path, offset, length);
}

bool AutoStorageReader::exists(const std::string& path) {
    return get_reader(path)->exists(path);
}

size_t AutoStorageReader::size(const std::string& path) {
    return get_reader(path)->size(path);
}

// ============================================================================
// StorageURL
// ============================================================================

StorageURL StorageURL::parse(const std::string& url) {
    StorageURL result;

    // Find scheme
    auto scheme_end = url.find("://");
    if (scheme_end != std::string::npos) {
        result.scheme = url.substr(0, scheme_end);

        // Parse rest based on scheme
        std::string rest = url.substr(scheme_end + 3);

        if (result.scheme == "s3" || result.scheme == "gs") {
            // Format: s3://bucket/key or gs://bucket/key
            auto slash = rest.find('/');
            if (slash != std::string::npos) {
                result.bucket = rest.substr(0, slash);
                result.key = rest.substr(slash + 1);
            } else {
                result.bucket = rest;
            }
        } else if (result.scheme == "http" || result.scheme == "https") {
            // Format: http://host/path
            auto slash = rest.find('/');
            if (slash != std::string::npos) {
                result.host = rest.substr(0, slash);
                result.key = rest.substr(slash + 1);
            } else {
                result.host = rest;
            }
        } else if (result.scheme == "file") {
            // Format: file:///path
            result.key = rest;
        }
    } else {
        // No scheme - assume local file
        result.scheme = "file";
        result.key = url;
    }

    return result;
}

bool StorageURL::is_cloud() const {
    return scheme == "s3" || scheme == "gs" || scheme == "http" || scheme == "https";
}

bool StorageURL::is_local() const {
    return scheme == "file" || scheme.empty();
}

}  // namespace turboloader
