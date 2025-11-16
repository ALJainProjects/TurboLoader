#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <string>
#include <memory>

namespace turboloader {

/**
 * Abstract storage reader interface
 *
 * Supports both local files and cloud storage (S3, GCS, Azure)
 */
class StorageReader {
public:
    virtual ~StorageReader() = default;

    /**
     * Read entire file
     * @param path File path or URL
     * @return File contents
     */
    virtual std::vector<uint8_t> read(const std::string& path) = 0;

    /**
     * Read file range
     * @param path File path or URL
     * @param offset Byte offset
     * @param length Number of bytes to read
     * @return File contents
     */
    virtual std::vector<uint8_t> read_range(
        const std::string& path,
        size_t offset,
        size_t length
    ) = 0;

    /**
     * Check if file exists
     */
    virtual bool exists(const std::string& path) = 0;

    /**
     * Get file size
     */
    virtual size_t size(const std::string& path) = 0;

    /**
     * Get reader type name
     */
    virtual std::string type() const = 0;
};

/**
 * Local filesystem reader
 */
class LocalStorageReader : public StorageReader {
public:
    LocalStorageReader() = default;
    ~LocalStorageReader() override = default;

    std::vector<uint8_t> read(const std::string& path) override;

    std::vector<uint8_t> read_range(
        const std::string& path,
        size_t offset,
        size_t length
    ) override;

    bool exists(const std::string& path) override;
    size_t size(const std::string& path) override;
    std::string type() const override { return "local"; }
};

/**
 * S3 storage reader (AWS S3, MinIO, etc.)
 *
 * URL format: s3://bucket/key
 */
class S3StorageReader : public StorageReader {
public:
    struct Config {
        std::string access_key;
        std::string secret_key;
        std::string region{"us-east-1"};
        std::string endpoint;  // Optional: for MinIO, etc.
        bool use_ssl{true};
    };

    explicit S3StorageReader(const Config& config);
    ~S3StorageReader() override;

    std::vector<uint8_t> read(const std::string& path) override;

    std::vector<uint8_t> read_range(
        const std::string& path,
        size_t offset,
        size_t length
    ) override;

    bool exists(const std::string& path) override;
    size_t size(const std::string& path) override;
    std::string type() const override { return "s3"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * GCS storage reader (Google Cloud Storage)
 *
 * URL format: gs://bucket/key
 */
class GCSStorageReader : public StorageReader {
public:
    struct Config {
        std::string credentials_json;  // Path to service account JSON
        std::string project_id;
    };

    explicit GCSStorageReader(const Config& config);
    ~GCSStorageReader() override;

    std::vector<uint8_t> read(const std::string& path) override;

    std::vector<uint8_t> read_range(
        const std::string& path,
        size_t offset,
        size_t length
    ) override;

    bool exists(const std::string& path) override;
    size_t size(const std::string& path) override;
    std::string type() const override { return "gcs"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * Auto-detecting storage reader
 *
 * Routes requests based on URL scheme:
 * - file:// or /path/to/file -> Local
 * - s3:// -> S3
 * - gs:// -> GCS
 * - http(s):// -> HTTP download
 */
class AutoStorageReader : public StorageReader {
public:
    AutoStorageReader();
    ~AutoStorageReader() override;

    /**
     * Configure S3 access
     */
    void configure_s3(const S3StorageReader::Config& config);

    /**
     * Configure GCS access
     */
    void configure_gcs(const GCSStorageReader::Config& config);

    std::vector<uint8_t> read(const std::string& path) override;

    std::vector<uint8_t> read_range(
        const std::string& path,
        size_t offset,
        size_t length
    ) override;

    bool exists(const std::string& path) override;
    size_t size(const std::string& path) override;
    std::string type() const override { return "auto"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    StorageReader* get_reader(const std::string& path);
};

/**
 * Parse storage URL into components
 */
struct StorageURL {
    std::string scheme;   // "file", "s3", "gs", "http", "https"
    std::string bucket;   // For cloud storage
    std::string key;      // Object key / file path
    std::string host;     // For HTTP URLs

    static StorageURL parse(const std::string& url);
    bool is_cloud() const;
    bool is_local() const;
};

}  // namespace turboloader
