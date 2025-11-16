#pragma once

#include "turboloader/readers/storage_reader.hpp"
#include <memory>
#include <string>

namespace turboloader {

/**
 * HTTP/HTTPS storage reader using libcurl
 *
 * Supports:
 * - GET requests for full file downloads
 * - Range requests for partial reads
 * - Connection reuse and pooling
 * - Automatic redirects
 * - SSL/TLS verification
 */
class HttpStorageReader : public StorageReader {
public:
    struct Config {
        // Connection timeout in seconds
        long connect_timeout{30};

        // Total request timeout in seconds
        long request_timeout{300};

        // Follow redirects
        bool follow_redirects{true};

        // Maximum redirects to follow
        long max_redirects{5};

        // Verify SSL certificates
        bool verify_ssl{true};

        // User agent string
        std::string user_agent{"TurboLoader/0.1.0"};

        // Enable verbose output for debugging
        bool verbose{false};
    };

    HttpStorageReader();
    explicit HttpStorageReader(const Config& config);
    ~HttpStorageReader() override;

    // Non-copyable but movable
    HttpStorageReader(const HttpStorageReader&) = delete;
    HttpStorageReader& operator=(const HttpStorageReader&) = delete;
    HttpStorageReader(HttpStorageReader&&) noexcept;
    HttpStorageReader& operator=(HttpStorageReader&&) noexcept;

    /**
     * Read entire file from URL
     * @param url HTTP(S) URL (e.g., "https://example.com/data.tar")
     */
    std::vector<uint8_t> read(const std::string& url) override;

    /**
     * Read byte range from URL
     * @param url HTTP(S) URL
     * @param offset Starting byte offset
     * @param length Number of bytes to read
     */
    std::vector<uint8_t> read_range(const std::string& url,
                                     size_t offset,
                                     size_t length) override;

    /**
     * Check if URL exists (HEAD request)
     */
    bool exists(const std::string& url) override;

    /**
     * Get file size from URL (HEAD request for Content-Length)
     */
    size_t size(const std::string& url) override;

    /**
     * Get configuration
     */
    const Config& config() const;

    /**
     * Update configuration
     */
    void set_config(const Config& config);

    /**
     * Get reader type
     */
    std::string type() const override { return "http"; }

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace turboloader
