#include "turboloader/readers/http_reader.hpp"
#include <stdexcept>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

namespace turboloader {

#ifdef HAVE_CURL

// RAII wrapper for CURL handle
class CurlHandle {
public:
    CurlHandle() : curl_(curl_easy_init()) {
        if (!curl_) {
            throw std::runtime_error("Failed to initialize libcurl");
        }
    }

    ~CurlHandle() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
    }

    // Non-copyable, movable
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;

    CurlHandle(CurlHandle&& other) noexcept : curl_(other.curl_) {
        other.curl_ = nullptr;
    }

    CurlHandle& operator=(CurlHandle&& other) noexcept {
        if (this != &other) {
            if (curl_) {
                curl_easy_cleanup(curl_);
            }
            curl_ = other.curl_;
            other.curl_ = nullptr;
        }
        return *this;
    }

    CURL* get() { return curl_; }

private:
    CURL* curl_;
};

// Global libcurl initialization
struct CurlGlobal {
    CurlGlobal() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }

    ~CurlGlobal() {
        curl_global_cleanup();
    }
};

// Ensure global init happens once
static CurlGlobal global_curl_init;

// Callback for writing response data
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    auto* buffer = static_cast<std::vector<uint8_t>*>(userp);

    size_t old_size = buffer->size();
    buffer->resize(old_size + total_size);

    std::memcpy(buffer->data() + old_size, contents, total_size);

    return total_size;
}

struct HttpStorageReader::Impl {
    Config config_;
    CurlHandle curl_;

    explicit Impl(const Config& config) : config_(config) {
        // Configure curl handle with default settings
        apply_config();
    }

    void apply_config() {
        CURL* curl = curl_.get();

        // Timeouts
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, config_.connect_timeout);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, config_.request_timeout);

        // Redirects
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, config_.follow_redirects ? 1L : 0L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, config_.max_redirects);

        // SSL verification
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, config_.verify_ssl ? 1L : 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, config_.verify_ssl ? 2L : 0L);

        // User agent
        curl_easy_setopt(curl, CURLOPT_USERAGENT, config_.user_agent.c_str());

        // Verbose
        curl_easy_setopt(curl, CURLOPT_VERBOSE, config_.verbose ? 1L : 0L);

        // Enable automatic decompression (gzip, deflate, br, zstd)
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    }

    std::vector<uint8_t> get(const std::string& url, const std::string* range = nullptr) {
        CURL* curl = curl_.get();

        // Reset options that might change per request
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl, CURLOPT_NOBODY, 0L);

        // Set range if provided
        if (range) {
            curl_easy_setopt(curl, CURLOPT_RANGE, range->c_str());
        } else {
            curl_easy_setopt(curl, CURLOPT_RANGE, nullptr);
        }

        // Response buffer
        std::vector<uint8_t> response;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Perform request
        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("HTTP request failed: ") +
                                     curl_easy_strerror(res));
        }

        // Check HTTP status code
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code >= 400) {
            throw std::runtime_error("HTTP error " + std::to_string(http_code) +
                                     " for URL: " + url);
        }

        return response;
    }

    bool head(const std::string& url, size_t* out_size = nullptr) {
        CURL* curl = curl_.get();

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, nullptr);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            return false;
        }

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code >= 400) {
            return false;
        }

        // Get content length if requested
        if (out_size) {
            curl_off_t content_length = 0;
            curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
            *out_size = static_cast<size_t>(content_length);
        }

        return true;
    }
};

#else  // !HAVE_CURL

struct HttpStorageReader::Impl {
    Config config_;

    explicit Impl(const Config& config) : config_(config) {}

    void apply_config() {}

    std::vector<uint8_t> get(const std::string& url, const std::string* range = nullptr) {
        throw std::runtime_error(
            "HTTP reader not available - install libcurl and rebuild with -DHAVE_CURL");
    }

    bool head(const std::string& url, size_t* out_size = nullptr) {
        throw std::runtime_error(
            "HTTP reader not available - install libcurl and rebuild with -DHAVE_CURL");
    }
};

#endif  // HAVE_CURL

HttpStorageReader::HttpStorageReader()
    : pimpl_(std::make_unique<Impl>(Config{})) {
}

HttpStorageReader::HttpStorageReader(const Config& config)
    : pimpl_(std::make_unique<Impl>(config)) {
}

HttpStorageReader::~HttpStorageReader() = default;
HttpStorageReader::HttpStorageReader(HttpStorageReader&&) noexcept = default;
HttpStorageReader& HttpStorageReader::operator=(HttpStorageReader&&) noexcept = default;

std::vector<uint8_t> HttpStorageReader::read(const std::string& url) {
    return pimpl_->get(url);
}

std::vector<uint8_t> HttpStorageReader::read_range(const std::string& url,
                                                     size_t offset,
                                                     size_t length) {
    // HTTP range format: "bytes=offset-end"
    std::string range = std::to_string(offset) + "-" + std::to_string(offset + length - 1);
    return pimpl_->get(url, &range);
}

bool HttpStorageReader::exists(const std::string& url) {
    return pimpl_->head(url);
}

size_t HttpStorageReader::size(const std::string& url) {
    size_t file_size = 0;
    if (!pimpl_->head(url, &file_size)) {
        throw std::runtime_error("Failed to get size for URL: " + url);
    }
    return file_size;
}

const HttpStorageReader::Config& HttpStorageReader::config() const {
    return pimpl_->config_;
}

void HttpStorageReader::set_config(const Config& config) {
    pimpl_->config_ = config;
    pimpl_->apply_config();
}

}  // namespace turboloader
