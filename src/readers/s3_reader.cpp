#include "turboloader/readers/storage_reader.hpp"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

namespace turboloader {

#ifdef HAVE_CURL

// AWS Signature V4 implementation
class AWS4Signer {
public:
    AWS4Signer(const std::string& access_key, const std::string& secret_key,
               const std::string& region, const std::string& service = "s3")
        : access_key_(access_key)
        , secret_key_(secret_key)
        , region_(region)
        , service_(service) {}

    std::string sign_request(const std::string& method,
                             const std::string& host,
                             const std::string& path,
                             const std::string& query = "") {
        // Get current time
        auto now = std::time(nullptr);
        struct tm* gmt = std::gmtime(&now);

        char date_stamp[9];
        char time_stamp[17];
        std::strftime(date_stamp, sizeof(date_stamp), "%Y%m%d", gmt);
        std::strftime(time_stamp, sizeof(time_stamp), "%Y%m%dT%H%M%SZ", gmt);

        // Canonical request
        std::string canonical_headers = "host:" + host + "\n" +
                                       "x-amz-date:" + time_stamp + "\n";
        std::string signed_headers = "host;x-amz-date";

        std::string canonical_request =
            method + "\n" +
            path + "\n" +
            query + "\n" +
            canonical_headers + "\n" +
            signed_headers + "\n" +
            sha256_hex("");  // Empty payload hash

        // String to sign
        std::string credential_scope = std::string(date_stamp) + "/" +
                                       region_ + "/" + service_ + "/aws4_request";

        std::string string_to_sign =
            "AWS4-HMAC-SHA256\n" +
            time_stamp + "\n" +
            credential_scope + "\n" +
            sha256_hex(canonical_request);

        // Signing key
        std::string k_date = hmac_sha256("AWS4" + secret_key_, date_stamp);
        std::string k_region = hmac_sha256(k_date, region_);
        std::string k_service = hmac_sha256(k_region, service_);
        std::string k_signing = hmac_sha256(k_service, "aws4_request");

        // Signature
        std::string signature = hmac_sha256_hex(k_signing, string_to_sign);

        // Authorization header
        std::string authorization =
            "AWS4-HMAC-SHA256 Credential=" + access_key_ + "/" + credential_scope +
            ", SignedHeaders=" + signed_headers +
            ", Signature=" + signature;

        return "Authorization: " + authorization + "\r\n" +
               "x-amz-date: " + time_stamp + "\r\n";
    }

private:
    std::string access_key_;
    std::string secret_key_;
    std::string region_;
    std::string service_;

    std::string sha256_hex(const std::string& data) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash);

        std::stringstream ss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return ss.str();
    }

    std::string hmac_sha256(const std::string& key, const std::string& data) {
        unsigned char hash[EVP_MAX_MD_SIZE];
        unsigned int hash_len;

        HMAC(EVP_sha256(),
             key.data(), key.size(),
             reinterpret_cast<const unsigned char*>(data.data()), data.size(),
             hash, &hash_len);

        return std::string(reinterpret_cast<char*>(hash), hash_len);
    }

    std::string hmac_sha256_hex(const std::string& key, const std::string& data) {
        std::string hash = hmac_sha256(key, data);

        std::stringstream ss;
        for (unsigned char c : hash) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)c;
        }
        return ss.str();
    }
};

// S3 Storage Reader implementation
class S3StorageReader : public StorageReader {
public:
    struct Config {
        std::string access_key;
        std::string secret_key;
        std::string region{"us-east-1"};
        long timeout{300};
    };

    S3StorageReader(const Config& config) : config_(config) {
        curl_ = curl_easy_init();
        if (!curl_) {
            throw std::runtime_error("Failed to initialize libcurl for S3");
        }
    }

    ~S3StorageReader() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
    }

    std::vector<uint8_t> read(const std::string& path) override {
        // Parse s3://bucket/key
        auto [bucket, key] = parse_s3_url(path);
        std::string host = bucket + ".s3." + config_.region + ".amazonaws.com";
        std::string url = "https://" + host + "/" + key;

        // Sign request
        AWS4Signer signer(config_.access_key, config_.secret_key, config_.region);
        std::string auth_headers = signer.sign_request("GET", host, "/" + key);

        // Make request
        std::vector<uint8_t> response;

        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, config_.timeout);

        // Add auth headers
        struct curl_slist* headers = nullptr;
        std::istringstream header_stream(auth_headers);
        std::string header_line;
        while (std::getline(header_stream, header_line)) {
            if (!header_line.empty() && header_line.back() == '\r') {
                header_line.pop_back();
            }
            if (!header_line.empty()) {
                headers = curl_slist_append(headers, header_line.c_str());
            }
        }

        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(headers);

        if (res != CURLE_OK) {
            throw std::runtime_error("S3 request failed: " + std::string(curl_easy_strerror(res)));
        }

        long http_code = 0;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

        if (http_code >= 400) {
            throw std::runtime_error("S3 error " + std::to_string(http_code) + " for " + path);
        }

        return response;
    }

    std::vector<uint8_t> read_range(const std::string& path,
                                     size_t offset,
                                     size_t length) override {
        // Similar to read but with Range header
        auto [bucket, key] = parse_s3_url(path);
        std::string host = bucket + ".s3." + config_.region + ".amazonaws.com";
        std::string url = "https://" + host + "/" + key;

        AWS4Signer signer(config_.access_key, config_.secret_key, config_.region);
        std::string auth_headers = signer.sign_request("GET", host, "/" + key);

        std::vector<uint8_t> response;

        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);

        struct curl_slist* headers = nullptr;
        std::istringstream header_stream(auth_headers);
        std::string header_line;
        while (std::getline(header_stream, header_line)) {
            if (!header_line.empty() && header_line.back() == '\r') {
                header_line.pop_back();
            }
            if (!header_line.empty()) {
                headers = curl_slist_append(headers, header_line.c_str());
            }
        }

        // Add range header
        std::string range = "Range: bytes=" + std::to_string(offset) + "-" +
                           std::to_string(offset + length - 1);
        headers = curl_slist_append(headers, range.c_str());

        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(headers);

        if (res != CURLE_OK) {
            throw std::runtime_error("S3 range request failed: " + std::string(curl_easy_strerror(res)));
        }

        return response;
    }

    bool exists(const std::string& path) override {
        try {
            size(path);
            return true;
        } catch (...) {
            return false;
        }
    }

    size_t size(const std::string& path) override {
        auto [bucket, key] = parse_s3_url(path);
        std::string host = bucket + ".s3." + config_.region + ".amazonaws.com";
        std::string url = "https://" + host + "/" + key;

        AWS4Signer signer(config_.access_key, config_.secret_key, config_.region);
        std::string auth_headers = signer.sign_request("HEAD", host, "/" + key);

        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_NOBODY, 1L);

        struct curl_slist* headers = nullptr;
        std::istringstream header_stream(auth_headers);
        std::string header_line;
        while (std::getline(header_stream, header_line)) {
            if (!header_line.empty() && header_line.back() == '\r') {
                header_line.pop_back();
            }
            if (!header_line.empty()) {
                headers = curl_slist_append(headers, header_line.c_str());
            }
        }

        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);

        CURLcode res = curl_easy_perform(curl_);
        curl_slist_free_all(headers);

        if (res != CURLE_OK) {
            throw std::runtime_error("S3 HEAD request failed");
        }

        curl_off_t content_length = 0;
        curl_easy_getinfo(curl_, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);

        return static_cast<size_t>(content_length);
    }

    std::string type() const override { return "s3"; }

private:
    Config config_;
    CURL* curl_;

    std::pair<std::string, std::string> parse_s3_url(const std::string& url) {
        // Parse s3://bucket/key/path
        if (url.substr(0, 5) != "s3://") {
            throw std::invalid_argument("Invalid S3 URL: " + url);
        }

        size_t bucket_start = 5;
        size_t key_start = url.find('/', bucket_start);

        if (key_start == std::string::npos) {
            throw std::invalid_argument("Invalid S3 URL (no key): " + url);
        }

        std::string bucket = url.substr(bucket_start, key_start - bucket_start);
        std::string key = url.substr(key_start + 1);

        return {bucket, key};
    }

    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t total_size = size * nmemb;
        auto* buffer = static_cast<std::vector<uint8_t>*>(userp);

        size_t old_size = buffer->size();
        buffer->resize(old_size + total_size);

        std::memcpy(buffer->data() + old_size, contents, total_size);

        return total_size;
    }
};

#endif  // HAVE_CURL

}  // namespace turboloader
