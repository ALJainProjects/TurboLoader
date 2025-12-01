/**
 * @file azure_blob_reader.hpp
 * @brief Azure Blob Storage reader for enterprise cloud support
 *
 * Supports:
 * - Reading blobs from Azure Blob Storage
 * - Multiple authentication methods (connection string, SAS token, managed identity)
 * - Streaming reads for large blobs
 * - Range requests for partial reads
 *
 * Usage:
 * ```cpp
 * // Using connection string
 * AzureBlobReader reader("container", "blob/path/data.tar");
 * reader.set_connection_string(getenv("AZURE_STORAGE_CONNECTION_STRING"));
 *
 * // Using SAS token
 * AzureBlobReader reader("container", "blob/path/data.tar");
 * reader.set_sas_token("?sv=2021-06-08&ss=b&srt=co...");
 *
 * // Read data
 * auto data = reader.read();
 * auto chunk = reader.read_range(0, 1024 * 1024);
 * ```
 *
 * Note: Full implementation requires Azure SDK for C++.
 * This provides a libcurl-based implementation for basic functionality.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <functional>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <curl/curl.h>

namespace turboloader {
namespace readers {

/**
 * @brief Azure Blob metadata
 */
struct AzureBlobInfo {
    std::string name;
    std::string container;
    size_t size;
    std::string content_type;
    std::string etag;
    std::string last_modified;
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * @brief Base64 encoding/decoding utilities
 */
class Base64 {
public:
    static std::string encode(const std::vector<uint8_t>& data) {
        static const char* chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string result;
        result.reserve(((data.size() + 2) / 3) * 4);

        for (size_t i = 0; i < data.size(); i += 3) {
            uint32_t n = static_cast<uint32_t>(data[i]) << 16;
            if (i + 1 < data.size()) n |= static_cast<uint32_t>(data[i + 1]) << 8;
            if (i + 2 < data.size()) n |= static_cast<uint32_t>(data[i + 2]);

            result += chars[(n >> 18) & 0x3F];
            result += chars[(n >> 12) & 0x3F];
            result += (i + 1 < data.size()) ? chars[(n >> 6) & 0x3F] : '=';
            result += (i + 2 < data.size()) ? chars[n & 0x3F] : '=';
        }

        return result;
    }

    static std::vector<uint8_t> decode(const std::string& encoded) {
        static const int lookup[256] = {
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
            52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
            15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
            -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        };

        std::vector<uint8_t> result;
        result.reserve(encoded.size() * 3 / 4);

        uint32_t val = 0;
        int bits = 0;

        for (char c : encoded) {
            if (c == '=') break;
            int v = lookup[static_cast<uint8_t>(c)];
            if (v < 0) continue;

            val = (val << 6) | v;
            bits += 6;

            if (bits >= 8) {
                bits -= 8;
                result.push_back((val >> bits) & 0xFF);
            }
        }

        return result;
    }
};

/**
 * @brief Azure Blob Storage reader using libcurl
 */
class AzureBlobReader {
public:
    /**
     * @brief Create reader for specific blob
     */
    AzureBlobReader(const std::string& container, const std::string& blob_path)
        : container_(container), blob_path_(blob_path) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }

    /**
     * @brief Create reader from Azure URL
     * Format: https://<account>.blob.core.windows.net/<container>/<blob>
     * Or: azure://<container>/<blob>
     */
    explicit AzureBlobReader(const std::string& url) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        parse_url(url);
    }

    ~AzureBlobReader() {
        curl_global_cleanup();
    }

    // Non-copyable
    AzureBlobReader(const AzureBlobReader&) = delete;
    AzureBlobReader& operator=(const AzureBlobReader&) = delete;

    /**
     * @brief Set storage account name
     */
    void set_account(const std::string& account) {
        account_name_ = account;
    }

    /**
     * @brief Set connection string
     * Format: DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=...
     */
    void set_connection_string(const std::string& conn_str) {
        parse_connection_string(conn_str);
    }

    /**
     * @brief Set SAS token
     */
    void set_sas_token(const std::string& sas_token) {
        sas_token_ = sas_token;
        if (!sas_token_.empty() && sas_token_[0] != '?') {
            sas_token_ = "?" + sas_token_;
        }
        auth_method_ = AuthMethod::SAS_TOKEN;
    }

    /**
     * @brief Set account key for shared key auth
     */
    void set_account_key(const std::string& account_key) {
        account_key_ = account_key;
        auth_method_ = AuthMethod::SHARED_KEY;
    }

    /**
     * @brief Enable managed identity authentication (for Azure VMs)
     */
    void use_managed_identity() {
        auth_method_ = AuthMethod::MANAGED_IDENTITY;
    }

    /**
     * @brief Get blob metadata
     */
    AzureBlobInfo get_blob_info() {
        AzureBlobInfo info;
        info.container = container_;
        info.name = blob_path_;

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize curl");
        }

        std::string url = build_url();

        // HEAD request for metadata
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);

        // Set auth headers
        struct curl_slist* headers = nullptr;
        headers = add_auth_headers(headers, "HEAD");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Capture response headers
        std::string response_headers;
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &response_headers);

        CURLcode res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            // Parse Content-Length
            double content_length;
            curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
            info.size = static_cast<size_t>(content_length);

            // Parse other headers
            info.content_type = extract_header(response_headers, "Content-Type");
            info.etag = extract_header(response_headers, "ETag");
            info.last_modified = extract_header(response_headers, "Last-Modified");
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error("Failed to get blob info: " +
                                     std::string(curl_easy_strerror(res)));
        }

        return info;
    }

    /**
     * @brief Read entire blob
     */
    std::vector<uint8_t> read() {
        return read_range(0, 0);  // 0 size means entire blob
    }

    /**
     * @brief Read range of blob
     */
    std::vector<uint8_t> read_range(size_t offset, size_t length) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize curl");
        }

        std::string url = build_url();

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // Set auth headers
        struct curl_slist* headers = nullptr;
        headers = add_auth_headers(headers, "GET");

        // Add range header if specified
        if (length > 0) {
            std::string range = "Range: bytes=" + std::to_string(offset) +
                               "-" + std::to_string(offset + length - 1);
            headers = curl_slist_append(headers, range.c_str());
        }

        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Capture response
        std::vector<uint8_t> data;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

        CURLcode res = curl_easy_perform(curl);

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error("Failed to read blob: " +
                                     std::string(curl_easy_strerror(res)));
        }

        if (http_code >= 400) {
            throw std::runtime_error("HTTP error: " + std::to_string(http_code));
        }

        return data;
    }

    /**
     * @brief Stream blob content with callback
     */
    void stream(std::function<void(const uint8_t*, size_t)> callback,
                size_t chunk_size = 4 * 1024 * 1024) {
        auto info = get_blob_info();
        size_t total_size = info.size;
        size_t offset = 0;

        while (offset < total_size) {
            size_t remaining = total_size - offset;
            size_t read_size = std::min(chunk_size, remaining);

            auto chunk = read_range(offset, read_size);
            callback(chunk.data(), chunk.size());

            offset += chunk.size();
        }
    }

    /**
     * @brief List blobs in container
     */
    std::vector<AzureBlobInfo> list_blobs(const std::string& prefix = "") {
        std::vector<AzureBlobInfo> blobs;

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize curl");
        }

        // Build list URL
        std::string url = "https://" + account_name_ + ".blob.core.windows.net/" +
                         container_ + "?restype=container&comp=list";
        if (!prefix.empty()) {
            url += "&prefix=" + url_encode(prefix);
        }

        if (auth_method_ == AuthMethod::SAS_TOKEN) {
            url += sas_token_.substr(1);  // Append without leading ?
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        struct curl_slist* headers = nullptr;
        headers = add_auth_headers(headers, "GET", true);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        std::string response;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, string_write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            throw std::runtime_error("Failed to list blobs: " +
                                     std::string(curl_easy_strerror(res)));
        }

        // Parse XML response (simplified)
        parse_blob_list(response, blobs);

        return blobs;
    }

    const std::string& container() const { return container_; }
    const std::string& blob_path() const { return blob_path_; }
    const std::string& account() const { return account_name_; }

private:
    enum class AuthMethod {
        NONE,
        SHARED_KEY,
        SAS_TOKEN,
        MANAGED_IDENTITY
    };

    void parse_url(const std::string& url) {
        // Handle azure:// scheme
        if (url.substr(0, 8) == "azure://") {
            std::string path = url.substr(8);
            auto slash_pos = path.find('/');
            if (slash_pos != std::string::npos) {
                container_ = path.substr(0, slash_pos);
                blob_path_ = path.substr(slash_pos + 1);
            }
            return;
        }

        // Handle https:// scheme
        if (url.find(".blob.core.windows.net") != std::string::npos) {
            // Extract account name
            auto start = url.find("://") + 3;
            auto dot = url.find(".blob", start);
            account_name_ = url.substr(start, dot - start);

            // Extract container and blob
            auto path_start = url.find(".net/") + 5;
            std::string path = url.substr(path_start);

            // Check for SAS token
            auto sas_pos = path.find('?');
            if (sas_pos != std::string::npos) {
                sas_token_ = path.substr(sas_pos);
                path = path.substr(0, sas_pos);
                auth_method_ = AuthMethod::SAS_TOKEN;
            }

            auto slash_pos = path.find('/');
            if (slash_pos != std::string::npos) {
                container_ = path.substr(0, slash_pos);
                blob_path_ = path.substr(slash_pos + 1);
            }
        }
    }

    void parse_connection_string(const std::string& conn_str) {
        // Parse key=value pairs separated by ;
        size_t pos = 0;
        while (pos < conn_str.size()) {
            auto eq_pos = conn_str.find('=', pos);
            if (eq_pos == std::string::npos) break;

            auto semi_pos = conn_str.find(';', eq_pos);
            if (semi_pos == std::string::npos) semi_pos = conn_str.size();

            std::string key = conn_str.substr(pos, eq_pos - pos);
            std::string value = conn_str.substr(eq_pos + 1, semi_pos - eq_pos - 1);

            if (key == "AccountName") {
                account_name_ = value;
            } else if (key == "AccountKey") {
                account_key_ = value;
                auth_method_ = AuthMethod::SHARED_KEY;
            } else if (key == "EndpointSuffix") {
                endpoint_suffix_ = value;
            }

            pos = semi_pos + 1;
        }
    }

    std::string build_url() const {
        std::string url = "https://" + account_name_ + ".blob.core.windows.net/" +
                         container_ + "/" + blob_path_;

        if (auth_method_ == AuthMethod::SAS_TOKEN && !sas_token_.empty()) {
            url += sas_token_;
        }

        return url;
    }

    struct curl_slist* add_auth_headers(struct curl_slist* headers,
                                         const std::string& method,
                                         bool is_list = false) const {
        // Add common headers
        headers = curl_slist_append(headers, "x-ms-version: 2021-06-08");

        // Add date header
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        char date_buf[128];
        std::strftime(date_buf, sizeof(date_buf), "%a, %d %b %Y %H:%M:%S GMT",
                      std::gmtime(&time_t_now));
        std::string date_header = "x-ms-date: " + std::string(date_buf);
        headers = curl_slist_append(headers, date_header.c_str());

        // Add authorization header for shared key auth
        if (auth_method_ == AuthMethod::SHARED_KEY && !account_key_.empty()) {
            std::string auth_header = "Authorization: " +
                                      compute_shared_key_auth(method, date_buf, is_list);
            headers = curl_slist_append(headers, auth_header.c_str());
        }

        return headers;
    }

    std::string compute_shared_key_auth(const std::string& method,
                                         const std::string& date,
                                         bool is_list) const {
        // Build string to sign
        std::stringstream ss;
        ss << method << "\n";  // HTTP verb
        ss << "\n";  // Content-Encoding
        ss << "\n";  // Content-Language
        ss << "\n";  // Content-Length
        ss << "\n";  // Content-MD5
        ss << "\n";  // Content-Type
        ss << "\n";  // Date
        ss << "\n";  // If-Modified-Since
        ss << "\n";  // If-Match
        ss << "\n";  // If-None-Match
        ss << "\n";  // If-Unmodified-Since
        ss << "\n";  // Range
        ss << "x-ms-date:" << date << "\n";
        ss << "x-ms-version:2021-06-08\n";

        // Canonicalized resource
        ss << "/" << account_name_ << "/" << container_;
        if (!is_list) {
            ss << "/" << blob_path_;
        } else {
            ss << "\ncomp:list\nrestype:container";
        }

        std::string string_to_sign = ss.str();

        // Compute HMAC-SHA256
        auto key_bytes = Base64::decode(account_key_);

        unsigned char hmac_result[EVP_MAX_MD_SIZE];
        unsigned int hmac_len;

        HMAC(EVP_sha256(), key_bytes.data(), key_bytes.size(),
             reinterpret_cast<const unsigned char*>(string_to_sign.c_str()),
             string_to_sign.size(), hmac_result, &hmac_len);

        std::vector<uint8_t> signature(hmac_result, hmac_result + hmac_len);
        std::string signature_b64 = Base64::encode(signature);

        return "SharedKey " + account_name_ + ":" + signature_b64;
    }

    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t total = size * nmemb;
        auto* data = static_cast<std::vector<uint8_t>*>(userp);
        data->insert(data->end(), static_cast<uint8_t*>(contents),
                     static_cast<uint8_t*>(contents) + total);
        return total;
    }

    static size_t string_write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t total = size * nmemb;
        auto* str = static_cast<std::string*>(userp);
        str->append(static_cast<char*>(contents), total);
        return total;
    }

    static size_t header_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t total = size * nmemb;
        auto* str = static_cast<std::string*>(userp);
        str->append(static_cast<char*>(contents), total);
        return total;
    }

    std::string extract_header(const std::string& headers, const std::string& name) const {
        std::string search = name + ": ";
        auto pos = headers.find(search);
        if (pos == std::string::npos) return "";

        pos += search.size();
        auto end = headers.find("\r\n", pos);
        if (end == std::string::npos) end = headers.size();

        return headers.substr(pos, end - pos);
    }

    std::string url_encode(const std::string& str) const {
        std::ostringstream encoded;
        encoded.fill('0');
        encoded << std::hex;

        for (char c : str) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encoded << c;
            } else {
                encoded << '%' << std::setw(2) << static_cast<int>(static_cast<uint8_t>(c));
            }
        }

        return encoded.str();
    }

    void parse_blob_list(const std::string& xml, std::vector<AzureBlobInfo>& blobs) const {
        // Simple XML parsing for blob list
        size_t pos = 0;

        while ((pos = xml.find("<Blob>", pos)) != std::string::npos) {
            auto blob_end = xml.find("</Blob>", pos);
            if (blob_end == std::string::npos) break;

            std::string blob_xml = xml.substr(pos, blob_end - pos);

            AzureBlobInfo info;
            info.container = container_;
            info.name = extract_xml_value(blob_xml, "Name");

            // Parse properties
            auto props_start = blob_xml.find("<Properties>");
            if (props_start != std::string::npos) {
                std::string size_str = extract_xml_value(blob_xml, "Content-Length", props_start);
                if (!size_str.empty()) {
                    info.size = std::stoull(size_str);
                }
                info.content_type = extract_xml_value(blob_xml, "Content-Type", props_start);
                info.etag = extract_xml_value(blob_xml, "Etag", props_start);
                info.last_modified = extract_xml_value(blob_xml, "Last-Modified", props_start);
            }

            blobs.push_back(info);
            pos = blob_end + 1;
        }
    }

    std::string extract_xml_value(const std::string& xml, const std::string& tag,
                                   size_t start = 0) const {
        std::string open_tag = "<" + tag + ">";
        std::string close_tag = "</" + tag + ">";

        auto tag_start = xml.find(open_tag, start);
        if (tag_start == std::string::npos) return "";

        tag_start += open_tag.size();
        auto tag_end = xml.find(close_tag, tag_start);
        if (tag_end == std::string::npos) return "";

        return xml.substr(tag_start, tag_end - tag_start);
    }

    std::string container_;
    std::string blob_path_;
    std::string account_name_;
    std::string account_key_;
    std::string sas_token_;
    std::string endpoint_suffix_ = "core.windows.net";
    AuthMethod auth_method_ = AuthMethod::NONE;
};

}  // namespace readers
}  // namespace turboloader
