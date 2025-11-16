/**
 * @file reader_orchestrator.hpp
 * @brief Unified reader orchestrator - auto-selects appropriate reader based on data source
 *
 * Architecture:
 * - Auto-detects data source from path (local file, HTTP, S3, GCS)
 * - Auto-selects appropriate reader (local file I/O, HTTPReader, S3Reader, GCSReader)
 * - Provides unified interface for all data sources
 * - Handles fallback strategies when native SDKs unavailable
 *
 * Features:
 * - Zero-configuration: just provide a path
 * - Automatic protocol detection (file://, http://, https://, s3://, gs://)
 * - Connection pooling for HTTP-based sources
 * - Range request support for all sources
 * - Retry logic with exponential backoff
 * - Thread-safe operations
 *
 * Usage:
 * ```cpp
 * ReaderOrchestrator reader;
 *
 * // Local file
 * auto data = reader.read("/path/to/file.tar");
 *
 * // HTTP/HTTPS
 * auto data = reader.read("https://example.com/dataset.tar");
 *
 * // S3
 * auto data = reader.read("s3://bucket/path/to/dataset.tar");
 *
 * // GCS
 * auto data = reader.read("gs://bucket/path/to/dataset.tar");
 * ```
 */

#pragma once

#include "http_reader.hpp"
#include "s3_reader.hpp"
#include "gcs_reader.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace turboloader {

/**
 * @brief Data source type enumeration
 */
enum class SourceType {
    LOCAL_FILE,     // Local filesystem
    HTTP,           // HTTP/HTTPS URL
    S3,             // Amazon S3 (s3://)
    GCS,            // Google Cloud Storage (gs://)
    UNKNOWN
};

/**
 * @brief Reader configuration
 */
struct ReaderConfig {
    // HTTP/S3/GCS options
    size_t max_connections = 25;
    int timeout_ms = 30000;
    int max_retries = 3;

    // S3-specific options
    std::string aws_region = "us-east-1";
    std::string aws_access_key_id;
    std::string aws_secret_access_key;
    std::string s3_endpoint_url;  // For S3-compatible storage

    // GCS-specific options
    std::string gcs_project_id;
    std::string gcs_service_account_json_path;
    std::string gcs_access_token;
    std::string gcs_endpoint_url;  // For GCS-compatible storage

    // Local file options
    bool use_mmap = true;  // Use memory-mapped I/O for large files
};

/**
 * @brief Reader response
 */
struct ReaderResponse {
    std::vector<uint8_t> data;
    std::string error_message;
    double read_time_ms = 0.0;
    size_t bytes_read = 0;
    SourceType source_type = SourceType::UNKNOWN;

    bool is_success() const { return error_message.empty(); }
};

/**
 * @brief Unified reader orchestrator
 *
 * Auto-selects the appropriate reader based on the data source URL/path.
 */
class ReaderOrchestrator {
private:
    ReaderConfig config_;
    std::unique_ptr<HTTPReader> http_reader_;

public:
    explicit ReaderOrchestrator(const ReaderConfig& config = ReaderConfig())
        : config_(config) {
        // Pre-create HTTP reader for HTTP/S3/GCS fallback
        http_reader_ = std::make_unique<HTTPReader>(config_.max_connections);
    }

    /**
     * @brief Detect source type from path/URL
     */
    static SourceType detect_source(const std::string& path) {
        if (path.find("s3://") == 0) {
            return SourceType::S3;
        } else if (path.find("gs://") == 0) {
            return SourceType::GCS;
        } else if (path.find("http://") == 0 || path.find("https://") == 0) {
            return SourceType::HTTP;
        } else if (path.find("file://") == 0 || path[0] == '/' || path[0] == '.' ||
                   (path.length() >= 2 && path[1] == ':')) {
            // file://, absolute path /, relative path ., or Windows drive C:
            return SourceType::LOCAL_FILE;
        }

        // Default to local file for unqualified paths
        return SourceType::LOCAL_FILE;
    }

    /**
     * @brief Read entire file/object from any source
     */
    bool read(const std::string& path, ReaderResponse& response) {
        auto start = std::chrono::high_resolution_clock::now();

        SourceType source_type = detect_source(path);
        response.source_type = source_type;

        bool success = false;

        switch (source_type) {
            case SourceType::LOCAL_FILE:
                success = read_local_file(path, response);
                break;

            case SourceType::HTTP:
                success = read_http(path, response);
                break;

            case SourceType::S3:
                success = read_s3(path, response);
                break;

            case SourceType::GCS:
                success = read_gcs(path, response);
                break;

            default:
                response.error_message = "Unknown source type for path: " + path;
                success = false;
        }

        auto end = std::chrono::high_resolution_clock::now();
        response.read_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return success;
    }

    /**
     * @brief Read entire file/object (convenience method)
     */
    std::vector<uint8_t> read(const std::string& path) {
        ReaderResponse response;
        if (read(path, response)) {
            return std::move(response.data);
        }
        throw std::runtime_error("Failed to read from " + path + ": " + response.error_message);
    }

    /**
     * @brief Read byte range from any source
     */
    bool read_range(const std::string& path, size_t offset, size_t size, ReaderResponse& response) {
        auto start = std::chrono::high_resolution_clock::now();

        SourceType source_type = detect_source(path);
        response.source_type = source_type;

        bool success = false;

        switch (source_type) {
            case SourceType::LOCAL_FILE:
                success = read_local_file_range(path, offset, size, response);
                break;

            case SourceType::HTTP:
                success = read_http_range(path, offset, size, response);
                break;

            case SourceType::S3:
                success = read_s3_range(path, offset, size, response);
                break;

            case SourceType::GCS:
                success = read_gcs_range(path, offset, size, response);
                break;

            default:
                response.error_message = "Unknown source type for path: " + path;
                success = false;
        }

        auto end = std::chrono::high_resolution_clock::now();
        response.read_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        return success;
    }

    /**
     * @brief Get file/object size
     */
    bool get_size(const std::string& path, size_t& size) {
        SourceType source_type = detect_source(path);

        switch (source_type) {
            case SourceType::LOCAL_FILE:
                return get_local_file_size(path, size);

            case SourceType::HTTP:
                return http_reader_->get_file_size(path, size);

            case SourceType::S3:
                return get_s3_size(path, size);

            case SourceType::GCS:
                return get_gcs_size(path, size);

            default:
                return false;
        }
    }

    /**
     * @brief Get source type for path (static utility)
     */
    static std::string source_type_name(SourceType type) {
        switch (type) {
            case SourceType::LOCAL_FILE: return "Local File";
            case SourceType::HTTP: return "HTTP/HTTPS";
            case SourceType::S3: return "Amazon S3";
            case SourceType::GCS: return "Google Cloud Storage";
            default: return "Unknown";
        }
    }

private:
    /**
     * @brief Read local file
     */
    bool read_local_file(const std::string& path, ReaderResponse& response) {
        // Remove file:// prefix if present
        std::string actual_path = path;
        if (path.find("file://") == 0) {
            actual_path = path.substr(7);
        }

        std::ifstream file(actual_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            response.error_message = "Failed to open file: " + actual_path;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        response.data.resize(size);
        if (!file.read(reinterpret_cast<char*>(response.data.data()), size)) {
            response.error_message = "Failed to read file: " + actual_path;
            return false;
        }

        response.bytes_read = size;
        return true;
    }

    /**
     * @brief Read range from local file
     */
    bool read_local_file_range(const std::string& path, size_t offset, size_t size,
                                ReaderResponse& response) {
        std::string actual_path = path;
        if (path.find("file://") == 0) {
            actual_path = path.substr(7);
        }

        std::ifstream file(actual_path, std::ios::binary);
        if (!file.is_open()) {
            response.error_message = "Failed to open file: " + actual_path;
            return false;
        }

        file.seekg(offset, std::ios::beg);

        response.data.resize(size);
        if (!file.read(reinterpret_cast<char*>(response.data.data()), size)) {
            // Handle partial reads
            std::streamsize bytes_read = file.gcount();
            response.data.resize(bytes_read);
        }

        response.bytes_read = response.data.size();
        return true;
    }

    /**
     * @brief Get local file size
     */
    bool get_local_file_size(const std::string& path, size_t& size) {
        std::string actual_path = path;
        if (path.find("file://") == 0) {
            actual_path = path.substr(7);
        }

        struct stat stat_buf;
        if (stat(actual_path.c_str(), &stat_buf) != 0) {
            return false;
        }

        size = stat_buf.st_size;
        return true;
    }

    /**
     * @brief Read from HTTP/HTTPS
     */
    bool read_http(const std::string& url, ReaderResponse& response) {
        HTTPRequest request;
        request.url = url;
        request.timeout_ms = config_.timeout_ms;
        request.max_retries = config_.max_retries;

        HTTPResponse http_response;
        if (http_reader_->fetch(request, http_response)) {
            response.data = std::move(http_response.data);
            response.bytes_read = http_response.bytes_downloaded;
            return true;
        }

        response.error_message = http_response.error_message;
        return false;
    }

    /**
     * @brief Read range from HTTP/HTTPS
     */
    bool read_http_range(const std::string& url, size_t offset, size_t size,
                         ReaderResponse& response) {
        HTTPRequest request;
        request.url = url;
        request.offset = offset;
        request.size = size;
        request.timeout_ms = config_.timeout_ms;
        request.max_retries = config_.max_retries;

        HTTPResponse http_response;
        if (http_reader_->fetch(request, http_response)) {
            response.data = std::move(http_response.data);
            response.bytes_read = http_response.bytes_downloaded;
            return true;
        }

        response.error_message = http_response.error_message;
        return false;
    }

    /**
     * @brief Parse S3 URL into bucket and key
     */
    bool parse_s3_url(const std::string& url, std::string& bucket, std::string& key) {
        if (url.find("s3://") != 0) {
            return false;
        }

        std::string path = url.substr(5);  // Remove "s3://"
        size_t slash_pos = path.find('/');

        if (slash_pos == std::string::npos) {
            bucket = path;
            key = "";
        } else {
            bucket = path.substr(0, slash_pos);
            key = path.substr(slash_pos + 1);
        }

        return !bucket.empty();
    }

    /**
     * @brief Read from S3
     */
    bool read_s3(const std::string& url, ReaderResponse& response) {
        std::string bucket, key;
        if (!parse_s3_url(url, bucket, key)) {
            response.error_message = "Invalid S3 URL: " + url;
            return false;
        }

        S3Config s3_config;
        s3_config.bucket = bucket;
        s3_config.key = key;
        s3_config.region = config_.aws_region;
        s3_config.access_key_id = config_.aws_access_key_id;
        s3_config.secret_access_key = config_.aws_secret_access_key;
        s3_config.endpoint_url = config_.s3_endpoint_url;
        s3_config.timeout_ms = config_.timeout_ms;
        s3_config.max_connections = config_.max_connections;

        S3Reader s3_reader(s3_config);

        S3Request request;
        request.config = s3_config;
        request.max_retries = config_.max_retries;

        S3Response s3_response;
        if (s3_reader.fetch(request, s3_response)) {
            response.data = std::move(s3_response.data);
            response.bytes_read = s3_response.bytes_downloaded;
            return true;
        }

        response.error_message = s3_response.error_message;
        return false;
    }

    /**
     * @brief Read range from S3
     */
    bool read_s3_range(const std::string& url, size_t offset, size_t size,
                       ReaderResponse& response) {
        std::string bucket, key;
        if (!parse_s3_url(url, bucket, key)) {
            response.error_message = "Invalid S3 URL: " + url;
            return false;
        }

        S3Config s3_config;
        s3_config.bucket = bucket;
        s3_config.key = key;
        s3_config.region = config_.aws_region;
        s3_config.access_key_id = config_.aws_access_key_id;
        s3_config.secret_access_key = config_.aws_secret_access_key;
        s3_config.endpoint_url = config_.s3_endpoint_url;
        s3_config.timeout_ms = config_.timeout_ms;
        s3_config.max_connections = config_.max_connections;

        S3Reader s3_reader(s3_config);

        S3Request request;
        request.config = s3_config;
        request.offset = offset;
        request.size = size;
        request.max_retries = config_.max_retries;

        S3Response s3_response;
        if (s3_reader.fetch(request, s3_response)) {
            response.data = std::move(s3_response.data);
            response.bytes_read = s3_response.bytes_downloaded;
            return true;
        }

        response.error_message = s3_response.error_message;
        return false;
    }

    /**
     * @brief Get S3 object size
     */
    bool get_s3_size(const std::string& url, size_t& size) {
        std::string bucket, key;
        if (!parse_s3_url(url, bucket, key)) {
            return false;
        }

        S3Config s3_config;
        s3_config.bucket = bucket;
        s3_config.key = key;
        s3_config.region = config_.aws_region;
        s3_config.access_key_id = config_.aws_access_key_id;
        s3_config.secret_access_key = config_.aws_secret_access_key;
        s3_config.endpoint_url = config_.s3_endpoint_url;

        S3Reader s3_reader(s3_config);
        return s3_reader.get_object_size(s3_config, size);
    }

    /**
     * @brief Parse GCS URL into bucket and object
     */
    bool parse_gcs_url(const std::string& url, std::string& bucket, std::string& object) {
        if (url.find("gs://") != 0) {
            return false;
        }

        std::string path = url.substr(5);  // Remove "gs://"
        size_t slash_pos = path.find('/');

        if (slash_pos == std::string::npos) {
            bucket = path;
            object = "";
        } else {
            bucket = path.substr(0, slash_pos);
            object = path.substr(slash_pos + 1);
        }

        return !bucket.empty();
    }

    /**
     * @brief Read from GCS
     */
    bool read_gcs(const std::string& url, ReaderResponse& response) {
        std::string bucket, object;
        if (!parse_gcs_url(url, bucket, object)) {
            response.error_message = "Invalid GCS URL: " + url;
            return false;
        }

        GCSConfig gcs_config;
        gcs_config.bucket = bucket;
        gcs_config.object = object;
        gcs_config.project_id = config_.gcs_project_id;
        gcs_config.service_account_json_path = config_.gcs_service_account_json_path;
        gcs_config.access_token = config_.gcs_access_token;
        gcs_config.endpoint_url = config_.gcs_endpoint_url;
        gcs_config.timeout_ms = config_.timeout_ms;
        gcs_config.max_connections = config_.max_connections;

        GCSReader gcs_reader(gcs_config);

        GCSRequest request;
        request.config = gcs_config;
        request.max_retries = config_.max_retries;

        GCSResponse gcs_response;
        if (gcs_reader.fetch(request, gcs_response)) {
            response.data = std::move(gcs_response.data);
            response.bytes_read = gcs_response.bytes_downloaded;
            return true;
        }

        response.error_message = gcs_response.error_message;
        return false;
    }

    /**
     * @brief Read range from GCS
     */
    bool read_gcs_range(const std::string& url, size_t offset, size_t size,
                        ReaderResponse& response) {
        std::string bucket, object;
        if (!parse_gcs_url(url, bucket, object)) {
            response.error_message = "Invalid GCS URL: " + url;
            return false;
        }

        GCSConfig gcs_config;
        gcs_config.bucket = bucket;
        gcs_config.object = object;
        gcs_config.project_id = config_.gcs_project_id;
        gcs_config.service_account_json_path = config_.gcs_service_account_json_path;
        gcs_config.access_token = config_.gcs_access_token;
        gcs_config.endpoint_url = config_.gcs_endpoint_url;
        gcs_config.timeout_ms = config_.timeout_ms;
        gcs_config.max_connections = config_.max_connections;

        GCSReader gcs_reader(gcs_config);

        GCSRequest request;
        request.config = gcs_config;
        request.offset = offset;
        request.size = size;
        request.max_retries = config_.max_retries;

        GCSResponse gcs_response;
        if (gcs_reader.fetch(request, gcs_response)) {
            response.data = std::move(gcs_response.data);
            response.bytes_read = gcs_response.bytes_downloaded;
            return true;
        }

        response.error_message = gcs_response.error_message;
        return false;
    }

    /**
     * @brief Get GCS object size
     */
    bool get_gcs_size(const std::string& url, size_t& size) {
        std::string bucket, object;
        if (!parse_gcs_url(url, bucket, object)) {
            return false;
        }

        GCSConfig gcs_config;
        gcs_config.bucket = bucket;
        gcs_config.object = object;
        gcs_config.project_id = config_.gcs_project_id;
        gcs_config.service_account_json_path = config_.gcs_service_account_json_path;
        gcs_config.access_token = config_.gcs_access_token;
        gcs_config.endpoint_url = config_.gcs_endpoint_url;

        GCSReader gcs_reader(gcs_config);
        return gcs_reader.get_object_size(gcs_config, size);
    }
};

}  // namespace turboloader
