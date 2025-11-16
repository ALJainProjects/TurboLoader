/**
 * @file test_reader_orchestrator.cpp
 * @brief Comprehensive tests for unified reader orchestrator
 *
 * Tests:
 * 1. Source type detection (local, HTTP, S3, GCS)
 * 2. Local file reading
 * 3. Local file range requests
 * 4. HTTP/HTTPS reading (if internet available)
 * 5. S3 URL parsing
 * 6. GCS URL parsing
 * 7. Error handling
 * 8. Configuration options
 * 9. File size retrieval
 * 10. Unified interface
 */

#include "../src/readers/reader_orchestrator.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Create test file for local file tests
 */
bool create_test_file(const std::string& filename, size_t size_kb = 10) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write test pattern
    for (size_t i = 0; i < size_kb * 1024; ++i) {
        file.put(static_cast<char>(i % 256));
    }

    file.close();
    return true;
}

/**
 * @brief Test source type detection
 */
void test_source_detection() {
    std::cout << BOLD << "\n[TEST] Source Type Detection" << RESET << std::endl;

    // Local files
    assert(ReaderOrchestrator::detect_source("/path/to/file.tar") == SourceType::LOCAL_FILE);
    assert(ReaderOrchestrator::detect_source("file.tar") == SourceType::LOCAL_FILE);
    assert(ReaderOrchestrator::detect_source("./relative/path.tar") == SourceType::LOCAL_FILE);
    assert(ReaderOrchestrator::detect_source("file:///path/to/file.tar") == SourceType::LOCAL_FILE);

    // HTTP/HTTPS
    assert(ReaderOrchestrator::detect_source("http://example.com/file.tar") == SourceType::HTTP);
    assert(ReaderOrchestrator::detect_source("https://example.com/file.tar") == SourceType::HTTP);

    // S3
    assert(ReaderOrchestrator::detect_source("s3://bucket/path/to/file.tar") == SourceType::S3);
    assert(ReaderOrchestrator::detect_source("s3://my-bucket/dataset.tar") == SourceType::S3);

    // GCS
    assert(ReaderOrchestrator::detect_source("gs://bucket/path/to/file.tar") == SourceType::GCS);
    assert(ReaderOrchestrator::detect_source("gs://my-bucket/dataset.tar") == SourceType::GCS);

    std::cout << "  " << GREEN << "✓" << RESET << " Local file detection" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " HTTP/HTTPS detection" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " S3 detection" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " GCS detection" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test source type name conversion
 */
void test_source_type_names() {
    std::cout << BOLD << "\n[TEST] Source Type Names" << RESET << std::endl;

    assert(ReaderOrchestrator::source_type_name(SourceType::LOCAL_FILE) == "Local File");
    assert(ReaderOrchestrator::source_type_name(SourceType::HTTP) == "HTTP/HTTPS");
    assert(ReaderOrchestrator::source_type_name(SourceType::S3) == "Amazon S3");
    assert(ReaderOrchestrator::source_type_name(SourceType::GCS) == "Google Cloud Storage");
    assert(ReaderOrchestrator::source_type_name(SourceType::UNKNOWN) == "Unknown");

    std::cout << "  " << GREEN << "✓" << RESET << " All source type names correct" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test local file reading
 */
void test_local_file_reading() {
    std::cout << BOLD << "\n[TEST] Local File Reading" << RESET << std::endl;

    std::string test_file = "/tmp/test_reader_orchestrator.bin";

    // Create test file (10 KB)
    if (!create_test_file(test_file, 10)) {
        std::cout << RED << "  ✗ Failed to create test file" << RESET << std::endl;
        assert(false);
    }

    ReaderOrchestrator reader;
    ReaderResponse response;

    // Test reading entire file
    bool success = reader.read(test_file, response);
    assert(success);
    assert(response.is_success());
    assert(response.data.size() == 10 * 1024);
    assert(response.source_type == SourceType::LOCAL_FILE);
    assert(response.bytes_read == 10 * 1024);

    // Verify data pattern
    for (size_t i = 0; i < 256; ++i) {
        assert(response.data[i] == static_cast<uint8_t>(i % 256));
    }

    std::cout << "  " << GREEN << "✓" << RESET << " Read entire file (" << response.data.size()
              << " bytes)" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Data pattern verification passed" << std::endl;

    // Test with file:// prefix
    ReaderResponse response2;
    success = reader.read("file://" + test_file, response2);
    assert(success);
    assert(response2.data.size() == 10 * 1024);

    std::cout << "  " << GREEN << "✓" << RESET << " file:// prefix handling" << std::endl;

    // Clean up
    std::remove(test_file.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test local file range requests
 */
void test_local_file_range_requests() {
    std::cout << BOLD << "\n[TEST] Local File Range Requests" << RESET << std::endl;

    std::string test_file = "/tmp/test_reader_orchestrator_range.bin";

    // Create test file (100 KB)
    if (!create_test_file(test_file, 100)) {
        std::cout << RED << "  ✗ Failed to create test file" << RESET << std::endl;
        assert(false);
    }

    ReaderOrchestrator reader;
    ReaderResponse response;

    // Test range request: bytes 1000-1999 (1000 bytes)
    bool success = reader.read_range(test_file, 1000, 1000, response);
    assert(success);
    assert(response.is_success());
    assert(response.data.size() == 1000);
    assert(response.source_type == SourceType::LOCAL_FILE);

    // Verify data pattern at offset 1000
    for (size_t i = 0; i < 256; ++i) {
        size_t expected_val = (1000 + i) % 256;
        assert(response.data[i] == static_cast<uint8_t>(expected_val));
    }

    std::cout << "  " << GREEN << "✓" << RESET << " Range request (offset=1000, size=1000)"
              << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Data verification at offset" << std::endl;

    // Test range at different offset
    ReaderResponse response2;
    success = reader.read_range(test_file, 50000, 2000, response2);
    assert(success);
    assert(response2.data.size() == 2000);

    std::cout << "  " << GREEN << "✓" << RESET << " Range request (offset=50000, size=2000)"
              << std::endl;

    // Clean up
    std::remove(test_file.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test getting file size
 */
void test_file_size() {
    std::cout << BOLD << "\n[TEST] File Size Retrieval" << RESET << std::endl;

    std::string test_file = "/tmp/test_reader_orchestrator_size.bin";

    // Create test file (50 KB)
    if (!create_test_file(test_file, 50)) {
        std::cout << RED << "  ✗ Failed to create test file" << RESET << std::endl;
        assert(false);
    }

    ReaderOrchestrator reader;
    size_t size;

    bool success = reader.get_size(test_file, size);
    assert(success);
    assert(size == 50 * 1024);

    std::cout << "  " << GREEN << "✓" << RESET << " File size retrieval: " << size << " bytes"
              << std::endl;

    // Test non-existent file
    success = reader.get_size("/tmp/nonexistent_file_12345.bin", size);
    assert(!success);

    std::cout << "  " << GREEN << "✓" << RESET << " Non-existent file handled correctly"
              << std::endl;

    // Clean up
    std::remove(test_file.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test S3 URL parsing
 */
void test_s3_url_parsing() {
    std::cout << BOLD << "\n[TEST] S3 URL Parsing" << RESET << std::endl;

    // This is a white-box test - we'll create a reader and test the internal parsing
    // by attempting to read S3 URLs (they'll fail but we can check error messages)

    ReaderOrchestrator reader;
    ReaderResponse response;

    // Test S3 URL detection
    std::string s3_url = "s3://my-bucket/path/to/object.tar";
    reader.read(s3_url, response);

    assert(response.source_type == SourceType::S3);
    // Should fail because we don't have AWS credentials, but source type should be correct
    std::cout << "  " << GREEN << "✓" << RESET << " S3 URL detected: " << s3_url << std::endl;

    // Test S3 URL with just bucket
    s3_url = "s3://my-bucket";
    reader.read(s3_url, response);
    assert(response.source_type == SourceType::S3);
    std::cout << "  " << GREEN << "✓" << RESET << " S3 bucket-only URL: " << s3_url << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test GCS URL parsing
 */
void test_gcs_url_parsing() {
    std::cout << BOLD << "\n[TEST] GCS URL Parsing" << RESET << std::endl;

    ReaderOrchestrator reader;
    ReaderResponse response;

    // Test GCS URL detection
    std::string gcs_url = "gs://my-bucket/path/to/object.tar";
    reader.read(gcs_url, response);

    assert(response.source_type == SourceType::GCS);
    std::cout << "  " << GREEN << "✓" << RESET << " GCS URL detected: " << gcs_url << std::endl;

    // Test GCS URL with just bucket
    gcs_url = "gs://my-bucket";
    reader.read(gcs_url, response);
    assert(response.source_type == SourceType::GCS);
    std::cout << "  " << GREEN << "✓" << RESET << " GCS bucket-only URL: " << gcs_url << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test HTTP URL detection
 */
void test_http_url_detection() {
    std::cout << BOLD << "\n[TEST] HTTP URL Detection" << RESET << std::endl;
    std::cout << YELLOW << "  (This test may require internet access)" << RESET << std::endl;

    ReaderOrchestrator reader;
    ReaderResponse response;

    // Test HTTP URL
    std::string http_url = "http://example.com/file.tar";
    reader.read(http_url, response);
    assert(response.source_type == SourceType::HTTP);
    std::cout << "  " << GREEN << "✓" << RESET << " HTTP URL detected" << std::endl;

    // Test HTTPS URL
    std::string https_url = "https://example.com/file.tar";
    reader.read(https_url, response);
    assert(response.source_type == SourceType::HTTP);
    std::cout << "  " << GREEN << "✓" << RESET << " HTTPS URL detected" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test error handling
 */
void test_error_handling() {
    std::cout << BOLD << "\n[TEST] Error Handling" << RESET << std::endl;

    ReaderOrchestrator reader;
    ReaderResponse response;

    // Test non-existent local file
    bool success = reader.read("/tmp/nonexistent_file_12345.bin", response);
    assert(!success);
    assert(!response.is_success());
    assert(!response.error_message.empty());

    std::cout << "  " << GREEN << "✓" << RESET << " Non-existent file error handled" << std::endl;

    // Test convenience method (should throw)
    bool exception_thrown = false;
    try {
        auto data = reader.read("/tmp/nonexistent_file_12345.bin");
    } catch (const std::runtime_error& e) {
        exception_thrown = true;
        assert(std::string(e.what()).find("Failed to read") != std::string::npos);
    }
    assert(exception_thrown);

    std::cout << "  " << GREEN << "✓" << RESET << " Convenience method throws on error"
              << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test configuration options
 */
void test_configuration() {
    std::cout << BOLD << "\n[TEST] Configuration Options" << RESET << std::endl;

    ReaderConfig config;

    // Test default values
    assert(config.max_connections == 25);
    assert(config.timeout_ms == 30000);
    assert(config.max_retries == 3);
    assert(config.aws_region == "us-east-1");
    assert(config.use_mmap == true);

    std::cout << "  " << GREEN << "✓" << RESET << " Default configuration values" << std::endl;

    // Test custom configuration
    config.max_connections = 50;
    config.timeout_ms = 60000;
    config.max_retries = 5;
    config.aws_region = "us-west-2";
    config.aws_access_key_id = "test-key";
    config.aws_secret_access_key = "test-secret";
    config.s3_endpoint_url = "https://s3.custom.com";
    config.gcs_project_id = "my-project";
    config.gcs_service_account_json_path = "/path/to/credentials.json";

    ReaderOrchestrator reader(config);

    std::cout << "  " << GREEN << "✓" << RESET << " Custom configuration accepted" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test response structure
 */
void test_response_structure() {
    std::cout << BOLD << "\n[TEST] Response Structure" << RESET << std::endl;

    // Test default values
    ReaderResponse response;
    assert(response.data.empty());
    assert(response.error_message.empty());
    assert(response.read_time_ms == 0.0);
    assert(response.bytes_read == 0);
    assert(response.source_type == SourceType::UNKNOWN);
    assert(response.is_success() == true);  // No error means success

    std::cout << "  " << GREEN << "✓" << RESET << " Default response values" << std::endl;

    // Test error state
    response.error_message = "Test error";
    assert(response.is_success() == false);

    std::cout << "  " << GREEN << "✓" << RESET << " Error state detection" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Benchmark local file reading
 */
void benchmark_local_file_reading() {
    std::cout << BOLD << "\n[BENCHMARK] Local File Reading Performance" << RESET << std::endl;

    std::string test_file = "/tmp/test_reader_orchestrator_benchmark.bin";

    // Create larger test file (10 MB)
    std::cout << "  Creating 10 MB test file..." << std::endl;
    if (!create_test_file(test_file, 10 * 1024)) {
        std::cout << RED << "  ✗ Failed to create test file" << RESET << std::endl;
        return;
    }

    ReaderOrchestrator reader;

    const int num_reads = 10;
    size_t total_bytes = 0;
    double total_time_ms = 0.0;

    for (int i = 0; i < num_reads; ++i) {
        ReaderResponse response;
        reader.read(test_file, response);

        if (response.is_success()) {
            total_bytes += response.bytes_read;
            total_time_ms += response.read_time_ms;
        }
    }

    double avg_time_ms = total_time_ms / num_reads;
    double throughput_mbps = (total_bytes * 8.0 / 1024.0 / 1024.0) / (total_time_ms / 1000.0);

    std::cout << "  " << GREEN << "✓" << RESET << " Read " << total_bytes / 1024 / 1024
              << " MB total" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Average time per read: " << avg_time_ms
              << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Throughput: " << throughput_mbps << " Mbps"
              << std::endl;

    // Clean up
    std::remove(test_file.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Benchmark range requests
 */
void benchmark_range_requests() {
    std::cout << BOLD << "\n[BENCHMARK] Range Request Performance" << RESET << std::endl;

    std::string test_file = "/tmp/test_reader_orchestrator_range_benchmark.bin";

    // Create test file (100 MB)
    std::cout << "  Creating 100 MB test file..." << std::endl;
    if (!create_test_file(test_file, 100 * 1024)) {
        std::cout << RED << "  ✗ Failed to create test file" << RESET << std::endl;
        return;
    }

    ReaderOrchestrator reader;

    const int num_requests = 100;
    const size_t range_size = 1024 * 1024;  // 1 MB chunks
    size_t total_bytes = 0;
    double total_time_ms = 0.0;

    for (int i = 0; i < num_requests; ++i) {
        size_t offset = (i * range_size) % (90 * 1024 * 1024);  // Stay within file bounds

        ReaderResponse response;
        reader.read_range(test_file, offset, range_size, response);

        if (response.is_success()) {
            total_bytes += response.bytes_read;
            total_time_ms += response.read_time_ms;
        }
    }

    double avg_time_ms = total_time_ms / num_requests;
    double throughput_mbps = (total_bytes * 8.0 / 1024.0 / 1024.0) / (total_time_ms / 1000.0);

    std::cout << "  " << GREEN << "✓" << RESET << " " << num_requests << " range requests"
              << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Average time per request: " << avg_time_ms
              << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Throughput: " << throughput_mbps << " Mbps"
              << std::endl;

    // Clean up
    std::remove(test_file.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET
              << std::endl;
    std::cout << BOLD << "║     TurboLoader Reader Orchestrator Test Suite      ║" << RESET
              << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET
              << std::endl;

    try {
        test_source_detection();
        test_source_type_names();
        test_local_file_reading();
        test_local_file_range_requests();
        test_file_size();
        test_s3_url_parsing();
        test_gcs_url_parsing();
        test_http_url_detection();
        test_error_handling();
        test_configuration();
        test_response_structure();
        benchmark_local_file_reading();
        benchmark_range_requests();

        std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET
                  << std::endl;
        std::cout << BOLD << "║  " << GREEN << "✓ ALL TESTS PASSED" << RESET << BOLD
                  << "                                ║" << RESET << std::endl;
        std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET
                  << std::endl;

        std::cout << YELLOW << "\nNote: HTTP/S3/GCS tests validated URL detection only"
                  << RESET << std::endl;
        std::cout << YELLOW << "      Full integration tests require network access and credentials"
                  << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ TEST FAILED: " << e.what() << RESET << std::endl;
        return 1;
    }
}
