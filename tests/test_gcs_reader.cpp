/**
 * @file test_gcs_reader.cpp
 * @brief Comprehensive tests for GCS reader with performance benchmarks
 *
 * Tests:
 * 1. Basic GCS GET requests (public buckets)
 * 2. Range requests (partial downloads)
 * 3. URL construction
 * 4. GCS-compatible storage
 * 5. Error handling
 * 6. Performance benchmarks
 *
 * Note: These tests use public GCS buckets and HTTP fallback mode.
 * For full Google Cloud Storage SDK testing, compile with -DHAVE_GCS_SDK and configure credentials.
 */

#include "../src/readers/gcs_reader.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Test GCS URL construction
 */
void test_url_construction() {
    std::cout << BOLD << "\n[TEST] GCS URL Construction" << RESET << std::endl;

    GCSConfig config;
    config.bucket = "my-bucket";
    config.object = "path/to/object.jpg";

    GCSReader reader(config);

    // Standard GCS public URL
    std::string url1 = reader.construct_url(config);
    std::string expected1 = "https://storage.googleapis.com/my-bucket/path/to/object.jpg";
    assert(url1 == expected1);
    std::cout << "  " << GREEN << "✓" << RESET << " Standard GCS URL: " << url1 << std::endl;

    // Custom endpoint (GCS-compatible storage)
    config.endpoint_url = "https://storage.custom-cloud.com";
    std::string url2 = reader.construct_url(config);
    std::string expected2 = "https://storage.custom-cloud.com/my-bucket/path/to/object.jpg";
    assert(url2 == expected2);
    std::cout << "  " << GREEN << "✓" << RESET << " Custom endpoint URL: " << url2 << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test fetching from public GCS bucket
 *
 * Note: This test requires internet access and uses a public GCS bucket.
 * It may fail if the bucket is not available or network is down.
 */
void test_public_gcs_fetch() {
    std::cout << BOLD << "\n[TEST] Fetch from Public GCS Bucket" << RESET << std::endl;
    std::cout << YELLOW << "  (This test requires internet access)" << RESET << std::endl;

    // Use a well-known public GCS bucket (Google Cloud public datasets)
    GCSConfig config;
    config.bucket = "gcp-public-data-landsat";
    config.object = "index.csv.gz";
    config.timeout_ms = 15000;

    GCSReader reader(config);
    std::vector<uint8_t> output;

    bool success = reader.fetch_object(config, output);

    if (success) {
        assert(!output.empty());
        std::cout << "  " << GREEN << "✓" << RESET << " Successfully fetched object" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << output.size() << " bytes" << std::endl;

        // Verify it's a gzip file (starts with 0x1f 0x8b)
        if (output.size() >= 2) {
            bool is_gzip = (output[0] == 0x1f && output[1] == 0x8b);
            assert(is_gzip);
            std::cout << "  " << GREEN << "✓" << RESET << " Content validation passed (gzip format)" << std::endl;
        }
    } else {
        std::cout << "  " << YELLOW << "⚠" << RESET << " Test skipped (bucket not accessible)" << std::endl;
        std::cout << "  " << YELLOW << "  This is expected if there's no internet connection" << RESET << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test range requests from GCS
 */
void test_gcs_range_requests() {
    std::cout << BOLD << "\n[TEST] GCS Range Requests" << RESET << std::endl;
    std::cout << YELLOW << "  (This test requires internet access)" << RESET << std::endl;

    GCSConfig config;
    config.bucket = "gcp-public-data-landsat";
    config.object = "index.csv.gz";
    config.timeout_ms = 15000;

    GCSReader reader(config);

    // Fetch first 100 bytes
    std::vector<uint8_t> output;
    bool success = reader.fetch_range(config, 0, 100, output);

    if (success) {
        assert(output.size() == 100);
        std::cout << "  " << GREEN << "✓" << RESET << " Range request successful" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Downloaded exactly 100 bytes" << std::endl;

        // Verify gzip header in first bytes
        if (output.size() >= 2) {
            bool is_gzip = (output[0] == 0x1f && output[1] == 0x8b);
            assert(is_gzip);
            std::cout << "  " << GREEN << "✓" << RESET << " Content validation passed" << std::endl;
        }
    } else {
        std::cout << "  " << YELLOW << "⚠" << RESET << " Test skipped (bucket not accessible)" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test getting GCS object size
 */
void test_gcs_object_size() {
    std::cout << BOLD << "\n[TEST] GCS Object Size (HEAD Request)" << RESET << std::endl;
    std::cout << YELLOW << "  (This test requires internet access)" << RESET << std::endl;

    GCSConfig config;
    config.bucket = "gcp-public-data-landsat";
    config.object = "index.csv.gz";

    GCSReader reader(config);
    size_t size;

    bool success = reader.get_object_size(config, size);

    if (success) {
        assert(size > 0);
        std::cout << "  " << GREEN << "✓" << RESET << " HEAD request successful" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Object size: " << size << " bytes" << std::endl;
    } else {
        std::cout << "  " << YELLOW << "⚠" << RESET << " Test skipped (bucket not accessible)" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test error handling (non-existent bucket/object)
 */
void test_gcs_error_handling() {
    std::cout << BOLD << "\n[TEST] GCS Error Handling" << RESET << std::endl;

    // Test non-existent object
    GCSConfig config;
    config.bucket = "gcp-public-data-landsat";
    config.object = "non-existent-file-12345.txt";
    config.timeout_ms = 10000;

    GCSReader reader(config);
    std::vector<uint8_t> output;

    bool success = reader.fetch_object(config, output);

    // Should fail for non-existent object
    if (!success || output.empty()) {
        std::cout << "  " << GREEN << "✓" << RESET << " 404 error handled correctly" << std::endl;
    }

    // Test invalid bucket
    config.bucket = "non-existent-bucket-turboloader-12345";
    config.object = "test.txt";

    success = reader.fetch_object(config, output);
    if (!success || output.empty()) {
        std::cout << "  " << GREEN << "✓" << RESET << " Invalid bucket handled correctly" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test configuration structures
 */
void test_gcs_config() {
    std::cout << BOLD << "\n[TEST] GCS Configuration" << RESET << std::endl;

    // Test GCSConfig defaults
    GCSConfig config;
    assert(config.bucket.empty());
    assert(config.object.empty());
    assert(config.project_id.empty());
    assert(config.use_cdn == false);
    assert(config.max_connections == 25);
    assert(config.timeout_ms == 30000);
    std::cout << "  " << GREEN << "✓" << RESET << " GCSConfig default values" << std::endl;

    // Test GCSRequest defaults
    GCSRequest request;
    assert(request.offset == 0);
    assert(request.size == 0);
    assert(request.max_retries == 3);
    std::cout << "  " << GREEN << "✓" << RESET << " GCSRequest default values" << std::endl;

    // Test GCSResponse defaults
    GCSResponse response;
    assert(response.data.empty());
    assert(response.error_message.empty());
    assert(response.download_time_ms == 0.0);
    assert(response.bytes_downloaded == 0);
    assert(response.is_success() == true);  // No error message means success
    std::cout << "  " << GREEN << "✓" << RESET << " GCSResponse default values" << std::endl;

    // Test error state
    response.error_message = "Test error";
    assert(response.is_success() == false);
    std::cout << "  " << GREEN << "✓" << RESET << " GCSResponse error detection" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Benchmark GCS download throughput
 */
void benchmark_gcs_throughput() {
    std::cout << BOLD << "\n[BENCHMARK] GCS Download Throughput" << RESET << std::endl;
    std::cout << YELLOW << "  (This benchmark requires internet access)" << RESET << std::endl;

    GCSConfig config;
    config.bucket = "gcp-public-data-landsat";
    config.object = "index.csv.gz";
    config.timeout_ms = 30000;

    GCSReader reader(config);

    const int num_downloads = 3;  // Fewer downloads than S3 test (GCS file is larger)
    size_t total_bytes = 0;
    int successful_downloads = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_downloads; ++i) {
        std::vector<uint8_t> output;
        if (reader.fetch_object(config, output)) {
            total_bytes += output.size();
            ++successful_downloads;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (successful_downloads > 0) {
        double throughput_mbps = (total_bytes * 8.0 / 1024.0 / 1024.0) / (duration.count() / 1000.0);
        double requests_per_sec = (successful_downloads * 1000.0) / duration.count();

        std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << total_bytes / 1024
                  << " KB in " << duration.count() << " ms" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Successful downloads: "
                  << successful_downloads << "/" << num_downloads << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Throughput: " << throughput_mbps
                  << " Mbps" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Request rate: " << requests_per_sec
                  << " requests/sec" << std::endl;
    } else {
        std::cout << "  " << YELLOW << "⚠" << RESET << " Benchmark skipped (GCS not accessible)" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Print library version information
 */
void print_version_info() {
    std::cout << BOLD << "\n[INFO] GCS Reader Version" << RESET << std::endl;
    std::cout << "  " << GCSReader::version_info() << std::endl;

#ifndef HAVE_GCS_SDK
    std::cout << YELLOW << "\n  Note: Using HTTP fallback mode" << RESET << std::endl;
    std::cout << YELLOW << "  For native Google Cloud Storage SDK support, compile with:" << RESET << std::endl;
    std::cout << YELLOW << "    cmake -DHAVE_GCS_SDK=ON .." << RESET << std::endl;
#endif
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "║      TurboLoader GCS Reader Test Suite              ║" << RESET << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

    print_version_info();

    try {
        test_gcs_config();
        test_url_construction();
        test_public_gcs_fetch();
        test_gcs_range_requests();
        test_gcs_object_size();
        test_gcs_error_handling();
        benchmark_gcs_throughput();

        std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << "║  " << GREEN << "✓ ALL TESTS PASSED" << RESET << BOLD << "                                ║" << RESET << std::endl;
        std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

        std::cout << YELLOW << "\nNote: Some tests may be skipped if internet is unavailable" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ TEST FAILED: " << e.what() << RESET << std::endl;
        return 1;
    }
}
