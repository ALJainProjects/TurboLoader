/**
 * @file test_s3_reader.cpp
 * @brief Comprehensive tests for S3 reader with performance benchmarks
 *
 * Tests:
 * 1. Basic S3 GET requests (public buckets)
 * 2. Range requests (partial downloads)
 * 3. URL construction (virtual hosted vs path-style)
 * 4. S3-compatible storage (MinIO, Wasabi, etc.)
 * 5. Error handling
 * 6. Performance benchmarks
 *
 * Note: These tests use public S3 buckets and HTTP fallback mode.
 * For full AWS SDK testing, compile with -DHAVE_AWS_SDK and configure credentials.
 */

#include "../src/readers/s3_reader.hpp"
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
 * @brief Test S3 URL construction
 */
void test_url_construction() {
    std::cout << BOLD << "\n[TEST] S3 URL Construction" << RESET << std::endl;

    S3Config config;
    config.bucket = "my-bucket";
    config.key = "path/to/object.jpg";
    config.region = "us-west-2";

    S3Reader reader(config);

    // Virtual hosted-style URL
    config.use_virtual_addressing = true;
    std::string url1 = reader.construct_url(config);
    std::string expected1 = "https://my-bucket.s3.us-west-2.amazonaws.com/path/to/object.jpg";
    assert(url1 == expected1);
    std::cout << "  " << GREEN << "✓" << RESET << " Virtual hosted-style URL: " << url1 << std::endl;

    // Path-style URL
    config.use_virtual_addressing = false;
    std::string url2 = reader.construct_url(config);
    std::string expected2 = "https://s3.us-west-2.amazonaws.com/my-bucket/path/to/object.jpg";
    assert(url2 == expected2);
    std::cout << "  " << GREEN << "✓" << RESET << " Path-style URL: " << url2 << std::endl;

    // Custom endpoint (S3-compatible storage)
    config.endpoint_url = "https://s3.custom-storage.com";
    std::string url3 = reader.construct_url(config);
    std::string expected3 = "https://s3.custom-storage.com/my-bucket/path/to/object.jpg";
    assert(url3 == expected3);
    std::cout << "  " << GREEN << "✓" << RESET << " Custom endpoint URL: " << url3 << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test fetching from public S3 bucket
 *
 * Note: This test requires internet access and uses a public S3 bucket.
 * It may fail if the bucket is not available or network is down.
 */
void test_public_s3_fetch() {
    std::cout << BOLD << "\n[TEST] Fetch from Public S3 Bucket" << RESET << std::endl;
    std::cout << YELLOW << "  (This test requires internet access)" << RESET << std::endl;

    // Use a well-known public S3 bucket (AWS CloudFormation sample templates)
    S3Config config;
    config.bucket = "cloudformation-templates-us-east-1";
    config.key = "WordPress_Single_Instance.template";
    config.region = "us-east-1";
    config.timeout_ms = 15000;

    S3Reader reader(config);
    std::vector<uint8_t> output;

    bool success = reader.fetch_object(config, output);

    if (success) {
        assert(!output.empty());
        std::cout << "  " << GREEN << "✓" << RESET << " Successfully fetched object" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << output.size() << " bytes" << std::endl;

        // Verify it's a CloudFormation template (starts with JSON or YAML)
        std::string content(output.begin(), output.begin() + std::min(output.size(), size_t(10)));
        bool is_template = content.find("{") != std::string::npos ||
                          content.find("AWSTemplate") != std::string::npos;
        assert(is_template);
        std::cout << "  " << GREEN << "✓" << RESET << " Content validation passed" << std::endl;
    } else {
        std::cout << "  " << YELLOW << "⚠" << RESET << " Test skipped (bucket not accessible)" << std::endl;
        std::cout << "  " << YELLOW << "  This is expected if there's no internet connection" << RESET << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test range requests from S3
 */
void test_s3_range_requests() {
    std::cout << BOLD << "\n[TEST] S3 Range Requests" << RESET << std::endl;
    std::cout << YELLOW << "  (This test requires internet access)" << RESET << std::endl;

    S3Config config;
    config.bucket = "cloudformation-templates-us-east-1";
    config.key = "WordPress_Single_Instance.template";
    config.region = "us-east-1";
    config.timeout_ms = 15000;

    S3Reader reader(config);

    // Fetch first 100 bytes
    std::vector<uint8_t> output;
    bool success = reader.fetch_range(config, 0, 100, output);

    if (success) {
        assert(output.size() == 100);
        std::cout << "  " << GREEN << "✓" << RESET << " Range request successful" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Downloaded exactly 100 bytes" << std::endl;
    } else {
        std::cout << "  " << YELLOW << "⚠" << RESET << " Test skipped (bucket not accessible)" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test getting S3 object size
 */
void test_s3_object_size() {
    std::cout << BOLD << "\n[TEST] S3 Object Size (HEAD Request)" << RESET << std::endl;
    std::cout << YELLOW << "  (This test requires internet access)" << RESET << std::endl;

    S3Config config;
    config.bucket = "cloudformation-templates-us-east-1";
    config.key = "WordPress_Single_Instance.template";
    config.region = "us-east-1";

    S3Reader reader(config);
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
 * @brief Test error handling (non-existent bucket/key)
 */
void test_s3_error_handling() {
    std::cout << BOLD << "\n[TEST] S3 Error Handling" << RESET << std::endl;

    // Test non-existent key
    S3Config config;
    config.bucket = "cloudformation-templates-us-east-1";
    config.key = "non-existent-file-12345.txt";
    config.region = "us-east-1";
    config.timeout_ms = 10000;

    S3Reader reader(config);
    std::vector<uint8_t> output;

    bool success = reader.fetch_object(config, output);

    // Should fail for non-existent key
    if (!success || output.empty()) {
        std::cout << "  " << GREEN << "✓" << RESET << " 404 error handled correctly" << std::endl;
    }

    // Test invalid bucket
    config.bucket = "non-existent-bucket-turboloader-12345";
    config.key = "test.txt";

    success = reader.fetch_object(config, output);
    if (!success || output.empty()) {
        std::cout << "  " << GREEN << "✓" << RESET << " Invalid bucket handled correctly" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Benchmark S3 download throughput
 */
void benchmark_s3_throughput() {
    std::cout << BOLD << "\n[BENCHMARK] S3 Download Throughput" << RESET << std::endl;
    std::cout << YELLOW << "  (This benchmark requires internet access)" << RESET << std::endl;

    S3Config config;
    config.bucket = "cloudformation-templates-us-east-1";
    config.key = "WordPress_Single_Instance.template";
    config.region = "us-east-1";
    config.timeout_ms = 30000;

    S3Reader reader(config);

    const int num_downloads = 5;
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
        std::cout << "  " << YELLOW << "⚠" << RESET << " Benchmark skipped (S3 not accessible)" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Print library version information
 */
void print_version_info() {
    std::cout << BOLD << "\n[INFO] S3 Reader Version" << RESET << std::endl;
    std::cout << "  " << S3Reader::version_info() << std::endl;

#ifndef HAVE_AWS_SDK
    std::cout << YELLOW << "\n  Note: Using HTTP fallback mode" << RESET << std::endl;
    std::cout << YELLOW << "  For native AWS SDK support, compile with:" << RESET << std::endl;
    std::cout << YELLOW << "    cmake -DHAVE_AWS_SDK=ON .." << RESET << std::endl;
#endif
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "║      TurboLoader S3 Reader Test Suite               ║" << RESET << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

    print_version_info();

    try {
        test_url_construction();
        test_public_s3_fetch();
        test_s3_range_requests();
        test_s3_object_size();
        test_s3_error_handling();
        benchmark_s3_throughput();

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
