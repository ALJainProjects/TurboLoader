/**
 * @file test_http_reader.cpp
 * @brief Comprehensive tests for HTTP reader with performance benchmarks
 *
 * Tests:
 * 1. Basic HTTP GET requests
 * 2. HTTPS support
 * 3. Range requests (partial downloads)
 * 4. Connection pooling and reuse
 * 5. Retry logic with exponential backoff
 * 6. Concurrent downloads
 * 7. Performance benchmarks
 * 8. Error handling
 */

#include "../src/readers/http_reader.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Test basic HTTP GET request
 */
void test_basic_http_get() {
    std::cout << BOLD << "\n[TEST] Basic HTTP GET Request" << RESET << std::endl;

    HTTPReader reader;
    std::vector<uint8_t> output;

    // Fetch a small JSON file from httpbin.org (public test API)
    bool success = reader.fetch_file("http://httpbin.org/json", output, 10000);

    assert(success);
    assert(!output.empty());

    std::cout << "  " << GREEN << "✓" << RESET << " Successfully fetched data" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << output.size() << " bytes" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test HTTPS support
 */
void test_https_support() {
    std::cout << BOLD << "\n[TEST] HTTPS Support" << RESET << std::endl;

    HTTPReader reader;
    std::vector<uint8_t> output;

    // Fetch from HTTPS endpoint
    bool success = reader.fetch_file("https://httpbin.org/json", output, 10000);

    assert(success);
    assert(!output.empty());

    std::cout << "  " << GREEN << "✓" << RESET << " HTTPS request successful" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << output.size() << " bytes" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test range requests (partial downloads)
 */
void test_range_requests() {
    std::cout << BOLD << "\n[TEST] Range Requests (Partial Downloads)" << RESET << std::endl;

    HTTPReader reader;
    std::vector<uint8_t> output;

    // Download only first 100 bytes
    bool success = reader.fetch_range("http://httpbin.org/bytes/1000", 0, 100, output, 10000);

    assert(success);
    assert(output.size() == 100);

    std::cout << "  " << GREEN << "✓" << RESET << " Range request successful" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Downloaded exactly 100 bytes" << std::endl;

    // Test middle range
    output.clear();
    success = reader.fetch_range("http://httpbin.org/bytes/1000", 100, 200, output, 10000);

    assert(success);
    assert(output.size() == 200);

    std::cout << "  " << GREEN << "✓" << RESET << " Middle range request successful" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test file size retrieval (HEAD request)
 */
void test_file_size() {
    std::cout << BOLD << "\n[TEST] File Size Retrieval (HEAD Request)" << RESET << std::endl;

    HTTPReader reader;
    size_t size;

    bool success = reader.get_file_size("http://httpbin.org/bytes/5000", size);

    assert(success);
    assert(size == 5000);

    std::cout << "  " << GREEN << "✓" << RESET << " HEAD request successful" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " File size: " << size << " bytes" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test connection reuse and pooling
 */
void test_connection_pooling() {
    std::cout << BOLD << "\n[TEST] Connection Pooling and Reuse" << RESET << std::endl;

    HTTPReader reader(4);  // Pool with 4 connections

    // Make multiple requests to the same host
    const int num_requests = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_requests; ++i) {
        std::vector<uint8_t> output;
        bool success = reader.fetch_file("http://httpbin.org/bytes/1000", output, 10000);
        assert(success);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "  " << GREEN << "✓" << RESET << " Completed " << num_requests
              << " requests in " << duration.count() << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Average: "
              << (duration.count() / num_requests) << " ms/request" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test concurrent downloads
 */
void test_concurrent_downloads() {
    std::cout << BOLD << "\n[TEST] Concurrent Downloads" << RESET << std::endl;

    HTTPReader reader(8);  // Pool with 8 connections
    const int num_threads = 4;
    const int requests_per_thread = 5;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&reader, &success_count, requests_per_thread]() {
            for (int i = 0; i < requests_per_thread; ++i) {
                std::vector<uint8_t> output;
                if (reader.fetch_file("http://httpbin.org/bytes/1000", output, 10000)) {
                    success_count++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    assert(success_count == num_threads * requests_per_thread);

    std::cout << "  " << GREEN << "✓" << RESET << " Completed " << success_count
              << " concurrent requests in " << duration.count() << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Throughput: "
              << (success_count.load() * 1000 / duration.count()) << " requests/sec" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test error handling (404, timeout, etc.)
 */
void test_error_handling() {
    std::cout << BOLD << "\n[TEST] Error Handling" << RESET << std::endl;

    HTTPReader reader;

    // Test 404 Not Found
    {
        std::vector<uint8_t> output;
        bool success = reader.fetch_file("http://httpbin.org/status/404", output, 10000);
        assert(!success);
        std::cout << "  " << GREEN << "✓" << RESET << " 404 error handled correctly" << std::endl;
    }

    // Test invalid URL
    {
        std::vector<uint8_t> output;
        bool success = reader.fetch_file("http://invalid-domain-12345.com/", output, 5000);
        assert(!success);
        std::cout << "  " << GREEN << "✓" << RESET << " Invalid URL handled correctly" << std::endl;
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test retry logic with exponential backoff
 */
void test_retry_logic() {
    std::cout << BOLD << "\n[TEST] Retry Logic with Exponential Backoff" << RESET << std::endl;

    HTTPReader reader;
    HTTPRequest request;
    request.url = "http://httpbin.org/status/503";  // Service Unavailable (retriable)
    request.max_retries = 3;
    request.timeout_ms = 5000;

    HTTPResponse response;
    auto start = std::chrono::high_resolution_clock::now();
    bool success = reader.fetch(request, response);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should fail after retries
    assert(!success);
    assert(response.http_code == 503);

    // Should take at least 100 + 200 ms (backoff times)
    assert(duration.count() >= 300);

    std::cout << "  " << GREEN << "✓" << RESET << " Retry logic executed correctly" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Total time: " << duration.count()
              << " ms (includes backoff)" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Performance benchmark for download throughput
 */
void benchmark_download_throughput() {
    std::cout << BOLD << "\n[BENCHMARK] Download Throughput" << RESET << std::endl;

    HTTPReader reader(8);
    const size_t file_size = 100 * 1024;  // 100KB
    const int num_downloads = 20;

    size_t total_bytes = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_downloads; ++i) {
        std::vector<uint8_t> output;
        std::string url = "http://httpbin.org/bytes/" + std::to_string(file_size);

        if (reader.fetch_file(url, output, 15000)) {
            total_bytes += output.size();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double throughput_mbps = (total_bytes * 8.0 / 1024.0 / 1024.0) / (duration.count() / 1000.0);
    double requests_per_sec = (num_downloads * 1000.0) / duration.count();

    std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << total_bytes / 1024
              << " KB in " << duration.count() << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Throughput: " << throughput_mbps
              << " Mbps" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Request rate: " << requests_per_sec
              << " requests/sec" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Benchmark concurrent download performance
 */
void benchmark_concurrent_throughput() {
    std::cout << BOLD << "\n[BENCHMARK] Concurrent Download Throughput" << RESET << std::endl;

    HTTPReader reader(16);
    const size_t file_size = 50 * 1024;  // 50KB per file
    const int num_threads = 8;
    const int downloads_per_thread = 5;

    std::atomic<size_t> total_bytes{0};
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&reader, &total_bytes, file_size, downloads_per_thread]() {
            for (int i = 0; i < downloads_per_thread; ++i) {
                std::vector<uint8_t> output;
                std::string url = "http://httpbin.org/bytes/" + std::to_string(file_size);

                if (reader.fetch_file(url, output, 15000)) {
                    total_bytes += output.size();
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double throughput_mbps = (total_bytes.load() * 8.0 / 1024.0 / 1024.0) / (duration.count() / 1000.0);
    int total_downloads = num_threads * downloads_per_thread;
    double requests_per_sec = (total_downloads * 1000.0) / duration.count();

    std::cout << "  " << GREEN << "✓" << RESET << " Downloaded " << total_bytes.load() / 1024
              << " KB using " << num_threads << " threads in " << duration.count() << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Throughput: " << throughput_mbps
              << " Mbps" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Request rate: " << requests_per_sec
              << " requests/sec" << std::endl;

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Print library version information
 */
void print_version_info() {
    std::cout << BOLD << "\n[INFO] HTTP Reader Version" << RESET << std::endl;
    std::cout << "  " << HTTPReader::version_info() << std::endl;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "║      TurboLoader HTTP Reader Test Suite             ║" << RESET << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

    print_version_info();

    try {
        test_basic_http_get();
        test_https_support();
        test_range_requests();
        test_file_size();
        test_connection_pooling();
        test_concurrent_downloads();
        test_error_handling();
        test_retry_logic();
        benchmark_download_throughput();
        benchmark_concurrent_throughput();

        std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << "║  " << GREEN << "✓ ALL TESTS PASSED" << RESET << BOLD << "                                ║" << RESET << std::endl;
        std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ TEST FAILED: " << e.what() << RESET << std::endl;
        return 1;
    }
}
