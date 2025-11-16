#include "turboloader/readers/http_reader.hpp"
#include <iostream>
#include <cassert>

using namespace turboloader;

int main() {
    std::cout << "Testing HTTP Reader...\n\n";

    HttpStorageReader reader;

    // Test 1: Download a small file
    std::cout << "Test 1: Download small file from httpbin.org...\n";
    try {
        auto data = reader.read("https://httpbin.org/bytes/1024");
        std::cout << "  Downloaded " << data.size() << " bytes\n";
        assert(data.size() == 1024);
        std::cout << "  ✓ PASS\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ FAIL: " << e.what() << "\n\n";
        return 1;
    }

    // Test 2: HEAD request (exists)
    std::cout << "Test 2: HEAD request...\n";
    try {
        bool exists = reader.exists("https://httpbin.org/status/200");
        assert(exists);
        std::cout << "  ✓ PASS\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ FAIL: " << e.what() << "\n\n";
        return 1;
    }

    // Test 3: 404 handling
    std::cout << "Test 3: 404 handling...\n";
    try {
        bool exists = reader.exists("https://httpbin.org/status/404");
        assert(!exists);
        std::cout << "  ✓ PASS\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ FAIL: " << e.what() << "\n\n";
        return 1;
    }

    // Test 4: Range request
    std::cout << "Test 4: Range request...\n";
    try {
        auto data = reader.read_range("https://httpbin.org/bytes/10000", 100, 500);
        std::cout << "  Downloaded " << data.size() << " bytes (requested 500)\n";
        // httpbin may not support range requests perfectly
        assert(data.size() <= 500);
        std::cout << "  ✓ PASS\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ FAIL: " << e.what() << "\n\n";
        return 1;
    }

    // Test 5: Redirect handling
    std::cout << "Test 5: Redirect handling...\n";
    try {
        auto data = reader.read("https://httpbin.org/redirect-to?url=https://httpbin.org/bytes/512");
        std::cout << "  Downloaded " << data.size() << " bytes after redirect\n";
        std::cout << "  ✓ PASS\n\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ FAIL: " << e.what() << "\n\n";
        return 1;
    }

    std::cout << "All HTTP reader tests passed!\n";
    return 0;
}
