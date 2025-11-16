/**
 * @file test_tar_reader.cpp
 * @brief Tests for TarReader
 *
 * Tests:
 * 1. Basic TAR reading and indexing
 * 2. Per-worker sample partitioning
 * 3. Zero-copy JPEG data access
 * 4. Edge cases (empty TAR, single worker, etc.)
 */

#include "../src/readers/tar_reader.hpp"
#include "../src/core/sample.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <cstring>

using namespace turboloader;

// ANSI color codes for pretty output
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Create a minimal valid TAR file with JPEG images for testing
 *
 * @param path Path to create TAR file at
 * @param num_files Number of JPEG files to include
 */
void create_test_tar(const std::string& path, size_t num_files) {
    std::ofstream tar(path, std::ios::binary);
    if (!tar) {
        throw std::runtime_error("Failed to create test TAR file");
    }

    // Minimal valid JPEG data (1x1 pixel)
    const uint8_t jpeg_data[] = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
    };
    const size_t jpeg_size = sizeof(jpeg_data);

    for (size_t i = 0; i < num_files; ++i) {
        // Create TAR header
        char header[512];
        std::memset(header, 0, sizeof(header));

        // File name
        std::string filename = "img_" + std::to_string(i) + ".jpg";
        std::strncpy(header, filename.c_str(), 100);

        // File mode (644 octal)
        std::snprintf(header + 100, 8, "%07o", 0644);

        // Owner UID/GID (0)
        std::snprintf(header + 108, 8, "%07o", 0);
        std::snprintf(header + 116, 8, "%07o", 0);

        // File size (octal)
        std::snprintf(header + 124, 12, "%011lo", jpeg_size);

        // Modification time (octal)
        std::snprintf(header + 136, 12, "%011lo", 1234567890UL);

        // Checksum (space-filled initially)
        std::memset(header + 148, ' ', 8);

        // Type flag (regular file)
        header[156] = '0';

        // Magic "ustar\0"
        std::memcpy(header + 257, "ustar\0", 6);

        // Version "00"
        std::memcpy(header + 263, "00", 2);

        // Calculate checksum
        unsigned int checksum = 0;
        for (int j = 0; j < 512; ++j) {
            checksum += static_cast<unsigned char>(header[j]);
        }

        // Write checksum
        std::snprintf(header + 148, 8, "%06o", checksum);
        header[154] = '\0';
        header[155] = ' ';

        // Write header
        tar.write(header, sizeof(header));

        // Write file data
        tar.write(reinterpret_cast<const char*>(jpeg_data), jpeg_size);

        // Pad to 512-byte boundary
        size_t padding = (512 - (jpeg_size % 512)) % 512;
        if (padding > 0) {
            char pad[512] = {0};
            tar.write(pad, padding);
        }
    }

    // Write two zero blocks to mark end of TAR
    char zero_block[1024] = {0};
    tar.write(zero_block, sizeof(zero_block));

    tar.close();
}

/**
 * @brief Test: Basic TAR file indexing
 */
void test_basic_indexing() {
    std::cout << BOLD << "\n[TEST] Basic TAR Indexing" << RESET << std::endl;

    const std::string tar_path = "/tmp/test_tar_basic.tar";
    const size_t num_files = 10;

    // Create test TAR with 10 JPEG files
    create_test_tar(tar_path, num_files);

    // Open TAR reader (single worker)
    TarReader reader(tar_path, 0, 1);

    // Verify correct number of samples indexed
    assert(reader.total_samples() == num_files);
    assert(reader.num_samples() == num_files);

    std::cout << "  " << GREEN << "✓" << RESET << " Indexed " << num_files << " JPEG files correctly" << std::endl;

    // Verify we can read each sample
    for (size_t i = 0; i < num_files; ++i) {
        auto data = reader.get_sample(i);
        assert(!data.empty());

        // Check JPEG magic bytes (0xFF 0xD8)
        assert(data[0] == 0xFF);
        assert(data[1] == 0xD8);
    }

    std::cout << "  " << GREEN << "✓" << RESET << " All samples readable with valid JPEG headers" << std::endl;

    // Cleanup
    std::remove(tar_path.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test: Worker partitioning
 */
void test_worker_partitioning() {
    std::cout << BOLD << "\n[TEST] Worker Partitioning" << RESET << std::endl;

    const std::string tar_path = "/tmp/test_tar_partitioning.tar";
    const size_t num_files = 100;
    const size_t num_workers = 4;

    // Create test TAR with 100 JPEG files
    create_test_tar(tar_path, num_files);

    // Create readers for each worker
    std::vector<std::unique_ptr<TarReader>> readers;
    for (size_t i = 0; i < num_workers; ++i) {
        readers.push_back(std::make_unique<TarReader>(tar_path, i, num_workers));
    }

    // Verify each worker sees correct total
    for (const auto& reader : readers) {
        assert(reader->total_samples() == num_files);
    }

    std::cout << "  " << GREEN << "✓" << RESET << " All workers see total of " << num_files << " samples" << std::endl;

    // Verify partitioning
    size_t total_partitioned = 0;
    for (size_t i = 0; i < num_workers; ++i) {
        size_t worker_samples = readers[i]->num_samples();
        total_partitioned += worker_samples;
        std::cout << "  " << GREEN << "✓" << RESET << " Worker " << i << " has " << worker_samples << " samples" << std::endl;
    }

    assert(total_partitioned == num_files);
    std::cout << "  " << GREEN << "✓" << RESET << " Partitioning complete: all samples assigned" << std::endl;

    // Verify no overlapping samples between workers
    std::vector<bool> sample_seen(num_files, false);
    for (size_t worker_id = 0; worker_id < num_workers; ++worker_id) {
        for (size_t i = 0; i < readers[worker_id]->num_samples(); ++i) {
            const auto& entry = readers[worker_id]->get_entry(i);
            assert(!sample_seen[entry.index]);
            sample_seen[entry.index] = true;
        }
    }

    std::cout << "  " << GREEN << "✓" << RESET << " No sample overlaps between workers" << std::endl;

    // Cleanup
    std::remove(tar_path.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test: Zero-copy access
 */
void test_zero_copy_access() {
    std::cout << BOLD << "\n[TEST] Zero-Copy Access" << RESET << std::endl;

    const std::string tar_path = "/tmp/test_tar_zero_copy.tar";
    const size_t num_files = 5;

    // Create test TAR
    create_test_tar(tar_path, num_files);

    // Open TAR reader
    TarReader reader(tar_path, 0, 1);

    // Get sample data multiple times - should return same pointer
    auto span1 = reader.get_sample(0);
    auto span2 = reader.get_sample(0);

    // Both spans should point to same memory address (zero-copy)
    assert(span1.data() == span2.data());
    assert(span1.size() == span2.size());

    std::cout << "  " << GREEN << "✓" << RESET << " Multiple accesses return same memory address (zero-copy)" << std::endl;

    // Verify data is read-only (const)
    static_assert(std::is_const_v<std::remove_reference_t<decltype(*span1.data())>>,
                  "Span data should be const");

    std::cout << "  " << GREEN << "✓" << RESET << " Data is const (read-only)" << std::endl;

    // Cleanup
    std::remove(tar_path.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test: Edge cases
 */
void test_edge_cases() {
    std::cout << BOLD << "\n[TEST] Edge Cases" << RESET << std::endl;

    // Test 1: Empty TAR (only end markers)
    {
        const std::string tar_path = "/tmp/test_tar_empty.tar";
        std::ofstream tar(tar_path, std::ios::binary);
        char zero_block[1024] = {0};
        tar.write(zero_block, sizeof(zero_block));
        tar.close();

        TarReader reader(tar_path, 0, 1);
        assert(reader.total_samples() == 0);
        assert(reader.num_samples() == 0);

        std::cout << "  " << GREEN << "✓" << RESET << " Empty TAR handled correctly" << std::endl;
        std::remove(tar_path.c_str());
    }

    // Test 2: Single sample
    {
        const std::string tar_path = "/tmp/test_tar_single.tar";
        create_test_tar(tar_path, 1);

        TarReader reader(tar_path, 0, 1);
        assert(reader.total_samples() == 1);
        assert(reader.num_samples() == 1);

        auto data = reader.get_sample(0);
        assert(data[0] == 0xFF && data[1] == 0xD8);

        std::cout << "  " << GREEN << "✓" << RESET << " Single sample TAR handled correctly" << std::endl;
        std::remove(tar_path.c_str());
    }

    // Test 3: More workers than samples
    {
        const std::string tar_path = "/tmp/test_tar_few_samples.tar";
        create_test_tar(tar_path, 2);

        // 4 workers but only 2 samples
        TarReader reader1(tar_path, 0, 4);  // Should get 1 sample
        TarReader reader2(tar_path, 1, 4);  // Should get 1 sample
        TarReader reader3(tar_path, 2, 4);  // Should get 0 samples
        TarReader reader4(tar_path, 3, 4);  // Should get 0 samples

        assert(reader1.num_samples() + reader2.num_samples() +
               reader3.num_samples() + reader4.num_samples() == 2);

        std::cout << "  " << GREEN << "✓" << RESET << " More workers than samples handled correctly" << std::endl;
        std::remove(tar_path.c_str());
    }

    // Test 4: Out of bounds access
    {
        const std::string tar_path = "/tmp/test_tar_bounds.tar";
        create_test_tar(tar_path, 5);

        TarReader reader(tar_path, 0, 1);

        bool threw_exception = false;
        try {
            reader.get_sample(10);  // Out of bounds
        } catch (const std::out_of_range&) {
            threw_exception = true;
        }

        assert(threw_exception);
        std::cout << "  " << GREEN << "✓" << RESET << " Out of bounds access throws exception" << std::endl;
        std::remove(tar_path.c_str());
    }

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test: Integration with Sample structure
 */
void test_sample_integration() {
    std::cout << BOLD << "\n[TEST] Sample Integration" << RESET << std::endl;

    const std::string tar_path = "/tmp/test_tar_sample.tar";
    const size_t num_files = 3;

    create_test_tar(tar_path, num_files);

    TarReader reader(tar_path, 0, 1);

    // Create Sample objects from TarReader data
    std::vector<Sample> samples;
    for (size_t i = 0; i < reader.num_samples(); ++i) {
        auto jpeg_data = reader.get_sample(i);
        samples.emplace_back(i, jpeg_data);
    }

    // Verify samples
    assert(samples.size() == num_files);

    for (size_t i = 0; i < samples.size(); ++i) {
        assert(samples[i].index == i);
        assert(!samples[i].jpeg_data.empty());
        assert(samples[i].jpeg_data[0] == 0xFF);
        assert(samples[i].jpeg_data[1] == 0xD8);
        assert(!samples[i].is_decoded());
    }

    std::cout << "  " << GREEN << "✓" << RESET << " Created " << num_files << " Sample objects from TAR data" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " All samples have valid JPEG data spans" << std::endl;

    // Cleanup
    std::remove(tar_path.c_str());

    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "║         TurboLoader TAR Reader Test Suite           ║" << RESET << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

    try {
        test_basic_indexing();
        test_worker_partitioning();
        test_zero_copy_access();
        test_edge_cases();
        test_sample_integration();

        std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << "║  " << GREEN << "✓ ALL TESTS PASSED" << RESET << BOLD << "                                ║" << RESET << std::endl;
        std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ TEST FAILED: " << e.what() << RESET << std::endl;
        return 1;
    }
}
