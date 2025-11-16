/**
 * @file test_tar_reader.cpp
 * @brief Unit tests for TarReaderV2
 */

#include "../../src/io_v2/tar_reader_v2.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <cstdio>

using namespace turboloader::v2;

// Test framework macros
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "  " << #name << "... "; \
    test_##name(); \
    std::cout << "PASSED" << std::endl; \
} while(0)

// Helper: Create a minimal TAR file for testing
std::string create_test_tar() {
    const char* tar_path = "/tmp/test_v2.tar";
    std::ofstream tar(tar_path, std::ios::binary);

    // Helper to write TAR header
    auto write_header = [&](const char* name, size_t size) {
        TarHeader header;
        std::memset(&header, 0, sizeof(header));

        // Copy filename
        std::strncpy(header.name, name, sizeof(header.name) - 1);

        // Write size as octal
        std::snprintf(header.size, sizeof(header.size), "%011zo", size);

        // Set file mode
        std::strncpy(header.mode, "0000644", sizeof(header.mode) - 1);

        // Set magic and version
        std::strncpy(header.magic, "ustar", 5);
        header.magic[5] = '\0';
        header.version[0] = '0';
        header.version[1] = '0';

        // Set type to regular file
        header.typeflag = '0';

        // Calculate checksum
        unsigned int checksum = 0;
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&header);
        for (size_t i = 0; i < sizeof(header); ++i) {
            checksum += bytes[i];
        }
        std::snprintf(header.checksum, sizeof(header.checksum), "%06o", checksum);

        tar.write(reinterpret_cast<const char*>(&header), sizeof(header));
    };

    // Helper to write file data (padded to 512 bytes)
    auto write_data = [&](const std::vector<uint8_t>& data) {
        tar.write(reinterpret_cast<const char*>(data.data()), data.size());

        // Pad to 512-byte boundary
        size_t padding = (512 - (data.size() % 512)) % 512;
        std::vector<uint8_t> pad(padding, 0);
        tar.write(reinterpret_cast<const char*>(pad.data()), padding);
    };

    // Create fake JPEG files (just some bytes with JPEG magic)
    std::vector<uint8_t> jpeg1 = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10};  // JPEG magic
    std::vector<uint8_t> jpeg2 = {0xFF, 0xD8, 0xFF, 0xE1, 0x01, 0x02, 0x03, 0x04};
    std::vector<uint8_t> jpeg3 = {0xFF, 0xD8, 0xFF, 0xDB};

    // Write entries
    write_header("image1.jpg", jpeg1.size());
    write_data(jpeg1);

    write_header("image2.jpg", jpeg2.size());
    write_data(jpeg2);

    write_header("image3.jpg", jpeg3.size());
    write_data(jpeg3);

    // Write two zero blocks to mark end of archive
    std::vector<uint8_t> zero_block(512, 0);
    tar.write(reinterpret_cast<const char*>(zero_block.data()), 512);
    tar.write(reinterpret_cast<const char*>(zero_block.data()), 512);

    tar.close();

    return tar_path;
}

// ============================================================================
// Basic Tests
// ============================================================================

TEST(tar_header_parsing) {
    TarHeader header;
    std::memset(&header, 0, sizeof(header));

    // Test octal parsing
    std::strcpy(header.size, "00000000100");  // 64 in octal
    assert(header.get_size() == 64);

    std::strcpy(header.size, "00001000000");  // 262144 in octal
    assert(header.get_size() == 262144);

    // Test magic validation
    std::strcpy(header.magic, "ustar");
    assert(header.is_valid());

    std::strcpy(header.magic, "xxxxx");
    assert(!header.is_valid());
}

TEST(tar_reader_basic) {
    std::string tar_path = create_test_tar();

    // Single worker reads all samples
    TarReaderV2 reader(tar_path, 0, 1);

    assert(reader.total_samples() == 3);
    assert(reader.num_samples() == 3);
    assert(reader.worker_id() == 0);

    // Cleanup
    std::remove(tar_path.c_str());
}

TEST(tar_reader_zero_copy) {
    std::string tar_path = create_test_tar();
    TarReaderV2 reader(tar_path, 0, 1);

    // Get zero-copy view of first sample
    auto data = reader.get_sample(0);
    assert(data.size() == 6);
    assert(data[0] == 0xFF);
    assert(data[1] == 0xD8);

    // Get second sample
    auto data2 = reader.get_sample(1);
    assert(data2.size() == 8);
    assert(data2[0] == 0xFF);

    std::remove(tar_path.c_str());
}

TEST(tar_reader_entry_metadata) {
    std::string tar_path = create_test_tar();
    TarReaderV2 reader(tar_path, 0, 1);

    const TarEntry& entry = reader.get_entry(0);
    assert(entry.name == "image1.jpg");
    assert(entry.size == 6);
    assert(entry.index == 0);

    const TarEntry& entry2 = reader.get_entry(1);
    assert(entry2.name == "image2.jpg");
    assert(entry2.size == 8);

    std::remove(tar_path.c_str());
}

// ============================================================================
// Worker Partitioning Tests
// ============================================================================

TEST(single_worker_partition) {
    std::string tar_path = create_test_tar();

    TarReaderV2 reader(tar_path, 0, 1);
    assert(reader.num_samples() == 3);

    // Should get all samples
    for (size_t i = 0; i < 3; ++i) {
        auto data = reader.get_sample(i);
        assert(data.size() > 0);
        assert(data[0] == 0xFF);
    }

    std::remove(tar_path.c_str());
}

TEST(two_worker_partition) {
    std::string tar_path = create_test_tar();

    // Worker 0 should get samples [0, 2)
    TarReaderV2 worker0(tar_path, 0, 2);
    assert(worker0.num_samples() == 2);
    assert(worker0.get_entry(0).index == 0);
    assert(worker0.get_entry(1).index == 1);

    // Worker 1 should get samples [2, 3)
    TarReaderV2 worker1(tar_path, 1, 2);
    assert(worker1.num_samples() == 1);
    assert(worker1.get_entry(0).index == 2);

    std::remove(tar_path.c_str());
}

TEST(four_worker_partition) {
    std::string tar_path = create_test_tar();

    // Create 4 workers
    TarReaderV2 w0(tar_path, 0, 4);
    TarReaderV2 w1(tar_path, 1, 4);
    TarReaderV2 w2(tar_path, 2, 4);
    TarReaderV2 w3(tar_path, 3, 4);

    // Check partition sizes (3 samples / 4 workers = 1 per worker, with some empty)
    assert(w0.num_samples() == 1);
    assert(w1.num_samples() == 1);
    assert(w2.num_samples() == 1);
    assert(w3.num_samples() == 0);

    // Verify indices
    assert(w0.get_entry(0).index == 0);
    assert(w1.get_entry(0).index == 1);
    assert(w2.get_entry(0).index == 2);

    std::remove(tar_path.c_str());
}

// ============================================================================
// Multi-threaded Tests
// ============================================================================

TEST(concurrent_readers) {
    std::string tar_path = create_test_tar();
    constexpr size_t NUM_WORKERS = 4;

    std::atomic<size_t> total_samples_read{0};
    std::vector<std::thread> threads;

    // Launch workers
    for (size_t i = 0; i < NUM_WORKERS; ++i) {
        threads.emplace_back([&, worker_id = i]() {
            TarReaderV2 reader(tar_path, worker_id, NUM_WORKERS);

            // Read all samples for this worker
            for (size_t j = 0; j < reader.num_samples(); ++j) {
                auto data = reader.get_sample(j);
                assert(data.size() > 0);
                assert(data[0] == 0xFF);  // JPEG magic
                total_samples_read++;
            }
        });
    }

    // Wait for all workers
    for (auto& t : threads) {
        t.join();
    }

    // All samples should be read exactly once
    assert(total_samples_read == 3);

    std::remove(tar_path.c_str());
}

TEST(independent_file_descriptors) {
    std::string tar_path = create_test_tar();

    // Create multiple readers simultaneously
    TarReaderV2 r1(tar_path, 0, 2);
    TarReaderV2 r2(tar_path, 1, 2);

    // Both should be able to read independently
    auto d1 = r1.get_sample(0);
    auto d2 = r2.get_sample(0);

    assert(d1.size() > 0);
    assert(d2.size() > 0);

    // Data pointers should be in same mmap region but different offsets
    assert(d1.data() != d2.data());

    std::remove(tar_path.c_str());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(invalid_tar_path) {
    try {
        TarReaderV2 reader("/nonexistent/file.tar", 0, 1);
        assert(false);  // Should throw
    } catch (const std::runtime_error& e) {
        // Expected
        assert(std::string(e.what()).find("Failed to open") != std::string::npos);
    }
}

TEST(out_of_range_sample) {
    std::string tar_path = create_test_tar();
    TarReaderV2 reader(tar_path, 0, 1);

    try {
        auto data = reader.get_sample(999);
        assert(false);  // Should throw
    } catch (const std::out_of_range&) {
        // Expected
    }

    std::remove(tar_path.c_str());
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(large_dataset_partitioning) {
    // Test with larger dataset simulation
    std::string tar_path = create_test_tar();
    constexpr size_t NUM_WORKERS = 8;

    size_t total = 0;

    // Create readers sequentially and count samples
    for (size_t i = 0; i < NUM_WORKERS; ++i) {
        TarReaderV2 reader(tar_path, i, NUM_WORKERS);
        total += reader.num_samples();
        // Reader destroyed here, freeing resources
    }

    // All workers combined should cover all samples
    assert(total == 3);

    std::remove(tar_path.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  TarReaderV2 Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nBasic Tests:" << std::endl;
    RUN_TEST(tar_header_parsing);
    RUN_TEST(tar_reader_basic);
    RUN_TEST(tar_reader_zero_copy);
    RUN_TEST(tar_reader_entry_metadata);

    std::cout << "\nWorker Partitioning:" << std::endl;
    RUN_TEST(single_worker_partition);
    RUN_TEST(two_worker_partition);
    RUN_TEST(four_worker_partition);

    std::cout << "\nMulti-threaded:" << std::endl;
    RUN_TEST(concurrent_readers);
    RUN_TEST(independent_file_descriptors);

    std::cout << "\nError Handling:" << std::endl;
    RUN_TEST(invalid_tar_path);
    RUN_TEST(out_of_range_sample);

    std::cout << "\nPerformance:" << std::endl;
    RUN_TEST(large_dataset_partitioning);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  ALL TAR READER TESTS PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
