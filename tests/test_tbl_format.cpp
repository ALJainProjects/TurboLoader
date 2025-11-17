/**
 * @file test_tbl_format.cpp
 * @brief Unit tests for TBL format implementation
 *
 * Tests cover:
 * - Header writing/reading
 * - Index table correctness
 * - Sample data integrity
 * - TAR to TBL conversion
 * - Performance benchmarks
 */

#include "../src/formats/tbl_format.hpp"
#include "../src/readers/tbl_reader.hpp"
#include "../src/writers/tbl_writer.hpp"
#include "../src/readers/tar_reader.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>

using namespace turboloader;
using namespace turboloader::formats;
using namespace turboloader::readers;
using namespace turboloader::writers;

// Test helper: create a sample JPEG header (minimal valid JPEG)
std::vector<uint8_t> create_test_jpeg(size_t width, size_t height) {
    std::vector<uint8_t> jpeg;

    // JPEG SOI marker
    jpeg.push_back(0xFF);
    jpeg.push_back(0xD8);

    // JPEG APP0 marker (JFIF)
    jpeg.push_back(0xFF);
    jpeg.push_back(0xE0);
    jpeg.push_back(0x00);
    jpeg.push_back(0x10);
    jpeg.push_back('J');
    jpeg.push_back('F');
    jpeg.push_back('I');
    jpeg.push_back('F');
    jpeg.push_back(0x00);
    jpeg.push_back(0x01);
    jpeg.push_back(0x01);
    jpeg.push_back(0x00);
    jpeg.push_back(0x00);
    jpeg.push_back(0x01);
    jpeg.push_back(0x00);
    jpeg.push_back(0x01);
    jpeg.push_back(0x00);
    jpeg.push_back(0x00);

    // Add some dummy data
    for (size_t i = 0; i < width * height / 100; ++i) {
        jpeg.push_back(static_cast<uint8_t>(i & 0xFF));
    }

    // JPEG EOI marker
    jpeg.push_back(0xFF);
    jpeg.push_back(0xD9);

    return jpeg;
}

// Test 1: Basic header writing and reading
void test_header_write_read() {
    std::cout << "\n[TEST 1] Header Write/Read" << std::endl;

    const std::string filepath = "/tmp/test_tbl_header.tbl";

    // Write TBL file
    {
        TblWriter writer(filepath);

        // Add 100 test samples
        for (size_t i = 0; i < 100; ++i) {
            auto jpeg = create_test_jpeg(256, 256);
            writer.add_sample(jpeg.data(), jpeg.size(), SampleFormat::JPEG);
        }

        writer.finalize();

        assert(writer.num_samples() == 100);
        std::cout << "  ✓ Wrote 100 samples" << std::endl;
    }

    // Read TBL file
    {
        TblReader reader(filepath);

        assert(reader.num_samples() == 100);
        std::cout << "  ✓ Read header: 100 samples" << std::endl;

        // Verify we can read all samples
        for (size_t i = 0; i < 100; ++i) {
            auto [data, size] = reader.read_sample(i);
            assert(data != nullptr);
            assert(size > 0);

            // Verify JPEG markers
            assert(data[0] == 0xFF && data[1] == 0xD8);  // SOI
            assert(data[size-2] == 0xFF && data[size-1] == 0xD9);  // EOI
        }

        std::cout << "  ✓ Verified all samples readable" << std::endl;
    }

    std::cout << "✅ PASS" << std::endl;
}

// Test 2: Index table correctness
void test_index_table() {
    std::cout << "\n[TEST 2] Index Table Correctness" << std::endl;

    const std::string filepath = "/tmp/test_tbl_index.tbl";

    // Create samples of different sizes
    std::vector<std::vector<uint8_t>> test_samples;
    std::vector<size_t> expected_sizes;

    for (size_t i = 0; i < 50; ++i) {
        size_t width = 128 + (i * 10);  // Varying sizes
        size_t height = 128 + (i * 5);
        auto jpeg = create_test_jpeg(width, height);
        test_samples.push_back(jpeg);
        expected_sizes.push_back(jpeg.size());
    }

    // Write
    {
        TblWriter writer(filepath);
        for (const auto& sample : test_samples) {
            writer.add_sample(sample.data(), sample.size(), SampleFormat::JPEG);
        }
        writer.finalize();
    }

    // Read and verify sizes
    {
        TblReader reader(filepath);
        assert(reader.num_samples() == test_samples.size());

        for (size_t i = 0; i < test_samples.size(); ++i) {
            auto [data, size] = reader.read_sample(i);
            assert(size == expected_sizes[i]);

            // Verify content matches
            assert(std::memcmp(data, test_samples[i].data(), size) == 0);
        }

        std::cout << "  ✓ All sample sizes correct" << std::endl;
        std::cout << "  ✓ All sample data matches" << std::endl;
    }

    std::cout << "✅ PASS" << std::endl;
}

// Test 3: Multiple format support
void test_multiple_formats() {
    std::cout << "\n[TEST 3] Multiple Format Support" << std::endl;

    const std::string filepath = "/tmp/test_tbl_formats.tbl";

    std::vector<SampleFormat> formats = {
        SampleFormat::JPEG,
        SampleFormat::PNG,
        SampleFormat::WEBP,
        SampleFormat::BMP,
        SampleFormat::TIFF
    };

    // Write
    {
        TblWriter writer(filepath);
        for (auto format : formats) {
            auto jpeg = create_test_jpeg(256, 256);
            writer.add_sample(jpeg.data(), jpeg.size(), format);
        }
        writer.finalize();
    }

    // Read and verify formats
    {
        TblReader reader(filepath);
        assert(reader.num_samples() == formats.size());

        for (size_t i = 0; i < formats.size(); ++i) {
            SampleFormat format = reader.get_sample_format(i);
            assert(format == formats[i]);
            std::cout << "  ✓ Sample " << i << ": " << format_to_string(format) << std::endl;
        }
    }

    std::cout << "✅ PASS" << std::endl;
}

// Test 4: Extension to format conversion
void test_extension_detection() {
    std::cout << "\n[TEST 4] Extension to Format Conversion" << std::endl;

    assert(extension_to_format("image.jpg") == SampleFormat::JPEG);
    assert(extension_to_format("image.jpeg") == SampleFormat::JPEG);
    assert(extension_to_format("image.JPG") == SampleFormat::JPEG);
    assert(extension_to_format("image.png") == SampleFormat::PNG);
    assert(extension_to_format("image.PNG") == SampleFormat::PNG);
    assert(extension_to_format("image.webp") == SampleFormat::WEBP);
    assert(extension_to_format("image.bmp") == SampleFormat::BMP);
    assert(extension_to_format("image.tif") == SampleFormat::TIFF);
    assert(extension_to_format("image.tiff") == SampleFormat::TIFF);
    assert(extension_to_format("no_extension") == SampleFormat::UNKNOWN);

    std::cout << "  ✓ JPEG extensions" << std::endl;
    std::cout << "  ✓ PNG extensions" << std::endl;
    std::cout << "  ✓ WebP extensions" << std::endl;
    std::cout << "  ✓ BMP extensions" << std::endl;
    std::cout << "  ✓ TIFF extensions" << std::endl;
    std::cout << "  ✓ Unknown handling" << std::endl;

    std::cout << "✅ PASS" << std::endl;
}

// Test 5: Large file handling
void test_large_file() {
    std::cout << "\n[TEST 5] Large File Handling" << std::endl;

    const std::string filepath = "/tmp/test_tbl_large.tbl";
    const size_t num_samples = 1000;

    auto start_write = std::chrono::high_resolution_clock::now();

    // Write
    {
        TblWriter writer(filepath);
        for (size_t i = 0; i < num_samples; ++i) {
            auto jpeg = create_test_jpeg(256, 256);
            writer.add_sample(jpeg.data(), jpeg.size(), SampleFormat::JPEG);
        }
        writer.finalize();
    }

    auto end_write = std::chrono::high_resolution_clock::now();
    auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_write - start_write).count();

    std::cout << "  ✓ Wrote " << num_samples << " samples in " << write_ms << " ms" << std::endl;
    std::cout << "  ✓ Write rate: " << (num_samples * 1000.0 / write_ms) << " samples/s" << std::endl;

    auto start_read = std::chrono::high_resolution_clock::now();

    // Read (sequential)
    {
        TblReader reader(filepath);
        assert(reader.num_samples() == num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            auto [data, size] = reader.read_sample(i);
            assert(data != nullptr);
            assert(size > 0);
        }
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count();

    std::cout << "  ✓ Read " << num_samples << " samples in " << read_ms << " ms" << std::endl;
    std::cout << "  ✓ Read rate: " << (num_samples * 1000.0 / read_ms) << " samples/s" << std::endl;

    // Get file size
    struct stat st;
    stat(filepath.c_str(), &st);
    double size_mb = st.st_size / (1024.0 * 1024.0);
    std::cout << "  ✓ File size: " << size_mb << " MB" << std::endl;
    std::cout << "  ✓ Overhead per sample: " << (st.st_size / num_samples - 200) << " bytes (header + index)" << std::endl;

    std::cout << "✅ PASS" << std::endl;
}

// Test 6: Random access performance
void test_random_access() {
    std::cout << "\n[TEST 6] Random Access Performance" << std::endl;

    const std::string filepath = "/tmp/test_tbl_random.tbl";
    const size_t num_samples = 1000;

    // Create TBL file
    {
        TblWriter writer(filepath);
        for (size_t i = 0; i < num_samples; ++i) {
            auto jpeg = create_test_jpeg(256, 256);
            writer.add_sample(jpeg.data(), jpeg.size(), SampleFormat::JPEG);
        }
        writer.finalize();
    }

    // Random access test
    {
        TblReader reader(filepath);

        // Generate random indices
        std::vector<size_t> random_indices;
        for (size_t i = 0; i < 500; ++i) {
            random_indices.push_back(rand() % num_samples);
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t idx : random_indices) {
            auto [data, size] = reader.read_sample(idx);
            assert(data != nullptr);
            (void)size;  // Avoid unused warning
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "  ✓ Random access to 500 samples: " << elapsed_us << " μs" << std::endl;
        std::cout << "  ✓ Average access time: " << (elapsed_us / 500.0) << " μs/sample" << std::endl;
    }

    std::cout << "✅ PASS" << std::endl;
}

// Test 7: Error handling
void test_error_handling() {
    std::cout << "\n[TEST 7] Error Handling" << std::endl;

    const std::string filepath = "/tmp/test_tbl_errors.tbl";

    // Create valid file
    {
        TblWriter writer(filepath);
        for (size_t i = 0; i < 10; ++i) {
            auto jpeg = create_test_jpeg(256, 256);
            writer.add_sample(jpeg.data(), jpeg.size(), SampleFormat::JPEG);
        }
        writer.finalize();
    }

    // Test out-of-range access
    {
        TblReader reader(filepath);

        try {
            reader.read_sample(100);  // Out of range (only 10 samples)
            assert(false && "Should have thrown out_of_range");
        } catch (const std::out_of_range& e) {
            std::cout << "  ✓ Caught out-of-range error: " << e.what() << std::endl;
        }
    }

    // Test invalid file
    {
        try {
            TblReader reader("/nonexistent/file.tbl");
            assert(false && "Should have thrown error");
        } catch (const std::runtime_error& e) {
            std::cout << "  ✓ Caught file open error: " << e.what() << std::endl;
        }
    }

    // Test corrupted header
    {
        const std::string corrupt_filepath = "/tmp/test_tbl_corrupt.tbl";

        // Create file with invalid magic
        std::ofstream corrupt(corrupt_filepath, std::ios::binary);
        corrupt.write("INVALID", 7);
        corrupt.close();

        try {
            TblReader reader(corrupt_filepath);
            assert(false && "Should have thrown error");
        } catch (const std::runtime_error& e) {
            std::cout << "  ✓ Caught invalid header error: " << e.what() << std::endl;
        }
    }

    std::cout << "✅ PASS" << std::endl;
}

// Test 8: TAR to TBL conversion (if TAR file exists)
void test_tar_conversion() {
    std::cout << "\n[TEST 8] TAR to TBL Conversion" << std::endl;

    // Check if test TAR exists
    const std::string tar_path = "/tmp/test_prefetch.tar";
    const std::string tbl_path = "/tmp/test_converted.tbl";

    struct stat st;
    if (stat(tar_path.c_str(), &st) != 0) {
        std::cout << "  ⚠️  SKIPPED (no test TAR file at " << tar_path << ")" << std::endl;
        return;
    }

    // Open TAR (worker 0, 1 total worker - to get all samples)
    TarReader tar_reader(tar_path, 0, 1);
    const size_t num_samples = tar_reader.num_samples();

    std::cout << "  Found " << num_samples << " files in TAR" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Convert to TBL
    {
        TblWriter tbl_writer(tbl_path);

        for (size_t i = 0; i < num_samples; ++i) {
            auto sample_data = tar_reader.get_sample(i);
            const auto& entry = tar_reader.get_entry(i);
            SampleFormat format = extension_to_format(entry.name);

            tbl_writer.add_sample(
                sample_data.data(),
                sample_data.size(),
                format
            );
        }

        tbl_writer.finalize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Verify conversion
    {
        TblReader tbl_reader(tbl_path);
        assert(tbl_reader.num_samples() == num_samples);

        // Spot check a few samples
        for (size_t i = 0; i < std::min(num_samples, size_t(10)); ++i) {
            auto tar_sample = tar_reader.get_sample(i);
            auto [tbl_data, tbl_size] = tbl_reader.read_sample(i);

            assert(tbl_size == tar_sample.size());
            assert(std::memcmp(tbl_data, tar_sample.data(), tbl_size) == 0);
        }
    }

    // Get file sizes
    struct stat tar_st, tbl_st;
    stat(tar_path.c_str(), &tar_st);
    stat(tbl_path.c_str(), &tbl_st);

    double tar_mb = tar_st.st_size / (1024.0 * 1024.0);
    double tbl_mb = tbl_st.st_size / (1024.0 * 1024.0);
    double overhead_reduction = 100.0 * (1.0 - tbl_mb / tar_mb);

    std::cout << "  ✓ Converted " << num_samples << " samples in " << elapsed_ms << " ms" << std::endl;
    std::cout << "  ✓ Conversion rate: " << (num_samples * 1000.0 / elapsed_ms) << " samples/s" << std::endl;
    std::cout << "  ✓ TAR size: " << tar_mb << " MB" << std::endl;
    std::cout << "  ✓ TBL size: " << tbl_mb << " MB" << std::endl;
    std::cout << "  ✓ Size reduction: " << overhead_reduction << "%" << std::endl;
    std::cout << "  ✓ Spot-checked sample data integrity" << std::endl;

    std::cout << "✅ PASS" << std::endl;
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "TBL FORMAT UNIT TESTS\n";
    std::cout << "================================================================================\n";

    try {
        test_header_write_read();
        test_index_table();
        test_multiple_formats();
        test_extension_detection();
        test_large_file();
        test_random_access();
        test_error_handling();
        test_tar_conversion();

        std::cout << "\n================================================================================\n";
        std::cout << "ALL TESTS PASSED ✅\n";
        std::cout << "================================================================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
