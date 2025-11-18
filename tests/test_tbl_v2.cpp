/**
 * @file test_tbl_v2.cpp
 * @brief Tests for TBL v2 format, writer, and reader
 */

#include "../src/formats/tbl_v2_format.hpp"
#include "../src/writers/tbl_v2_writer.hpp"
#include "../src/readers/tbl_v2_reader.hpp"
#include "../src/utils/image_dimensions.hpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

using namespace turboloader;

// Simple JPEG header for testing (minimal valid JPEG)
const uint8_t JPEG_HEADER[] = {
    0xFF, 0xD8, 0xFF, 0xE0,  // SOI + APP0
    0x00, 0x10,              // APP0 length
    'J', 'F', 'I', 'F', 0x00,
    0x01, 0x01,              // Version
    0x00,                     // Units
    0x00, 0x01, 0x00, 0x01,  // X/Y density
    0x00, 0x00,              // Thumbnail
    0xFF, 0xC0,              // SOF0 (Start of Frame - Baseline DCT)
    0x00, 0x11,              // Length
    0x08,                     // Precision
    0x01, 0x00,              // Height: 256
    0x01, 0x00,              // Width: 256
    0x03,                     // Components
    0x01, 0x11, 0x00,        // Component 1
    0x02, 0x11, 0x01,        // Component 2
    0x03, 0x11, 0x01,        // Component 3
    0xFF, 0xD9               // EOI
};

void test_format_structures() {
    std::cout << "Testing format structures...\n";

    // Test header size
    assert(sizeof(formats::TblHeaderV2) == 64);
    std::cout << "  ✓ TblHeaderV2 is 64 bytes\n";

    // Test index entry size
    assert(sizeof(formats::TblIndexEntryV2) == 24);
    std::cout << "  ✓ TblIndexEntryV2 is 24 bytes\n";

    // Test metadata block header size
    assert(sizeof(formats::MetadataBlockHeader) == 16);
    std::cout << "  ✓ MetadataBlockHeader is 16 bytes\n";

    // Test header initialization
    formats::TblHeaderV2 header;
    assert(header.magic[0] == 'T' && header.magic[1] == 'B' &&
           header.magic[2] == 'L' && header.magic[3] == 0x02);
    assert(header.version == 2);
    assert(header.is_valid());
    std::cout << "  ✓ Header initialization correct\n";

    std::cout << "Format structures: PASSED\n\n";
}

void test_checksum_functions() {
    std::cout << "Testing checksum functions...\n";

    const uint8_t test_data[] = "Hello, TurboLoader!";
    size_t test_size = strlen(reinterpret_cast<const char*>(test_data));

    // Test CRC32
    uint32_t crc32 = formats::calculate_crc32(test_data, test_size);
    assert(crc32 != 0);
    std::cout << "  ✓ CRC32 calculation works\n";

    // Test CRC16
    uint16_t crc16 = formats::calculate_crc16(test_data, test_size);
    assert(crc16 != 0);
    std::cout << "  ✓ CRC16 calculation works\n";

    // Test consistency
    uint32_t crc32_2 = formats::calculate_crc32(test_data, test_size);
    assert(crc32 == crc32_2);
    std::cout << "  ✓ Checksum consistency verified\n";

    std::cout << "Checksum functions: PASSED\n\n";
}

void test_image_dimension_detection() {
    std::cout << "Testing image dimension detection...\n";

    // Test JPEG detection
    auto [width, height] = utils::detect_jpeg_dimensions(
        JPEG_HEADER, sizeof(JPEG_HEADER));

    assert(width == 256);
    assert(height == 256);
    std::cout << "  ✓ JPEG dimension detection works (" << width << "x" << height << ")\n";

    // Test auto-detection
    auto [w2, h2] = utils::detect_image_dimensions(
        JPEG_HEADER, sizeof(JPEG_HEADER), formats::SampleFormat::UNKNOWN);

    assert(w2 == 256);
    assert(h2 == 256);
    std::cout << "  ✓ Auto-detection works\n";

    std::cout << "Image dimension detection: PASSED\n\n";
}

void test_writer_and_reader() {
    std::cout << "Testing TBL v2 writer and reader...\n";

    const char* test_file = "/tmp/test_tbl_v2.tbl";

    // Create some test data
    std::vector<std::vector<uint8_t>> test_samples;
    for (int i = 0; i < 10; ++i) {
        std::vector<uint8_t> sample(JPEG_HEADER, JPEG_HEADER + sizeof(JPEG_HEADER));
        // Modify slightly to make samples different
        sample.push_back(static_cast<uint8_t>(i));
        test_samples.push_back(sample);
    }

    // Test writing
    {
        writers::TblWriterV2 writer(test_file, true);  // With compression

        for (size_t i = 0; i < test_samples.size(); ++i) {
            const auto& sample = test_samples[i];
            writer.add_sample(sample.data(), sample.size(),
                            formats::SampleFormat::JPEG, 256, 256);

            // Add metadata for every other sample
            if (i % 2 == 0) {
                std::string metadata = "{\"index\": " + std::to_string(i) + "}";
                writer.add_metadata(i, metadata);
            }
        }

        writer.finalize();
        std::cout << "  ✓ Writer created file successfully\n";
    }

    // Test reading
    {
        readers::TblReaderV2 reader(test_file, true);  // With checksum verification

        assert(reader.num_samples() == test_samples.size());
        std::cout << "  ✓ Reader reports correct sample count\n";

        assert(reader.is_compressed());
        std::cout << "  ✓ Reader detects compression\n";

        // Read and verify each sample
        for (size_t i = 0; i < test_samples.size(); ++i) {
            auto [data, size] = reader.read_sample(i);

            assert(size == test_samples[i].size());
            assert(std::memcmp(data, test_samples[i].data(), size) == 0);

            // Check sample info
            const auto& info = reader.get_sample_info(i);
            assert(info.format == formats::SampleFormat::JPEG);
            assert(info.width == 256);
            assert(info.height == 256);

            if (i == 0) {
                std::cout << "  ✓ Sample " << i << " read correctly (decompressed)\n";
            }
        }

        // Test metadata reading
        for (size_t i = 0; i < test_samples.size(); ++i) {
            auto [metadata, type] = reader.read_metadata(i);

            if (i % 2 == 0) {
                assert(!metadata.empty());
                assert(type == formats::MetadataType::JSON);
                if (i == 0) {
                    std::cout << "  ✓ Metadata read correctly: " << metadata << "\n";
                }
            } else {
                assert(metadata.empty());
            }
        }

        // Test dimension filtering
        auto jpeg_samples = reader.filter_by_format(formats::SampleFormat::JPEG);
        assert(jpeg_samples.size() == test_samples.size());
        std::cout << "  ✓ Format filtering works\n";

        auto sized_samples = reader.filter_by_dimensions(256, 256, 256, 256);
        assert(sized_samples.size() == test_samples.size());
        std::cout << "  ✓ Dimension filtering works\n";
    }

    // Clean up
    std::remove(test_file);

    std::cout << "Writer and reader: PASSED\n\n";
}

void test_uncompressed_mode() {
    std::cout << "Testing uncompressed mode...\n";

    const char* test_file = "/tmp/test_tbl_v2_uncompressed.tbl";

    // Write uncompressed
    {
        writers::TblWriterV2 writer(test_file, false);  // No compression

        std::vector<uint8_t> sample(JPEG_HEADER, JPEG_HEADER + sizeof(JPEG_HEADER));
        writer.add_sample(sample.data(), sample.size(),
                        formats::SampleFormat::JPEG, 256, 256);
        writer.finalize();
    }

    // Read and verify
    {
        readers::TblReaderV2 reader(test_file, true);

        assert(reader.num_samples() == 1);
        assert(!reader.is_compressed());
        std::cout << "  ✓ Uncompressed mode works\n";

        auto [data, size] = reader.read_sample(0);
        assert(size == sizeof(JPEG_HEADER));
        std::cout << "  ✓ Uncompressed sample read correctly\n";
    }

    std::remove(test_file);

    std::cout << "Uncompressed mode: PASSED\n\n";
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "TBL V2 FORMAT TEST SUITE\n";
    std::cout << "================================================================================\n\n";

    try {
        test_format_structures();
        test_checksum_functions();
        test_image_dimension_detection();
        test_writer_and_reader();
        test_uncompressed_mode();

        std::cout << "================================================================================\n";
        std::cout << "ALL TESTS PASSED\n";
        std::cout << "================================================================================\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
