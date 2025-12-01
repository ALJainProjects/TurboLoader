/**
 * @file tfrecord_reader.hpp
 * @brief TFRecord format reader for TensorFlow ecosystem compatibility
 *
 * TFRecord is TensorFlow's native binary format for storing datasets.
 * This reader supports:
 * - Reading single and sharded TFRecords
 * - Parsing Example and SequenceExample protos
 * - Feature extraction (bytes, float, int64)
 *
 * Note: This is a lightweight implementation that doesn't require protobuf.
 * It parses the wire format directly.
 *
 * Usage:
 * ```cpp
 * TFRecordReader reader("/path/to/data.tfrecord");
 * while (reader.has_next()) {
 *     auto example = reader.next();
 *     auto image_data = example.get_bytes("image/encoded");
 *     auto label = example.get_int64("image/class/label");
 * }
 * ```
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <glob.h>

namespace turboloader {
namespace readers {

/**
 * @brief CRC32C checksum (Castagnoli polynomial)
 */
class CRC32C {
public:
    static uint32_t compute(const uint8_t* data, size_t length) {
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < length; i++) {
            crc = table_[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFF;
    }

    static uint32_t masked(uint32_t crc) {
        return ((crc >> 15) | (crc << 17)) + 0xa282ead8;
    }

private:
    static constexpr uint32_t table_[256] = {
        0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4, 0xc79a971f, 0x35f1141c,
        0x26a1e7e8, 0xd4ca64eb, 0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
        0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24, 0x105ec76f, 0xe235446c,
        0xf165b798, 0x030e349b, 0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
        0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54, 0x5d1d08bf, 0xaf768bbc,
        0xbc267848, 0x4e4dfb4b, 0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
        0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35, 0xaa64d611, 0x580f5512,
        0x4b5fa6e6, 0xb93425e5, 0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
        0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45, 0xf779deae, 0x05125dad,
        0x1642ae59, 0xe4292d5a, 0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
        0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595, 0x417b1dbc, 0xb3109ebf,
        0xa0406d4b, 0x522bee48, 0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
        0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687, 0x0c38d26c, 0xfe53516f,
        0xed03a29b, 0x1f682198, 0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
        0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38, 0xdbfc821c, 0x2997011f,
        0x3ac7f2eb, 0xc8ac71e8, 0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
        0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096, 0xa65c047d, 0x5437877e,
        0x4767748a, 0xb50cf789, 0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
        0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46, 0x7198540d, 0x83f3d70e,
        0x90a324fa, 0x62c8a7f9, 0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
        0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36, 0x3cdb9bdd, 0xceb018de,
        0xdde0eb2a, 0x2f8b6829, 0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
        0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93, 0x082f63b7, 0xfa44e0b4,
        0xe9141340, 0x1b7f9043, 0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
        0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3, 0x55326b08, 0xa759e80b,
        0xb4091bff, 0x466298fc, 0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
        0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033, 0xa24bb5a6, 0x502036a5,
        0x4370c551, 0xb11b4652, 0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x76819f4d,
        0x2892a369, 0xdaf9206a, 0xc9a9d39e, 0x3bc2509d, 0xef083476, 0x1d63b775,
        0x0e330a81, 0xfc588982, 0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
        0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622, 0x38cc2a06, 0xcaa7a905,
        0xd9f75af1, 0x2b9cd9f2, 0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
        0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530, 0x0417b1db, 0xf67c32d8,
        0xe52cc12c, 0x1747422f, 0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a048dff,
        0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0, 0xd3d3e1ab, 0x21b862a8,
        0x32e8915c, 0xc083125f, 0x144976b4, 0xe622f5b7, 0xf5720643, 0x07198540,
        0x590ab964, 0xab613a67, 0xb831c993, 0x4a5a4a90, 0x9e902e7b, 0x6cfbad78,
        0x7fab5e8c, 0x8dc0dd8f, 0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
        0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1, 0x69e9f0d5, 0x9b8273d6,
        0x88d28022, 0x7ab90321, 0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
        0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81, 0x34f4f86a, 0xc69f7b69,
        0xd5cf889d, 0x27a40b9e, 0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
        0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351
    };
};

/**
 * @brief Parsed TensorFlow Example
 */
class TFExample {
public:
    /**
     * @brief Get bytes feature
     */
    std::vector<std::string> get_bytes_list(const std::string& key) const {
        auto it = bytes_features_.find(key);
        if (it != bytes_features_.end()) {
            return it->second;
        }
        return {};
    }

    std::string get_bytes(const std::string& key) const {
        auto list = get_bytes_list(key);
        return list.empty() ? "" : list[0];
    }

    /**
     * @brief Get float feature
     */
    std::vector<float> get_float_list(const std::string& key) const {
        auto it = float_features_.find(key);
        if (it != float_features_.end()) {
            return it->second;
        }
        return {};
    }

    float get_float(const std::string& key, float default_val = 0.0f) const {
        auto list = get_float_list(key);
        return list.empty() ? default_val : list[0];
    }

    /**
     * @brief Get int64 feature
     */
    std::vector<int64_t> get_int64_list(const std::string& key) const {
        auto it = int64_features_.find(key);
        if (it != int64_features_.end()) {
            return it->second;
        }
        return {};
    }

    int64_t get_int64(const std::string& key, int64_t default_val = 0) const {
        auto list = get_int64_list(key);
        return list.empty() ? default_val : list[0];
    }

    /**
     * @brief Check if feature exists
     */
    bool has_feature(const std::string& key) const {
        return bytes_features_.count(key) ||
               float_features_.count(key) ||
               int64_features_.count(key);
    }

    /**
     * @brief Get all feature names
     */
    std::vector<std::string> feature_names() const {
        std::vector<std::string> names;
        for (const auto& [key, _] : bytes_features_) names.push_back(key);
        for (const auto& [key, _] : float_features_) names.push_back(key);
        for (const auto& [key, _] : int64_features_) names.push_back(key);
        return names;
    }

    // Internal setters used by parser
    void set_bytes(const std::string& key, std::vector<std::string> value) {
        bytes_features_[key] = std::move(value);
    }

    void set_float(const std::string& key, std::vector<float> value) {
        float_features_[key] = std::move(value);
    }

    void set_int64(const std::string& key, std::vector<int64_t> value) {
        int64_features_[key] = std::move(value);
    }

private:
    std::unordered_map<std::string, std::vector<std::string>> bytes_features_;
    std::unordered_map<std::string, std::vector<float>> float_features_;
    std::unordered_map<std::string, std::vector<int64_t>> int64_features_;
};

/**
 * @brief TFRecord file reader
 *
 * TFRecord format:
 * - uint64 length
 * - uint32 masked_crc32_of_length
 * - data[length]
 * - uint32 masked_crc32_of_data
 */
class TFRecordReader {
public:
    /**
     * @brief Open single TFRecord file
     */
    explicit TFRecordReader(const std::string& file_path)
        : current_file_idx_(0) {
        files_.push_back(file_path);
        open_current_file();
    }

    /**
     * @brief Open sharded TFRecord files matching pattern
     * Pattern like "/path/to/data-*.tfrecord"
     */
    static std::unique_ptr<TFRecordReader> from_pattern(const std::string& pattern) {
        glob_t glob_result;
        glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);

        std::vector<std::string> files;
        for (size_t i = 0; i < glob_result.gl_pathc; i++) {
            files.push_back(glob_result.gl_pathv[i]);
        }
        globfree(&glob_result);

        if (files.empty()) {
            throw std::runtime_error("No files matching pattern: " + pattern);
        }

        std::sort(files.begin(), files.end());

        auto reader = std::make_unique<TFRecordReader>();
        reader->files_ = std::move(files);
        reader->open_current_file();
        return reader;
    }

    ~TFRecordReader() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    /**
     * @brief Check if more records are available
     */
    bool has_next() {
        if (!file_.is_open()) return false;

        // Check if current file has more data
        if (!file_.eof() && file_.peek() != EOF) {
            return true;
        }

        // Try next file
        current_file_idx_++;
        if (current_file_idx_ < files_.size()) {
            open_current_file();
            return has_next();
        }

        return false;
    }

    /**
     * @brief Read next record as raw bytes
     */
    std::vector<uint8_t> next_raw() {
        if (!file_.is_open()) {
            throw std::runtime_error("No file open");
        }

        // Read length (8 bytes)
        uint64_t length;
        file_.read(reinterpret_cast<char*>(&length), sizeof(length));
        if (file_.gcount() != sizeof(length)) {
            throw std::runtime_error("Failed to read record length");
        }

        // Read length CRC (4 bytes)
        uint32_t length_crc;
        file_.read(reinterpret_cast<char*>(&length_crc), sizeof(length_crc));

        // Verify length CRC
        uint32_t computed_crc = CRC32C::masked(
            CRC32C::compute(reinterpret_cast<uint8_t*>(&length), sizeof(length)));
        if (length_crc != computed_crc) {
            throw std::runtime_error("Length CRC mismatch");
        }

        // Read data
        std::vector<uint8_t> data(length);
        file_.read(reinterpret_cast<char*>(data.data()), length);
        if (static_cast<size_t>(file_.gcount()) != length) {
            throw std::runtime_error("Failed to read record data");
        }

        // Read data CRC (4 bytes)
        uint32_t data_crc;
        file_.read(reinterpret_cast<char*>(&data_crc), sizeof(data_crc));

        // Verify data CRC
        computed_crc = CRC32C::masked(CRC32C::compute(data.data(), data.size()));
        if (data_crc != computed_crc) {
            throw std::runtime_error("Data CRC mismatch");
        }

        records_read_++;
        return data;
    }

    /**
     * @brief Read and parse next Example
     */
    TFExample next() {
        auto data = next_raw();
        return parse_example(data.data(), data.size());
    }

    /**
     * @brief Reset to beginning
     */
    void reset() {
        current_file_idx_ = 0;
        records_read_ = 0;
        open_current_file();
    }

    /**
     * @brief Get number of files
     */
    size_t num_files() const { return files_.size(); }

    /**
     * @brief Get current file index
     */
    size_t current_file() const { return current_file_idx_; }

    /**
     * @brief Get total records read
     */
    size_t records_read() const { return records_read_; }

private:
    TFRecordReader() : current_file_idx_(0), records_read_(0) {}

    void open_current_file() {
        if (file_.is_open()) {
            file_.close();
        }

        if (current_file_idx_ < files_.size()) {
            file_.open(files_[current_file_idx_], std::ios::binary);
            if (!file_.is_open()) {
                throw std::runtime_error("Failed to open: " + files_[current_file_idx_]);
            }
        }
    }

    /**
     * @brief Parse Example protobuf
     * Wire format parsing without protobuf library
     */
    TFExample parse_example(const uint8_t* data, size_t size) {
        TFExample example;
        size_t pos = 0;

        while (pos < size) {
            // Read field tag
            uint64_t tag = read_varint(data, size, pos);
            uint32_t field_number = tag >> 3;
            uint32_t wire_type = tag & 0x7;

            if (field_number == 1 && wire_type == 2) {
                // Features field (length-delimited)
                uint64_t features_len = read_varint(data, size, pos);
                size_t features_end = pos + features_len;

                parse_features(example, data + pos, features_len);
                pos = features_end;
            } else {
                // Skip unknown field
                skip_field(data, size, pos, wire_type);
            }
        }

        return example;
    }

    void parse_features(TFExample& example, const uint8_t* data, size_t size) {
        size_t pos = 0;

        while (pos < size) {
            uint64_t tag = read_varint(data, size, pos);
            uint32_t field_number = tag >> 3;
            uint32_t wire_type = tag & 0x7;

            if (field_number == 1 && wire_type == 2) {
                // Feature map entry
                uint64_t entry_len = read_varint(data, size, pos);
                size_t entry_end = pos + entry_len;

                std::string key;
                const uint8_t* value_data = nullptr;
                size_t value_len = 0;

                // Parse map entry
                while (pos < entry_end) {
                    uint64_t entry_tag = read_varint(data, size, pos);
                    uint32_t entry_field = entry_tag >> 3;
                    uint32_t entry_wire = entry_tag & 0x7;

                    if (entry_field == 1 && entry_wire == 2) {
                        // Key (string)
                        uint64_t key_len = read_varint(data, size, pos);
                        key = std::string(reinterpret_cast<const char*>(data + pos), key_len);
                        pos += key_len;
                    } else if (entry_field == 2 && entry_wire == 2) {
                        // Value (Feature message)
                        value_len = read_varint(data, size, pos);
                        value_data = data + pos;
                        pos += value_len;
                    } else {
                        skip_field(data, size, pos, entry_wire);
                    }
                }

                if (!key.empty() && value_data) {
                    parse_feature(example, key, value_data, value_len);
                }

                pos = entry_end;
            } else {
                skip_field(data, size, pos, wire_type);
            }
        }
    }

    void parse_feature(TFExample& example, const std::string& key,
                       const uint8_t* data, size_t size) {
        size_t pos = 0;

        while (pos < size) {
            uint64_t tag = read_varint(data, size, pos);
            uint32_t field_number = tag >> 3;
            uint32_t wire_type = tag & 0x7;

            if (wire_type == 2) {
                uint64_t list_len = read_varint(data, size, pos);
                const uint8_t* list_data = data + pos;

                if (field_number == 1) {
                    // BytesList
                    example.set_bytes(key, parse_bytes_list(list_data, list_len));
                } else if (field_number == 2) {
                    // FloatList
                    example.set_float(key, parse_float_list(list_data, list_len));
                } else if (field_number == 3) {
                    // Int64List
                    example.set_int64(key, parse_int64_list(list_data, list_len));
                }

                pos += list_len;
            } else {
                skip_field(data, size, pos, wire_type);
            }
        }
    }

    std::vector<std::string> parse_bytes_list(const uint8_t* data, size_t size) {
        std::vector<std::string> result;
        size_t pos = 0;

        while (pos < size) {
            uint64_t tag = read_varint(data, size, pos);
            uint32_t wire_type = tag & 0x7;

            if (wire_type == 2) {
                uint64_t len = read_varint(data, size, pos);
                result.emplace_back(reinterpret_cast<const char*>(data + pos), len);
                pos += len;
            } else {
                skip_field(data, size, pos, wire_type);
            }
        }

        return result;
    }

    std::vector<float> parse_float_list(const uint8_t* data, size_t size) {
        std::vector<float> result;
        size_t pos = 0;

        while (pos < size) {
            uint64_t tag = read_varint(data, size, pos);
            uint32_t field_number = tag >> 3;
            uint32_t wire_type = tag & 0x7;

            if (field_number == 1) {
                if (wire_type == 5) {
                    // Single float (32-bit fixed)
                    float value;
                    std::memcpy(&value, data + pos, sizeof(float));
                    result.push_back(value);
                    pos += sizeof(float);
                } else if (wire_type == 2) {
                    // Packed floats
                    uint64_t len = read_varint(data, size, pos);
                    size_t count = len / sizeof(float);
                    for (size_t i = 0; i < count; i++) {
                        float value;
                        std::memcpy(&value, data + pos + i * sizeof(float), sizeof(float));
                        result.push_back(value);
                    }
                    pos += len;
                }
            } else {
                skip_field(data, size, pos, wire_type);
            }
        }

        return result;
    }

    std::vector<int64_t> parse_int64_list(const uint8_t* data, size_t size) {
        std::vector<int64_t> result;
        size_t pos = 0;

        while (pos < size) {
            uint64_t tag = read_varint(data, size, pos);
            uint32_t field_number = tag >> 3;
            uint32_t wire_type = tag & 0x7;

            if (field_number == 1) {
                if (wire_type == 0) {
                    // Single varint
                    int64_t value = static_cast<int64_t>(read_varint(data, size, pos));
                    result.push_back(value);
                } else if (wire_type == 2) {
                    // Packed varints
                    uint64_t len = read_varint(data, size, pos);
                    size_t end = pos + len;
                    while (pos < end) {
                        int64_t value = static_cast<int64_t>(read_varint(data, size, pos));
                        result.push_back(value);
                    }
                }
            } else {
                skip_field(data, size, pos, wire_type);
            }
        }

        return result;
    }

    uint64_t read_varint(const uint8_t* data, size_t size, size_t& pos) {
        uint64_t result = 0;
        int shift = 0;

        while (pos < size) {
            uint8_t byte = data[pos++];
            result |= static_cast<uint64_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) {
                return result;
            }
            shift += 7;
        }

        throw std::runtime_error("Truncated varint");
    }

    void skip_field(const uint8_t* data, size_t size, size_t& pos, uint32_t wire_type) {
        switch (wire_type) {
            case 0:  // Varint
                read_varint(data, size, pos);
                break;
            case 1:  // 64-bit
                pos += 8;
                break;
            case 2:  // Length-delimited
                pos += read_varint(data, size, pos);
                break;
            case 5:  // 32-bit
                pos += 4;
                break;
            default:
                throw std::runtime_error("Unknown wire type: " + std::to_string(wire_type));
        }
    }

    std::vector<std::string> files_;
    size_t current_file_idx_;
    std::ifstream file_;
    size_t records_read_ = 0;
};

}  // namespace readers
}  // namespace turboloader
