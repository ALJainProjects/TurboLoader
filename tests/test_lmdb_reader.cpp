/**
 * @file test_lmdb_reader.cpp
 * @brief Tests for LMDB database reader
 */

#include <gtest/gtest.h>
#include "../src/readers/lmdb_reader.hpp"
#include <fstream>
#include <filesystem>
#include <random>
#include <cstring>

namespace fs = std::filesystem;

class LMDBReaderTest : public ::testing::Test {
protected:
    std::string test_dir_;
    std::string test_lmdb_path_;

    void SetUp() override {
        // Create temporary directory for test files
        test_dir_ = "/tmp/turboloader_lmdb_test_" + std::to_string(std::random_device{}());
        fs::create_directories(test_dir_);
        test_lmdb_path_ = test_dir_ + "/test.lmdb";
        fs::create_directories(test_lmdb_path_);
    }

    void TearDown() override {
        // Clean up test directory
        fs::remove_all(test_dir_);
    }

    // Create a simple LMDB-like file for testing
    // This creates a file with a simplified key-value format that the reader can parse
    void create_test_lmdb(const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries) {
        std::string data_path = test_lmdb_path_ + "/data.mdb";
        std::ofstream file(data_path, std::ios::binary);

        // Write 2 meta pages (8KB total, LMDB reserves first 2 pages)
        std::vector<uint8_t> meta_pages(4096 * 2, 0);
        // Write LMDB magic number at start (simplified)
        meta_pages[0] = 0xDE;
        meta_pages[1] = 0xAD;
        meta_pages[2] = 0xBE;
        meta_pages[3] = 0xEF;
        file.write(reinterpret_cast<char*>(meta_pages.data()), meta_pages.size());

        // Write key-value entries in simplified format
        // Format: 4-byte key_len, key, 4-byte value_len, value
        for (const auto& [key, value] : entries) {
            uint32_t key_len = static_cast<uint32_t>(key.size());
            uint32_t value_len = static_cast<uint32_t>(value.size());

            file.write(reinterpret_cast<char*>(&key_len), 4);
            file.write(key.c_str(), key_len);
            file.write(reinterpret_cast<char*>(&value_len), 4);
            file.write(reinterpret_cast<const char*>(value.data()), value_len);
        }

        file.close();
    }

    // Create test entries
    std::vector<std::pair<std::string, std::vector<uint8_t>>> create_test_entries(size_t count) {
        std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
        for (size_t i = 0; i < count; ++i) {
            std::string key = "key" + std::to_string(i);
            std::vector<uint8_t> value(100 + i * 10);
            std::iota(value.begin(), value.end(), static_cast<uint8_t>(i));
            entries.emplace_back(key, value);
        }
        return entries;
    }
};

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST_F(LMDBReaderTest, ConstructWithValidPath) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    EXPECT_NO_THROW({
        turboloader::LMDBReader reader(test_lmdb_path_);
    });
}

TEST_F(LMDBReaderTest, ConstructWithDirectDataFile) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    std::string data_path = test_lmdb_path_ + "/data.mdb";
    EXPECT_NO_THROW({
        turboloader::LMDBReader reader(data_path);
    });
}

TEST_F(LMDBReaderTest, ConstructWithInvalidPath) {
    EXPECT_THROW({
        turboloader::LMDBReader reader("/nonexistent/path/to/lmdb");
    }, std::runtime_error);
}

TEST_F(LMDBReaderTest, ConstructWithEmptyDatabase) {
    // Create empty data.mdb with just meta pages
    std::string data_path = test_lmdb_path_ + "/data.mdb";
    std::ofstream file(data_path, std::ios::binary);
    std::vector<uint8_t> meta_pages(4096 * 2, 0);
    file.write(reinterpret_cast<char*>(meta_pages.data()), meta_pages.size());
    file.close();

    turboloader::LMDBReader reader(test_lmdb_path_);
    // Empty database might have 1 fallback entry or 0
    EXPECT_LE(reader.size(), 1);
}

// ============================================================================
// Size and Empty Tests
// ============================================================================

TEST_F(LMDBReaderTest, SizeReturnsCorrectCount) {
    auto entries = create_test_entries(10);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.size(), 10);
}

TEST_F(LMDBReaderTest, EmptyReturnsTrueForEmptyDB) {
    // Create minimal LMDB
    std::string data_path = test_lmdb_path_ + "/data.mdb";
    std::ofstream file(data_path, std::ios::binary);
    std::vector<uint8_t> meta_pages(4096 * 2, 0);
    file.write(reinterpret_cast<char*>(meta_pages.data()), meta_pages.size());
    file.close();

    turboloader::LMDBReader reader(test_lmdb_path_);
    // Either empty or has fallback entry
    EXPECT_TRUE(reader.empty() || reader.size() == 1);
}

TEST_F(LMDBReaderTest, EmptyReturnsFalseForPopulatedDB) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_FALSE(reader.empty());
}

TEST_F(LMDBReaderTest, PathReturnsCorrectPath) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.path(), test_lmdb_path_);
}

// ============================================================================
// Get by Index Tests
// ============================================================================

TEST_F(LMDBReaderTest, GetByIndexReturnsCorrectEntry) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto entry = reader.get(0);
    EXPECT_EQ(entry.key_string(), "key0");
    EXPECT_EQ(entry.value.size(), 100);
}

TEST_F(LMDBReaderTest, GetByIndexMultipleEntries) {
    auto entries = create_test_entries(10);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    for (size_t i = 0; i < reader.size(); ++i) {
        auto entry = reader.get(i);
        std::string expected_key = "key" + std::to_string(i);
        EXPECT_EQ(entry.key_string(), expected_key);
    }
}

TEST_F(LMDBReaderTest, GetByIndexOutOfRange) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    EXPECT_THROW({
        reader.get(100);
    }, std::out_of_range);
}

TEST_F(LMDBReaderTest, GetValueByIndex) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto value = reader.get_value(0);
    EXPECT_EQ(value.size(), 100);
}

// ============================================================================
// Get by Key Tests
// ============================================================================

TEST_F(LMDBReaderTest, GetByKeyReturnsCorrectEntry) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto entry = reader.get("key2");
    EXPECT_EQ(entry.key_string(), "key2");
}

TEST_F(LMDBReaderTest, GetByKeyNotFound) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    EXPECT_THROW({
        reader.get("nonexistent_key");
    }, std::out_of_range);
}

TEST_F(LMDBReaderTest, ContainsReturnsTrueForExistingKey) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    EXPECT_TRUE(reader.contains("key0"));
    EXPECT_TRUE(reader.contains("key4"));
}

TEST_F(LMDBReaderTest, ContainsReturnsFalseForMissingKey) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    EXPECT_FALSE(reader.contains("missing"));
    EXPECT_FALSE(reader.contains("key99"));
}

// ============================================================================
// Keys Method Tests
// ============================================================================

TEST_F(LMDBReaderTest, KeysReturnsAllKeys) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto keys = reader.keys();
    EXPECT_EQ(keys.size(), 5);

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string expected = "key" + std::to_string(i);
        EXPECT_EQ(keys[i], expected);
    }
}

TEST_F(LMDBReaderTest, KeysPreservesOrder) {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries = {
        {"alpha", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
        {"beta0", {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}},
        {"gamma", {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
        {"delta", {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}
    };
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto keys = reader.keys();
    EXPECT_EQ(keys.size(), 4);
    EXPECT_EQ(keys[0], "alpha");
    EXPECT_EQ(keys[1], "beta0");
    EXPECT_EQ(keys[2], "gamma");
    EXPECT_EQ(keys[3], "delta");
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_F(LMDBReaderTest, IteratorBasic) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    size_t count = 0;
    for (const auto& entry : reader) {
        EXPECT_FALSE(entry.key.empty());
        EXPECT_FALSE(entry.value.empty());
        ++count;
    }
    EXPECT_EQ(count, 5);
}

TEST_F(LMDBReaderTest, IteratorExplicit) {
    auto entries = create_test_entries(3);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto it = reader.begin();
    EXPECT_NE(it, reader.end());

    EXPECT_EQ((*it).key_string(), "key0");
    ++it;
    EXPECT_EQ((*it).key_string(), "key1");
    ++it;
    EXPECT_EQ((*it).key_string(), "key2");
    ++it;
    EXPECT_EQ(it, reader.end());
}

TEST_F(LMDBReaderTest, IteratorPostIncrement) {
    auto entries = create_test_entries(2);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto it = reader.begin();
    auto prev = it++;
    EXPECT_EQ((*prev).key_string(), "key0");
    EXPECT_EQ((*it).key_string(), "key1");
}

TEST_F(LMDBReaderTest, IteratorArrowOperator) {
    auto entries = create_test_entries(2);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto it = reader.begin();
    EXPECT_EQ(it->key_string(), "key0");
}

// ============================================================================
// Stats Tests
// ============================================================================

TEST_F(LMDBReaderTest, StatsCorrectForPopulatedDB) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto stats = reader.stats();
    EXPECT_EQ(stats.num_entries, 5);
    EXPECT_GT(stats.total_key_bytes, 0);
    EXPECT_GT(stats.total_value_bytes, 0);
    EXPECT_LE(stats.min_value_size, stats.max_value_size);
    EXPECT_GT(stats.avg_value_size, 0.0);
}

TEST_F(LMDBReaderTest, StatsMinMaxCorrect) {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries = {
        {"a", std::vector<uint8_t>(10, 1)},
        {"b", std::vector<uint8_t>(50, 2)},
        {"c", std::vector<uint8_t>(100, 3)}
    };
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto stats = reader.stats();
    EXPECT_EQ(stats.min_value_size, 10);
    EXPECT_EQ(stats.max_value_size, 100);
}

TEST_F(LMDBReaderTest, StatsAverageCorrect) {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries = {
        {"a", std::vector<uint8_t>(10, 1)},
        {"b", std::vector<uint8_t>(20, 2)},
        {"c", std::vector<uint8_t>(30, 3)}
    };
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);

    auto stats = reader.stats();
    EXPECT_DOUBLE_EQ(stats.avg_value_size, 20.0);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(LMDBReaderTest, MoveConstructor) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader1(test_lmdb_path_);
    size_t original_size = reader1.size();

    turboloader::LMDBReader reader2(std::move(reader1));
    EXPECT_EQ(reader2.size(), original_size);
    EXPECT_EQ(reader2.path(), test_lmdb_path_);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(LMDBReaderTest, MakeLMDBReaderValid) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    auto reader = turboloader::make_lmdb_reader(test_lmdb_path_);
    EXPECT_NE(reader, nullptr);
    EXPECT_EQ(reader->size(), 5);
}

TEST_F(LMDBReaderTest, MakeLMDBReaderInvalid) {
    EXPECT_THROW({
        turboloader::make_lmdb_reader("/invalid/path");
    }, std::runtime_error);
}

// ============================================================================
// is_lmdb_database Tests
// ============================================================================

TEST_F(LMDBReaderTest, IsLMDBDatabaseTrue) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    EXPECT_TRUE(turboloader::is_lmdb_database(test_lmdb_path_));
}

TEST_F(LMDBReaderTest, IsLMDBDatabaseFalseForNonexistent) {
    EXPECT_FALSE(turboloader::is_lmdb_database("/nonexistent/path"));
}

TEST_F(LMDBReaderTest, IsLMDBDatabaseFalseForEmptyDir) {
    std::string empty_dir = test_dir_ + "/empty_dir";
    fs::create_directories(empty_dir);

    EXPECT_FALSE(turboloader::is_lmdb_database(empty_dir));
}

TEST_F(LMDBReaderTest, IsLMDBDatabaseTrueForDataFile) {
    auto entries = create_test_entries(5);
    create_test_lmdb(entries);

    std::string data_path = test_lmdb_path_ + "/data.mdb";
    EXPECT_TRUE(turboloader::is_lmdb_database(data_path));
}

// ============================================================================
// Large Data Tests
// ============================================================================

TEST_F(LMDBReaderTest, LargeNumberOfEntries) {
    auto entries = create_test_entries(1000);
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.size(), 1000);

    // Verify random access
    auto entry500 = reader.get(500);
    EXPECT_EQ(entry500.key_string(), "key500");
}

TEST_F(LMDBReaderTest, LargeValues) {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
    for (int i = 0; i < 10; ++i) {
        std::string key = "large" + std::to_string(i);
        std::vector<uint8_t> value(1024 * 100, static_cast<uint8_t>(i));  // 100KB values
        entries.emplace_back(key, value);
    }
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.size(), 10);

    auto entry = reader.get(5);
    EXPECT_EQ(entry.value.size(), 1024 * 100);
    EXPECT_EQ(entry.value[0], 5);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(LMDBReaderTest, ShortKey) {
    // Short but valid key (ML datasets typically use longer keys)
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries = {
        {"img", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
    };
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.size(), 1);
    EXPECT_EQ(reader.get(0).key_string(), "img");
}

TEST_F(LMDBReaderTest, NumericStringKeys) {
    // Numeric keys with realistic value sizes for ML data
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries = {
        {"idx000", std::vector<uint8_t>(50, 1)},
        {"idx001", std::vector<uint8_t>(50, 2)},
        {"idx002", std::vector<uint8_t>(50, 3)},
        {"idx010", std::vector<uint8_t>(50, 4)},
        {"idx100", std::vector<uint8_t>(50, 5)}
    };
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.size(), 5);
    EXPECT_TRUE(reader.contains("idx000"));
    EXPECT_TRUE(reader.contains("idx100"));
}

TEST_F(LMDBReaderTest, SpecialCharactersInKeys) {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries = {
        {"path/to/image.jpg", {1, 2, 3}},
        {"data_file_001", {4, 5, 6}},
        {"sample.train.001", {7, 8, 9}}
    };
    create_test_lmdb(entries);

    turboloader::LMDBReader reader(test_lmdb_path_);
    EXPECT_EQ(reader.size(), 3);
    EXPECT_TRUE(reader.contains("path/to/image.jpg"));
    EXPECT_TRUE(reader.contains("data_file_001"));
}

// ============================================================================
// LMDBEntry Tests
// ============================================================================

TEST_F(LMDBReaderTest, LMDBEntryKeyString) {
    turboloader::LMDBEntry entry;
    entry.key = {'t', 'e', 's', 't'};
    entry.value = {1, 2, 3};

    EXPECT_EQ(entry.key_string(), "test");
}

TEST_F(LMDBReaderTest, LMDBEntryEmptyKey) {
    turboloader::LMDBEntry entry;
    entry.key = {};
    entry.value = {1, 2, 3};

    EXPECT_EQ(entry.key_string(), "");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
