#include "turboloader/readers/mmap_reader.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

using namespace turboloader;

class MmapReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary test file
        test_file_ = "/tmp/turboloader_test.txt";
        std::ofstream ofs(test_file_);
        ofs << "Hello, TurboLoader!";
        ofs.close();
    }

    void TearDown() override {
        std::filesystem::remove(test_file_);
    }

    std::string test_file_;
};

TEST_F(MmapReaderTest, BasicRead) {
    MmapReader reader(test_file_);

    EXPECT_TRUE(reader.is_open());
    EXPECT_EQ(reader.size(), 19);  // Length of "Hello, TurboLoader!"

    auto data = reader.read(0);
    std::string content(reinterpret_cast<const char*>(data.data()), data.size());
    EXPECT_EQ(content, "Hello, TurboLoader!");
}

TEST_F(MmapReaderTest, PartialRead) {
    MmapReader reader(test_file_);

    auto data = reader.read(0, 5);
    std::string content(reinterpret_cast<const char*>(data.data()), data.size());
    EXPECT_EQ(content, "Hello");
}

TEST_F(MmapReaderTest, ReadString) {
    MmapReader reader(test_file_);

    auto content = reader.read_string(7, 11);
    EXPECT_EQ(content, "TurboLoader");
}

TEST_F(MmapReaderTest, NonExistentFile) {
    EXPECT_THROW(MmapReader("/nonexistent/file.txt"), std::system_error);
}
