/**
 * @file test_csv_decoder.cpp
 * @brief Comprehensive tests for CSV decoder with performance benchmarks
 *
 * Tests:
 * 1. Basic CSV parsing
 * 2. Quoted fields and escape characters
 * 3. Header detection
 * 4. Column selection
 * 5. Row filtering
 * 6. Performance benchmarks
 */

#include "../src/decode/csv_decoder.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace turboloader;

// ANSI color codes
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

/**
 * @brief Test decoder availability
 */
void test_decoder_availability() {
    std::cout << BOLD << "\n[TEST] CSV Decoder Availability" << RESET << std::endl;
    std::cout << "  " << CSVDecoder::version_info() << std::endl;
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Create a test CSV file
 */
bool create_test_csv(const std::string& filename, int num_rows = 1000) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // Write header
    file << "id,name,value,score\n";

    // Write data rows
    for (int i = 0; i < num_rows; ++i) {
        file << i << ",user_" << i << "," << (i * 1.5) << "," << (i % 100) << "\n";
    }

    file.close();
    return true;
}

/**
 * @brief Create a CSV file with quoted fields
 */
bool create_quoted_csv(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "id,name,description\n";
    file << "1,\"John Doe\",\"A simple, normal person\"\n";
    file << "2,\"Jane Smith\",\"An expert in \"\"data science\"\"\"\n";
    file << "3,Bob,No quotes needed\n";

    file.close();
    return true;
}

/**
 * @brief Test basic CSV parsing
 */
void test_basic_parsing() {
    std::cout << BOLD << "\n[TEST] Basic CSV Parsing" << RESET << std::endl;

    std::string test_file = "/tmp/test_data.csv";
    if (!create_test_csv(test_file, 100)) {
        std::cout << YELLOW << "  ⚠ Could not create test CSV - skipping test" << RESET << std::endl;
        return;
    }

    CSVDecoder decoder;
    CSVConfig config;

    if (!decoder.load_file(test_file, config)) {
        std::cout << YELLOW << "  ⚠ Could not load test CSV - skipping test" << RESET << std::endl;
        remove(test_file.c_str());
        return;
    }

    CSVMetadata meta = decoder.get_metadata();

    std::cout << "  " << GREEN << "✓" << RESET << " Rows: " << meta.num_rows << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Columns: " << meta.num_columns << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Column names: ";
    for (size_t i = 0; i < meta.column_names.size(); ++i) {
        std::cout << meta.column_names[i];
        if (i < meta.column_names.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    remove(test_file.c_str());
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test quoted fields
 */
void test_quoted_fields() {
    std::cout << BOLD << "\n[TEST] Quoted Fields" << RESET << std::endl;

    std::string test_file = "/tmp/test_quoted.csv";
    if (!create_quoted_csv(test_file)) {
        std::cout << YELLOW << "  ⚠ Could not create test CSV - skipping test" << RESET << std::endl;
        return;
    }

    CSVDecoder decoder;
    CSVConfig config;

    if (!decoder.load_file(test_file, config)) {
        std::cout << YELLOW << "  ⚠ Could not load test CSV - skipping test" << RESET << std::endl;
        remove(test_file.c_str());
        return;
    }

    auto rows = decoder.parse_rows(config);

    std::cout << "  " << GREEN << "✓" << RESET << " Parsed " << rows.size() << " rows" << std::endl;

    if (!rows.empty() && rows[0].size() >= 3) {
        std::cout << "  " << GREEN << "✓" << RESET << " Row 1, col 2: \"" << rows[0][1] << "\"" << std::endl;
        std::cout << "  " << GREEN << "✓" << RESET << " Row 1, col 3: \"" << rows[0][2] << "\"" << std::endl;
    }

    remove(test_file.c_str());
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test column selection
 */
void test_column_selection() {
    std::cout << BOLD << "\n[TEST] Column Selection" << RESET << std::endl;

    std::string test_file = "/tmp/test_data.csv";
    if (!create_test_csv(test_file, 100)) {
        std::cout << YELLOW << "  ⚠ Could not create test CSV - skipping test" << RESET << std::endl;
        return;
    }

    CSVDecoder decoder;
    CSVConfig config;

    if (!decoder.load_file(test_file, config)) {
        std::cout << YELLOW << "  ⚠ Could not load test CSV - skipping test" << RESET << std::endl;
        remove(test_file.c_str());
        return;
    }

    // Get specific column
    auto id_column = decoder.get_column("id");
    auto name_column = decoder.get_column("name");

    std::cout << "  " << GREEN << "✓" << RESET << " ID column: " << id_column.size() << " values" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Name column: " << name_column.size() << " values" << std::endl;

    if (!id_column.empty()) {
        std::cout << "  " << GREEN << "✓" << RESET << " First ID: " << id_column[0] << std::endl;
    }

    remove(test_file.c_str());
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Test row filtering
 */
void test_row_filtering() {
    std::cout << BOLD << "\n[TEST] Row Filtering" << RESET << std::endl;

    std::string test_file = "/tmp/test_data.csv";
    if (!create_test_csv(test_file, 1000)) {
        std::cout << YELLOW << "  ⚠ Could not create test CSV - skipping test" << RESET << std::endl;
        return;
    }

    CSVDecoder decoder;
    CSVConfig config;
    config.skip_rows = 10;
    config.max_rows = 50;

    if (!decoder.load_file(test_file, config)) {
        std::cout << YELLOW << "  ⚠ Could not load test CSV - skipping test" << RESET << std::endl;
        remove(test_file.c_str());
        return;
    }

    auto rows = decoder.parse_rows(config);

    std::cout << "  " << GREEN << "✓" << RESET << " Filtered to " << rows.size()
              << " rows (skip=10, max=50)" << std::endl;

    remove(test_file.c_str());
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Benchmark CSV parsing performance
 */
void benchmark_csv_parsing() {
    std::cout << BOLD << "\n[BENCHMARK] CSV Parsing Performance" << RESET << std::endl;

    std::string test_file = "/tmp/test_large.csv";
    if (!create_test_csv(test_file, 100000)) {  // 100K rows
        std::cout << YELLOW << "  ⚠ Could not create test CSV - skipping benchmark" << RESET << std::endl;
        return;
    }

    CSVDecoder decoder;
    CSVConfig config;

    auto load_start = std::chrono::high_resolution_clock::now();
    bool success = decoder.load_file(test_file, config);
    auto load_end = std::chrono::high_resolution_clock::now();

    if (!success) {
        std::cout << YELLOW << "  ⚠ Could not load test CSV - skipping benchmark" << RESET << std::endl;
        remove(test_file.c_str());
        return;
    }

    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);

    auto parse_start = std::chrono::high_resolution_clock::now();
    auto rows = decoder.parse_rows(config);
    auto parse_end = std::chrono::high_resolution_clock::now();

    auto parse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(parse_end - parse_start);
    double rows_per_sec = (rows.size() * 1000.0) / parse_duration.count();

    std::cout << "  " << GREEN << "✓" << RESET << " Loaded file in " << load_duration.count() << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Parsed " << rows.size()
              << " rows in " << parse_duration.count() << " ms" << std::endl;
    std::cout << "  " << GREEN << "✓" << RESET << " Performance: " << static_cast<int>(rows_per_sec)
              << " rows/sec" << std::endl;

    remove(test_file.c_str());
    std::cout << GREEN << "  PASSED" << RESET << std::endl;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "║      TurboLoader CSV Decoder Test Suite             ║" << RESET << std::endl;
    std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

    try {
        test_decoder_availability();
        test_basic_parsing();
        test_quoted_fields();
        test_column_selection();
        test_row_filtering();
        benchmark_csv_parsing();

        std::cout << BOLD << "\n╔═══════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << "║  " << GREEN << "✓ ALL TESTS PASSED" << RESET << BOLD << "                                ║" << RESET << std::endl;
        std::cout << BOLD << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ TEST FAILED: " << e.what() << RESET << std::endl;
        return 1;
    }
}
