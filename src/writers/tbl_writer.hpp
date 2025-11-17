/**
 * @file tbl_writer.hpp
 * @brief TurboLoader Binary (.tbl) format writer
 *
 * Converts TAR archives and other formats to optimized .tbl format.
 */

#pragma once

#include "../formats/tbl_format.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

namespace turboloader {
namespace writers {

using namespace formats;

/**
 * @brief TBL file writer
 */
class TblWriter {
public:
    explicit TblWriter(const std::string& filepath)
        : filepath_(filepath)
        , header_()
        , index_()
        , data_offset_(0)
        , finalized_(false)
    {
        // Open output file
        output_.open(filepath, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!output_) {
            throw std::runtime_error("Failed to create file: " + filepath);
        }

        // Reserve space for header (will write later)
        output_.seekp(sizeof(TblHeader));
        data_offset_ = sizeof(TblHeader);
    }

    ~TblWriter() {
        if (!finalized_) {
            try {
                finalize();
            } catch (...) {
                // Ignore exceptions in destructor
            }
        }
    }

    // Disable copy
    TblWriter(const TblWriter&) = delete;
    TblWriter& operator=(const TblWriter&) = delete;

    /**
     * @brief Add a sample to the TBL file
     *
     * @param data Sample data buffer
     * @param size Size of sample data
     * @param format Sample format (JPEG, PNG, etc.)
     */
    void add_sample(const uint8_t* data, size_t size, SampleFormat format) {
        if (finalized_) {
            throw std::runtime_error("Cannot add samples after finalization");
        }

        // Calculate current data offset (after index table)
        const uint64_t sample_offset = data_offset_ +
            (index_.size() * sizeof(TblIndexEntry)) +
            calculate_data_size();

        // Create index entry
        TblIndexEntry entry(sample_offset, static_cast<uint32_t>(size), format);
        index_.push_back(entry);

        // Buffer sample data (will write during finalize)
        sample_data_.push_back(std::vector<uint8_t>(data, data + size));

        header_.num_samples++;
    }

    /**
     * @brief Add a sample from file
     */
    void add_sample_from_file(const std::string& filename) {
        // Read file
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        size_t size = file.tellg();
        file.seekg(0);

        std::vector<uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size);

        // Detect format from filename
        SampleFormat format = extension_to_format(filename);

        add_sample(data.data(), size, format);
    }

    /**
     * @brief Finalize the TBL file (write header and index)
     */
    void finalize() {
        if (finalized_) {
            return;
        }

        // Update header
        header_.header_size = sizeof(TblHeader) + (index_.size() * sizeof(TblIndexEntry));

        // Recalculate offsets now that we know final index size
        uint64_t current_offset = header_.header_size;
        for (size_t i = 0; i < index_.size(); ++i) {
            index_[i].offset = current_offset;
            current_offset += index_[i].size;
        }

        // Write header
        output_.seekp(0);
        output_.write(reinterpret_cast<const char*>(&header_), sizeof(TblHeader));

        // Write index table
        output_.write(reinterpret_cast<const char*>(index_.data()),
                     index_.size() * sizeof(TblIndexEntry));

        // Write sample data
        for (const auto& sample : sample_data_) {
            output_.write(reinterpret_cast<const char*>(sample.data()), sample.size());
        }

        output_.flush();
        output_.close();

        // Clear sample data buffer to free memory
        sample_data_.clear();

        finalized_ = true;
    }

    /**
     * @brief Get number of samples added
     */
    size_t num_samples() const {
        return header_.num_samples;
    }

    /**
     * @brief Get output file path
     */
    const std::string& filepath() const {
        return filepath_;
    }

private:
    uint64_t calculate_data_size() const {
        uint64_t total = 0;
        for (const auto& sample : sample_data_) {
            total += sample.size();
        }
        return total;
    }

    std::string filepath_;
    std::ofstream output_;
    TblHeader header_;
    std::vector<TblIndexEntry> index_;
    std::vector<std::vector<uint8_t>> sample_data_;  // Buffered sample data
    uint64_t data_offset_;
    bool finalized_;
};

} // namespace writers
} // namespace turboloader
