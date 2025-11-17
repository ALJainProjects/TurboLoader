/**
 * @file tar_to_tbl.cpp
 * @brief Convert TAR archives to TBL format
 *
 * Usage: tar_to_tbl input.tar output.tbl
 */

#include "../src/readers/tar_reader.hpp"
#include "../src/writers/tbl_writer.hpp"
#include "../src/formats/tbl_format.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace turboloader;
using namespace turboloader::readers;
using namespace turboloader::writers;
using namespace turboloader::formats;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.tar> <output.tbl>" << std::endl;
        return 1;
    }

    const std::string input_tar = argv[1];
    const std::string output_tbl = argv[2];

    try {
        std::cout << "Converting TAR to TBL format..." << std::endl;
        std::cout << "  Input:  " << input_tar << std::endl;
        std::cout << "  Output: " << output_tbl << std::endl;
        std::cout << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Open TAR reader (worker 0, 1 total worker - to get all samples)
        readers::TarReader tar_reader(input_tar, 0, 1);
        const size_t num_samples = tar_reader.num_samples();

        std::cout << "Found " << num_samples << " files in TAR archive" << std::endl;
        std::cout << "Converting..." << std::endl;

        // Create TBL writer
        writers::TblWriter tbl_writer(output_tbl);

        // Convert each file
        size_t converted = 0;
        for (size_t i = 0; i < num_samples; ++i) {
            auto sample_data = tar_reader.get_sample(i);
            const auto& entry = tar_reader.get_entry(i);

            // Detect format from filename
            formats::SampleFormat format = formats::extension_to_format(entry.name);

            // Add to TBL
            tbl_writer.add_sample(
                sample_data.data(),
                sample_data.size(),
                format
            );

            converted++;

            // Progress indicator
            if (converted % 100 == 0 || converted == num_samples) {
                std::cout << "  Progress: " << converted << "/" << num_samples
                          << " (" << (100 * converted / num_samples) << "%)" << std::endl;
            }
        }

        // Finalize TBL file
        std::cout << "Finalizing TBL file..." << std::endl;
        tbl_writer.finalize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Get file sizes
        struct stat tar_stat, tbl_stat;
        stat(input_tar.c_str(), &tar_stat);
        stat(output_tbl.c_str(), &tbl_stat);

        double tar_size_mb = tar_stat.st_size / (1024.0 * 1024.0);
        double tbl_size_mb = tbl_stat.st_size / (1024.0 * 1024.0);
        double compression_ratio = 100.0 * (1.0 - tbl_size_mb / tar_size_mb);

        std::cout << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "Conversion Complete!" << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "  Samples converted: " << converted << std::endl;
        std::cout << "  Time elapsed:      " << duration.count() << " ms" << std::endl;
        std::cout << "  Conversion rate:   " << (converted * 1000.0 / duration.count()) << " samples/s" << std::endl;
        std::cout << std::endl;
        std::cout << "  TAR size:  " << std::fixed << std::setprecision(2) << tar_size_mb << " MB" << std::endl;
        std::cout << "  TBL size:  " << std::fixed << std::setprecision(2) << tbl_size_mb << " MB" << std::endl;
        std::cout << "  Size reduction: " << std::fixed << std::setprecision(1) << compression_ratio << "%" << std::endl;
        std::cout << "================================================================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
