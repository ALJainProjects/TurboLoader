#include "turboloader/pipeline/pipeline.hpp"
#include <iostream>

int main() {
    using namespace turboloader;

    // Configure pipeline
    Pipeline::Config config;
    config.num_workers = 4;
    config.queue_size = 256;
    config.prefetch_factor = 2;
    config.shuffle = true;

    try {
        // Create pipeline for TAR datasets
        std::vector<std::string> tar_files = {
            // Add your TAR file paths here
            // "/path/to/shard0.tar",
            // "/path/to/shard1.tar",
        };

        if (tar_files.empty()) {
            std::cout << "No TAR files specified. Add paths to tar_files vector.\n";
            return 0;
        }

        Pipeline pipeline(tar_files, config);

        std::cout << "Total samples: " << pipeline.total_samples() << "\n";

        // Start pipeline
        pipeline.start();

        // Fetch batches
        size_t batch_size = 32;
        size_t total_batches = 0;

        while (true) {
            auto batch = pipeline.next_batch(batch_size);

            if (batch.empty()) {
                break;  // No more data
            }

            total_batches++;

            // Process batch
            for (const auto& sample : batch) {
                std::cout << "Sample " << sample.index << " has " << sample.data.size() << " files\n";

                // Access sample data
                for (const auto& [ext, data] : sample.data) {
                    std::cout << "  ." << ext << ": " << data.size() << " bytes\n";
                }
            }

            if (total_batches >= 5) {
                // Just show first 5 batches
                break;
            }
        }

        pipeline.stop();

        std::cout << "Processed " << total_batches << " batches\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
