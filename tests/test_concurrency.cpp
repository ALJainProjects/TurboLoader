// Test file to detect race conditions with ThreadSanitizer
#include "turboloader/pipeline/pipeline.hpp"
#include <iostream>
#include <chrono>

int main() {
    // Create a small test to trigger race conditions
    std::vector<std::string> tar_paths = {"/tmp/test_webdataset.tar"};

    turboloader::Pipeline::Config config;
    config.num_workers = 8;  // Use 8 workers to test stability
    config.decode_jpeg = true;
    config.queue_size = 128;

    std::cout << "Creating pipeline with " << config.num_workers << " workers..." << std::endl;

    turboloader::Pipeline pipeline(tar_paths, config);

    std::cout << "Total samples: " << pipeline.total_samples() << std::endl;
    std::cout << "Starting pipeline..." << std::endl;

    pipeline.start();

    // Give pipeline time to start populating queue
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    size_t processed = 0;
    size_t target = std::min(pipeline.total_samples(), static_cast<size_t>(100));

    auto start = std::chrono::steady_clock::now();

    while (processed < target) {
        auto batch = pipeline.next_batch(64);
        if (batch.empty()) {
            // Wait a bit for more data
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // Check if we've waited too long (pipeline might be done)
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30) {
                std::cout << "Timeout waiting for data" << std::endl;
                break;
            }
            continue;
        }
        processed += batch.size();

        if (processed % 100 == 0) {
            std::cout << "Processed: " << processed << "/" << target << std::endl;
        }
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    pipeline.stop();

    std::cout << "Processed " << processed << " samples in " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (processed * 1000.0 / ms) << " samples/sec" << std::endl;

    return 0;
}
