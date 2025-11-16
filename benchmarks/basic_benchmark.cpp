/**
 * Basic TurboLoader Benchmark
 *
 * Measures raw data loading throughput:
 * - TAR file reading and parsing
 * - JPEG decoding with libjpeg-turbo
 * - Multi-threaded pipeline performance
 *
 * Usage: ./basic_benchmark <num_workers> <tar_file>
 * Example: ./basic_benchmark 8 /path/to/dataset.tar
 */

#include <iostream>
#include <chrono>
#include <turboloader/pipeline/pipeline.hpp>

using namespace turboloader;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_workers> <tar_file>\n";
        std::cerr << "Example: " << argv[0] << " 8 /tmp/dataset.tar\n";
        return 1;
    }

    int num_workers = std::stoi(argv[1]);
    std::string tar_path = argv[2];

    std::cout << "=== TurboLoader Basic Benchmark ===\n";
    std::cout << "Workers: " << num_workers << "\n";
    std::cout << "Dataset: " << tar_path << "\n\n";

    // Create pipeline
    Pipeline::Config config{
        .num_workers = static_cast<size_t>(num_workers),
        .queue_size = 256,
        .prefetch_factor = 2,
        .shuffle = false,
        .decode_jpeg = true
    };

    Pipeline pipeline({tar_path}, config);
    pipeline.start();

    const size_t batch_size = 32;

    // Warmup
    std::cout << "Warming up...\n";
    for (int i = 0; i < 5; i++) {
        auto batch = pipeline.next_batch(batch_size);
    }

    // Benchmark
    std::cout << "Running benchmark...\n";
    int total_samples = 0;
    int num_batches = 0;

    auto start = high_resolution_clock::now();

    while (true) {
        auto batch = pipeline.next_batch(batch_size);
        if (batch.empty()) {
            break;  // End of epoch
        }
        total_samples += batch.size();
        num_batches++;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count() / 1000.0;

    // Results
    double throughput = total_samples / duration;
    double avg_batch_time = (duration / num_batches) * 1000.0;  // ms

    std::cout << "\n=== Results ===\n";
    std::cout << "Total samples: " << total_samples << "\n";
    std::cout << "Total time: " << duration << " seconds\n";
    std::cout << "Throughput: " << static_cast<int>(throughput) << " img/s\n";
    std::cout << "Avg batch time: " << avg_batch_time << " ms\n";
    std::cout << "Batches processed: " << num_batches << "\n";

    pipeline.stop();
    return 0;
}
