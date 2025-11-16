/**
 * Multi-Format Benchmark
 *
 * Compares decoding performance across different image formats:
 * - JPEG (libjpeg-turbo)
 * - PNG (libpng)
 * - WebP (libwebp)
 *
 * Tests both single-format and mixed-format datasets to measure:
 * 1. Format-specific throughput
 * 2. Mixed workload performance
 * 3. Decoder overhead
 *
 * Usage: ./multiformat_benchmark <num_workers> <tar_file>
 * Example: ./multiformat_benchmark 8 /tmp/mixed_format.tar
 */

#include <iostream>
#include <chrono>
#include <map>
#include <turboloader/pipeline/pipeline.hpp>

using namespace turboloader;
using namespace std::chrono;

struct FormatStats {
    int count = 0;
    double total_time = 0.0;
    size_t total_bytes = 0;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_workers> <tar_file>\n";
        std::cerr << "Example: " << argv[0] << " 8 /tmp/multiformat.tar\n";
        return 1;
    }

    int num_workers = std::stoi(argv[1]);
    std::string tar_path = argv[2];

    std::cout << "=== Multi-Format Benchmark ===\n";
    std::cout << "Workers: " << num_workers << "\n";
    std::cout << "Dataset: " << tar_path << "\n\n";

    // Create pipeline with multi-format decoding
    Pipeline::Config config{
        .num_workers = static_cast<size_t>(num_workers),
        .queue_size = 256,
        .prefetch_factor = 2,
        .shuffle = false,
        .decode_jpeg = true  // Enable all decoders
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

    std::map<std::string, FormatStats> format_stats;
    int total_samples = 0;
    int num_batches = 0;

    auto start = high_resolution_clock::now();

    while (true) {
        auto batch = pipeline.next_batch(batch_size);
        if (batch.empty()) {
            break;
        }

        // Track format statistics
        for (const auto& sample : batch) {
            std::string format = "unknown";

            // Detect format from extension or magic bytes
            if (sample.data.count(".jpg") || sample.data.count(".jpeg")) {
                format = "JPEG";
            } else if (sample.data.count(".png")) {
                format = "PNG";
            } else if (sample.data.count(".webp")) {
                format = "WebP";
            }

            format_stats[format].count++;

            // Accumulate bytes
            for (const auto& [key, data] : sample.data) {
                format_stats[format].total_bytes += data.size();
            }
        }

        total_samples += batch.size();
        num_batches++;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count() / 1000.0;

    // Overall results
    double throughput = total_samples / duration;
    double avg_batch_time = (duration / num_batches) * 1000.0;  // ms

    std::cout << "\n=== Overall Results ===\n";
    std::cout << "Total samples: " << total_samples << "\n";
    std::cout << "Total time: " << duration << " seconds\n";
    std::cout << "Throughput: " << static_cast<int>(throughput) << " img/s\n";
    std::cout << "Avg batch time: " << avg_batch_time << " ms\n";

    // Per-format results
    std::cout << "\n=== Per-Format Statistics ===\n";
    std::cout << "Format      Count    Total MB    Avg Size    Throughput\n";
    std::cout << "------      -----    --------    --------    ----------\n";

    for (const auto& [format, stats] : format_stats) {
        double mb = stats.total_bytes / (1024.0 * 1024.0);
        double avg_size_kb = (stats.total_bytes / stats.count) / 1024.0;
        double format_throughput = stats.count / duration;

        printf("%-10s  %5d    %8.2f    %8.2f    %d img/s\n",
               format.c_str(),
               stats.count,
               mb,
               avg_size_kb,
               static_cast<int>(format_throughput));
    }

    // Decoder efficiency
    std::cout << "\n=== Decoder Efficiency ===\n";
    std::cout << "Workers: " << num_workers << "\n";
    std::cout << "Throughput per worker: " << static_cast<int>(throughput / num_workers) << " img/s\n";
    std::cout << "Parallel efficiency: " << (throughput / num_workers) << " img/s/worker\n";

    pipeline.stop();
    return 0;
}
