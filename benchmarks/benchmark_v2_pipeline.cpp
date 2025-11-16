/**
 * @file benchmark_v2_pipeline.cpp
 * @brief Benchmark for PipelineV2 throughput measurement
 *
 * Measures raw throughput of the v2.0 pipeline for comparison with PyTorch.
 */

#include "../src/pipeline_v2/pipeline_v2.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

using namespace turboloader::v2;
using namespace std::chrono;

/**
 * @brief Benchmark configuration
 */
struct BenchmarkConfig {
    std::string tar_path;
    size_t num_workers = 4;
    size_t batch_size = 32;
    size_t num_epochs = 2;
    bool warmup = true;
};

/**
 * @brief Run single epoch benchmark
 */
double benchmark_epoch(PipelineV2& pipeline, size_t total_samples, bool print_progress = false) {
    auto start = high_resolution_clock::now();

    size_t batches_processed = 0;
    size_t samples_processed = 0;

    while (!pipeline.is_finished()) {
        auto batch = pipeline.next_batch();

        if (batch.empty()) {
            break;
        }

        samples_processed += batch.size();
        batches_processed++;

        if (print_progress && batches_processed % 10 == 0) {
            auto elapsed = high_resolution_clock::now() - start;
            auto ms = duration_cast<milliseconds>(elapsed).count();
            double throughput = samples_processed / (ms / 1000.0);
            std::cout << "  Batch " << batches_processed
                      << ": " << samples_processed << "/" << total_samples
                      << " samples (" << std::fixed << std::setprecision(2)
                      << throughput << " img/s)" << std::endl;
        }
    }

    auto end = high_resolution_clock::now();
    auto duration_ms = duration_cast<milliseconds>(end - start).count();

    return samples_processed / (duration_ms / 1000.0);
}

/**
 * @brief Main benchmark routine
 */
void run_benchmark(const BenchmarkConfig& config) {
    std::cout << "========================================" << std::endl;
    std::cout << "  TurboLoader v2.0 Pipeline Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  TAR file: " << config.tar_path << std::endl;
    std::cout << "  Workers: " << config.num_workers << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Epochs: " << config.num_epochs << std::endl;
    std::cout << std::endl;

    // Create pipeline config
    PipelineConfig pipeline_config;
    pipeline_config.tar_path = config.tar_path;
    pipeline_config.num_workers = config.num_workers;
    pipeline_config.batch_size = config.batch_size;

    // Create pipeline
    std::cout << "Initializing pipeline..." << std::endl;
    PipelineV2 pipeline(pipeline_config);

    size_t total_samples = pipeline.total_samples();
    std::cout << "Total samples: " << total_samples << std::endl;
    std::cout << std::endl;

    // Warmup epoch (optional)
    if (config.warmup) {
        std::cout << "Warmup epoch (results not counted):" << std::endl;
        double warmup_throughput = benchmark_epoch(pipeline, total_samples, false);
        std::cout << "  Warmup throughput: " << std::fixed << std::setprecision(2)
                  << warmup_throughput << " img/s" << std::endl;
        std::cout << std::endl;

        // Reset pipeline for actual benchmark
        // Note: Need to recreate pipeline as workers have finished
        pipeline.stop();
    }

    // Run benchmark epochs
    std::vector<double> throughputs;

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << config.num_epochs << ":" << std::endl;

        // Recreate pipeline for each epoch
        PipelineV2 epoch_pipeline(pipeline_config);

        double throughput = benchmark_epoch(epoch_pipeline, total_samples, true);
        throughputs.push_back(throughput);

        std::cout << "  Epoch throughput: " << std::fixed << std::setprecision(2)
                  << throughput << " img/s" << std::endl;
        std::cout << std::endl;

        epoch_pipeline.stop();
    }

    // Calculate statistics
    double mean = 0.0;
    for (double t : throughputs) {
        mean += t;
    }
    mean /= throughputs.size();

    double variance = 0.0;
    for (double t : throughputs) {
        variance += (t - mean) * (t - mean);
    }
    variance /= throughputs.size();
    double stddev = std::sqrt(variance);

    double min_throughput = *std::min_element(throughputs.begin(), throughputs.end());
    double max_throughput = *std::max_element(throughputs.begin(), throughputs.end());

    // Print summary
    std::cout << "========================================" << std::endl;
    std::cout << "  Benchmark Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mean throughput:   " << mean << " img/s" << std::endl;
    std::cout << "Std deviation:     " << stddev << " img/s" << std::endl;
    std::cout << "Min throughput:    " << min_throughput << " img/s" << std::endl;
    std::cout << "Max throughput:    " << max_throughput << " img/s" << std::endl;
    std::cout << std::endl;

    // Comparison with PyTorch baseline
    double pytorch_baseline = 48.07;  // From previous benchmarks
    double speedup = mean / pytorch_baseline;

    std::cout << "Comparison to PyTorch DataLoader:" << std::endl;
    std::cout << "  PyTorch baseline:  " << pytorch_baseline << " img/s" << std::endl;
    std::cout << "  TurboLoader v2.0:  " << mean << " img/s" << std::endl;
    std::cout << "  Speedup:           " << speedup << "x" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    BenchmarkConfig config;

    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <tar_path> [num_workers] [batch_size] [num_epochs]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " /tmp/benchmark_dataset/test.tar 4 32 3" << std::endl;
        return 1;
    }

    config.tar_path = argv[1];

    if (argc >= 3) {
        config.num_workers = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        config.batch_size = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        config.num_epochs = std::atoi(argv[4]);
    }

    try {
        run_benchmark(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
