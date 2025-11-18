/**
 * @file pipeline.hpp
 * @brief Unified pipeline supporting all TurboLoader formats
 *
 * This is the main pipeline for TurboLoader, combining:
 * - Multi-format support (images, videos, CSV, Parquet)
 * - Multi-source support (local, HTTP, S3, GCS)
 * - Auto-detection and routing
 * - Multi-threaded processing with standard C++ primitives
 *
 * ARCHITECTURE:
 * ```
 * TAR Mode (for image datasets):
 * [Worker 0] --\
 * [Worker 1] ----> [Thread-safe Queue] --> [Main Thread] --> Batches
 * [Worker 2] --/
 * [Worker 3]
 *
 * Video Mode:
 * [FFmpeg Decoder] --> [Frame Queue] --> [Batch Assembler] --> Batches
 *
 * Tabular Mode:
 * [CSV/Parquet Reader] --> [Row Queue] --> [Batch Assembler] --> Batches
 * ```
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Per-worker TAR readers (eliminates mutex bottleneck)
 * - SIMD-accelerated JPEG decoding (libjpeg-turbo)
 * - Object pooling for buffer reuse
 * - Hardware-accelerated video decoding (NVDEC, VAAPI, VideoToolbox)
 * - Zero-copy Parquet reading via Apache Arrow
 *
 * USAGE:
 * ```cpp
 * // TAR archive with images
 * UnifiedPipelineConfig config;
 * config.data_path = "/path/to/images.tar";
 * config.format = DataFormat::TAR;  // Auto-detected
 * config.num_workers = 4;
 * config.batch_size = 32;
 *
 * UnifiedPipeline pipeline(config);
 * pipeline.start();
 *
 * while (!pipeline.is_finished()) {
 *     auto batch = pipeline.next_batch();
 *     // Process batch...
 * }
 * ```
 */

#pragma once

// Core types
#include "../core/sample.hpp"
#include "../core/object_pool.hpp"
#include "../core/spsc_ring_buffer.hpp"

// Readers
#include "../readers/tar_reader.hpp"
#include "../readers/http_reader.hpp"
#include "../readers/s3_reader.hpp"
#include "../readers/gcs_reader.hpp"
#include "../readers/reader_orchestrator.hpp"

// Decoders
#include "../decode/jpeg_decoder.hpp"
#ifdef HAVE_NVJPEG
#include "../decode/nvjpeg_decoder.hpp"
#endif
#include "../decode/png_decoder.hpp"
#ifdef HAVE_WEBP
#include "../decode/webp_decoder.hpp"
#endif
#include "../decode/bmp_decoder.hpp"
#include "../decode/tiff_decoder.hpp"
#ifdef HAVE_FFMPEG
#include "../decode/video_decoder.hpp"
#endif
#include "../decode/csv_decoder.hpp"
#ifdef HAVE_ARROW
#include "../decode/parquet_decoder.hpp"
#endif

#include <atomic>
#include <condition_variable>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <algorithm>

namespace turboloader {

/**
 * @brief Data format types
 */
enum class DataFormat {
    UNKNOWN,
    // Images
    JPEG,
    PNG,
    WEBP,
    BMP,
    TIFF,
    // Videos
    MP4,
    AVI,
    MKV,
    MOV,
    // Tabular
    CSV,
    PARQUET,
    // Archives (fastest mode for images)
    TAR
};

/**
 * @brief Data source types
 */
enum class DataSource {
    LOCAL_FILE,
    HTTP,
    S3,
    GCS
};

/**
 * @brief Unified pipeline configuration
 */
struct UnifiedPipelineConfig {
    // ===== Data source =====
    std::string data_path;                      // Path (local, http://, s3://, gs://)
    DataFormat format = DataFormat::UNKNOWN;    // Auto-detect if UNKNOWN

    // ===== Pipeline settings =====
    size_t num_workers = 4;                     // Worker threads
    size_t batch_size = 32;                     // Samples per batch
    size_t queue_size = 256;                    // Sample queue size
    size_t buffer_pool_size = 128;              // Buffer pool size
    bool prefetch = true;                       // Enable prefetching
    size_t prefetch_batches = 2;                // Batches to prefetch
    bool shuffle = false;                       // Shuffle (future)

    // ===== Image processing =====
    bool resize_images = false;                 // Resize images
    int target_width = 256;                     // Target width
    int target_height = 256;                    // Target height
    bool normalize = false;                     // Normalize to [0,1]
    bool use_gpu_decode = true;                 // GPU-accelerated JPEG (nvJPEG)

    // ===== Video processing =====
    int video_fps = 30;                         // Target FPS for extraction
    int max_video_frames = -1;                  // -1 = all frames
    bool video_hw_accel = true;                 // Hardware acceleration

    // ===== CSV processing =====
    char csv_delimiter = ',';                   // Delimiter
    bool csv_has_header = true;                 // Header row
    bool csv_skip_empty = true;                 // Skip empty lines

    // ===== Parquet processing =====
    bool parquet_use_threads = true;            // Multi-threaded reading
    bool parquet_use_mmap = true;               // Memory-mapped I/O
};

/**
 * @brief Unified sample (supports all data types)
 */
struct UnifiedSample {
    // Metadata
    size_t index;
    DataFormat format;
    std::string filename;

    // Image/Video data
    std::vector<uint8_t> image_data;
    int width;
    int height;
    int channels;

    // Tabular data
    std::vector<std::string> row_data;
    std::vector<std::string> column_names;

    // Construction
    UnifiedSample() : index(0), format(DataFormat::UNKNOWN),
                     width(0), height(0), channels(0) {}

    UnifiedSample(size_t idx, DataFormat fmt)
        : index(idx), format(fmt), width(0), height(0), channels(0) {}
};

/**
 * @brief Unified batch
 */
struct UnifiedBatch {
    std::vector<UnifiedSample> samples;
    size_t batch_id;

    explicit UnifiedBatch(size_t max_size = 32) : batch_id(0) {
        samples.reserve(max_size);
    }

    void add(UnifiedSample&& sample) {
        samples.push_back(std::move(sample));
    }

    size_t size() const { return samples.size(); }
    bool empty() const { return samples.empty(); }
    void clear() { samples.clear(); }
};

/**
 * @brief Format and source detection
 */
class FormatDetector {
public:
    static DataFormat detect_from_path(const std::string& path) {
        std::string ext = get_extension(path);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Images
        if (ext == ".jpg" || ext == ".jpeg") return DataFormat::JPEG;
        if (ext == ".png") return DataFormat::PNG;
        if (ext == ".webp") return DataFormat::WEBP;
        if (ext == ".bmp") return DataFormat::BMP;
        if (ext == ".tiff" || ext == ".tif") return DataFormat::TIFF;

        // Videos
        if (ext == ".mp4") return DataFormat::MP4;
        if (ext == ".avi") return DataFormat::AVI;
        if (ext == ".mkv") return DataFormat::MKV;
        if (ext == ".mov") return DataFormat::MOV;

        // Tabular
        if (ext == ".csv") return DataFormat::CSV;
        if (ext == ".parquet") return DataFormat::PARQUET;

        // Archives
        if (ext == ".tar") return DataFormat::TAR;

        return DataFormat::UNKNOWN;
    }

    static DataSource detect_source(const std::string& path) {
        if (path.substr(0, 7) == "http://" || path.substr(0, 8) == "https://")
            return DataSource::HTTP;
        if (path.substr(0, 5) == "s3://")
            return DataSource::S3;
        if (path.substr(0, 5) == "gs://")
            return DataSource::GCS;
        return DataSource::LOCAL_FILE;
    }

private:
    static std::string get_extension(const std::string& path) {
        size_t dot_pos = path.find_last_of('.');
        if (dot_pos == std::string::npos) return "";
        return path.substr(dot_pos);
    }
};

/**
 * @brief Lock-free per-worker sample queue using SPSC ring buffer
 *
 * Each worker gets its own SPSC queue (Single-Producer Single-Consumer).
 * The main thread round-robins across worker queues to collect samples.
 * This eliminates all mutex contention for maximum throughput.
 */
using WorkerQueue = SPSCRingBuffer<UnifiedSample, 256>;

/**
 * @brief TAR worker for multi-threaded image processing
 *
 * Each worker has its own:
 * - TAR reader (no mutex contention)
 * - JPEG decoder (no mutex contention)
 * - SPSC queue (lock-free, no mutex contention)
 */
class TarWorker {
public:
    TarWorker(const UnifiedPipelineConfig& config,
              size_t worker_id,
              BufferPool* buffer_pool,
              std::shared_ptr<std::vector<uint8_t>> remote_tar_data = nullptr)
        : config_(config),
          worker_id_(worker_id),
          buffer_pool_(buffer_pool),
          queue_(std::make_unique<WorkerQueue>()),
          running_(false),
          samples_processed_(0) {

        // Per-worker TAR reader
        // Check if using remote TAR data (already fetched)
        if (remote_tar_data) {
            tar_reader_ = std::make_unique<TarReader>(
                remote_tar_data,
                worker_id,
                config.num_workers
            );
        } else {
            // Local file path
            tar_reader_ = std::make_unique<TarReader>(
                config.data_path,
                worker_id,
                config.num_workers
            );
        }

        // Per-worker JPEG decoder (GPU-accelerated if available and enabled)
#ifdef HAVE_NVJPEG
        if (config.use_gpu_decode) {
            nvjpeg_decoder_ = std::make_unique<NvJpegDecoder>();
            if (nvjpeg_decoder_->is_available()) {
                use_gpu_ = true;
            } else {
                // Fall back to CPU if GPU unavailable
                nvjpeg_decoder_.reset();
                decoder_ = std::make_unique<JPEGDecoder>(buffer_pool);
                use_gpu_ = false;
            }
        } else {
            decoder_ = std::make_unique<JPEGDecoder>(buffer_pool);
            use_gpu_ = false;
        }
#else
        // No nvJPEG: CPU-only decode
        decoder_ = std::make_unique<JPEGDecoder>(buffer_pool);
#endif
    }

    void start() {
        running_ = true;
        thread_ = std::thread(&TarWorker::run, this);
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool is_finished() const {
        return !running_;
    }

    size_t samples_processed() const {
        return samples_processed_.load();
    }

    size_t num_samples() const {
        return tar_reader_->num_samples();
    }

    WorkerQueue* get_queue() {
        return queue_.get();
    }

private:
    void run() {
        size_t total_samples = tar_reader_->num_samples();

        for (size_t i = 0; i < total_samples && running_; ++i) {
            // Zero-copy JPEG data from TAR
            auto jpeg_data = tar_reader_->get_sample(i);
            const auto& entry = tar_reader_->get_entry(i);

            // Create UnifiedSample
            UnifiedSample usample;
            usample.index = entry.index;
            usample.format = DataFormat::JPEG;
            usample.filename = entry.name;

#ifdef HAVE_NVJPEG
            // GPU-accelerated JPEG decoding (if enabled and available)
            if (use_gpu_ && nvjpeg_decoder_) {
                NvJpegResult gpu_result;
                if (nvjpeg_decoder_->decode(
                    reinterpret_cast<const uint8_t*>(jpeg_data.data()),
                    jpeg_data.size(),
                    gpu_result)) {
                    // Successful GPU decode
                    usample.image_data = std::move(gpu_result.rgb_data);
                    usample.width = gpu_result.width;
                    usample.height = gpu_result.height;
                    usample.channels = gpu_result.channels;
                } else {
                    continue;  // Skip corrupted
                }
            } else
#endif
            {
                // CPU SIMD-accelerated JPEG decoding (fallback)
                Sample sample(entry.index, jpeg_data);
                try {
                    decoder_->decode_sample(sample);
                } catch (const std::exception& e) {
                    continue;  // Skip corrupted
                }
                usample.image_data = std::move(sample.decoded_rgb);
                usample.width = sample.width;
                usample.height = sample.height;
                usample.channels = sample.channels;
            }

            // Push to lock-free SPSC queue (busy-wait if full)
            while (running_ && !queue_->try_push(std::move(usample))) {
                std::this_thread::yield();  // Yield if queue is full
            }

            if (!running_) {
                break;
            }

            samples_processed_++;
        }

        running_ = false;
    }

    UnifiedPipelineConfig config_;
    [[maybe_unused]] size_t worker_id_;
    [[maybe_unused]] BufferPool* buffer_pool_;

    std::unique_ptr<TarReader> tar_reader_;
    std::unique_ptr<JPEGDecoder> decoder_;
#ifdef HAVE_NVJPEG
    std::unique_ptr<NvJpegDecoder> nvjpeg_decoder_;
    bool use_gpu_ = false;
#endif
    std::unique_ptr<WorkerQueue> queue_;

    std::thread thread_;
    std::atomic<bool> running_;
    std::atomic<size_t> samples_processed_;
};

/**
 * @brief UNIFIED PIPELINE
 *
 * Production-ready pipeline supporting:
 * - TAR archives (fastest mode with multi-threaded processing)
 * - Videos (MP4, AVI, MKV with hardware acceleration)
 * - CSV/Parquet (tabular data)
 * - Single images
 */
class UnifiedPipeline {
public:
    explicit UnifiedPipeline(const UnifiedPipelineConfig& config)
        : config_(config),
          running_(false),
          samples_processed_(0),
          batches_produced_(0) {

        // Auto-detect format
        if (config_.format == DataFormat::UNKNOWN) {
            config_.format = FormatDetector::detect_from_path(config_.data_path);
            if (config_.format == DataFormat::UNKNOWN) {
                throw std::runtime_error("Cannot auto-detect format: " + config_.data_path);
            }
        }

        // Detect source
        data_source_ = FormatDetector::detect_source(config_.data_path);

        // Initialize
        initialize();
    }

    ~UnifiedPipeline() {
        stop();
    }

    void start() {
        running_ = true;

        if (config_.format == DataFormat::TAR) {
            // TAR mode: Check if remote TAR (http://, https://, s3://, gs://)
            std::shared_ptr<std::vector<uint8_t>> remote_tar_data = nullptr;

            if (is_remote_path(config_.data_path)) {
                // Fetch remote TAR data once, share across workers
                remote_tar_data = fetch_remote_tar(config_.data_path);
            }

            // Create workers with per-worker lock-free queues
            for (size_t i = 0; i < config_.num_workers; ++i) {
                tar_workers_.push_back(std::make_unique<TarWorker>(
                    config_, i, buffer_pool_.get(), remote_tar_data
                ));
                tar_workers_.back()->start();
            }
        } else if (is_video_format(config_.format)) {
            // Video mode: Single producer
            fallback_queue_ = std::make_unique<WorkerQueue>();
            workers_.emplace_back(&UnifiedPipeline::video_worker, this);
        } else if (is_tabular_format(config_.format)) {
            // Tabular mode: Single producer
            fallback_queue_ = std::make_unique<WorkerQueue>();
            workers_.emplace_back(&UnifiedPipeline::tabular_worker, this);
        }
    }

    void stop() {
        // Set running to false first
        bool was_running = running_.exchange(false);

        // If already stopped, nothing to do
        if (!was_running && tar_workers_.empty() && workers_.empty()) {
            return;
        }

        // Stop TAR workers (each has own queue)
        for (auto& worker : tar_workers_) {
            worker->stop();
        }
        tar_workers_.clear();

        // Join other worker threads FIRST before destroying resources
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();

        // Now safe to destroy decoders (threads have finished)
        csv_decoder_.reset();
#ifdef HAVE_ARROW
        parquet_decoder_.reset();
#endif
#ifdef HAVE_FFMPEG
        video_decoder_.reset();
#endif
        fallback_queue_.reset();
    }

    UnifiedBatch next_batch() {
        UnifiedBatch batch(config_.batch_size);
        batch.batch_id = batches_produced_++;

        if (config_.format == DataFormat::TAR) {
            // TAR mode: Round-robin across worker queues (lock-free)
            size_t consecutive_failures = 0;
            size_t max_failures = tar_workers_.size() * 2;  // Give up after 2 full rounds

            while (batch.size() < config_.batch_size && consecutive_failures < max_failures) {
                bool got_sample = false;

                for (auto& worker : tar_workers_) {
                    if (batch.size() >= config_.batch_size) break;

                    UnifiedSample sample;
                    if (worker->get_queue()->try_pop(sample)) {
                        batch.add(std::move(sample));
                        got_sample = true;
                        consecutive_failures = 0;
                    }
                }

                if (!got_sample) {
                    consecutive_failures++;
                    if (is_finished()) break;
                    std::this_thread::yield();
                }
            }
        } else {
            // Video/Tabular mode: Single queue
            while (batch.size() < config_.batch_size) {
                UnifiedSample sample;
                if (!fallback_queue_->try_pop(sample)) {
                    if (is_finished()) break;
                    std::this_thread::yield();
                    continue;
                }
                batch.add(std::move(sample));
            }
        }

        return batch;
    }

    bool is_finished() const {
        if (config_.format == DataFormat::TAR) {
            // Check if all workers are finished AND all queues are empty
            for (const auto& worker : tar_workers_) {
                if (!worker->is_finished()) {
                    return false;
                }
                if (!worker->get_queue()->empty()) {
                    return false;
                }
            }
            return true;
        } else {
            return !running_ && fallback_queue_ && fallback_queue_->empty();
        }
    }

    size_t total_samples_processed() const {
        if (config_.format == DataFormat::TAR) {
            size_t total = 0;
            for (const auto& worker : tar_workers_) {
                total += worker->samples_processed();
            }
            return total;
        } else {
            return samples_processed_.load();
        }
    }

private:
    void initialize() {
        if (config_.format == DataFormat::TAR) {
            // Create buffer pool for TAR mode
            buffer_pool_ = std::make_unique<BufferPool>(
                256 * 256 * 3,
                config_.buffer_pool_size,
                config_.buffer_pool_size * 2
            );
        } else if (is_video_format(config_.format)) {
            init_video_decoder();
        } else if (is_tabular_format(config_.format)) {
            init_tabular_decoder();
        }
    }

    /**
     * @brief Check if path is a remote TAR (http://, https://, s3://, gs://)
     */
    bool is_remote_path(const std::string& path) const {
        return (path.substr(0, 7) == "http://" ||
                path.substr(0, 8) == "https://" ||
                path.substr(0, 5) == "s3://" ||
                path.substr(0, 5) == "gs://");
    }

    /**
     * @brief Fetch remote TAR data using ReaderOrchestrator
     */
    std::shared_ptr<std::vector<uint8_t>> fetch_remote_tar(const std::string& path) {
        ReaderOrchestrator reader;

        try {
            auto data_vec = reader.read(path);
            return std::make_shared<std::vector<uint8_t>>(std::move(data_vec));
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to fetch remote TAR from " + path + ": " + e.what());
        }
    }

    void init_video_decoder() {
#ifdef HAVE_FFMPEG
        video_decoder_ = std::make_unique<VideoDecoder>();
        VideoConfig video_config;
        video_config.frame_step = 1;
        video_config.max_frames = config_.max_video_frames;

        if (!video_decoder_->open(config_.data_path, video_config)) {
            throw std::runtime_error("Failed to open video: " + config_.data_path);
        }
#else
        throw std::runtime_error("Video support requires FFmpeg (-DHAVE_FFMPEG)");
#endif
    }

    void init_tabular_decoder() {
        if (config_.format == DataFormat::CSV) {
            csv_decoder_ = std::make_unique<CSVDecoder>();
            CSVConfig csv_config;
            csv_config.delimiter = config_.csv_delimiter;
            csv_config.has_header = config_.csv_has_header;

            if (!csv_decoder_->load_file(config_.data_path, csv_config)) {
                throw std::runtime_error("Failed to load CSV: " + config_.data_path);
            }
        } else if (config_.format == DataFormat::PARQUET) {
#ifdef HAVE_ARROW
            parquet_decoder_ = std::make_unique<ParquetDecoder>();
            ParquetConfig parquet_config;
            parquet_config.use_threads = config_.parquet_use_threads;
            parquet_config.use_memory_map = config_.parquet_use_mmap;

            if (!parquet_decoder_->open(config_.data_path, parquet_config)) {
                throw std::runtime_error("Failed to open Parquet: " + config_.data_path);
            }
#else
            throw std::runtime_error("Parquet support requires Arrow (-DHAVE_ARROW)");
#endif
        }
    }

    void video_worker() {
#ifdef HAVE_FFMPEG
        if (!video_decoder_ || !fallback_queue_) {
            running_ = false;
            return;
        }

        VideoConfig config;
        config.frame_step = 1;
        config.max_frames = config_.max_video_frames;

        std::vector<std::vector<uint8_t>> frames;
        std::vector<int> widths, heights;

        int num_frames = video_decoder_->extract_frames(config, frames, widths, heights);

        for (int i = 0; i < num_frames && running_; ++i) {
            UnifiedSample sample;
            sample.index = i;
            sample.format = config_.format;
            sample.image_data = std::move(frames[i]);
            sample.width = widths[i];
            sample.height = heights[i];
            sample.channels = 3;

            // Push to lock-free queue (busy-wait if full)
            while (running_ && fallback_queue_ && !fallback_queue_->try_push(std::move(sample))) {
                std::this_thread::yield();
            }
            if (!running_ || !fallback_queue_) break;
            samples_processed_++;
        }
#endif
        running_ = false;
    }

    void tabular_worker() {
        if (config_.format == DataFormat::CSV && csv_decoder_) {
            CSVConfig config;
            auto rows = csv_decoder_->parse_rows(config);
            auto metadata = csv_decoder_->get_metadata();

            for (size_t i = 0; i < rows.size() && running_; ++i) {
                UnifiedSample sample;
                sample.index = i;
                sample.format = DataFormat::CSV;
                sample.row_data = rows[i];
                sample.column_names = metadata.column_names;

                // Push to lock-free queue (busy-wait if full)
                while (running_ && fallback_queue_ && !fallback_queue_->try_push(std::move(sample))) {
                    std::this_thread::yield();
                }
                if (!running_ || !fallback_queue_) break;
                samples_processed_++;
            }
        }

        running_ = false;
    }

    static bool is_video_format(DataFormat format) {
        return format == DataFormat::MP4 || format == DataFormat::AVI ||
               format == DataFormat::MKV || format == DataFormat::MOV;
    }

    static bool is_tabular_format(DataFormat format) {
        return format == DataFormat::CSV || format == DataFormat::PARQUET;
    }

    UnifiedPipelineConfig config_;
    DataSource data_source_;

    // TAR mode
    std::unique_ptr<BufferPool> buffer_pool_;
    std::vector<std::unique_ptr<TarWorker>> tar_workers_;

    // Video mode
#ifdef HAVE_FFMPEG
    std::unique_ptr<VideoDecoder> video_decoder_;
#endif

    // Tabular mode
    std::unique_ptr<CSVDecoder> csv_decoder_;
#ifdef HAVE_ARROW
    std::unique_ptr<ParquetDecoder> parquet_decoder_;
#endif

    // Lock-free queue for video/tabular modes (single producer)
    std::unique_ptr<WorkerQueue> fallback_queue_;

    // Worker threads (for non-TAR modes)
    std::vector<std::thread> workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> samples_processed_;
    std::atomic<size_t> batches_produced_;
};

}  // namespace turboloader
