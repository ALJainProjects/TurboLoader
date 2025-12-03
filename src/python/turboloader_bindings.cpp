/**
 * @file turboloader_bindings.cpp
 * @brief Python bindings for TurboLoader v2.1.0
 *
 * Provides PyTorch-compatible DataLoader interface using pybind11.
 * Includes all SIMD-accelerated transforms with comprehensive docstrings.
 *
 * v1.8.0: ARM NEON optimizations, modern augmentations, error recovery, logging
 * v1.8.1: Full Python bindings for MixUp/CutMix/Mosaic source image methods
 * v2.0.0: Pipe operator for transforms, HDF5/TFRecord/Zarr support, COCO/VOC,
 *         Azure Blob Storage, GPU transforms, multi-platform wheels
 * v2.1.0: Fixed double-partitioning bug, fixed smart batcher race condition,
 *         exposed enable_smart_batching in Python bindings
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../pipeline/pipeline.hpp"
#include "../transforms/transforms.hpp"
#include "../readers/tbl_v2_reader.hpp"
#include "../writers/tbl_v2_writer.hpp"
#include "../formats/tbl_v2_format.hpp"
#include "../pipeline/smart_batching.hpp"
#include <thread>
#include <chrono>

namespace py = pybind11;
using namespace turboloader;
using namespace turboloader::transforms;
using namespace turboloader::pipeline;

/**
 * @brief Convert UnifiedSample to Python dict with NumPy array
 */
py::dict sample_to_dict(const UnifiedSample& sample) {
    py::dict result;

    result["index"] = sample.index;
    result["filename"] = sample.filename;
    result["width"] = sample.width;
    result["height"] = sample.height;
    result["channels"] = sample.channels;

    // Convert image data to NumPy array (zero-copy when possible)
    if (!sample.image_data.empty() && sample.width > 0 && sample.height > 0) {
        // Create NumPy array with shape (H, W, C)
        py::array_t<uint8_t> img_array({
            static_cast<py::ssize_t>(sample.height),
            static_cast<py::ssize_t>(sample.width),
            static_cast<py::ssize_t>(sample.channels)
        });

        // Copy data (necessary because C++ sample will be destroyed)
        auto buf = img_array.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        std::memcpy(ptr, sample.image_data.data(), sample.image_data.size());

        result["image"] = img_array;
    } else {
        result["image"] = py::none();
    }

    return result;
}

/**
 * @brief Python-friendly DataLoader wrapper for UnifiedPipeline
 *
 * Drop-in replacement for PyTorch DataLoader with TurboLoader performance.
 */
class DataLoader {
public:
    /**
     * @brief Constructor - PyTorch-compatible interface
     *
     * @param data_path Path to data (TAR, video, CSV, etc.)
     * @param batch_size Batch size (default: 32)
     * @param num_workers Number of worker threads (default: 4)
     * @param shuffle Enable shuffling (future feature, default: false)
     * @param enable_distributed Enable distributed training (default: false)
     * @param world_rank Rank of this process in distributed training (default: 0)
     * @param world_size Total number of processes in distributed training (default: 1)
     * @param drop_last Drop incomplete batches at end (default: false)
     * @param distributed_seed Seed for shuffling across ranks (default: 42)
     * @param enable_cache Enable tiered caching (NEW in v2.0.0, default: false)
     * @param cache_l1_mb L1 memory cache size in MB (default: 512)
     * @param cache_l2_gb L2 disk cache size in GB (default: 0 = disabled)
     * @param cache_dir L2 disk cache directory (default: /tmp/turboloader_cache)
     * @param auto_smart_batching Auto-detect if smart batching is beneficial (NEW in v2.3.0, default: true)
     * @param enable_smart_batching Manual override for smart batching (ignored if auto_smart_batching=true)
     * @param prefetch_batches Number of batches to prefetch (default: 4)
     */
    DataLoader(
        const std::string& data_path,
        size_t batch_size = 32,
        size_t num_workers = 4,
        bool shuffle = false,
        bool enable_distributed = false,
        int world_rank = 0,
        int world_size = 1,
        bool drop_last = false,
        int distributed_seed = 42,
        bool enable_cache = false,
        size_t cache_l1_mb = 512,
        size_t cache_l2_gb = 0,
        const std::string& cache_dir = "/tmp/turboloader_cache",
        bool auto_smart_batching = true,
        bool enable_smart_batching = false,
        size_t prefetch_batches = 4
    ) {
        config_.data_path = data_path;
        config_.batch_size = batch_size;
        config_.num_workers = num_workers;
        config_.shuffle = shuffle;
        config_.queue_size = 256;  // Good default for high throughput

        // Distributed Training
        config_.enable_distributed = enable_distributed;
        config_.world_rank = world_rank;
        config_.world_size = world_size;
        config_.drop_last = drop_last;
        config_.distributed_seed = distributed_seed;

        // Caching (NEW in v2.0.0)
        config_.enable_cache = enable_cache;
        config_.cache_l1_mb = cache_l1_mb;
        config_.cache_l2_gb = cache_l2_gb;
        config_.cache_dir = cache_dir;

        // Smart Batching (NEW in v2.3.0 - auto-detection)
        config_.auto_smart_batching = auto_smart_batching;
        config_.enable_smart_batching = enable_smart_batching;
        config_.prefetch_batches = prefetch_batches;

        // Create pipeline (will auto-detect format)
        pipeline_ = std::make_unique<UnifiedPipeline>(config_);
        pipeline_->start();
        started_ = true;
    }

    ~DataLoader() {
        if (pipeline_) {
            pipeline_->stop();
        }
    }

    /**
     * @brief Get next batch
     *
     * @return List of sample dictionaries
     */
    py::list next_batch() {
        if (!pipeline_) {
            throw std::runtime_error("DataLoader not initialized");
        }

        auto batch = pipeline_->next_batch();
        py::list result;

        for (const auto& sample : batch.samples) {
            result.append(sample_to_dict(sample));
        }

        return result;
    }

    /**
     * @brief Get next batch as contiguous NumPy array (HIGH PERFORMANCE)
     *
     * Returns batch as single contiguous array instead of list of dicts.
     * This is 8-12% faster than next_batch() due to:
     * - Single allocation for entire batch
     * - Parallel memcpy with OpenMP
     * - No per-sample Python dict creation overhead
     *
     * @param chw_format If true, return (N, C, H, W) for PyTorch; else (N, H, W, C)
     * @param target_height Target height (0 = use first image's height)
     * @param target_width Target width (0 = use first image's width)
     * @return Tuple of (batch_array, metadata_dict)
     */
    std::tuple<py::array_t<uint8_t>, py::dict> next_batch_array(
        bool chw_format = false,
        int target_height = 0,
        int target_width = 0
    ) {
        if (!pipeline_) {
            throw std::runtime_error("DataLoader not initialized");
        }

        // Get batch from pipeline (with GIL released)
        std::vector<UnifiedSample> samples;
        {
            py::gil_scoped_release release;
            auto batch = pipeline_->next_batch();
            samples = std::move(batch.samples);
        }

        if (samples.empty()) {
            return {py::array_t<uint8_t>(), py::dict()};
        }

        // Determine dimensions from first sample or targets
        int height = target_height > 0 ? target_height : samples[0].height;
        int width = target_width > 0 ? target_width : samples[0].width;
        int channels = samples[0].channels;
        size_t batch_size = samples.size();
        size_t single_image_size = static_cast<size_t>(height) * width * channels;

        // Allocate single contiguous array for entire batch
        py::array_t<uint8_t> array;

        if (chw_format) {
            // PyTorch format: (N, C, H, W)
            array = py::array_t<uint8_t>({
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(channels),
                static_cast<py::ssize_t>(height),
                static_cast<py::ssize_t>(width)
            });
        } else {
            // TensorFlow format: (N, H, W, C)
            array = py::array_t<uint8_t>({
                static_cast<py::ssize_t>(batch_size),
                static_cast<py::ssize_t>(height),
                static_cast<py::ssize_t>(width),
                static_cast<py::ssize_t>(channels)
            });
        }

        auto buf = array.request();
        uint8_t* dst = static_cast<uint8_t*>(buf.ptr);

        // Parallel copy all samples (GIL released)
        {
            py::gil_scoped_release release;

            #pragma omp parallel for if(batch_size > 4)
            for (size_t i = 0; i < batch_size; ++i) {
                const auto& sample = samples[i];

                // Skip samples with wrong dimensions
                if (sample.width != width || sample.height != height ||
                    sample.channels != channels || sample.image_data.empty()) {
                    continue;
                }

                if (chw_format) {
                    // HWC -> CHW transpose per image
                    size_t num_pixels = static_cast<size_t>(height) * width;
                    uint8_t* img_dst = dst + i * single_image_size;

                    for (int c = 0; c < channels; ++c) {
                        uint8_t* channel_dst = img_dst + c * num_pixels;
                        const uint8_t* src = sample.image_data.data();

                        for (size_t p = 0; p < num_pixels; ++p) {
                            channel_dst[p] = src[p * channels + c];
                        }
                    }
                } else {
                    // Direct copy for HWC format
                    std::memcpy(dst + i * single_image_size,
                               sample.image_data.data(),
                               single_image_size);
                }
            }
        }

        // Build lightweight metadata dict
        py::dict metadata;
        py::list indices, filenames;
        for (const auto& sample : samples) {
            indices.append(sample.index);
            filenames.append(sample.filename);
        }
        metadata["indices"] = indices;
        metadata["filenames"] = filenames;
        metadata["batch_size"] = batch_size;
        metadata["height"] = height;
        metadata["width"] = width;
        metadata["channels"] = channels;

        return {array, metadata};
    }

    /**
     * @brief Fill pre-allocated buffer with next batch (ZERO ALLOCATION)
     *
     * Even faster than next_batch_array() by reusing a pre-allocated buffer.
     *
     * @param buffer Pre-allocated NumPy array of shape (N, H, W, C)
     * @return Number of samples actually filled (may be < N for last batch)
     */
    size_t next_batch_into(py::array_t<uint8_t, py::array::c_style>& buffer) {
        if (!pipeline_) {
            throw std::runtime_error("DataLoader not initialized");
        }

        auto buf = buffer.request();
        if (buf.ndim != 4) {
            throw std::runtime_error("Buffer must be 4D (N, H, W, C)");
        }

        size_t max_batch = buf.shape[0];
        int height = buf.shape[1];
        int width = buf.shape[2];
        int channels = buf.shape[3];
        size_t image_size = static_cast<size_t>(height) * width * channels;

        // Get batch from pipeline (GIL released)
        std::vector<UnifiedSample> samples;
        {
            py::gil_scoped_release release;
            auto batch = pipeline_->next_batch();
            samples = std::move(batch.samples);
        }

        size_t count = std::min(samples.size(), max_batch);
        uint8_t* dst = static_cast<uint8_t*>(buf.ptr);

        // Parallel copy (GIL released)
        {
            py::gil_scoped_release release;

            #pragma omp parallel for if(count > 4)
            for (size_t i = 0; i < count; ++i) {
                const auto& sample = samples[i];
                if (sample.width == width && sample.height == height &&
                    sample.channels == channels && !sample.image_data.empty()) {
                    std::memcpy(dst + i * image_size,
                               sample.image_data.data(),
                               image_size);
                }
            }
        }

        return count;
    }

    /**
     * @brief Check if finished
     */
    bool is_finished() const {
        if (!pipeline_) {
            return true;
        }
        return pipeline_->is_finished();
    }

    /**
     * @brief Check if smart batching is active (NEW in v2.3.0)
     *
     * With auto_smart_batching=True, this tells you if smart batching
     * was enabled based on image size variation detection.
     */
    bool smart_batching_enabled() const {
        if (!pipeline_) {
            return false;
        }
        return pipeline_->smart_batching_enabled();
    }

    /**
     * @brief Stop the pipeline
     */
    void stop() {
        if (pipeline_) {
            pipeline_->stop();
        }
    }

    /**
     * @brief Context manager: __enter__
     */
    DataLoader& enter() {
        return *this;
    }

    /**
     * @brief Context manager: __exit__
     */
    void exit(py::object exc_type, py::object exc_value, py::object traceback) {
        stop();
    }

    /**
     * @brief Iterator: __iter__
     */
    DataLoader& iter() {
        return *this;
    }

    /**
     * @brief Iterator: __next__
     */
    py::list next() {
        // Don't check is_finished() at the start - let next_batch() handle it
        // This avoids race conditions where workers finish between checks

        // Keep trying to get a batch, with small sleep between attempts
        // This handles the case where workers need time to process
        py::list batch;
        int attempts = 0;
        const int max_attempts = 100;  // Up to 10 seconds (100ms * 100)

        while (attempts < max_attempts) {
            batch = next_batch();

            // If we got samples, return them
            if (py::len(batch) > 0) {
                return batch;
            }

            // If pipeline is finished and no samples, stop iteration
            if (is_finished()) {
                throw py::stop_iteration();
            }

            // Give workers time to process (release GIL while sleeping)
            py::gil_scoped_release release;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            attempts++;
        }

        // Timeout - if we got here, something might be wrong
        // Return empty batch rather than hanging forever
        return batch;
    }

    /**
     * @brief Set epoch for reproducible shuffling (NEW in v2.8.0)
     *
     * When shuffle=True, call this at the start of each epoch to get
     * reproducible shuffling. Different epochs produce different orderings,
     * but the same epoch + seed = same ordering.
     */
    void set_epoch(size_t epoch) {
        if (pipeline_) {
            pipeline_->set_epoch(epoch);
        }
    }

private:
    UnifiedPipelineConfig config_;
    std::unique_ptr<UnifiedPipeline> pipeline_;
    bool started_ = false;
};

/**
 * @brief Helper to convert ImageData to NumPy array
 */
py::array_t<uint8_t> imagedata_to_numpy(const ImageData& img) {
    py::array_t<uint8_t> array({
        static_cast<py::ssize_t>(img.height),
        static_cast<py::ssize_t>(img.width),
        static_cast<py::ssize_t>(img.channels)
    });

    auto buf = array.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    std::memcpy(ptr, img.data, img.width * img.height * img.channels);

    return array;
}

/**
 * @brief Helper to convert NumPy array to ImageData
 */
std::unique_ptr<ImageData> numpy_to_imagedata(py::array_t<uint8_t> array) {
    auto buf = array.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Array must be 3D (H, W, C)");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];

    size_t size = height * width * channels;
    auto data = new uint8_t[size];
    std::memcpy(data, buf.ptr, size);

    return std::make_unique<ImageData>(data, width, height, channels,
                                       width * channels, true);
}

/**
 * @brief pybind11 module definition
 *
 * Module is named _turboloader (with underscore) to avoid conflicts
 * with the turboloader package. The Python __init__.py re-exports the API.
 */
PYBIND11_MODULE(_turboloader, m) {
    m.doc() = "TurboLoader v2.1.0 - High-performance data loading for ML\n\n"
              "Drop-in replacement for PyTorch DataLoader with 12x speedup.\n\n"
              "Features:\n"
              "- ARM NEON optimizations (3-5x speedup on Apple Silicon)\n"
              "- Modern augmentations: MixUp, CutMix, Mosaic, RandAugment\n"
              "- Error recovery: graceful handling of corrupted files\n"
              "- Logging framework with profiling support\n"
              "- Distributed Training: Multi-node with deterministic sharding\n"
              "- TBL v2 format with LZ4 compression (40-60% space savings)\n"
              "- 23+ SIMD-accelerated transforms (AVX2/AVX-512/NEON)\n"
              "- Smart Batching (1.2x throughput, 15-25% less memory)\n"
              "- TAR archives (52+ Gbps local, HTTP/S3/GCS remote)\n"
              "- Multi-threaded with lock-free queues\n"
              "- AutoAugment policies (ImageNet, CIFAR10, SVHN)\n"
              "- PyTorch & TensorFlow tensor conversion\n"
              "- Data integrity validation (CRC32/CRC16)\n"
              "- Zero-copy where possible\n\n"
              "Usage:\n"
              "    import turboloader\n"
              "    # Single-node training\n"
              "    loader = turboloader.DataLoader('data.tar', batch_size=32, num_workers=8)\n"
              "    for batch in loader:\n"
              "        # batch is list of dicts with 'image' (numpy array) and metadata\n"
              "        pass\n\n"
              "    # Distributed training (PyTorch DDP)\n"
              "    loader = turboloader.DataLoader(\n"
              "        'data.tar',\n"
              "        batch_size=32,\n"
              "        num_workers=8,\n"
              "        enable_distributed=True,\n"
              "        world_rank=torch.distributed.get_rank(),\n"
              "        world_size=torch.distributed.get_world_size()\n"
              "    )\n\n"
              "Transforms:\n"
              "    transform = turboloader.Resize(224, 224, turboloader.InterpolationMode.BILINEAR)\n"
              "    output = transform.apply(image)\n\n"
              "Documentation:\n"
              "    https://github.com/ALJainProjects/TurboLoader/tree/main/docs";

    // DataLoader class (PyTorch-compatible)
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const std::string&, size_t, size_t, bool, bool, int, int, bool, int, bool, size_t, size_t, const std::string&, bool, bool, size_t>(),
             py::arg("data_path"),
             py::arg("batch_size") = 32,
             py::arg("num_workers") = 4,
             py::arg("shuffle") = false,
             py::arg("enable_distributed") = false,
             py::arg("world_rank") = 0,
             py::arg("world_size") = 1,
             py::arg("drop_last") = false,
             py::arg("distributed_seed") = 42,
             py::arg("enable_cache") = false,
             py::arg("cache_l1_mb") = 512,
             py::arg("cache_l2_gb") = 0,
             py::arg("cache_dir") = "/tmp/turboloader_cache",
             py::arg("auto_smart_batching") = true,
             py::arg("enable_smart_batching") = false,
             py::arg("prefetch_batches") = 4,
             "Create TurboLoader DataLoader (PyTorch-compatible)\n\n"
             "Args:\n"
             "    data_path (str): Path to data (TAR, video, CSV, Parquet)\n"
             "                    Supports: local files, http://, https://, s3://, gs://\n"
             "    batch_size (int): Samples per batch (default: 32)\n"
             "    num_workers (int): Worker threads (default: 4)\n"
             "    shuffle (bool): Shuffle samples (future feature, default: False)\n"
             "    enable_distributed (bool): Enable distributed training (NEW in v1.7.1, default: False)\n"
             "    world_rank (int): Rank of this process (0 to world_size-1, default: 0)\n"
             "    world_size (int): Total number of processes (default: 1)\n"
             "    drop_last (bool): Drop incomplete batches at end (default: False)\n"
             "    distributed_seed (int): Seed for shuffling (same across ranks, default: 42)\n"
             "    enable_cache (bool): Enable tiered caching (NEW in v2.0.0, default: False)\n"
             "    cache_l1_mb (int): L1 memory cache size in MB (default: 512)\n"
             "    cache_l2_gb (int): L2 disk cache size in GB, 0=disabled (default: 0)\n"
             "    cache_dir (str): L2 disk cache directory (default: /tmp/turboloader_cache)\n"
             "    auto_smart_batching (bool): Auto-detect if smart batching is beneficial (NEW in v2.3.0, default: True)\n"
             "    enable_smart_batching (bool): Manual override for smart batching, ignored if auto_smart_batching=True (default: False)\n"
             "    prefetch_batches (int): Number of batches to prefetch (default: 4)\n\n"
             "Returns:\n"
             "    DataLoader: Iterable that yields batches\n\n"
             "Example:\n"
             "    >>> # Single-node training (auto smart batching detects if sizes vary)\n"
             "    >>> loader = turboloader.DataLoader('imagenet.tar', batch_size=128, num_workers=8)\n"
             "    >>> for batch in loader:\n"
             "    >>>     images = [sample['image'] for sample in batch]  # NumPy arrays\n"
             "    >>>     # Train your model...\n\n"
             "    >>> # Force smart batching OFF (for uniform-size datasets)\n"
             "    >>> loader = turboloader.DataLoader(\n"
             "    >>>     'imagenet.tar',\n"
             "    >>>     auto_smart_batching=False,\n"
             "    >>>     enable_smart_batching=False\n"
             "    >>> )\n\n"
             "    >>> # Force smart batching ON\n"
             "    >>> loader = turboloader.DataLoader(\n"
             "    >>>     'imagenet.tar',\n"
             "    >>>     auto_smart_batching=False,\n"
             "    >>>     enable_smart_batching=True\n"
             "    >>> )"
        )
        .def("next_batch", &DataLoader::next_batch,
             "Get next batch\n\n"
             "Returns:\n"
             "    list: Batch of samples, each a dict with:\n"
             "        - 'index' (int): Sample index\n"
             "        - 'filename' (str): Original filename\n"
             "        - 'width' (int): Image width\n"
             "        - 'height' (int): Image height\n"
             "        - 'channels' (int): Number of channels (3 for RGB)\n"
             "        - 'image' (np.ndarray): Image data (H, W, C) uint8"
        )
        .def("next_batch_array", &DataLoader::next_batch_array,
             py::arg("chw_format") = false,
             py::arg("target_height") = 0,
             py::arg("target_width") = 0,
             "Get next batch as contiguous NumPy array (HIGH PERFORMANCE)\n\n"
             "Returns batch as single contiguous array instead of list of dicts.\n"
             "This is 8-12% faster than next_batch() due to:\n"
             "- Single allocation for entire batch\n"
             "- Parallel memcpy with OpenMP\n"
             "- No per-sample Python dict creation overhead\n\n"
             "Args:\n"
             "    chw_format (bool): If True, return (N, C, H, W) for PyTorch; else (N, H, W, C)\n"
             "    target_height (int): Target height (0 = use first image's height)\n"
             "    target_width (int): Target width (0 = use first image's width)\n\n"
             "Returns:\n"
             "    tuple: (batch_array, metadata_dict)\n"
             "        - batch_array: np.ndarray of shape (N, H, W, C) or (N, C, H, W)\n"
             "        - metadata_dict: {'indices': [...], 'filenames': [...], ...}"
        )
        .def("next_batch_into", &DataLoader::next_batch_into,
             py::arg("buffer"),
             "Fill pre-allocated buffer with next batch (ZERO ALLOCATION)\n\n"
             "Even faster than next_batch_array() by reusing a pre-allocated buffer.\n\n"
             "Args:\n"
             "    buffer (np.ndarray): Pre-allocated array of shape (N, H, W, C)\n\n"
             "Returns:\n"
             "    int: Number of samples actually filled (may be < N for last batch)"
        )
        .def("is_finished", &DataLoader::is_finished,
             "Check if all data has been processed\n\n"
             "Returns:\n"
             "    bool: True if pipeline finished")
        .def("smart_batching_enabled", &DataLoader::smart_batching_enabled,
             "Check if smart batching is active (NEW in v2.3.0)\n\n"
             "With auto_smart_batching=True, this returns True if smart batching\n"
             "was enabled based on image size variation detection.\n\n"
             "Returns:\n"
             "    bool: True if smart batching is currently active")
        .def("set_epoch", &DataLoader::set_epoch,
             py::arg("epoch"),
             "Set epoch for reproducible shuffling (NEW in v2.8.0)\n\n"
             "When shuffle=True, call this at the start of each epoch to get\n"
             "reproducible shuffling. Different epochs produce different orderings,\n"
             "but the same epoch + seed = same ordering.\n\n"
             "Args:\n"
             "    epoch (int): The epoch number (0, 1, 2, ...)\n\n"
             "Example:\n"
             "    >>> loader = turboloader.DataLoader('data.tar', shuffle=True)\n"
             "    >>> for epoch in range(10):\n"
             "    ...     loader.set_epoch(epoch)\n"
             "    ...     for batch in loader:\n"
             "    ...         train(batch)")
        .def("stop", &DataLoader::stop,
             "Stop the pipeline and clean up resources")
        .def("__enter__", &DataLoader::enter,
             "Context manager entry")
        .def("__exit__", &DataLoader::exit,
             "Context manager exit")
        .def("__iter__", &DataLoader::iter,
             "Make DataLoader iterable")
        .def("__next__", &DataLoader::next,
             "Get next batch (iterator protocol)");

    // Module-level functions
    m.def("version", []() { return "2.5.0"; },
          "Get TurboLoader version\n\n"
          "Returns:\n"
          "    str: Version string (e.g., '2.3.23')");

    m.def("features", []() {
        py::dict features;
        features["version"] = "2.5.0";
        features["distributed_training"] = true;
        features["tar_support"] = true;
        features["remote_tar"] = true;
        features["http_support"] = true;
        features["s3_support"] = true;
        features["gcs_support"] = true;
        features["azure_support"] = true;
        features["jpeg_decode"] = true;
        features["png_decode"] = true;
        features["webp_decode"] = true;
        features["simd_acceleration"] = true;
        features["lock_free_queues"] = true;
        features["num_transforms"] = 24;
        features["autoaugment"] = true;
        features["pytorch_tensors"] = true;
        features["tensorflow_tensors"] = true;
        features["lanczos_interpolation"] = true;
        features["smart_batching"] = true;
        features["pipe_operator"] = true;
        features["hdf5_support"] = true;
        features["tfrecord_support"] = true;
        features["zarr_support"] = true;
        features["coco_voc_support"] = true;
#ifdef __linux__
        features["io_uring"] = true;
#else
        features["io_uring"] = false;
#endif
#ifdef TURBOLOADER_HAS_CUDA
        features["gpu_transforms"] = true;
#else
        features["gpu_transforms"] = false;
#endif
        return features;
    }, "Get TurboLoader feature support\n\n"
       "Returns:\n"
       "    dict: Feature flags and capabilities\n\n"
       "Example:\n"
       "    >>> import turboloader\n"
       "    >>> features = turboloader.features()\n"
       "    >>> print(f\"Version: {features['version']}\")\n"
       "    >>> print(f\"Transforms: {features['num_transforms']}\")\n"
       "    >>> print(f\"Distributed: {features['distributed_training']}\")");

    m.def("list_transforms", []() {
        py::list transforms;
        transforms.append("Resize");
        transforms.append("Normalize");
        transforms.append("ImageNetNormalize");
        transforms.append("RandomHorizontalFlip");
        transforms.append("RandomVerticalFlip");
        transforms.append("CenterCrop");
        transforms.append("RandomCrop");
        transforms.append("ColorJitter");
        transforms.append("Grayscale");
        transforms.append("Pad");
        transforms.append("RandomRotation");
        transforms.append("RandomAffine");
        transforms.append("GaussianBlur");
        transforms.append("RandomErasing");
        transforms.append("RandomPosterize");
        transforms.append("RandomSolarize");
        transforms.append("RandomPerspective");
        transforms.append("AutoAugment");
        transforms.append("ToTensor");
        return transforms;
    }, "List all available transforms\n\n"
       "Returns:\n"
       "    list[str]: Names of all 19 transforms\n\n"
       "Example:\n"
       "    >>> import turboloader\n"
       "    >>> print(turboloader.list_transforms())");

    // ========================================================================
    // TRANSFORM BINDINGS
    // ========================================================================

    // Enums
    py::enum_<InterpolationMode>(m, "InterpolationMode",
                 "Interpolation modes for image resizing\n\n"
                 "Available modes:\n"
                 "  NEAREST: Nearest-neighbor (fastest, lowest quality)\n"
                 "  BILINEAR: Bilinear interpolation (good balance)\n"
                 "  BICUBIC: Bicubic interpolation (higher quality)\n"
                 "  LANCZOS: Lanczos resampling (highest quality, best for downsampling)")
        .value("NEAREST", InterpolationMode::NEAREST)
        .value("BILINEAR", InterpolationMode::BILINEAR)
        .value("BICUBIC", InterpolationMode::BICUBIC)
        .value("LANCZOS", InterpolationMode::LANCZOS);

    py::enum_<PaddingMode>(m, "PaddingMode",
                 "Padding modes for image operations\n\n"
                 "Available modes:\n"
                 "  CONSTANT: Pad with constant value\n"
                 "  EDGE: Pad with edge pixel values\n"
                 "  REFLECT: Reflect pixels at border")
        .value("CONSTANT", PaddingMode::CONSTANT)
        .value("EDGE", PaddingMode::EDGE)
        .value("REFLECT", PaddingMode::REFLECT);

    py::enum_<TensorFormat>(m, "TensorFormat",
                 "Tensor format for framework compatibility\n\n"
                 "Available formats:\n"
                 "  NONE: Keep as HWC uint8\n"
                 "  PYTORCH_CHW: Convert to CHW float32 (PyTorch)\n"
                 "  TENSORFLOW_HWC: Convert to HWC float32 (TensorFlow)")
        .value("NONE", TensorFormat::NONE)
        .value("PYTORCH_CHW", TensorFormat::PYTORCH_CHW)
        .value("TENSORFLOW_HWC", TensorFormat::TENSORFLOW_HWC);

    // Base Transform class with pipe operator support
    py::class_<Transform>(m, "Transform",
             "Base class for all image transforms\n\n"
             "All transforms inherit from this class and provide SIMD-accelerated operations.\n"
             "Transforms can be composed into pipelines for efficient batch processing.\n\n"
             "Pipe operator (|) support:\n"
             "    >>> pipeline = Resize(224, 224) | RandomHorizontalFlip(0.5) | ImageNetNormalize()\n"
             "    >>> output = pipeline.apply(image)")
        .def("apply", [](Transform& self, py::array_t<uint8_t> img) {
            auto input = numpy_to_imagedata(img);
            auto output = self.apply(*input);
            return imagedata_to_numpy(*output);
        }, "Apply transform to image\n\n"
           "Args:\n"
           "    img (np.ndarray): Input image (H, W, C) uint8\n\n"
           "Returns:\n"
           "    np.ndarray: Transformed image (H, W, C) uint8")
        .def("name", &Transform::name,
             "Get transform name\n\n"
             "Returns:\n"
             "    str: Name of the transform")
        .def("is_deterministic", &Transform::is_deterministic,
             "Check if transform is deterministic\n\n"
             "Returns:\n"
             "    bool: True if transform produces same output for same input")
        .def("__or__", [](py::object self, py::object other) {
            // Create a new ComposedTransforms from two transforms
            py::list transforms;
            transforms.append(self);
            transforms.append(other);
            // Get the Compose function from the module
            py::module_ m = py::module_::import("_turboloader");
            return m.attr("Compose")(transforms);
        }, py::arg("other"),
           "Pipe operator for composing transforms\n\n"
           "Usage:\n"
           "    >>> pipeline = Resize(224, 224) | RandomHorizontalFlip(0.5) | ImageNetNormalize()\n"
           "    >>> output = pipeline.apply(image)\n\n"
           "Args:\n"
           "    other: Another transform to chain\n\n"
           "Returns:\n"
           "    ComposedTransforms: A pipeline combining both transforms");

    // Resize
    py::class_<ResizeTransform, Transform>(m, "Resize",
             "SIMD-accelerated image resizing transform\n\n"
             "Resizes images to target dimensions using optimized interpolation.\n"
             "Performance: 3.2x faster than torchvision\n\n"
             "Example:\n"
             "    >>> transform = turboloader.Resize(224, 224, turboloader.InterpolationMode.BILINEAR)\n"
             "    >>> output = transform.apply(image)")
        .def(py::init<int, int, InterpolationMode>(),
             py::arg("width"), py::arg("height"),
             py::arg("interpolation") = InterpolationMode::BILINEAR,
             "Create Resize transform\n\n"
             "Args:\n"
             "    width (int): Target width in pixels\n"
             "    height (int): Target height in pixels\n"
             "    interpolation (InterpolationMode): Interpolation method (default: BILINEAR)");

    // Normalize
    py::class_<NormalizeTransform, Transform>(m, "Normalize",
             "SIMD-accelerated normalization transform\n\n"
             "Normalizes images using mean and standard deviation.\n"
             "Formula: output = (input - mean) / std\n\n"
             "Example:\n"
             "    >>> transform = turboloader.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])")
        .def(py::init<const std::vector<float>&, const std::vector<float>&, bool>(),
             py::arg("mean"), py::arg("std"), py::arg("to_float") = false,
             "Create Normalize transform\n\n"
             "Args:\n"
             "    mean (list[float]): Mean values for each channel\n"
             "    std (list[float]): Standard deviation for each channel\n"
             "    to_float (bool): Convert to float32 (default: False)");

    py::class_<ImageNetNormalize, NormalizeTransform>(m, "ImageNetNormalize",
             "ImageNet normalization preset\n\n"
             "Uses ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].\n"
             "Convenient shorthand for ImageNet training/validation.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.ImageNetNormalize(to_float=True)")
        .def(py::init<bool>(), py::arg("to_float") = false,
             "Create ImageNet normalization transform\n\n"
             "Args:\n"
             "    to_float (bool): Convert to float32 (default: False)");

    // Flips
    py::class_<RandomHorizontalFlipTransform, Transform>(m, "RandomHorizontalFlip",
             "SIMD-accelerated random horizontal flip\n\n"
             "Randomly flips images horizontally with probability p.\n"
             "Performance: 10,000+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomHorizontalFlip(p=0.5)")
        .def(py::init<float, unsigned>(),
             py::arg("p") = 0.5f, py::arg("seed") = std::random_device{}(),
             "Create RandomHorizontalFlip transform\n\n"
             "Args:\n"
             "    p (float): Probability of flipping (default: 0.5)\n"
             "    seed (int): Random seed for reproducibility");

    py::class_<RandomVerticalFlipTransform, Transform>(m, "RandomVerticalFlip",
             "SIMD-accelerated random vertical flip\n\n"
             "Randomly flips images vertically with probability p.\n"
             "Performance: 10,000+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomVerticalFlip(p=0.5)")
        .def(py::init<float, unsigned>(),
             py::arg("p") = 0.5f, py::arg("seed") = std::random_device{}(),
             "Create RandomVerticalFlip transform\n\n"
             "Args:\n"
             "    p (float): Probability of flipping (default: 0.5)\n"
             "    seed (int): Random seed for reproducibility");

    // Crops
    py::class_<CenterCropTransform, Transform>(m, "CenterCrop",
             "Center crop transform\n\n"
             "Crops the center region of the image to target size.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.CenterCrop(224, 224)")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"),
             "Create CenterCrop transform\n\n"
             "Args:\n"
             "    width (int): Target width\n"
             "    height (int): Target height");

    py::class_<RandomCropTransform, Transform>(m, "RandomCrop",
             "Random crop with optional padding\n\n"
             "Randomly crops a region from the image with optional padding.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomCrop(224, 224, padding=32)")
        .def(py::init<int, int, int, PaddingMode, uint8_t, unsigned>(),
             py::arg("width"), py::arg("height"),
             py::arg("padding") = 0,
             py::arg("pad_mode") = PaddingMode::CONSTANT,
             py::arg("pad_value") = 0,
             py::arg("seed") = std::random_device{}(),
             "Create RandomCrop transform\n\n"
             "Args:\n"
             "    width (int): Target crop width\n"
             "    height (int): Target crop height\n"
             "    padding (int): Padding size (default: 0)\n"
             "    pad_mode (PaddingMode): Padding mode (default: CONSTANT)\n"
             "    pad_value (int): Padding value for CONSTANT mode (default: 0)\n"
             "    seed (int): Random seed");

    // ColorJitter
    py::class_<ColorJitterTransform, Transform>(m, "ColorJitter",
             "SIMD-accelerated color jitter transform\n\n"
             "Randomly adjusts brightness, contrast, saturation, and hue.\n"
             "Performance: 5,000+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.ColorJitter(brightness=0.2, contrast=0.2)")
        .def(py::init<float, float, float, float, unsigned>(),
             py::arg("brightness") = 0.0f,
             py::arg("contrast") = 0.0f,
             py::arg("saturation") = 0.0f,
             py::arg("hue") = 0.0f,
             py::arg("seed") = std::random_device{}(),
             "Create ColorJitter transform\n\n"
             "Args:\n"
             "    brightness (float): Brightness adjustment factor (0.0 = no change)\n"
             "    contrast (float): Contrast adjustment factor (0.0 = no change)\n"
             "    saturation (float): Saturation adjustment factor (0.0 = no change)\n"
             "    hue (float): Hue adjustment factor (0.0 = no change)\n"
             "    seed (int): Random seed");

    // Grayscale
    py::class_<GrayscaleTransform, Transform>(m, "Grayscale",
             "Convert image to grayscale\n\n"
             "Converts RGB images to grayscale using weighted average.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.Grayscale(num_output_channels=1)")
        .def(py::init<int>(), py::arg("num_output_channels") = 1,
             "Create Grayscale transform\n\n"
             "Args:\n"
             "    num_output_channels (int): Output channels (1 or 3, default: 1)");

    // Pad
    py::class_<PadTransform, Transform>(m, "Pad",
             "Pad image with specified mode\n\n"
             "Adds padding around the image border.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.Pad(32, mode=turboloader.PaddingMode.REFLECT)")
        .def(py::init<int, PaddingMode, uint8_t>(),
             py::arg("padding"),
             py::arg("mode") = PaddingMode::CONSTANT,
             py::arg("value") = 0,
             "Create Pad transform\n\n"
             "Args:\n"
             "    padding (int): Padding size on all sides\n"
             "    mode (PaddingMode): Padding mode (default: CONSTANT)\n"
             "    value (int): Padding value for CONSTANT mode (default: 0)");

    // Rotation
    py::class_<RandomRotationTransform, Transform>(m, "RandomRotation",
             "Random rotation transform\n\n"
             "Randomly rotates images by angle in [-degrees, +degrees].\n"
             "Uses SIMD-accelerated bilinear interpolation.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomRotation(15.0)")
        .def(py::init<float, bool, uint8_t, unsigned>(),
             py::arg("degrees"),
             py::arg("expand") = false,
             py::arg("fill") = 0,
             py::arg("seed") = std::random_device{}(),
             "Create RandomRotation transform\n\n"
             "Args:\n"
             "    degrees (float): Rotation range [-degrees, +degrees]\n"
             "    expand (bool): Expand output to fit rotated image (default: False)\n"
             "    fill (int): Fill value for empty areas (default: 0)\n"
             "    seed (int): Random seed");

    // Affine
    py::class_<RandomAffineTransform, Transform>(m, "RandomAffine",
             "Random affine transformation\n\n"
             "Applies random affine transformations (rotation, translation, scale, shear).\n"
             "Uses SIMD-accelerated interpolation.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomAffine(degrees=15, scale_min=0.8, scale_max=1.2)")
        .def(py::init<float, float, float, float, float, float, uint8_t, unsigned>(),
             py::arg("degrees") = 0.0f,
             py::arg("translate_x") = 0.0f,
             py::arg("translate_y") = 0.0f,
             py::arg("scale_min") = 1.0f,
             py::arg("scale_max") = 1.0f,
             py::arg("shear") = 0.0f,
             py::arg("fill") = 0,
             py::arg("seed") = std::random_device{}(),
             "Create RandomAffine transform\n\n"
             "Args:\n"
             "    degrees (float): Rotation range (default: 0.0)\n"
             "    translate_x (float): Horizontal translation fraction (default: 0.0)\n"
             "    translate_y (float): Vertical translation fraction (default: 0.0)\n"
             "    scale_min (float): Minimum scale factor (default: 1.0)\n"
             "    scale_max (float): Maximum scale factor (default: 1.0)\n"
             "    shear (float): Shear angle in degrees (default: 0.0)\n"
             "    fill (int): Fill value for empty areas (default: 0)\n"
             "    seed (int): Random seed");

    // Blur
    py::class_<GaussianBlurTransform, Transform>(m, "GaussianBlur",
             "SIMD-accelerated Gaussian blur\n\n"
             "Applies Gaussian blur using separable convolution.\n"
             "Performance: 2,000+ img/s (kernel_size=5)\n\n"
             "Example:\n"
             "    >>> transform = turboloader.GaussianBlur(kernel_size=5, sigma=1.5)")
        .def(py::init<int, float>(),
             py::arg("kernel_size"), py::arg("sigma") = 0.0f,
             "Create GaussianBlur transform\n\n"
             "Args:\n"
             "    kernel_size (int): Blur kernel size (must be odd)\n"
             "    sigma (float): Gaussian sigma (default: auto-calculate from kernel_size)");

    // Erasing
    py::class_<RandomErasingTransform, Transform>(m, "RandomErasing",
             "Random erasing augmentation (Cutout)\n\n"
             "Randomly erases rectangular regions for data augmentation.\n"
             "Performance: 8,000+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomErasing(p=0.5, scale_min=0.02, scale_max=0.33)")
        .def(py::init<float, float, float, float, float, uint8_t, unsigned>(),
             py::arg("p") = 0.5f,
             py::arg("scale_min") = 0.02f,
             py::arg("scale_max") = 0.33f,
             py::arg("ratio_min") = 0.3f,
             py::arg("ratio_max") = 3.33f,
             py::arg("value") = 0,
             py::arg("seed") = std::random_device{}(),
             "Create RandomErasing transform\n\n"
             "Args:\n"
             "    p (float): Probability of applying erasing (default: 0.5)\n"
             "    scale_min (float): Minimum erased area relative to image (default: 0.02)\n"
             "    scale_max (float): Maximum erased area relative to image (default: 0.33)\n"
             "    ratio_min (float): Minimum aspect ratio (default: 0.3)\n"
             "    ratio_max (float): Maximum aspect ratio (default: 3.33)\n"
             "    value (int): Fill value for erased region (default: 0)\n"
             "    seed (int): Random seed");

    // Advanced Transforms (v0.7.0)

    // Posterize
    py::class_<RandomPosterizeTransform, Transform>(m, "RandomPosterize",
             "Random posterize transform (NEW in v0.7.0)\n\n"
             "Reduces bit depth for a posterization effect.\n"
             "Ultra-fast bitwise operations.\n"
             "Performance: 336,000+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomPosterize(bits=4, p=0.5)")
        .def(py::init<int, float, unsigned>(),
             py::arg("bits"), py::arg("p") = 1.0f, py::arg("seed") = std::random_device{}(),
             "Create RandomPosterize transform\n\n"
             "Args:\n"
             "    bits (int): Number of bits to keep (1-8)\n"
             "    p (float): Probability of applying (default: 1.0)\n"
             "    seed (int): Random seed");

    // Solarize
    py::class_<RandomSolarizeTransform, Transform>(m, "RandomSolarize",
             "Random solarize transform (NEW in v0.7.0)\n\n"
             "Inverts pixels above threshold for a solarization effect.\n"
             "Performance: 21,000+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomSolarize(threshold=128, p=0.5)")
        .def(py::init<uint8_t, float, unsigned>(),
             py::arg("threshold"), py::arg("p") = 1.0f, py::arg("seed") = std::random_device{}(),
             "Create RandomSolarize transform\n\n"
             "Args:\n"
             "    threshold (int): Inversion threshold (0-255)\n"
             "    p (float): Probability of applying (default: 1.0)\n"
             "    seed (int): Random seed");

    // Perspective
    py::class_<RandomPerspectiveTransform, Transform>(m, "RandomPerspective",
             "Random perspective warp (NEW in v0.7.0)\n\n"
             "Applies random perspective transformation with SIMD interpolation.\n"
             "Performance: 9,900+ img/s\n\n"
             "Example:\n"
             "    >>> transform = turboloader.RandomPerspective(distortion_scale=0.5, p=0.5)")
        .def(py::init<float, float, uint8_t, unsigned>(),
             py::arg("distortion_scale"), py::arg("p") = 0.5f,
             py::arg("fill") = 0, py::arg("seed") = std::random_device{}(),
             "Create RandomPerspective transform\n\n"
             "Args:\n"
             "    distortion_scale (float): Perspective distortion amount (0.0-1.0)\n"
             "    p (float): Probability of applying (default: 0.5)\n"
             "    fill (int): Fill value for empty areas (default: 0)\n"
             "    seed (int): Random seed");

    // AutoAugment
    py::enum_<AutoAugmentPolicy>(m, "AutoAugmentPolicy",
                 "AutoAugment policy presets (NEW in v0.7.0)\n\n"
                 "Learned augmentation policies for different datasets:\n"
                 "  IMAGENET: Optimized for ImageNet classification\n"
                 "  CIFAR10: Optimized for CIFAR-10\n"
                 "  SVHN: Optimized for Street View House Numbers")
        .value("IMAGENET", AutoAugmentPolicy::IMAGENET)
        .value("CIFAR10", AutoAugmentPolicy::CIFAR10)
        .value("SVHN", AutoAugmentPolicy::SVHN);

    py::class_<AutoAugmentTransform, Transform>(m, "AutoAugment",
             "AutoAugment learned policies (NEW in v0.7.0)\n\n"
             "State-of-the-art learned augmentation policies.\n"
             "Performance: 19,800+ img/s (ImageNet policy)\n\n"
             "Example:\n"
             "    >>> transform = turboloader.AutoAugment(policy=turboloader.AutoAugmentPolicy.IMAGENET)")
        .def(py::init<AutoAugmentPolicy, unsigned>(),
             py::arg("policy") = AutoAugmentPolicy::IMAGENET,
             py::arg("seed") = std::random_device{}(),
             "Create AutoAugment transform\n\n"
             "Args:\n"
             "    policy (AutoAugmentPolicy): Augmentation policy (default: IMAGENET)\n"
             "    seed (int): Random seed");

    // ToTensor
    py::class_<ToTensorTransform, Transform>(m, "ToTensor",
             "Convert to tensor format\n\n"
             "Converts uint8 images to float32 tensors in PyTorch (CHW) or TensorFlow (HWC) format.\n"
             "Includes normalization to [0,1] range.\n\n"
             "Example:\n"
             "    >>> transform = turboloader.ToTensor(format=turboloader.TensorFormat.PYTORCH_CHW)")
        .def(py::init<TensorFormat, bool>(),
             py::arg("format") = TensorFormat::PYTORCH_CHW,
             py::arg("normalize") = true,
             "Create ToTensor transform\n\n"
             "Args:\n"
             "    format (TensorFormat): Output tensor format (default: PYTORCH_CHW)\n"
             "    normalize (bool): Normalize to [0,1] range (default: True)");

    // Python-side Compose helper (stores Python object references)
    // Since C++ Transform objects can't be easily cloned across the Python/C++ boundary,
    // we create a Python-side wrapper class that holds references to Python transform objects
    class PyTransformPipeline {
    private:
        std::vector<py::object> transforms_;

    public:
        explicit PyTransformPipeline(py::list transforms) {
            for (auto transform_obj : transforms) {
                transforms_.push_back(py::reinterpret_borrow<py::object>(transform_obj));
            }
        }

        py::array_t<uint8_t> apply(py::array_t<uint8_t> img) {
            py::array_t<uint8_t> current = img;

            // Apply each transform sequentially
            for (const auto& transform_obj : transforms_) {
                // Call the transform's apply method
                current = transform_obj.attr("apply")(current).cast<py::array_t<uint8_t>>();
            }

            return current;
        }

        size_t size() const { return transforms_.size(); }

        // Get transforms for pipe operator
        const std::vector<py::object>& get_transforms() const { return transforms_; }
    };

    // Bind PyTransformPipeline
    py::class_<PyTransformPipeline>(m, "ComposedTransforms",
                  "Transform pipeline that applies multiple transforms sequentially\n\n"
                  "This class composes multiple transforms into a single operation.\n"
                  "Transforms are applied in the order they were added.\n\n"
                  "Pipe operator (|) support:\n"
                  "    >>> pipeline = Resize(224, 224) | RandomHorizontalFlip(0.5)\n"
                  "    >>> extended = pipeline | ImageNetNormalize()")
        .def("apply", &PyTransformPipeline::apply,
             py::arg("img"),
             "Apply all transforms in sequence\n\n"
             "Args:\n"
             "    img (np.ndarray): Input image (H, W, C) uint8\n\n"
             "Returns:\n"
             "    np.ndarray: Transformed image")
        .def("__len__", &PyTransformPipeline::size,
             "Get number of transforms in pipeline")
        .def("__call__", &PyTransformPipeline::apply,
             py::arg("img"),
             "Apply pipeline (callable interface)")
        .def("__or__", [](PyTransformPipeline& self, py::object other) {
            // Create a new ComposedTransforms with all existing transforms + the new one
            py::list transforms;
            for (const auto& t : self.get_transforms()) {
                transforms.append(t);
            }
            transforms.append(other);
            return PyTransformPipeline(transforms);
        }, py::arg("other"),
           "Pipe operator for extending pipelines\n\n"
           "Usage:\n"
           "    >>> pipeline = Resize(224, 224) | RandomHorizontalFlip(0.5)\n"
           "    >>> extended = pipeline | ImageNetNormalize()\n\n"
           "Args:\n"
           "    other: Another transform to add to the pipeline\n\n"
           "Returns:\n"
           "    ComposedTransforms: Extended pipeline with the new transform");

    // Compose helper function
    m.def("Compose", [](py::list transforms) -> PyTransformPipeline {
        if (transforms.size() == 0) {
            throw std::runtime_error("Compose() requires at least one transform");
        }

        // Validate that all items have an 'apply' method
        for (auto transform_obj : transforms) {
            if (!py::hasattr(transform_obj, "apply")) {
                throw std::runtime_error(
                    "Compose() requires all items to have an 'apply' method. "
                    "Make sure you're passing Transform objects."
                );
            }
        }

        return PyTransformPipeline(transforms);
    }, py::arg("transforms"),
       "Create a transform pipeline (Compose multiple transforms)\n\n"
       "Combines multiple transforms into a single pipeline that applies them sequentially.\n"
       "This is equivalent to calling each transform individually, but more convenient.\n\n"
       "Args:\n"
       "    transforms (list[Transform]): List of transforms to apply in order\n\n"
       "Returns:\n"
       "    ComposedTransforms: Pipeline that applies all transforms sequentially\n\n"
       "Example:\n"
       "    >>> import turboloader\n"
       "    >>> import numpy as np\n"
       "    >>> \n"
       "    >>> # Create individual transforms\n"
       "    >>> resize = turboloader.Resize(224, 224)\n"
       "    >>> flip = turboloader.RandomHorizontalFlip(0.5)\n"
       "    >>> normalize = turboloader.ImageNetNormalize()\n"
       "    >>> \n"
       "    >>> # Compose them into a pipeline\n"
       "    >>> pipeline = turboloader.Compose([resize, flip, normalize])\n"
       "    >>> \n"
       "    >>> # Apply pipeline to an image\n"
       "    >>> img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)\n"
       "    >>> transformed = pipeline.apply(img)\n"
       "    >>> # Or use callable interface\n"
       "    >>> transformed = pipeline(img)");

    // ========================================================================
    // TBL V2 FORMAT BINDINGS (NEW in v1.5.0)
    // ========================================================================

    // SampleFormat enum
    py::enum_<formats::SampleFormat>(m, "SampleFormat",
                 "Sample format types for TBL v2\n\n"
                 "Available formats:\n"
                 "  UNKNOWN: Unknown/unsupported format\n"
                 "  JPEG: JPEG image\n"
                 "  PNG: PNG image\n"
                 "  WEBP: WebP image\n"
                 "  BMP: BMP image\n"
                 "  TIFF: TIFF image\n"
                 "  VIDEO_MP4: MP4 video\n"
                 "  VIDEO_AVI: AVI video")
        .value("UNKNOWN", formats::SampleFormat::UNKNOWN)
        .value("JPEG", formats::SampleFormat::JPEG)
        .value("PNG", formats::SampleFormat::PNG)
        .value("WEBP", formats::SampleFormat::WEBP)
        .value("BMP", formats::SampleFormat::BMP)
        .value("TIFF", formats::SampleFormat::TIFF)
        .value("VIDEO_MP4", formats::SampleFormat::VIDEO_MP4)
        .value("VIDEO_AVI", formats::SampleFormat::VIDEO_AVI);

    // MetadataType enum
    py::enum_<formats::MetadataType>(m, "MetadataType",
                 "Metadata types for TBL v2\n\n"
                 "Available types:\n"
                 "  NONE: No metadata\n"
                 "  JSON: JSON-formatted metadata\n"
                 "  PROTOBUF: Protocol Buffers\n"
                 "  MSGPACK: MessagePack format\n"
                 "  CUSTOM: Custom binary format")
        .value("NONE", formats::MetadataType::NONE)
        .value("JSON", formats::MetadataType::JSON)
        .value("PROTOBUF", formats::MetadataType::PROTOBUF)
        .value("MSGPACK", formats::MetadataType::MSGPACK)
        .value("CUSTOM", formats::MetadataType::CUSTOM);

    // TblReaderV2 class
    py::class_<readers::TblReaderV2>(m, "TblReaderV2",
             "TBL v2 format reader with LZ4 decompression (NEW in v1.5.0)\n\n"
             "Features:\n"
             "- Memory-mapped I/O for zero-copy reads\n"
             "- Automatic LZ4 decompression\n"
             "- Checksum verification\n"
             "- Metadata access\n"
             "- Dimension-based filtering\n\n"
             "Example:\n"
             "    >>> reader = turboloader.TblReaderV2('dataset.tbl', verify_checksums=True)\n"
             "    >>> print(f'Samples: {reader.num_samples()}')\n"
             "    >>> data, size = reader.read_sample(0)\n"
             "    >>> metadata, meta_type = reader.read_metadata(0)")
        .def(py::init<const std::string&, bool>(),
             py::arg("path"),
             py::arg("verify_checksums") = true,
             "Create TBL v2 reader\n\n"
             "Args:\n"
             "    path (str): Path to TBL v2 file\n"
             "    verify_checksums (bool): Enable checksum verification (default: True)")
        .def("read_sample", [](readers::TblReaderV2& self, size_t index) {
            auto [data, size] = self.read_sample(index);
            // Return as Python bytes
            return py::bytes(reinterpret_cast<const char*>(data), size);
        }, py::arg("index"),
           "Read sample data by index\n\n"
           "Returns decompressed data if sample is compressed.\n"
           "Verifies checksum if verification is enabled.\n\n"
           "Args:\n"
           "    index (int): Sample index\n\n"
           "Returns:\n"
           "    bytes: Sample data (decompressed if needed)")
        .def("read_metadata", [](readers::TblReaderV2& self, size_t index) {
            auto [metadata, type] = self.read_metadata(index);
            return py::make_tuple(metadata, type);
        }, py::arg("index"),
           "Read metadata for a sample\n\n"
           "Args:\n"
           "    index (int): Sample index\n\n"
           "Returns:\n"
           "    tuple: (metadata_string, MetadataType)")
        .def("num_samples", &readers::TblReaderV2::num_samples,
             "Get number of samples in the file\n\n"
             "Returns:\n"
             "    int: Number of samples")
        .def("is_compressed", &readers::TblReaderV2::is_compressed,
             "Check if file uses compression\n\n"
             "Returns:\n"
             "    bool: True if compressed with LZ4")
        .def("has_metadata", &readers::TblReaderV2::has_metadata,
             "Check if file has metadata section\n\n"
             "Returns:\n"
             "    bool: True if file has metadata")
        .def("filter_by_dimensions", &readers::TblReaderV2::filter_by_dimensions,
             py::arg("min_width") = 0, py::arg("min_height") = 0,
             py::arg("max_width") = 0, py::arg("max_height") = 0,
             "Get indices of samples matching dimension filter\n\n"
             "Args:\n"
             "    min_width (int): Minimum width (0 = no filter)\n"
             "    min_height (int): Minimum height (0 = no filter)\n"
             "    max_width (int): Maximum width (0 = no filter)\n"
             "    max_height (int): Maximum height (0 = no filter)\n\n"
             "Returns:\n"
             "    list[int]: Matching sample indices")
        .def("filter_by_format", &readers::TblReaderV2::filter_by_format,
             py::arg("format"),
             "Get indices of samples matching format filter\n\n"
             "Args:\n"
             "    format (SampleFormat): Sample format to filter by\n\n"
             "Returns:\n"
             "    list[int]: Matching sample indices")
        .def("get_sample_info", [](readers::TblReaderV2& self, size_t index) {
            const auto& info = self.get_sample_info(index);
            py::dict result;
            result["offset"] = info.offset;
            result["size"] = info.size;
            result["uncompressed_size"] = info.uncompressed_size;
            result["width"] = info.width;
            result["height"] = info.height;
            result["format"] = info.format;
            result["is_compressed"] = info.is_compressed();
            result["has_metadata"] = info.has_metadata();
            return result;
        }, py::arg("index"),
           "Get sample information without reading data\n\n"
           "Args:\n"
           "    index (int): Sample index\n\n"
           "Returns:\n"
           "    dict: Sample info (offset, size, dimensions, format, flags)");

    // TblWriterV2 class
    py::class_<writers::TblWriterV2>(m, "TblWriterV2",
             "Streaming TBL v2 format writer with LZ4 compression (NEW in v1.5.0)\n\n"
             "Key improvements over v1:\n"
             "- Constant memory usage (streams samples directly to disk)\n"
             "- LZ4 compression support (40-60% additional space savings)\n"
             "- Metadata support (labels, dimensions, EXIF)\n"
             "- Data integrity checksums (CRC32/CRC16)\n"
             "- Dimension caching for fast filtered loading\n\n"
             "Example:\n"
             "    >>> writer = turboloader.TblWriterV2('output.tbl', enable_compression=True)\n"
             "    >>> with open('image.jpg', 'rb') as f:\n"
             "    >>>     data = f.read()\n"
             "    >>> idx = writer.add_sample(data, turboloader.SampleFormat.JPEG, width=256, height=256)\n"
             "    >>> writer.add_metadata(idx, '{\"label\": \"cat\"}', turboloader.MetadataType.JSON)\n"
             "    >>> writer.finalize()")
        .def(py::init<const std::string&, bool>(),
             py::arg("path"),
             py::arg("enable_compression") = true,
             "Create TBL v2 writer\n\n"
             "Args:\n"
             "    path (str): Output file path\n"
             "    enable_compression (bool): Enable LZ4 compression (default: True)")
        .def("add_sample", [](writers::TblWriterV2& self, py::bytes data,
                              formats::SampleFormat format, uint16_t width, uint16_t height) {
            // Convert Python bytes to C++ buffer
            char* buffer;
            Py_ssize_t length;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length)) {
                throw std::runtime_error("Failed to extract bytes data");
            }
            return self.add_sample(reinterpret_cast<const uint8_t*>(buffer),
                                  length, format, width, height);
        }, py::arg("data"), py::arg("format"),
           py::arg("width") = 0, py::arg("height") = 0,
           "Add a sample to the TBL file\n\n"
           "Args:\n"
           "    data (bytes): Sample data\n"
           "    format (SampleFormat): Sample format (JPEG, PNG, etc.)\n"
           "    width (int): Image width (0 if unknown/not image)\n"
           "    height (int): Image height (0 if unknown/not image)\n\n"
           "Returns:\n"
           "    int: Index of the added sample")
        .def("add_metadata", &writers::TblWriterV2::add_metadata,
             py::arg("sample_index"), py::arg("metadata"),
             py::arg("type") = formats::MetadataType::JSON,
             "Add metadata for a sample\n\n"
             "Args:\n"
             "    sample_index (int): Index of the sample\n"
             "    metadata (str): Metadata content\n"
             "    type (MetadataType): Metadata type (default: JSON)")
        .def("finalize", &writers::TblWriterV2::finalize,
             "Finalize the TBL file\n\n"
             "Writes the header, index, data, and metadata sections.\n"
             "Must be called before closing the file.")
        .def("num_samples", &writers::TblWriterV2::num_samples,
             "Get the number of samples written\n\n"
             "Returns:\n"
             "    int: Number of samples")
        .def("is_compression_enabled", &writers::TblWriterV2::is_compression_enabled,
             "Check if compression is enabled\n\n"
             "Returns:\n"
             "    bool: True if LZ4 compression is enabled")
        .def("__enter__", [](writers::TblWriterV2& self) -> writers::TblWriterV2& {
            return self;
        }, "Context manager entry")
        .def("__exit__", [](writers::TblWriterV2& self, py::object, py::object, py::object) {
            self.finalize();
        }, "Context manager exit (auto-finalizes)");

    // ========================================================================
    // SMART BATCHING BINDINGS (NEW in v1.7.0)
    // ========================================================================

    // SmartBatchConfig struct
    py::class_<SmartBatchConfig>(m, "SmartBatchConfig",
             "Configuration for Smart Batching (NEW in v1.7.0)\n\n"
             "Smart Batching groups samples by similar dimensions to minimize padding overhead,\n"
             "resulting in ~1.2x throughput improvement and 15-25% reduced memory usage.\n\n"
             "Performance benefits:\n"
             "- Reduces memory usage by 15-25% (less padding)\n"
             "- Improves throughput by ~1.2x (less wasted computation)\n"
             "- Better GPU utilization (more uniform batches)\n\n"
             "Example:\n"
             "    >>> config = turboloader.SmartBatchConfig()\n"
             "    >>> config.bucket_width_step = 64\n"
             "    >>> config.bucket_height_step = 64\n"
             "    >>> config.min_bucket_size = 32\n"
             "    >>> config.enable_dynamic_buckets = True")
        .def(py::init<>(),
             "Create SmartBatchConfig with default values\n\n"
             "Default configuration:\n"
             "  bucket_width_step = 32\n"
             "  bucket_height_step = 32\n"
             "  min_bucket_size = 16\n"
             "  max_bucket_size = 128\n"
             "  enable_dynamic_buckets = True\n"
             "  max_buckets = 100\n"
             "  strict_sizing = False")
        .def_readwrite("bucket_width_step", &SmartBatchConfig::bucket_width_step,
             "Width granularity for grouping samples (default: 32)\n\n"
             "Samples with width within ±bucket_width_step are grouped together.\n"
             "Smaller values = more precise grouping but more buckets.\n"
             "Larger values = fewer buckets but more padding.\n\n"
             "Recommended: 32 for natural images, 64 for larger images (>512px)")
        .def_readwrite("bucket_height_step", &SmartBatchConfig::bucket_height_step,
             "Height granularity for grouping samples (default: 32)\n\n"
             "Samples with height within ±bucket_height_step are grouped together.\n"
             "Smaller values = more precise grouping but more buckets.\n"
             "Larger values = fewer buckets but more padding.\n\n"
             "Recommended: 32 for natural images, 64 for larger images (>512px)")
        .def_readwrite("min_bucket_size", &SmartBatchConfig::min_bucket_size,
             "Minimum samples before creating batch (default: 16)\n\n"
             "Buckets with fewer than min_bucket_size samples will wait for more.\n"
             "Higher values = better batching efficiency but higher latency.\n"
             "Lower values = lower latency but potentially less efficient batches.\n\n"
             "Recommended: 8-32 depending on batch size and dataset size")
        .def_readwrite("max_bucket_size", &SmartBatchConfig::max_bucket_size,
             "Maximum samples per bucket (default: 128)\n\n"
             "Once a bucket reaches max_bucket_size, it will be flushed.\n"
             "Should be >= batch_size for best performance.\n\n"
             "Recommended: 2-4x your batch size")
        .def_readwrite("enable_dynamic_buckets", &SmartBatchConfig::enable_dynamic_buckets,
             "Create buckets on-demand (default: True)\n\n"
             "When True, buckets are created automatically as new size combinations appear.\n"
             "When False, only pre-defined buckets are used.\n\n"
             "Recommended: True for most use cases")
        .def_readwrite("max_buckets", &SmartBatchConfig::max_buckets,
             "Maximum number of buckets (default: 100)\n\n"
             "Limits memory usage by capping the number of active buckets.\n"
             "If max_buckets is reached, new samples may be dropped or use fallback batching.\n\n"
             "Recommended: 50-200 depending on dataset diversity")
        .def_readwrite("strict_sizing", &SmartBatchConfig::strict_sizing,
             "Only exact sizes in bucket (default: False)\n\n"
             "When True, only samples with exact matching dimensions go in same bucket.\n"
             "When False, samples within bucket_width_step/bucket_height_step are grouped.\n\n"
             "Recommended: False for most use cases");

    // ========================================================================
    // MODERN AUGMENTATIONS (NEW in v1.8.0)
    // ========================================================================

    // MixUp transform with Python wrapper for source image management
    py::class_<transforms::MixUpTransform, transforms::Transform>(m, "MixUp",
             "MixUp augmentation (NEW in v1.8.0)\n\n"
             "Blends two images using linear interpolation:\n"
             "  output = lambda * image1 + (1 - lambda) * image2\n\n"
             "Lambda is sampled from Beta(alpha, alpha) distribution.\n\n"
             "Reference: Zhang et al., 'mixup: Beyond Empirical Risk Minimization' (2017)\n\n"
             "Example:\n"
             "    >>> mixup = turboloader.MixUp(alpha=0.4)\n"
             "    >>> mixup.set_mix_image(other_image)\n"
             "    >>> blended = mixup.apply(image)\n"
             "    >>> lambda_val = mixup.get_lambda()")
        .def(py::init<float, unsigned>(),
             py::arg("alpha") = 0.4f,
             py::arg("seed") = std::random_device{}(),
             "Create MixUp transform\n\n"
             "Args:\n"
             "    alpha (float): Beta distribution parameter (default: 0.4)\n"
             "    seed (int): Random seed (default: random)")
        .def("set_mix_image", [](transforms::MixUpTransform& self, py::array_t<uint8_t> img) {
            auto buf = img.request();
            if (buf.ndim != 3) throw std::runtime_error("Image must be 3D (H, W, C)");
            int height = buf.shape[0];
            int width = buf.shape[1];
            int channels = buf.shape[2];
            // Store a copy of the image data (managed by the lambda capture)
            static thread_local std::vector<uint8_t> mix_data;
            static thread_local transforms::ImageData mix_image(nullptr, 0, 0, 0, 0, false);
            mix_data.assign(static_cast<uint8_t*>(buf.ptr),
                           static_cast<uint8_t*>(buf.ptr) + buf.size);
            mix_image = transforms::ImageData(mix_data.data(), width, height, channels, width * channels, false);
            self.set_mix_image(mix_image);
        }, py::arg("image"),
             "Set the second image for mixing\n\n"
             "Args:\n"
             "    image: NumPy array (H, W, C) uint8")
        .def("get_lambda", &transforms::MixUpTransform::get_lambda,
             "Get the lambda value used for the last mix\n\n"
             "Returns:\n"
             "    float: Lambda value in [0, 1]");

    // CutMix transform with Python wrapper for source image management
    py::class_<transforms::CutMixTransform, transforms::Transform>(m, "CutMix",
             "CutMix augmentation (NEW in v1.8.0)\n\n"
             "Cuts a rectangular patch from one image and pastes it onto another.\n"
             "Labels should be mixed proportionally to the area.\n\n"
             "Reference: Yun et al., 'CutMix: Regularization Strategy' (2019)\n\n"
             "Example:\n"
             "    >>> cutmix = turboloader.CutMix(alpha=1.0)\n"
             "    >>> cutmix.set_source_image(other_image)\n"
             "    >>> mixed = cutmix.apply(image)\n"
             "    >>> lambda_val = cutmix.get_lambda()")
        .def(py::init<float, unsigned>(),
             py::arg("alpha") = 1.0f,
             py::arg("seed") = std::random_device{}(),
             "Create CutMix transform\n\n"
             "Args:\n"
             "    alpha (float): Beta distribution parameter (default: 1.0)\n"
             "    seed (int): Random seed (default: random)")
        .def("set_source_image", [](transforms::CutMixTransform& self, py::array_t<uint8_t> img) {
            auto buf = img.request();
            if (buf.ndim != 3) throw std::runtime_error("Image must be 3D (H, W, C)");
            int height = buf.shape[0];
            int width = buf.shape[1];
            int channels = buf.shape[2];
            static thread_local std::vector<uint8_t> src_data;
            static thread_local transforms::ImageData src_image(nullptr, 0, 0, 0, 0, false);
            src_data.assign(static_cast<uint8_t*>(buf.ptr),
                           static_cast<uint8_t*>(buf.ptr) + buf.size);
            src_image = transforms::ImageData(src_data.data(), width, height, channels, width * channels, false);
            self.set_source_image(src_image);
        }, py::arg("image"),
             "Set the source image for the cut patch\n\n"
             "Args:\n"
             "    image: NumPy array (H, W, C) uint8")
        .def("get_lambda", &transforms::CutMixTransform::get_lambda,
             "Get the lambda value (ratio of mixed area)\n\n"
             "Returns:\n"
             "    float: Lambda value in [0, 1]")
        .def("get_bbox", [](const transforms::CutMixTransform& self) {
            int x1, y1, x2, y2;
            self.get_bbox(x1, y1, x2, y2);
            return py::make_tuple(x1, y1, x2, y2);
        }, "Get the bounding box of the cut region\n\n"
           "Returns:\n"
           "    tuple: (x1, y1, x2, y2) coordinates");

    // Mosaic transform with Python wrapper for multiple images
    py::class_<transforms::MosaicTransform, transforms::Transform>(m, "Mosaic",
             "Mosaic augmentation (NEW in v1.8.0)\n\n"
             "Creates a 2x2 grid from 4 images, commonly used in YOLO training.\n\n"
             "Reference: Bochkovskiy et al., 'YOLOv4' (2020)\n\n"
             "Example:\n"
             "    >>> mosaic = turboloader.Mosaic(output_size=640)\n"
             "    >>> mosaic.set_images([img1, img2, img3, img4])\n"
             "    >>> combined = mosaic.apply(img1)")
        .def(py::init<int, unsigned>(),
             py::arg("output_size") = 640,
             py::arg("seed") = std::random_device{}(),
             "Create Mosaic transform\n\n"
             "Args:\n"
             "    output_size (int): Size of output square image (default: 640)\n"
             "    seed (int): Random seed (default: random)")
        .def("set_images", [](transforms::MosaicTransform& self, py::list images) {
            if (py::len(images) != 4) throw std::runtime_error("Mosaic requires exactly 4 images");
            static thread_local std::vector<std::vector<uint8_t>> img_data(4);
            static thread_local std::vector<transforms::ImageData> img_objects;
            img_objects.clear();
            for (int i = 0; i < 4; ++i) {
                py::array_t<uint8_t> arr = images[i].cast<py::array_t<uint8_t>>();
                auto buf = arr.request();
                if (buf.ndim != 3) throw std::runtime_error("Each image must be 3D (H, W, C)");
                int height = buf.shape[0];
                int width = buf.shape[1];
                int channels = buf.shape[2];
                img_data[i].assign(static_cast<uint8_t*>(buf.ptr),
                                  static_cast<uint8_t*>(buf.ptr) + buf.size);
                img_objects.emplace_back(img_data[i].data(), width, height, channels, width * channels, false);
            }
            self.set_images(&img_objects[0], &img_objects[1], &img_objects[2], &img_objects[3]);
        }, py::arg("images"),
             "Set the 4 images for mosaic\n\n"
             "Args:\n"
             "    images: List of 4 NumPy arrays (H, W, C) uint8")
        .def("get_center", [](const transforms::MosaicTransform& self) {
            int cx, cy;
            self.get_center(cx, cy);
            return py::make_tuple(cx, cy);
        }, "Get the center point of the mosaic\n\n"
           "Returns:\n"
           "    tuple: (cx, cy) coordinates");

    // RandAugment transform
    py::class_<transforms::RandAugmentTransform, transforms::Transform>(m, "RandAugment",
             "RandAugment augmentation (NEW in v1.8.0)\n\n"
             "Applies N random augmentations with magnitude M.\n\n"
             "Reference: Cubuk et al., 'RandAugment: Practical automated data\n"
             "           augmentation with a reduced search space' (2020)\n\n"
             "Example:\n"
             "    >>> randaug = turboloader.RandAugment(num_ops=2, magnitude=9)\n"
             "    >>> augmented = randaug.apply(image)")
        .def(py::init<int, int, unsigned>(),
             py::arg("num_ops") = 2,
             py::arg("magnitude") = 9,
             py::arg("seed") = std::random_device{}(),
             "Create RandAugment transform\n\n"
             "Args:\n"
             "    num_ops (int): Number of operations to apply (default: 2)\n"
             "    magnitude (int): Magnitude of augmentations 0-30 (default: 9)\n"
             "    seed (int): Random seed (default: random)");

    // GridMask transform
    py::class_<transforms::GridMaskTransform, transforms::Transform>(m, "GridMask",
             "GridMask augmentation (NEW in v1.8.0)\n\n"
             "Applies a grid-based mask to the image for regularization.\n\n"
             "Reference: Chen et al., 'GridMask Data Augmentation' (2020)\n\n"
             "Example:\n"
             "    >>> gridmask = turboloader.GridMask(d=0.5, ratio=0.6, p=0.5)\n"
             "    >>> masked = gridmask.apply(image)")
        .def(py::init<float, float, float, unsigned>(),
             py::arg("d") = 0.5f,
             py::arg("ratio") = 0.6f,
             py::arg("p") = 0.5f,
             py::arg("seed") = std::random_device{}(),
             "Create GridMask transform\n\n"
             "Args:\n"
             "    d (float): Grid cell size ratio (default: 0.5)\n"
             "    ratio (float): Mask ratio within cell (default: 0.6)\n"
             "    p (float): Probability of applying (default: 0.5)\n"
             "    seed (int): Random seed (default: random)");

    // ========================================================================
    // LOGGING FRAMEWORK (NEW in v1.8.0)
    // ========================================================================

    // Log level enum
    py::enum_<pipeline::ErrorSeverity>(m, "LogLevel",
             "Log severity levels (NEW in v1.8.0)")
        .value("DEBUG", pipeline::ErrorSeverity::DEBUG)
        .value("INFO", pipeline::ErrorSeverity::INFO)
        .value("WARNING", pipeline::ErrorSeverity::WARNING)
        .value("ERROR", pipeline::ErrorSeverity::ERROR)
        .value("CRITICAL", pipeline::ErrorSeverity::CRITICAL);

    // Logger singleton access
    m.def("enable_logging", []() {
        pipeline::Logger::instance().enable();
    }, "Enable TurboLoader logging\n\n"
       "When enabled, TurboLoader will log debug info, warnings, and errors.\n\n"
       "Example:\n"
       "    >>> turboloader.enable_logging()\n"
       "    >>> loader = turboloader.DataLoader('data.tar', batch_size=32)");

    m.def("disable_logging", []() {
        pipeline::Logger::instance().disable();
    }, "Disable TurboLoader logging");

    m.def("set_log_level", [](pipeline::ErrorSeverity level) {
        pipeline::Logger::instance().set_level(level);
    }, py::arg("level"),
       "Set minimum log level\n\n"
       "Args:\n"
       "    level: LogLevel.DEBUG, INFO, WARNING, ERROR, or CRITICAL\n\n"
       "Example:\n"
       "    >>> turboloader.set_log_level(turboloader.LogLevel.DEBUG)");

    m.def("set_log_output", [](const std::string& path) {
        pipeline::Logger::instance().set_output(path);
    }, py::arg("path"),
       "Set log output file\n\n"
       "Args:\n"
       "    path (str): Path to log file (empty string = stderr)\n\n"
       "Example:\n"
       "    >>> turboloader.set_log_output('/var/log/turboloader.log')");
}
