#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "turboloader/pipeline/pipeline.hpp"
#include "turboloader/transforms/simd_transforms.hpp"
#include "turboloader/transforms/augmentation_transforms.hpp"
#include <thread>
#include <chrono>

namespace py = pybind11;
using namespace turboloader;
using namespace turboloader::transforms;

// Wrapper for Sample that returns NumPy arrays
class PySample {
public:
    std::unordered_map<std::string, py::array_t<uint8_t>> data;
    size_t index;
    int width;
    int height;
    int channels;

    PySample(const Sample& sample)
        : index(sample.index)
        , width(sample.width)
        , height(sample.height)
        , channels(sample.channels) {

        // Convert std::vector<uint8_t> to NumPy arrays
        for (const auto& [ext, vec] : sample.data) {
            // Create NumPy array (copy data)
            py::array_t<uint8_t> arr(vec.size());
            auto buf = arr.request();
            uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
            std::memcpy(ptr, vec.data(), vec.size());

            data[ext] = arr;
        }
    }

    // Get decoded image as NumPy array (H, W, C)
    py::array_t<uint8_t> get_image() const {
        if (width == 0 || height == 0) {
            return py::array_t<uint8_t>();
        }

        // Find the jpg data (which is now decoded RGB)
        auto it = data.find("jpg");
        if (it == data.end()) {
            return py::array_t<uint8_t>();
        }

        // Reshape to (H, W, C)
        auto arr = it->second;
        arr.resize({height, width, channels});
        return arr;
    }
};

// Wrapper for Pipeline
class PyPipeline {
public:
    PyPipeline(const std::vector<std::string>& tar_paths,
               size_t num_workers = 4,
               size_t queue_size = 256,
               bool shuffle = false,
               bool decode_jpeg = false,
               bool enable_simd_transforms = false,
               const TransformConfig* transform_config_ptr = nullptr,
               // Legacy parameters (deprecated)
               bool enable_resize = false,
               int resize_width = 224,
               int resize_height = 224) {

        Pipeline::Config config;
        config.num_workers = num_workers;
        config.queue_size = queue_size;
        config.shuffle = shuffle;
        config.decode_jpeg = decode_jpeg;
        config.enable_simd_transforms = enable_simd_transforms;

        // Use provided transform config or defaults
        if (transform_config_ptr) {
            config.transform_config = *transform_config_ptr;
        }

        // Legacy support
        config.enable_resize = enable_resize;
        config.resize_width = resize_width;
        config.resize_height = resize_height;

        pipeline_ = std::make_unique<Pipeline>(tar_paths, config);
    }

    void start() {
        pipeline_->start();
    }

    void stop() {
        pipeline_->stop();
    }

    void reset() {
        pipeline_->reset();
    }

    size_t total_samples() const {
        return pipeline_->total_samples();
    }

    // Get next batch as list of PySamples
    std::vector<PySample> next_batch(size_t batch_size) {
        // Add some wait time for initial batches (same issue as C++ benchmark)
        auto batch = pipeline_->next_batch(batch_size);

        // If empty, wait a bit and try again (handles initial pipeline startup)
        int retries = 0;
        while (batch.empty() && retries < 10) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            batch = pipeline_->next_batch(batch_size);
            retries++;
        }

        std::vector<PySample> py_batch;
        py_batch.reserve(batch.size());

        for (const auto& sample : batch) {
            py_batch.emplace_back(sample);
        }

        return py_batch;
    }

    // Python iterator interface
    class Iterator {
    public:
        Iterator(PyPipeline* pipeline, size_t batch_size)
            : pipeline_(pipeline), batch_size_(batch_size) {}

        std::vector<PySample> next() {
            auto batch = pipeline_->next_batch(batch_size_);
            if (batch.empty()) {
                throw py::stop_iteration();
            }
            return batch;
        }

    private:
        PyPipeline* pipeline_;
        size_t batch_size_;
    };

    Iterator iter(size_t batch_size = 32) {
        return Iterator(this, batch_size);
    }

private:
    std::unique_ptr<Pipeline> pipeline_;
};

PYBIND11_MODULE(turboloader, m) {
    m.doc() = "TurboLoader - High-performance ML data loading library";

    // Enums
    py::enum_<ResizeMethod>(m, "ResizeMethod")
        .value("NEAREST", ResizeMethod::NEAREST)
        .value("BILINEAR", ResizeMethod::BILINEAR)
        .value("BICUBIC", ResizeMethod::BICUBIC)
        .export_values();

    py::enum_<ColorSpace>(m, "ColorSpace")
        .value("RGB", ColorSpace::RGB)
        .value("BGR", ColorSpace::BGR)
        .value("YUV", ColorSpace::YUV)
        .value("GRAY", ColorSpace::GRAY)
        .export_values();

    // TransformConfig class
    py::class_<TransformConfig>(m, "TransformConfig")
        .def(py::init<>())
        .def_readwrite("enable_resize", &TransformConfig::enable_resize)
        .def_readwrite("resize_width", &TransformConfig::resize_width)
        .def_readwrite("resize_height", &TransformConfig::resize_height)
        .def_readwrite("resize_method", &TransformConfig::resize_method)
        .def_readwrite("enable_normalize", &TransformConfig::enable_normalize)
        .def_property("mean",
            [](const TransformConfig& config) {
                return std::vector<float>(config.mean, config.mean + 3);
            },
            [](TransformConfig& config, const std::vector<float>& values) {
                if (values.size() != 3) throw std::runtime_error("mean must have 3 values");
                std::copy(values.begin(), values.end(), config.mean);
            })
        .def_property("std",
            [](const TransformConfig& config) {
                return std::vector<float>(config.std, config.std + 3);
            },
            [](TransformConfig& config, const std::vector<float>& values) {
                if (values.size() != 3) throw std::runtime_error("std must have 3 values");
                std::copy(values.begin(), values.end(), config.std);
            })
        .def_readwrite("enable_color_convert", &TransformConfig::enable_color_convert)
        .def_readwrite("src_color", &TransformConfig::src_color)
        .def_readwrite("dst_color", &TransformConfig::dst_color)
        .def_readwrite("output_float", &TransformConfig::output_float)
        .def("__repr__", [](const TransformConfig& config) {
            return "<TransformConfig resize=" + std::string(config.enable_resize ? "on" : "off") +
                   " normalize=" + std::string(config.enable_normalize ? "on" : "off") + ">";
        });

    // PySample class
    py::class_<PySample>(m, "Sample")
        .def_readonly("index", &PySample::index)
        .def_readonly("width", &PySample::width)
        .def_readonly("height", &PySample::height)
        .def_readonly("channels", &PySample::channels)
        .def_readonly("data", &PySample::data)
        .def("get_image", &PySample::get_image,
             "Get decoded image as NumPy array (H, W, C)")
        .def("__repr__", [](const PySample& s) {
            return "<Sample index=" + std::to_string(s.index) +
                   " shape=(" + std::to_string(s.height) + ", " +
                   std::to_string(s.width) + ", " +
                   std::to_string(s.channels) + ")>";
        });

    // PyPipeline iterator
    py::class_<PyPipeline::Iterator>(m, "Iterator")
        .def("__iter__", [](PyPipeline::Iterator& it) -> PyPipeline::Iterator& { return it; })
        .def("__next__", &PyPipeline::Iterator::next);

    // PyPipeline class
    py::class_<PyPipeline>(m, "Pipeline")
        .def(py::init<const std::vector<std::string>&, size_t, size_t, bool, bool, bool, const TransformConfig*, bool, int, int>(),
             py::arg("tar_paths"),
             py::arg("num_workers") = 4,
             py::arg("queue_size") = 256,
             py::arg("shuffle") = false,
             py::arg("decode_jpeg") = false,
             py::arg("enable_simd_transforms") = false,
             py::arg("transform_config") = nullptr,
             py::arg("enable_resize") = false,
             py::arg("resize_width") = 224,
             py::arg("resize_height") = 224,
             R"doc(
                Create a data loading pipeline.

                Args:
                    tar_paths: List of TAR file paths
                    num_workers: Number of worker threads (default: 4)
                    queue_size: Size of output queue (default: 256)
                    shuffle: Shuffle samples (default: False)
                    decode_jpeg: Decode JPEG images to RGB (default: False)
                    enable_simd_transforms: Enable SIMD-accelerated transforms (default: False)
                    transform_config: TransformConfig object for SIMD transforms (default: None)
                    enable_resize: [DEPRECATED] Apply resize transform (default: False)
                    resize_width: [DEPRECATED] Target width for resize (default: 224)
                    resize_height: [DEPRECATED] Target height for resize (default: 224)
             )doc")
        .def("start", &PyPipeline::start, "Start the pipeline")
        .def("stop", &PyPipeline::stop, "Stop the pipeline")
        .def("reset", &PyPipeline::reset, "Reset for next epoch")
        .def("total_samples", &PyPipeline::total_samples, "Get total number of samples")
        .def("next_batch", &PyPipeline::next_batch,
             py::arg("batch_size"),
             "Get next batch of samples")
        .def("iter", &PyPipeline::iter,
             py::arg("batch_size") = 32,
             "Iterate over batches")
        .def("__iter__", [](PyPipeline& p) { return p.iter(32); },
             "Iterate over batches with default batch size")
        .def("__len__", &PyPipeline::total_samples)
        .def("__repr__", [](const PyPipeline& p) {
            return "<Pipeline samples=" + std::to_string(p.total_samples()) + ">";
        });

    // Augmentation Transforms
    py::class_<AugmentationTransform, std::shared_ptr<AugmentationTransform>>(m, "AugmentationTransform")
        .def("set_probability", &AugmentationTransform::set_probability)
        .def("get_probability", &AugmentationTransform::get_probability);

    py::class_<RandomHorizontalFlip, AugmentationTransform, std::shared_ptr<RandomHorizontalFlip>>(m, "RandomHorizontalFlip")
        .def(py::init<float>(), py::arg("probability") = 0.5f,
             "Random horizontal flip augmentation\n\n"
             "Args:\n"
             "    probability: Probability of applying the flip (default: 0.5)");

    py::class_<RandomVerticalFlip, AugmentationTransform, std::shared_ptr<RandomVerticalFlip>>(m, "RandomVerticalFlip")
        .def(py::init<float>(), py::arg("probability") = 0.5f,
             "Random vertical flip augmentation\n\n"
             "Args:\n"
             "    probability: Probability of applying the flip (default: 0.5)");

    py::class_<ColorJitter, AugmentationTransform, std::shared_ptr<ColorJitter>>(m, "ColorJitter")
        .def(py::init<float, float, float, float>(),
             py::arg("brightness") = 0.0f,
             py::arg("contrast") = 0.0f,
             py::arg("saturation") = 0.0f,
             py::arg("hue") = 0.0f,
             "Color jitter augmentation (SIMD optimized)\n\n"
             "Args:\n"
             "    brightness: Brightness adjustment range (default: 0.0)\n"
             "    contrast: Contrast adjustment range (default: 0.0)\n"
             "    saturation: Saturation adjustment range (default: 0.0)\n"
             "    hue: Hue adjustment range (default: 0.0)")
        .def("set_brightness_range", &ColorJitter::set_brightness_range)
        .def("set_contrast_range", &ColorJitter::set_contrast_range)
        .def("set_saturation_range", &ColorJitter::set_saturation_range)
        .def("set_hue_range", &ColorJitter::set_hue_range);

    py::class_<RandomRotation, AugmentationTransform, std::shared_ptr<RandomRotation>>(m, "RandomRotation")
        .def(py::init<float>(), py::arg("degrees") = 0.0f,
             "Random rotation augmentation with bilinear interpolation\n\n"
             "Args:\n"
             "    degrees: Maximum rotation angle in degrees (default: 0.0)")
        .def("set_degrees", &RandomRotation::set_degrees)
        .def("set_fill_color", &RandomRotation::set_fill_color,
             py::arg("r"), py::arg("g"), py::arg("b"));

    py::class_<RandomCrop, AugmentationTransform, std::shared_ptr<RandomCrop>>(m, "RandomCrop")
        .def(py::init<int, int>(),
             py::arg("crop_width"),
             py::arg("crop_height"),
             "Random crop augmentation\n\n"
             "Args:\n"
             "    crop_width: Width of the crop\n"
             "    crop_height: Height of the crop");

    py::class_<RandomErasing, AugmentationTransform, std::shared_ptr<RandomErasing>>(m, "RandomErasing")
        .def(py::init<float, float, float, float, float>(),
             py::arg("probability") = 0.5f,
             py::arg("scale_min") = 0.02f,
             py::arg("scale_max") = 0.33f,
             py::arg("ratio_min") = 0.3f,
             py::arg("ratio_max") = 3.3f,
             "Random erasing augmentation (Cutout)\n\n"
             "Args:\n"
             "    probability: Probability of applying erasing (default: 0.5)\n"
             "    scale_min: Minimum area scale (default: 0.02)\n"
             "    scale_max: Maximum area scale (default: 0.33)\n"
             "    ratio_min: Minimum aspect ratio (default: 0.3)\n"
             "    ratio_max: Maximum aspect ratio (default: 3.3)");

    py::class_<GaussianBlur, AugmentationTransform, std::shared_ptr<GaussianBlur>>(m, "GaussianBlur")
        .def(py::init<float, int>(),
             py::arg("sigma") = 1.0f,
             py::arg("kernel_size") = 5,
             "Gaussian blur augmentation (SIMD optimized)\n\n"
             "Args:\n"
             "    sigma: Gaussian kernel sigma (default: 1.0)\n"
             "    kernel_size: Kernel size (default: 5)")
        .def("set_sigma_range", &GaussianBlur::set_sigma_range,
             py::arg("min_sigma"), py::arg("max_sigma"));

    py::class_<AugmentationPipeline>(m, "AugmentationPipeline")
        .def(py::init<>(), "Create an augmentation pipeline")
        .def(py::init<uint64_t>(), py::arg("seed"), "Create pipeline with random seed")
        .def("add_transform", &AugmentationPipeline::add_transform,
             "Add a transform to the pipeline")
        .def("set_seed", &AugmentationPipeline::set_seed, "Set random seed")
        .def("num_transforms", &AugmentationPipeline::num_transforms,
             "Get number of transforms in pipeline")
        .def("clear", &AugmentationPipeline::clear,
             "Clear all transforms from pipeline");

    // Config class - add augmentation pipeline support
    py::class_<Pipeline::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("num_workers", &Pipeline::Config::num_workers)
        .def_readwrite("queue_size", &Pipeline::Config::queue_size)
        .def_readwrite("prefetch_factor", &Pipeline::Config::prefetch_factor)
        .def_readwrite("shuffle", &Pipeline::Config::shuffle)
        .def_readwrite("shuffle_buffer_size", &Pipeline::Config::shuffle_buffer_size)
        .def_readwrite("decode_jpeg", &Pipeline::Config::decode_jpeg)
        .def_readwrite("enable_simd_transforms", &Pipeline::Config::enable_simd_transforms)
        .def_readwrite("transform_config", &Pipeline::Config::transform_config)
        .def_readwrite("enable_resize", &Pipeline::Config::enable_resize)
        .def_readwrite("resize_width", &Pipeline::Config::resize_width)
        .def_readwrite("resize_height", &Pipeline::Config::resize_height)
        .def_readwrite("enable_normalize", &Pipeline::Config::enable_normalize);

    // Version info
    m.attr("__version__") = "0.3.5";
}
