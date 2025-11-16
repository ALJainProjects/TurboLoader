#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "turboloader/pipeline/pipeline.hpp"
#include "turboloader/transforms/simd_transforms.hpp"
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

    // Version info
    m.attr("__version__") = "0.2.0";
}
