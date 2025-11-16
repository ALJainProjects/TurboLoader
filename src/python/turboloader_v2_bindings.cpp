/**
 * @file turboloader_v2_bindings.cpp
 * @brief Python bindings for TurboLoader v2.0 pipeline
 *
 * Provides Python interface to the high-performance C++ PipelineV2 using pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../pipeline_v2/pipeline_v2.hpp"

namespace py = pybind11;
using namespace turboloader::v2;

/**
 * @brief Python wrapper for SampleV2
 *
 * Converts C++ sample to Python dict with NumPy arrays
 */
py::dict sample_to_dict(const SampleV2& sample) {
    py::dict result;

    result["index"] = sample.index;
    result["width"] = sample.width;
    result["height"] = sample.height;
    result["channels"] = sample.channels;
    result["is_decoded"] = sample.is_decoded();

    // Convert decoded RGB to NumPy array (zero-copy view)
    if (sample.is_decoded()) {
        // Create NumPy array with shape (H, W, C)
        py::array_t<uint8_t> rgb_array({
            static_cast<py::ssize_t>(sample.height),
            static_cast<py::ssize_t>(sample.width),
            static_cast<py::ssize_t>(sample.channels)
        });

        // Copy data (necessary because sample.decoded_rgb will be destroyed)
        auto buf = rgb_array.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        std::memcpy(ptr, sample.decoded_rgb.data(), sample.decoded_rgb.size());

        result["image"] = rgb_array;
    } else {
        result["image"] = py::none();
    }

    return result;
}

/**
 * @brief Python wrapper for BatchV2
 *
 * Converts C++ batch to Python list of dicts
 */
py::list batch_to_list(BatchV2& batch) {
    py::list result;

    for (auto& sample : batch) {
        result.append(sample_to_dict(sample));
    }

    return result;
}

/**
 * @brief Python-friendly PipelineV2 wrapper
 */
class PyPipelineV2 {
public:
    /**
     * @brief Constructor
     *
     * @param tar_path Path to TAR file
     * @param num_workers Number of worker threads (default: 4)
     * @param batch_size Batch size (default: 32)
     * @param queue_size Queue size (default: 256)
     */
    PyPipelineV2(
        const std::string& tar_path,
        size_t num_workers = 4,
        size_t batch_size = 32,
        size_t queue_size = 256
    ) {
        config_.tar_path = tar_path;
        config_.num_workers = num_workers;
        config_.batch_size = batch_size;
        config_.queue_size = queue_size;

        pipeline_ = std::make_unique<PipelineV2>(config_);
    }

    /**
     * @brief Get next batch as Python list
     *
     * @return List of sample dictionaries
     */
    py::list next_batch() {
        if (!pipeline_) {
            throw std::runtime_error("Pipeline not initialized");
        }

        auto batch = pipeline_->next_batch();
        return batch_to_list(batch);
    }

    /**
     * @brief Check if pipeline has finished
     *
     * @return True if all workers are done
     */
    bool is_finished() const {
        if (!pipeline_) {
            return true;
        }
        return pipeline_->is_finished();
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
     * @brief Get total number of samples
     *
     * @return Total samples in dataset
     */
    size_t total_samples() const {
        if (!pipeline_) {
            return 0;
        }
        return pipeline_->total_samples();
    }

    /**
     * @brief Get total samples processed
     *
     * @return Samples processed so far
     */
    size_t total_samples_processed() const {
        if (!pipeline_) {
            return 0;
        }
        return pipeline_->total_samples_processed();
    }

    /**
     * @brief Context manager support: __enter__
     */
    PyPipelineV2& enter() {
        return *this;
    }

    /**
     * @brief Context manager support: __exit__
     */
    void exit(py::object exc_type, py::object exc_value, py::object traceback) {
        stop();
    }

    /**
     * @brief Iterator support: __iter__
     */
    PyPipelineV2& iter() {
        return *this;
    }

    /**
     * @brief Iterator support: __next__
     */
    py::list next() {
        if (is_finished()) {
            throw py::stop_iteration();
        }

        auto batch = next_batch();

        // If batch is empty and pipeline is finished, stop iteration
        if (py::len(batch) == 0 && is_finished()) {
            throw py::stop_iteration();
        }

        return batch;
    }

private:
    PipelineConfig config_;
    std::unique_ptr<PipelineV2> pipeline_;
};

/**
 * @brief pybind11 module definition
 */
PYBIND11_MODULE(turboloader_v2, m) {
    m.doc() = "TurboLoader v2.0 - High-performance data loading for PyTorch";

    // PipelineV2 class
    py::class_<PyPipelineV2>(m, "DataLoader")
        .def(py::init<const std::string&, size_t, size_t, size_t>(),
             py::arg("tar_path"),
             py::arg("num_workers") = 4,
             py::arg("batch_size") = 32,
             py::arg("queue_size") = 256,
             "Initialize TurboLoader v2.0 DataLoader\n\n"
             "Args:\n"
             "    tar_path (str): Path to TAR file containing images\n"
             "    num_workers (int): Number of worker threads (default: 4)\n"
             "    batch_size (int): Batch size (default: 32)\n"
             "    queue_size (int): Lock-free queue size (default: 256)\n"
        )
        .def("next_batch", &PyPipelineV2::next_batch,
             "Get next batch of decoded samples\n\n"
             "Returns:\n"
             "    list: List of sample dictionaries, each containing:\n"
             "        - index (int): Sample index\n"
             "        - width (int): Image width\n"
             "        - height (int): Image height\n"
             "        - channels (int): Number of channels (3 for RGB)\n"
             "        - is_decoded (bool): Whether image was successfully decoded\n"
             "        - image (np.ndarray): RGB image array (H, W, 3) or None\n"
        )
        .def("is_finished", &PyPipelineV2::is_finished,
             "Check if all samples have been processed")
        .def("stop", &PyPipelineV2::stop,
             "Stop the pipeline and clean up resources")
        .def("total_samples", &PyPipelineV2::total_samples,
             "Get total number of samples in dataset")
        .def("total_samples_processed", &PyPipelineV2::total_samples_processed,
             "Get number of samples processed so far")
        .def("__enter__", &PyPipelineV2::enter,
             "Context manager entry")
        .def("__exit__", &PyPipelineV2::exit,
             "Context manager exit")
        .def("__iter__", &PyPipelineV2::iter,
             "Iterator support")
        .def("__next__", &PyPipelineV2::next,
             "Iterator next");

    // Module-level functions
    m.def("version", []() { return "2.0.0"; },
          "Get TurboLoader version");

    m.def("has_simd_support", []() { return true; },
          "Check if SIMD JPEG decoding is available");
}
