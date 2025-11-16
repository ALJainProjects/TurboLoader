/**
 * @file turboloader_bindings.cpp
 * @brief Minimal Python bindings stub for TurboLoader v0.4.0
 *
 * Note: Full Python API bindings will be available in v0.5.0
 * This minimal version exports version information only.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief pybind11 module definition
 */
PYBIND11_MODULE(turboloader, m) {
    m.doc() = "TurboLoader v0.4.0 - High-performance data loading (C++ library)\n\n"
              "This package provides the compiled C++ library for TurboLoader.\n"
              "Full Python bindings will be available in v0.5.0.\n\n"
              "Features in v0.4.0:\n"
              "- Remote TAR support (HTTP, S3, GCS)\n"
              "- GPU-accelerated JPEG decoding (nvJPEG)\n"
              "- Lock-free SPSC queues\n"
              "- 52+ Gbps local file throughput\n"
              "- Multi-format pipeline (images, video, tabular data)";

    m.def("version", []() { return "0.4.0"; },
          "Get TurboLoader version");

    m.def("features", []() {
        py::dict features;
        features["version"] = "0.4.0";
        features["remote_tar"] = true;
        features["local_tar"] = true;
        features["jpeg_decode"] = true;
        features["png_decode"] = true;
        features["webp_decode"] = true;
        features["http_support"] = true;
        features["s3_support"] = true;
        features["gcs_support"] = true;
        return features;
    }, "Get TurboLoader feature support");
}
