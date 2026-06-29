// Pure-C++ interface to the Metal GPU transform path (implemented in metal_transforms.mm).
// Deliberately contains NO Objective-C / Metal headers so the C++ pybind translation unit
// (turboloader_bindings.cpp) can include it directly. Only compiled/linked on macOS arm64
// when the build defines TURBOLOADER_METAL (see setup.py); everything degrades to an
// honest "not available" elsewhere.
#pragma once

#include <cstdint>
#include <vector>

namespace turboloader {
namespace metal {

// True if a usable Metal device is present and the compute pipeline built.
bool available();

// Short human-readable device name (e.g. "Apple M4 Max"), or "" if unavailable.
const char* device_name();

// One source image: tightly-packed HWC uint8 RGB (3 channels), `w`*`h` pixels.
struct ImageRef {
    const uint8_t* data;
    int w;
    int h;
};

// GPU bilinear-resize + per-channel normalize a batch of (variable-size) RGB images to a
// common (dst_h, dst_w), writing CHW float32 into `out` (must hold N*3*dst_h*dst_w floats,
// laid out [n][c][y][x]). Uses half-pixel sampling identical to the CPU path. mean/std are
// length-3 (per RGB channel). Returns false on any Metal error. Thread-safe.
bool resize_normalize_batch(const std::vector<ImageRef>& imgs,
                            int dst_h,
                            int dst_w,
                            const float mean[3],
                            const float std_[3],
                            float* out);

}  // namespace metal
}  // namespace turboloader
