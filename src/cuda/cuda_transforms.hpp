// Pure-C++ interface to the CUDA GPU transform path (implemented in cuda_transforms.cu,
// compiled with nvcc). Mirrors src/metal/metal_transforms.hpp so the two backends are
// drop-in interchangeable behind the loader.
//
// !!! UNVALIDATED !!!  These kernels are a faithful port of the bit-exact, validated Metal
// kernels (same half-pixel bilinear + normalize math), but there is no NVIDIA GPU on the
// dev/CI machines, so this code has NOT been compiled or run. Build only with
// TURBOLOADER_ENABLE_CUDA=1 + nvcc on a real CUDA box; expect to debug. See
// docs/GPU_ACCELERATION.md.
#pragma once

#include <cstdint>
#include <vector>

namespace turboloader {
namespace cuda {

// True if CUDA support was compiled in AND a CUDA device is present at runtime.
bool available();

// Short device name (e.g. "NVIDIA GeForce RTX 3090"), or "" if unavailable.
const char* device_name();

// Same layout as the Metal path: tightly-packed HWC uint8 RGB.
struct ImageRef {
    const uint8_t* data;
    int w;
    int h;
};

// Same per-image crop window as the Metal path.
struct CropParams {
    float x;
    float y;
    float w;
    float h;
    int flip;
};

// GPU bilinear-resize + per-channel normalize -> CHW float32 (N*3*dst_h*dst_w). Mirrors
// turboloader::metal::resize_normalize_batch exactly.
bool resize_normalize_batch(const std::vector<ImageRef>& imgs, int dst_h, int dst_w,
                            const float mean[3], const float std_[3], float* out);

// Fused crop + resize + (optional) hflip + normalize. Mirrors
// turboloader::metal::crop_resize_normalize_batch exactly.
bool crop_resize_normalize_batch(const std::vector<ImageRef>& imgs,
                                 const std::vector<CropParams>& crops, int dst_h, int dst_w,
                                 const float mean[3], const float std_[3], float* out);

// GPU-RESIDENT fused pipeline (needs HAVE_NVJPEG): nvJPEG decodes each JPEG straight into a
// device buffer, the resize+normalize kernel reads it in place, and a SINGLE D2H copies the
// (N,3,dst_h,dst_w) CHW float32 result to `out`. No per-image host round-trips. Returns
// false if not compiled with nvJPEG or on error.
bool decode_resize_normalize_batch(const std::vector<const uint8_t*>& jpegs,
                                   const std::vector<size_t>& sizes, int dst_h, int dst_w,
                                   const float mean[3], const float std_[3], float* out);

// Same fused pipeline but GPU-RESIDENT output: returns the device pointer of the result
// (N*3*dst_h*dst_w float32), valid until the next fused call (consume before the next
// batch, like DALI). Returns 0 on error / if not built with nvJPEG. Lets the consumer keep
// the batch on the GPU (zero-copy into a torch CUDA tensor) — no final D2H.
uintptr_t decode_resize_normalize_batch_gpu(const std::vector<const uint8_t*>& jpegs,
                                            const std::vector<size_t>& sizes, int dst_h, int dst_w,
                                            const float mean[3], const float std_[3]);

}  // namespace cuda
}  // namespace turboloader
