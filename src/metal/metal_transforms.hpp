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

// Per-image augmentation: a crop window in source pixels (floats, so the resize is exact)
// plus an optional horizontal flip. The randomness (RandomResizedCrop scale/ratio) is
// chosen by the caller and passed in here, so it stays seedable/reproducible.
struct CropParams {
    float x;  // crop top-left x in source pixels
    float y;  // crop top-left y in source pixels
    float w;  // crop width in source pixels
    float h;  // crop height in source pixels
    int flip;  // 1 = horizontal flip, 0 = none
};

// Fused crop + bilinear-resize + (optional) horizontal-flip + normalize, one GPU pass per
// image. crops[i] applies to imgs[i]. Output is CHW float32 (N*3*dst_h*dst_w), [n][c][y][x].
// This is the canonical train-time pipeline (RandomResizedCrop + RandomHorizontalFlip +
// Normalize) done entirely on the GPU. Returns false on any error.
bool crop_resize_normalize_batch(const std::vector<ImageRef>& imgs,
                                 const std::vector<CropParams>& crops,
                                 int dst_h,
                                 int dst_w,
                                 const float mean[3],
                                 const float std_[3],
                                 float* out);

// Per-image color-jitter factors (multiplicative; 1.0 = no change). Contrast is applied
// around mid-gray (0.5), saturation blends toward luminance.
struct JitterParams {
    float brightness;
    float contrast;
    float saturation;
};

// Fused crop + resize + (optional) hflip + color-jitter + normalize: the full ImageNet
// train-time pipeline in ONE GPU pass. crops[i]/jitter[i] apply to imgs[i]. Output CHW
// float32 (N*3*dst_h*dst_w). Returns false on error.
bool train_transform_batch(const std::vector<ImageRef>& imgs,
                           const std::vector<CropParams>& crops,
                           const std::vector<JitterParams>& jitter,
                           int dst_h,
                           int dst_w,
                           const float mean[3],
                           const float std_[3],
                           float* out);

// Hybrid GPU JPEG decode (implemented in metal_decode.mm): CPU (libjpeg) parse + Huffman
// entropy decode -> GPU dequant + 8x8 IDCT -> CPU chroma upsample + YCbCr->RGB. Writes
// HWC uint8 RGB (width*height*3) into `out`. Returns false on error / unsupported JPEG
// (caller should fall back to the CPU decode_jpeg). The GPU IDCT is bit-exact vs libjpeg.
bool decode_jpeg(const uint8_t* data, size_t size, std::vector<uint8_t>& out, int& width,
                 int& height);

}  // namespace metal
}  // namespace turboloader
