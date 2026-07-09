// Pure-C++ interface to the CUDA GPU transform path (implemented in cuda_transforms.cu,
// compiled with nvcc). Mirrors src/metal/metal_transforms.hpp so the two backends are
// drop-in interchangeable behind the loader.
//
// VALIDATED on real hardware: compiled and run on two Jetson AGX Orins (CUDA 11.4,
// aarch64) and an RTX 3090 (CUDA 13.3, x86_64). cuda_resize_normalize matches the numpy
// bilinear reference to 3.2e-05 (same half-pixel bilinear + normalize math as the
// bit-exact Metal kernels it ports). Build with TURBOLOADER_ENABLE_CUDA=1 + nvcc; see
// docs/GPU_ACCELERATION.md and experiments/cuda/RESULTS.md.
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

// Batched normalize of an ALREADY-DEVICE-RESIDENT, pre-resized NHWC uint8 batch (all N images
// H*W, contiguous at `src_dev`) -> NCHW float32, in ONE kernel launch. No resize, no H2D. For a
// pre-processed dataset uploaded to the GPU once (uint8) and normalized per epoch on the GPU.
// Returns the device pointer of the (N,3,H,W) float32 result (valid until the 4th-next call), or
// 0 on error / if CUDA is unavailable.
uintptr_t normalize_resident_batch(uintptr_t src_dev, int N, int H, int W, const float mean[3],
                                   const float std_[3]);

// Same, but output image n gathers from source image sel_dev[n] (device int64 index array) —
// a fused shuffle+normalize so a GPU-resident loader shuffles without a separate gather copy.
uintptr_t normalize_resident_gather_batch(uintptr_t src_dev, uintptr_t sel_dev, int N, int H, int W,
                                          const float mean[3], const float std_[3]);

// Streaming (dataset in host RAM, larger than VRAM): create `num_slots` async-H2D slots (each own
// stream + device scratch + output ring). Returns the number created (0 if CUDA unavailable).
int stream_normalize_init(int num_slots);

// Streaming per-batch on `slot`: async H2D of the pre-resized uint8 batch at host pointer
// `host_src` (best pinned), then normalize -> device pointer of the (N,3,H,W) float32 result.
// K slots on K threads overlap one batch's H2D with another's kernel. Returns 0 on error.
uintptr_t stream_normalize_batch(uintptr_t host_src, int N, int H, int W, const float mean[3],
                                 const float std_[3], int slot);

// Fully-in-C++ streaming loader for a pre-resized uint8 dataset in PINNED host RAM (larger than
// VRAM). A persistent pool of K worker threads runs the WHOLE iteration GIL-free: each pulls a
// batch, async-H2Ds it on its own stream, normalizes into a double-buffered output pool, and
// hands the ready device pointer to next_batch(). Python calls next_batch() once per batch (the
// only per-batch Python), so the GIL is not the bottleneck the way K Python worker threads are.
// PIMPL keeps CUDA/threading types out of this header (included by the non-CUDA bindings build).
class CudaStreamCore {
 public:
    // host_ptr: pinned host uint8 base of the (n, h, w, 3) NHWC dataset. num_slots = worker threads.
    CudaStreamCore(uintptr_t host_ptr, int n, int h, int w, int batch, const float mean[3],
                   const float std_[3], int num_slots, bool drop_last);
    ~CudaStreamCore();
    int num_batches() const;
    void begin_epoch();               // reset for a new pass (batches processed in order)
    uintptr_t next_batch(int* out_n); // blocking; device ptr of next ready (n,3,h,w) f32; 0 at end

 private:
    struct Impl;
    Impl* impl_;
};

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

// Transform-only, GPU-resident: the inputs are ALREADY-decoded device images (e.g. from
// nvImageCodec) given as device pointers + dims. Runs resize+normalize in place and returns
// the device pointer of the (N,3,dst_h,dst_w) float32 result (valid until the next call).
// Lets a fast external decoder (nvImageCodec) feed our transform with zero extra copies.
uintptr_t resize_normalize_device_batch(const std::vector<uintptr_t>& d_imgs,
                                        const std::vector<int>& ws, const std::vector<int>& hs,
                                        int dst_h, int dst_w, const float mean[3],
                                        const float std_[3]);

// ---- In-C++ nvImageCodec decode pipeline (needs HAVE_NVIMGCODEC) ----
// DALI moved off nvjpegDecodeBatched to nvImageCodec, NVIDIA's modern codec library
// (~21.6k img/s decode on a 3090). These run the WHOLE decode->resize->normalize->batch path
// inside one C++ call so a Python loader pays no per-image overhead and the GIL stays released
// the entire time — the last step to close the gap to DALI's GIL-free C++ data path.

// Initialize the pipeline once: dlopen libnvimgcodec at `lib_path` (the wheel's
// libnvimgcodec.so.0), load codec extensions from `ext_path` (the wheel's extensions dir, may
// be empty for auto-discovery), and create `num_slots` INDEPENDENT pipeline slots on `device_id`
// (NVIMGCODEC_DEVICE_CURRENT = -1) — each its own decoder + CUDA stream + buffers + output ring.
// K>1 slots, driven by K threads, overlap one batch's host decode with another's GPU work
// (DALI-style multi-batch-in-flight). Returns false if built without nvImageCodec or on error.
bool nvimgcodec_pipeline_init(int num_slots, const char* lib_path, const char* ext_path,
                              int device_id);

// Number of slots created by init (0 if not initialized / not built with nvImageCodec).
int nvimgcodec_num_slots();

// Decode a batch of JPEG byte buffers on SLOT `slot` and resize+normalize -> device pointer of
// the (N,3,dst_h,dst_w) CHW float32 result (valid until that slot's NV_OUT_RING-th next call).
// nvImageCodec decodes straight into the slot's device buffers; the resize_normalize kernel runs
// on the slot's CUDA stream (auto-ordered after decode), then the call syncs that stream so the
// returned pointer is fully ready (no async-handoff race). Each slot must be driven by at most
// ONE thread at a time. Returns 0 on error / bad slot / before init.
uintptr_t nvimgcodec_decode_resize_normalize_slot(int slot,
                                                  const std::vector<const uint8_t*>& jpegs,
                                                  const std::vector<size_t>& sizes, int dst_h,
                                                  int dst_w, const float mean[3],
                                                  const float std_[3]);

// Convenience: slot 0 (the single-slot synchronous path).
uintptr_t nvimgcodec_decode_resize_normalize(const std::vector<const uint8_t*>& jpegs,
                                             const std::vector<size_t>& sizes, int dst_h,
                                             int dst_w, const float mean[3], const float std_[3]);

}  // namespace cuda
}  // namespace turboloader
