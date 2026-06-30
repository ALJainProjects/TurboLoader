// CUDA implementation of the GPU transform path. Compiled with nvcc only when
// TURBOLOADER_ENABLE_CUDA=1 (see setup.py / docs/GPU_ACCELERATION.md).
//
// !!! UNVALIDATED !!!  This is a faithful, line-for-line port of the bit-exact, validated
// Metal kernels in src/metal/metal_transforms.mm — the SAME half-pixel bilinear + normalize
// math. But it has NOT been compiled or run: there is no NVIDIA GPU on the dev/CI machines.
// It is committed so it's ready to build, test, and debug the moment a CUDA box is online.
#include "cuda_transforms.hpp"

#include <cuda_runtime.h>
#ifdef HAVE_NVJPEG
#include <nvjpeg.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// Mirrors the Metal `resize_normalize` kernel exactly.
__global__ void resize_normalize_kernel(const uint8_t* src, float* dst, int srcW, int srcH,
                                        int dstW, int dstH, float3 mean, float3 invstd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;
    float sx = fmaxf(0.0f, (x + 0.5f) * (float)srcW / dstW - 0.5f);
    float sy = fmaxf(0.0f, (y + 0.5f) * (float)srcH / dstH - 0.5f);
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0, dy = sy - y0;
    float m[3] = {mean.x, mean.y, mean.z};
    float isd[3] = {invstd.x, invstd.y, invstd.z};
    for (int c = 0; c < 3; c++) {
        float p00 = src[(y0 * srcW + x0) * 3 + c], p10 = src[(y0 * srcW + x1) * 3 + c];
        float p01 = src[(y1 * srcW + x0) * 3 + c], p11 = src[(y1 * srcW + x1) * 3 + c];
        float top = p00 * (1 - dx) + p10 * dx, bot = p01 * (1 - dx) + p11 * dx;
        float v = (top * (1 - dy) + bot * dy) / 255.0f;
        dst[(c * dstH + y) * dstW + x] = (v - m[c]) * isd[c];
    }
}

// Mirrors the Metal `crop_resize_normalize` kernel exactly.
__global__ void crop_resize_normalize_kernel(const uint8_t* src, float* dst, int srcW, int srcH,
                                             int dstW, int dstH, float4 crop, int flip,
                                             float3 mean, float3 invstd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;
    int ox = flip ? (dstW - 1 - x) : x;
    float sx = crop.x + (ox + 0.5f) / (float)dstW * crop.z - 0.5f;
    float sy = crop.y + (y + 0.5f) / (float)dstH * crop.w - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), (float)(srcW - 1));
    sy = fminf(fmaxf(sy, 0.0f), (float)(srcH - 1));
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0, dy = sy - y0;
    float m[3] = {mean.x, mean.y, mean.z};
    float isd[3] = {invstd.x, invstd.y, invstd.z};
    for (int c = 0; c < 3; c++) {
        float p00 = src[(y0 * srcW + x0) * 3 + c], p10 = src[(y0 * srcW + x1) * 3 + c];
        float p01 = src[(y1 * srcW + x0) * 3 + c], p11 = src[(y1 * srcW + x1) * 3 + c];
        float top = p00 * (1 - dx) + p10 * dx, bot = p01 * (1 - dx) + p11 * dx;
        float v = (top * (1 - dy) + bot * dy) / 255.0f;
        dst[(c * dstH + y) * dstW + x] = (v - m[c]) * isd[c];
    }
}

std::string g_name;

// Pack variable-size images into one device buffer (offsets), like the Metal path.
bool pack_to_device(const std::vector<turboloader::cuda::ImageRef>& imgs,
                    std::vector<size_t>& off, uint8_t** d_src) {
    const size_t N = imgs.size();
    off.assign(N + 1, 0);
    for (size_t i = 0; i < N; i++) {
        if (!imgs[i].data || imgs[i].w <= 0 || imgs[i].h <= 0) return false;
        off[i + 1] = off[i] + (size_t)imgs[i].w * imgs[i].h * 3;
    }
    if (cudaMalloc(d_src, std::max<size_t>(off[N], 1)) != cudaSuccess) return false;
    for (size_t i = 0; i < N; i++)
        cudaMemcpy(*d_src + off[i], imgs[i].data, off[i + 1] - off[i], cudaMemcpyHostToDevice);
    return true;
}

}  // namespace

namespace turboloader {
namespace cuda {

bool available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

const char* device_name() {
    if (!available()) return "";
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "";
    g_name = prop.name;
    return g_name.c_str();
}

bool resize_normalize_batch(const std::vector<ImageRef>& imgs, int dst_h, int dst_w,
                            const float mean[3], const float std_[3], float* out) {
    if (!available() || imgs.empty() || dst_h <= 0 || dst_w <= 0) return false;
    const size_t N = imgs.size(), per_out = (size_t)3 * dst_h * dst_w;
    std::vector<size_t> off;
    uint8_t* d_src = nullptr;
    float* d_dst = nullptr;
    if (!pack_to_device(imgs, off, &d_src)) return false;
    if (cudaMalloc(&d_dst, N * per_out * sizeof(float)) != cudaSuccess) {
        cudaFree(d_src);
        return false;
    }
    float3 m = make_float3(mean[0], mean[1], mean[2]);
    float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    dim3 block(16, 16), grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (size_t i = 0; i < N; i++)
        resize_normalize_kernel<<<grid, block>>>(d_src + off[i], d_dst + i * per_out, imgs[i].w,
                                                 imgs[i].h, dst_w, dst_h, m, isd);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaGetLastError();
    bool ok = (err == cudaSuccess);
    if (!ok) fprintf(stderr, "[turboloader cuda] kernel error: %s\n", cudaGetErrorString(err));
    if (ok) cudaMemcpy(out, d_dst, N * per_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    return ok;
}

bool crop_resize_normalize_batch(const std::vector<ImageRef>& imgs,
                                 const std::vector<CropParams>& crops, int dst_h, int dst_w,
                                 const float mean[3], const float std_[3], float* out) {
    if (!available() || imgs.empty() || crops.size() != imgs.size() || dst_h <= 0 || dst_w <= 0)
        return false;
    const size_t N = imgs.size(), per_out = (size_t)3 * dst_h * dst_w;
    std::vector<size_t> off;
    uint8_t* d_src = nullptr;
    float* d_dst = nullptr;
    if (!pack_to_device(imgs, off, &d_src)) return false;
    if (cudaMalloc(&d_dst, N * per_out * sizeof(float)) != cudaSuccess) {
        cudaFree(d_src);
        return false;
    }
    float3 m = make_float3(mean[0], mean[1], mean[2]);
    float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    dim3 block(16, 16), grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (size_t i = 0; i < N; i++) {
        float4 crop = make_float4(crops[i].x, crops[i].y, crops[i].w, crops[i].h);
        crop_resize_normalize_kernel<<<grid, block>>>(d_src + off[i], d_dst + i * per_out,
                                                      imgs[i].w, imgs[i].h, dst_w, dst_h, crop,
                                                      crops[i].flip, m, isd);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaGetLastError();
    bool ok = (err == cudaSuccess);
    if (!ok) fprintf(stderr, "[turboloader cuda] kernel error: %s\n", cudaGetErrorString(err));
    if (ok) cudaMemcpy(out, d_dst, N * per_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    return ok;
}

#ifdef HAVE_NVJPEG
// nvJPEG batched decoder + persistent device pools (created once, grown as needed, reused
// across batches — no per-batch cudaMalloc). nvjpegDecodeBatched decodes the whole batch in
// parallel on the GPU, then the transform reads the pool in place; one D2H of the output.
static nvjpegHandle_t g_nvjpeg = nullptr;
static nvjpegJpegState_t g_batched_state = nullptr;
static cudaStream_t g_stream = nullptr;
static bool g_nvjpeg_ready = false;
static uint8_t* g_decode_pool = nullptr;
static size_t g_decode_cap = 0;
// Ring of output pools so a prefetched batch's result doesn't clobber the one still being
// consumed (each call returns a different slot; up to OUT_RING-1 batches stay valid).
static const int OUT_RING = 4;
static float* g_out_pool[OUT_RING] = {nullptr};
static size_t g_out_cap[OUT_RING] = {0};
static int g_out_idx = 0;
static int g_batched_max = 0;
static std::mutex g_pool_mutex;

// Advanced/decoupled nvJPEG API for host/device PIPELINING (DALI-style): double-buffered
// states + streams so image N+1's host Huffman (CPU) overlaps image N's device IDCT (GPU).
static nvjpegJpegDecoder_t g_decoder = nullptr;
static nvjpegJpegState_t g_pipe_state[2] = {nullptr, nullptr};
static nvjpegBufferPinned_t g_pinned[2] = {nullptr, nullptr};
static nvjpegBufferDevice_t g_device[2] = {nullptr, nullptr};
static nvjpegJpegStream_t g_jstream[2] = {nullptr, nullptr};
static cudaStream_t g_pstream[2] = {nullptr, nullptr};
static nvjpegDecodeParams_t g_dparams = nullptr;
static bool g_pipe_ready = false;

static void nvjpeg_init() {
    static bool done = false;
    if (done) return;
    done = true;
    // Prefer the hardware JPEG decoder (Ampere+), then GPU-hybrid, then the basic backend —
    // this is what DALI uses for its decode speed. nvjpegCreateEx falls through gracefully.
    nvjpegStatus_t s = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &g_nvjpeg);
    if (s != NVJPEG_STATUS_SUCCESS)
        s = nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr, nullptr, 0, &g_nvjpeg);
    if (s != NVJPEG_STATUS_SUCCESS) s = nvjpegCreateSimple(&g_nvjpeg);
    if (s != NVJPEG_STATUS_SUCCESS) return;
    if (nvjpegJpegStateCreate(g_nvjpeg, &g_batched_state) != NVJPEG_STATUS_SUCCESS) return;
    cudaStreamCreate(&g_stream);
    g_nvjpeg_ready = true;

    // Set up the decoupled (advanced) API for pipelined decode. Best-effort: if any of it
    // fails, g_pipe_ready stays false and we fall back to nvjpegDecodeBatched.
    if (nvjpegDecoderCreate(g_nvjpeg, NVJPEG_BACKEND_GPU_HYBRID, &g_decoder) ==
        NVJPEG_STATUS_SUCCESS) {
        bool ok = true;
        for (int b = 0; b < 2 && ok; b++) {
            ok = ok && nvjpegDecoderStateCreate(g_nvjpeg, g_decoder, &g_pipe_state[b]) ==
                           NVJPEG_STATUS_SUCCESS;
            ok = ok && nvjpegBufferPinnedCreate(g_nvjpeg, nullptr, &g_pinned[b]) ==
                           NVJPEG_STATUS_SUCCESS;
            ok = ok && nvjpegBufferDeviceCreate(g_nvjpeg, nullptr, &g_device[b]) ==
                           NVJPEG_STATUS_SUCCESS;
            ok = ok &&
                 nvjpegJpegStreamCreate(g_nvjpeg, &g_jstream[b]) == NVJPEG_STATUS_SUCCESS;
            cudaStreamCreate(&g_pstream[b]);
        }
        ok = ok &&
             nvjpegDecodeParamsCreate(g_nvjpeg, &g_dparams) == NVJPEG_STATUS_SUCCESS;
        if (ok) {
            nvjpegDecodeParamsSetOutputFormat(g_dparams, NVJPEG_OUTPUT_RGBI);
            g_pipe_ready = true;
        }
    }
}

// Pipelined decode via the decoupled API: image i+1's host Huffman overlaps image i's
// device IDCT (double-buffered states/streams). Decodes into decode_pool at off[i].
static bool pipelined_decode(const std::vector<const uint8_t*>& jpegs,
                             const std::vector<size_t>& sizes, const std::vector<int>& ws,
                             const std::vector<int>& hs, const std::vector<size_t>& off,
                             uint8_t* decode_pool) {
    const int N = (int)jpegs.size();
    for (int i = 0; i < N; i++) {
        const int b = i & 1;
        if (i >= 2) cudaStreamSynchronize(g_pstream[b]);  // image i-2's work done -> reuse buffer
        if (nvjpegJpegStreamParse(g_nvjpeg, jpegs[i], sizes[i], 0, 0, g_jstream[b]) !=
            NVJPEG_STATUS_SUCCESS)
            return false;
        nvjpegStateAttachPinnedBuffer(g_pipe_state[b], g_pinned[b]);
        nvjpegStateAttachDeviceBuffer(g_pipe_state[b], g_device[b]);
        if (nvjpegDecodeJpegHost(g_nvjpeg, g_decoder, g_pipe_state[b], g_dparams,
                                 g_jstream[b]) != NVJPEG_STATUS_SUCCESS)
            return false;  // CPU Huffman (overlaps the previous image's async GPU work)
        if (nvjpegDecodeJpegTransferToDevice(g_nvjpeg, g_decoder, g_pipe_state[b], g_jstream[b],
                                             g_pstream[b]) != NVJPEG_STATUS_SUCCESS)
            return false;
        nvjpegImage_t oi;
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            oi.channel[c] = nullptr;
            oi.pitch[c] = 0;
        }
        oi.channel[0] = decode_pool + off[i];
        oi.pitch[0] = (size_t)ws[i] * 3;
        if (nvjpegDecodeJpegDevice(g_nvjpeg, g_decoder, g_pipe_state[b], &oi, g_pstream[b]) !=
            NVJPEG_STATUS_SUCCESS)
            return false;  // async GPU IDCT
    }
    cudaStreamSynchronize(g_pstream[0]);
    cudaStreamSynchronize(g_pstream[1]);
    return true;
}

// Core fused decode+transform into a ring output slot (device, N*3*dst_h*dst_w float32).
// Caller must hold g_pool_mutex and have called nvjpeg_init(). Returns the device pointer of
// the slot used (no D2H), or nullptr on error.
static float* fused_into_pool(const std::vector<const uint8_t*>& jpegs,
                              const std::vector<size_t>& sizes, int dst_h, int dst_w,
                              const float mean[3], const float std_[3]) {
    if (jpegs.empty() || jpegs.size() != sizes.size() || dst_h <= 0 || dst_w <= 0) return nullptr;
    const int N = (int)jpegs.size();
    const size_t per_out = (size_t)3 * dst_h * dst_w;

    // Pass 1: dimensions + packed decode offsets.
    std::vector<int> ws(N), hs(N);
    std::vector<size_t> off(N + 1, 0);
    for (int i = 0; i < N; i++) {
        int nc;
        nvjpegChromaSubsampling_t ss;
        int w[NVJPEG_MAX_COMPONENT], h[NVJPEG_MAX_COMPONENT];
        if (nvjpegGetImageInfo(g_nvjpeg, jpegs[i], sizes[i], &nc, &ss, w, h) !=
            NVJPEG_STATUS_SUCCESS)
            return nullptr;
        ws[i] = w[0];
        hs[i] = h[0];
        off[i + 1] = off[i] + (size_t)ws[i] * hs[i] * 3;
    }

    // Grow the (transient) decode pool + the current ring output slot as needed.
    if (off[N] > g_decode_cap) {
        if (g_decode_pool) cudaFree(g_decode_pool);
        if (cudaMalloc(&g_decode_pool, off[N]) != cudaSuccess) {
            g_decode_cap = 0;
            return nullptr;
        }
        g_decode_cap = off[N];
    }
    const size_t out_bytes = (size_t)N * per_out * sizeof(float);
    float*& outp = g_out_pool[g_out_idx];
    if (out_bytes > g_out_cap[g_out_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_out_cap[g_out_idx] = 0;
            return nullptr;
        }
        g_out_cap[g_out_idx] = out_bytes;
    }
    // Decode the batch into the decode pool. Default: nvjpegDecodeBatched (fastest in
    // practice). The decoupled/pipelined API (opt-in via TURBOLOADER_NVJPEG_PIPELINED) does
    // host/device overlap but its per-image call overhead makes it ~10x slower here — DALI's
    // win is a far more optimized pipeline, not the naive split.
    if (g_pipe_ready && std::getenv("TURBOLOADER_NVJPEG_PIPELINED") != nullptr) {
        if (!pipelined_decode(jpegs, sizes, ws, hs, off, g_decode_pool)) return nullptr;
    } else {
        if (N != g_batched_max) {  // re-init on ANY batch-size change (e.g. a ragged last batch)
            int cpu_threads = (int)std::thread::hardware_concurrency();
            if (cpu_threads < 1) cpu_threads = 8;
            if (nvjpegDecodeBatchedInitialize(g_nvjpeg, g_batched_state, N, cpu_threads,
                                              NVJPEG_OUTPUT_RGBI) != NVJPEG_STATUS_SUCCESS)
                return nullptr;
            g_batched_max = N;
        }
        std::vector<nvjpegImage_t> imgs(N);
        for (int i = 0; i < N; i++) {
            for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
                imgs[i].channel[c] = nullptr;
                imgs[i].pitch[c] = 0;
            }
            imgs[i].channel[0] = g_decode_pool + off[i];
            imgs[i].pitch[0] = (size_t)ws[i] * 3;
        }
        if (nvjpegDecodeBatched(g_nvjpeg, g_batched_state, jpegs.data(), sizes.data(),
                                imgs.data(), g_stream) != NVJPEG_STATUS_SUCCESS)
            return nullptr;
    }

    // Transform reads the decode pool in place (no host round-trip).
    float3 m = make_float3(mean[0], mean[1], mean[2]);
    float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    dim3 block(16, 16), grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (int i = 0; i < N; i++)
        resize_normalize_kernel<<<grid, block, 0, g_stream>>>(
            g_decode_pool + off[i], outp + (size_t)i * per_out, ws[i], hs[i], dst_w, dst_h, m,
            isd);
    cudaError_t err = cudaStreamSynchronize(g_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] fused pipeline: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    g_out_idx = (g_out_idx + 1) % OUT_RING;  // next batch uses a different ring slot
    return outp;  // device result; no D2H
}

// CPU output: fused work + one D2H of the result to host `out`.
bool decode_resize_normalize_batch(const std::vector<const uint8_t*>& jpegs,
                                   const std::vector<size_t>& sizes, int dst_h, int dst_w,
                                   const float mean[3], const float std_[3], float* out) {
    if (!available()) return false;
    nvjpeg_init();
    if (!g_nvjpeg_ready) return false;
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    float* outp = fused_into_pool(jpegs, sizes, dst_h, dst_w, mean, std_);
    if (!outp) return false;
    const size_t bytes = (size_t)jpegs.size() * 3 * dst_h * dst_w * sizeof(float);
    cudaMemcpy(out, outp, bytes, cudaMemcpyDeviceToHost);
    return true;
}

// GPU output: fused work, return the device pointer of the result (valid until the next
// fused call — consume before the next batch, like DALI). Returns 0 on error.
uintptr_t decode_resize_normalize_batch_gpu(const std::vector<const uint8_t*>& jpegs,
                                            const std::vector<size_t>& sizes, int dst_h, int dst_w,
                                            const float mean[3], const float std_[3]) {
    if (!available()) return 0;
    nvjpeg_init();
    if (!g_nvjpeg_ready) return 0;
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    float* outp = fused_into_pool(jpegs, sizes, dst_h, dst_w, mean, std_);
    return outp ? reinterpret_cast<uintptr_t>(outp) : 0;
}

uintptr_t resize_normalize_device_batch(const std::vector<uintptr_t>& d_imgs,
                                        const std::vector<int>& ws, const std::vector<int>& hs,
                                        int dst_h, int dst_w, const float mean[3],
                                        const float std_[3]) {
    if (!available() || d_imgs.empty() || d_imgs.size() != ws.size() ||
        d_imgs.size() != hs.size() || dst_h <= 0 || dst_w <= 0)
        return 0;
    nvjpeg_init();  // ensures g_stream exists
    if (!g_nvjpeg_ready) return 0;
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    const int N = (int)d_imgs.size();
    const size_t per_out = (size_t)3 * dst_h * dst_w;
    float*& outp = g_out_pool[g_out_idx];
    const size_t out_bytes = (size_t)N * per_out * sizeof(float);
    if (out_bytes > g_out_cap[g_out_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_out_cap[g_out_idx] = 0;
            return 0;
        }
        g_out_cap[g_out_idx] = out_bytes;
    }
    float3 m = make_float3(mean[0], mean[1], mean[2]);
    float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    dim3 block(16, 16), grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (int i = 0; i < N; i++)  // transform reads the external device images in place
        resize_normalize_kernel<<<grid, block, 0, g_stream>>>(
            reinterpret_cast<const uint8_t*>(d_imgs[i]), outp + (size_t)i * per_out, ws[i], hs[i],
            dst_w, dst_h, m, isd);
    cudaError_t err = cudaStreamSynchronize(g_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] device transform: %s\n", cudaGetErrorString(err));
        return 0;
    }
    float* result = outp;
    g_out_idx = (g_out_idx + 1) % OUT_RING;
    return reinterpret_cast<uintptr_t>(result);
}
#else
bool decode_resize_normalize_batch(const std::vector<const uint8_t*>&,
                                   const std::vector<size_t>&, int, int, const float[3],
                                   const float[3], float*) {
    return false;  // built without nvJPEG
}
uintptr_t decode_resize_normalize_batch_gpu(const std::vector<const uint8_t*>&,
                                            const std::vector<size_t>&, int, int, const float[3],
                                            const float[3]) {
    return 0;  // built without nvJPEG
}
uintptr_t resize_normalize_device_batch(const std::vector<uintptr_t>&, const std::vector<int>&,
                                        const std::vector<int>&, int, int, const float[3],
                                        const float[3]) {
    return 0;  // built without nvJPEG
}
#endif

// ===========================================================================================
// In-C++ nvImageCodec decode pipeline. dlopen'd at runtime (the wheel ships only
// libnvimgcodec.so.0 with no linker symlink, and it is already loaded once Python imports
// nvidia.nvimgcodec), so we link nothing — we only need the header for the structs/enums and
// resolve the entry points with dlsym. Runs read-free decode->resize->normalize->batch in one
// GIL-released call: nvImageCodec decodes each JPEG straight into a persistent device buffer,
// then resize_normalize_kernel reads it in place on the SAME stream (auto-ordered after the
// decode's device work — no cross-stream sync).
// ===========================================================================================
#ifdef HAVE_NVIMGCODEC
#include <nvimgcodec.h>
#include <dlfcn.h>

namespace {

// dlsym'd entry points (only the ones this pipeline uses).
typedef nvimgcodecStatus_t (*pfn_InstanceCreate)(nvimgcodecInstance_t*,
                                                 const nvimgcodecInstanceCreateInfo_t*);
typedef nvimgcodecStatus_t (*pfn_DecoderCreate)(nvimgcodecInstance_t, nvimgcodecDecoder_t*,
                                                const nvimgcodecExecutionParams_t*, const char*);
typedef nvimgcodecStatus_t (*pfn_CodeStreamCreateFromHostMem)(nvimgcodecInstance_t,
                                                              nvimgcodecCodeStream_t*,
                                                              const unsigned char*, size_t,
                                                              const nvimgcodecCodeStreamView_t*);
typedef nvimgcodecStatus_t (*pfn_CodeStreamGetImageInfo)(nvimgcodecCodeStream_t,
                                                         nvimgcodecImageInfo_t*);
typedef nvimgcodecStatus_t (*pfn_ImageCreate)(nvimgcodecInstance_t, nvimgcodecImage_t*,
                                              const nvimgcodecImageInfo_t*);
typedef nvimgcodecStatus_t (*pfn_DecoderDecode)(nvimgcodecDecoder_t,
                                                const nvimgcodecCodeStream_t*,
                                                const nvimgcodecImage_t*, int,
                                                const nvimgcodecDecodeParams_t*,
                                                nvimgcodecFuture_t*);
typedef nvimgcodecStatus_t (*pfn_FutureWaitForAll)(nvimgcodecFuture_t);
typedef nvimgcodecStatus_t (*pfn_FutureGetProcessingStatus)(nvimgcodecFuture_t,
                                                            nvimgcodecProcessingStatus_t*,
                                                            size_t*);
typedef nvimgcodecStatus_t (*pfn_FutureDestroy)(nvimgcodecFuture_t);

static pfn_InstanceCreate nv_InstanceCreate = nullptr;
static pfn_DecoderCreate nv_DecoderCreate = nullptr;
static pfn_CodeStreamCreateFromHostMem nv_CodeStreamFromHostMem = nullptr;
static pfn_CodeStreamGetImageInfo nv_CodeStreamGetImageInfo = nullptr;
static pfn_ImageCreate nv_ImageCreate = nullptr;
static pfn_DecoderDecode nv_DecoderDecode = nullptr;
static pfn_FutureWaitForAll nv_FutureWaitForAll = nullptr;
static pfn_FutureGetProcessingStatus nv_FutureGetProcessingStatus = nullptr;
static pfn_FutureDestroy nv_FutureDestroy = nullptr;

static nvimgcodecInstance_t g_nv_instance = nullptr;
static nvimgcodecDecoder_t g_nv_decoder = nullptr;
static cudaStream_t g_nv_stream = nullptr;
static bool g_nv_ready = false;
static std::mutex g_nv_mutex;

// Reused per-image handles + persistent device input buffers (one decoded image each), grown
// on demand. A single decode thread calls the pipeline, so these are reused serially.
static std::vector<nvimgcodecCodeStream_t> g_nv_streams;
static std::vector<nvimgcodecImage_t> g_nv_images;
static std::vector<uint8_t*> g_nv_inbuf;
static std::vector<size_t> g_nv_incap;

// Output ring (independent of the nvJPEG path's): each call returns a different slot so up to
// NV_OUT_RING-1 prefetched batches stay valid while the consumer reads an earlier one.
static const int NV_OUT_RING = 4;
static float* g_nv_out_pool[NV_OUT_RING] = {nullptr};
static size_t g_nv_out_cap[NV_OUT_RING] = {0};
static int g_nv_out_idx = 0;

template <typename T>
static bool load_sym(void* h, T& fn, const char* name) {
    fn = reinterpret_cast<T>(dlsym(h, name));
    if (!fn) fprintf(stderr, "[turboloader nvimgcodec] missing symbol %s\n", name);
    return fn != nullptr;
}

}  // namespace

bool nvimgcodec_pipeline_init(const char* lib_path, const char* ext_path, int device_id) {
    std::lock_guard<std::mutex> lk(g_nv_mutex);
    if (g_nv_ready) return true;
    if (!available()) return false;
    void* h = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "[turboloader nvimgcodec] dlopen(%s): %s\n", lib_path, dlerror());
        return false;
    }
    bool ok = load_sym(h, nv_InstanceCreate, "nvimgcodecInstanceCreate") &&
              load_sym(h, nv_DecoderCreate, "nvimgcodecDecoderCreate") &&
              load_sym(h, nv_CodeStreamFromHostMem, "nvimgcodecCodeStreamCreateFromHostMem") &&
              load_sym(h, nv_CodeStreamGetImageInfo, "nvimgcodecCodeStreamGetImageInfo") &&
              load_sym(h, nv_ImageCreate, "nvimgcodecImageCreate") &&
              load_sym(h, nv_DecoderDecode, "nvimgcodecDecoderDecode") &&
              load_sym(h, nv_FutureWaitForAll, "nvimgcodecFutureWaitForAll") &&
              load_sym(h, nv_FutureGetProcessingStatus, "nvimgcodecFutureGetProcessingStatus") &&
              load_sym(h, nv_FutureDestroy, "nvimgcodecFutureDestroy");
    if (!ok) return false;

    nvimgcodecInstanceCreateInfo_t ci{};
    ci.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.struct_size = sizeof(ci);
    ci.load_builtin_modules = 1;
    ci.load_extension_modules = 1;
    ci.extension_modules_path = (ext_path && ext_path[0]) ? ext_path : nullptr;
    ci.create_debug_messenger = 0;
    if (nv_InstanceCreate(&g_nv_instance, &ci) != NVIMGCODEC_STATUS_SUCCESS) {
        fprintf(stderr, "[turboloader nvimgcodec] InstanceCreate failed\n");
        return false;
    }
    nvimgcodecExecutionParams_t ep{};
    ep.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
    ep.struct_size = sizeof(ep);
    ep.device_id = device_id;       // NVIMGCODEC_DEVICE_CURRENT = -1
    ep.max_num_cpu_threads = 0;     // default = number of cores
    ep.pre_init = 1;
    ep.num_backends = 0;            // all backends allowed (HW + GPU-hybrid + CPU fallback)
    ep.backends = nullptr;
    if (nv_DecoderCreate(g_nv_instance, &g_nv_decoder, &ep, nullptr) != NVIMGCODEC_STATUS_SUCCESS) {
        fprintf(stderr, "[turboloader nvimgcodec] DecoderCreate failed\n");
        return false;
    }
    if (cudaStreamCreate(&g_nv_stream) != cudaSuccess) return false;
    g_nv_ready = true;
    return true;
}

uintptr_t nvimgcodec_decode_resize_normalize(const std::vector<const uint8_t*>& jpegs,
                                             const std::vector<size_t>& sizes, int dst_h,
                                             int dst_w, const float mean[3], const float std_[3]) {
    if (!g_nv_ready || jpegs.empty() || jpegs.size() != sizes.size() || dst_h <= 0 || dst_w <= 0)
        return 0;
    std::lock_guard<std::mutex> lk(g_nv_mutex);
    const int N = (int)jpegs.size();
    if ((int)g_nv_streams.size() < N) {
        g_nv_streams.resize(N, nullptr);
        g_nv_images.resize(N, nullptr);
        g_nv_inbuf.resize(N, nullptr);
        g_nv_incap.resize(N, 0);
    }
    std::vector<int> ws(N), hs(N);

    nvimgcodecDecodeParams_t dp{};
    dp.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    dp.struct_size = sizeof(dp);
    dp.apply_exif_orientation = 1;

    for (int i = 0; i < N; i++) {
        // Reuse the handle if non-null (nvImageCodec rebinds it to the new source).
        if (nv_CodeStreamFromHostMem(g_nv_instance, &g_nv_streams[i], jpegs[i], sizes[i],
                                     nullptr) != NVIMGCODEC_STATUS_SUCCESS)
            return 0;
        nvimgcodecImageInfo_t si{};
        si.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        si.struct_size = sizeof(si);
        if (nv_CodeStreamGetImageInfo(g_nv_streams[i], &si) != NVIMGCODEC_STATUS_SUCCESS) return 0;
        const int W = (int)si.plane_info[0].width, H = (int)si.plane_info[0].height;
        if (W <= 0 || H <= 0) return 0;
        ws[i] = W;
        hs[i] = H;
        const size_t need = (size_t)W * H * 3;
        if (need > g_nv_incap[i]) {
            if (g_nv_inbuf[i]) cudaFree(g_nv_inbuf[i]);
            if (cudaMalloc(&g_nv_inbuf[i], need) != cudaSuccess) {
                g_nv_incap[i] = 0;
                return 0;
            }
            g_nv_incap[i] = need;
        }
        // Output image: interleaved RGB (HWC) uint8 in our device buffer, contiguous rows
        // (row_stride = W*3) so resize_normalize_kernel's implicit stride matches, decoded on
        // g_nv_stream so the kernel (same stream) is ordered after it.
        nvimgcodecImageInfo_t oi{};
        oi.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        oi.struct_size = sizeof(oi);
        oi.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        oi.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        oi.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        oi.num_planes = 1;
        oi.plane_info[0].struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO;
        oi.plane_info[0].struct_size = sizeof(oi.plane_info[0]);
        oi.plane_info[0].width = (uint32_t)W;
        oi.plane_info[0].height = (uint32_t)H;
        oi.plane_info[0].num_channels = 3;
        oi.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        oi.plane_info[0].row_stride = (size_t)W * 3;
        oi.plane_info[0].precision = 0;
        oi.buffer = g_nv_inbuf[i];
        oi.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        oi.cuda_stream = g_nv_stream;
        if (nv_ImageCreate(g_nv_instance, &g_nv_images[i], &oi) != NVIMGCODEC_STATUS_SUCCESS)
            return 0;
    }

    nvimgcodecFuture_t fut = nullptr;
    if (nv_DecoderDecode(g_nv_decoder, g_nv_streams.data(), g_nv_images.data(), N, &dp, &fut) !=
        NVIMGCODEC_STATUS_SUCCESS)
        return 0;
    nv_FutureWaitForAll(fut);
    std::vector<nvimgcodecProcessingStatus_t> st(N, 0);
    size_t st_n = (size_t)N;
    nv_FutureGetProcessingStatus(fut, st.data(), &st_n);
    nv_FutureDestroy(fut);
    for (int i = 0; i < N; i++) {
        if (st[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            fprintf(stderr, "[turboloader nvimgcodec] decode status[%d]=0x%x\n", i,
                    (unsigned)st[i]);
            return 0;
        }
    }

    const size_t per_out = (size_t)3 * dst_h * dst_w;
    float*& outp = g_nv_out_pool[g_nv_out_idx];
    const size_t out_bytes = (size_t)N * per_out * sizeof(float);
    if (out_bytes > g_nv_out_cap[g_nv_out_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_nv_out_cap[g_nv_out_idx] = 0;
            return 0;
        }
        g_nv_out_cap[g_nv_out_idx] = out_bytes;
    }
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    dim3 block(16, 16), grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (int i = 0; i < N; i++)
        resize_normalize_kernel<<<grid, block, 0, g_nv_stream>>>(
            g_nv_inbuf[i], outp + (size_t)i * per_out, ws[i], hs[i], dst_w, dst_h, m, isd);
    cudaError_t err = cudaStreamSynchronize(g_nv_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader nvimgcodec] kernel: %s\n", cudaGetErrorString(err));
        return 0;
    }
    float* result = outp;
    g_nv_out_idx = (g_nv_out_idx + 1) % NV_OUT_RING;
    return reinterpret_cast<uintptr_t>(result);
}
#else
bool nvimgcodec_pipeline_init(const char*, const char*, int) { return false; }
uintptr_t nvimgcodec_decode_resize_normalize(const std::vector<const uint8_t*>&,
                                             const std::vector<size_t>&, int, int, const float[3],
                                             const float[3]) {
    return 0;  // built without nvImageCodec
}
#endif

}  // namespace cuda
}  // namespace turboloader
