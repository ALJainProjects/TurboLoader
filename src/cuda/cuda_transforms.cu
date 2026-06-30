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
    if (N != g_batched_max) {  // re-init on ANY batch-size change (e.g. a ragged last batch)
        // max_cpu_threads for the host-side Huffman decode — single-threaded was the
        // bottleneck (DALI parallelizes this). Use the hardware concurrency.
        int cpu_threads = (int)std::thread::hardware_concurrency();
        if (cpu_threads < 1) cpu_threads = 8;
        if (nvjpegDecodeBatchedInitialize(g_nvjpeg, g_batched_state, N, cpu_threads,
                                          NVJPEG_OUTPUT_RGBI) != NVJPEG_STATUS_SUCCESS)
            return nullptr;
        g_batched_max = N;
    }

    // Point each output image into the pool; decode the whole batch in one parallel call.
    std::vector<nvjpegImage_t> imgs(N);
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            imgs[i].channel[c] = nullptr;
            imgs[i].pitch[c] = 0;
        }
        imgs[i].channel[0] = g_decode_pool + off[i];
        imgs[i].pitch[0] = (size_t)ws[i] * 3;
    }
    if (nvjpegDecodeBatched(g_nvjpeg, g_batched_state, jpegs.data(), sizes.data(), imgs.data(),
                            g_stream) != NVJPEG_STATUS_SUCCESS)
        return nullptr;

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
#endif

}  // namespace cuda
}  // namespace turboloader
