// CUDA implementation of the GPU transform path. Compiled with nvcc only when
// TURBOLOADER_ENABLE_CUDA=1 (see setup.py / docs/GPU_ACCELERATION.md).
//
// VALIDATED on real hardware (two Jetson AGX Orins + an RTX 3090): a line-for-line port
// of the bit-exact Metal kernels; cuda_resize_normalize matches the numpy reference to
// 3.2e-05. The nvImageCodec pipeline here beats DALI on-the-fly on a 3090 — see
// experiments/cuda/RESULTS.md for benchmarks and correctness proofs.

#include "cuda_transforms.hpp"

#include <cuda_runtime.h>
#ifdef HAVE_NVJPEG
#include <nvjpeg.h>
#endif
#ifdef HAVE_NVIMGCODEC
#include <nvimgcodec.h>
#include <dlfcn.h>
#endif

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
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

// Batched normalize: pre-resized NHWC uint8 (contiguous, all images same H*W) -> NCHW float32
// normalized, in ONE kernel launch for the whole batch of N images (no per-image kernel, no
// resize). For a pre-processed, GPU-resident dataset: upload uint8 once, then normalize per
// epoch on the GPU with zero H2D — the path that beats a pre-processed loader at its own game.
__global__ void normalize_nhwc_to_nchw_kernel(const uint8_t* src, float* dst, int N, int H, int W,
                                              float3 mean, float3 invstd) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)N * 3 * H * W;
    if (idx >= total) return;
    const int x = (int)(idx % W);
    const int y = (int)((idx / W) % H);
    const int c = (int)((idx / ((long)W * H)) % 3);
    const long n = idx / ((long)W * H * 3);
    const float m = (c == 0) ? mean.x : (c == 1) ? mean.y : mean.z;
    const float isd = (c == 0) ? invstd.x : (c == 1) ? invstd.y : invstd.z;
    const uint8_t v = src[(n * H * W + (long)y * W + x) * 3 + c];  // NHWC in
    dst[idx] = (v / 255.0f - m) * isd;                            // NCHW out (idx == n*3HW+c*HW+y*W+x)
}

// Same, but output image n gathers from source image `sel[n]` (a device int64 index array) —
// fuses a shuffle/gather into the normalize so a GPU-resident loader can shuffle without a
// separate gather copy.
__global__ void normalize_gather_kernel(const uint8_t* src, const long long* sel, float* dst,
                                        int N, int H, int W, float3 mean, float3 invstd) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)N * 3 * H * W;
    if (idx >= total) return;
    const int x = (int)(idx % W);
    const int y = (int)((idx / W) % H);
    const int c = (int)((idx / ((long)W * H)) % 3);
    const long n = idx / ((long)W * H * 3);
    const long s = (long)sel[n];  // source image this output row pulls from
    const float m = (c == 0) ? mean.x : (c == 1) ? mean.y : mean.z;
    const float isd = (c == 0) ? invstd.x : (c == 1) ? invstd.y : invstd.z;
    const uint8_t v = src[(s * H * W + (long)y * W + x) * 3 + c];
    dst[idx] = (v / 255.0f - m) * isd;
}

// Port of the (numpy-validated) Metal `nv12_resize_normalize` kernel, generalized
// over the 4:2:0 chroma layout: c_px_stride 2 with cb/cr addressing one interleaved
// plane = NV12 (NVDEC output); c_px_stride 1 with separate planes = I420 (what CPU
// decoders emit). Identical sampling math: half-pixel luma bilinear, MPEG chroma
// siting (horizontal co-sited cx = sx/2, vertical centered cy = sy/2 - 1/4),
// video-range BT.601/709 matrix, clip.
__global__ void yuv420_resize_normalize_kernel(const uint8_t* yp, const uint8_t* cbp,
                                               const uint8_t* crp, int y_stride, int c_stride,
                                               int c_px_stride, int srcW, int srcH, int dstW,
                                               int dstH, int bt709, float* dst, float3 mean,
                                               float3 invstd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;
    float sx = fmaxf(0.0f, (x + 0.5f) * (float)srcW / dstW - 0.5f);
    float sy = fmaxf(0.0f, (y + 0.5f) * (float)srcH / dstH - 0.5f);

    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0, dy = sy - y0;
    float Ytop = yp[y0 * y_stride + x0] * (1 - dx) + yp[y0 * y_stride + x1] * dx;
    float Ybot = yp[y1 * y_stride + x0] * (1 - dx) + yp[y1 * y_stride + x1] * dx;
    float Yv = Ytop * (1 - dy) + Ybot * dy;

    int cw = (srcW + 1) / 2, ch = (srcH + 1) / 2;
    float cxf = fminf(fmaxf(sx * 0.5f, 0.0f), (float)(cw - 1));
    float cyf = fminf(fmaxf(sy * 0.5f - 0.25f, 0.0f), (float)(ch - 1));
    int cx0 = (int)cxf, cy0 = (int)cyf;
    int cx1 = min(cx0 + 1, cw - 1), cy1 = min(cy0 + 1, ch - 1);
    float cdx = cxf - cx0, cdy = cyf - cy0;
    const uint8_t* cb00 = cbp + cy0 * c_stride;
    const uint8_t* cb10 = cbp + cy1 * c_stride;
    const uint8_t* cr00 = crp + cy0 * c_stride;
    const uint8_t* cr10 = crp + cy1 * c_stride;
    float Cb = ((float)cb00[cx0 * c_px_stride] * (1 - cdx) +
                (float)cb00[cx1 * c_px_stride] * cdx) * (1 - cdy) +
               ((float)cb10[cx0 * c_px_stride] * (1 - cdx) +
                (float)cb10[cx1 * c_px_stride] * cdx) * cdy;
    float Cr = ((float)cr00[cx0 * c_px_stride] * (1 - cdx) +
                (float)cr00[cx1 * c_px_stride] * cdx) * (1 - cdy) +
               ((float)cr10[cx0 * c_px_stride] * (1 - cdx) +
                (float)cr10[cx1 * c_px_stride] * cdx) * cdy;

    float yv = (Yv - 16.0f) * (255.0f / 219.0f);
    float cb = Cb - 128.0f, cr = Cr - 128.0f;
    float R, G, B;
    if (bt709) {
        R = yv + 1.792741f * cr;
        G = yv - 0.213249f * cb - 0.532909f * cr;
        B = yv + 2.112402f * cb;
    } else {
        R = yv + 1.596027f * cr;
        G = yv - 0.391762f * cb - 0.812968f * cr;
        B = yv + 2.017232f * cb;
    }
    float rgb[3] = {fminf(fmaxf(R / 255.0f, 0.0f), 1.0f), fminf(fmaxf(G / 255.0f, 0.0f), 1.0f),
                    fminf(fmaxf(B / 255.0f, 0.0f), 1.0f)};
    float m[3] = {mean.x, mean.y, mean.z};
    float isd[3] = {invstd.x, invstd.y, invstd.z};
    for (int c = 0; c < 3; c++) dst[(c * dstH + y) * dstW + x] = (rgb[c] - m[c]) * isd[c];
}

// Novel fused CLIP-assembly kernel: one launch builds a whole (T, 3, H, W)
// training clip from T decoded YUV frames, applying the SAME crop window and
// horizontal flip to every frame (the standard video-augmentation contract —
// spatial augmentation must be consistent across time) fused with the YUV->RGB
// conversion, bilinear resize, and normalize. Grid z = frame index; per-frame
// plane pointers arrive in device arrays. No standard loader fuses this — it is
// normally decode -> convert -> crop -> resize -> normalize as separate passes.
__global__ void yuv420_clip_crop_normalize_kernel(
    const uint8_t* const* y_ptrs, const uint8_t* const* cb_ptrs, const uint8_t* const* cr_ptrs,
    int y_stride, int c_stride, int c_px_stride, int srcW, int srcH, int dstW, int dstH,
    float4 crop, int flip, int bt709, float* dst, float3 mean, float3 invstd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z;
    if (x >= dstW || y >= dstH) return;
    const uint8_t* yp = y_ptrs[t];
    const uint8_t* cbp = cb_ptrs[t];
    const uint8_t* crp = cr_ptrs[t];

    int ox = flip ? (dstW - 1 - x) : x;
    float sx = crop.x + (ox + 0.5f) / (float)dstW * crop.z - 0.5f;
    float sy = crop.y + (y + 0.5f) / (float)dstH * crop.w - 0.5f;
    sx = fminf(fmaxf(sx, 0.0f), (float)(srcW - 1));
    sy = fminf(fmaxf(sy, 0.0f), (float)(srcH - 1));

    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    float dx = sx - x0, dy = sy - y0;
    float Ytop = yp[y0 * y_stride + x0] * (1 - dx) + yp[y0 * y_stride + x1] * dx;
    float Ybot = yp[y1 * y_stride + x0] * (1 - dx) + yp[y1 * y_stride + x1] * dx;
    float Yv = Ytop * (1 - dy) + Ybot * dy;

    int cw = (srcW + 1) / 2, ch = (srcH + 1) / 2;
    float cxf = fminf(fmaxf(sx * 0.5f, 0.0f), (float)(cw - 1));
    float cyf = fminf(fmaxf(sy * 0.5f - 0.25f, 0.0f), (float)(ch - 1));
    int cx0 = (int)cxf, cy0 = (int)cyf;
    int cx1 = min(cx0 + 1, cw - 1), cy1 = min(cy0 + 1, ch - 1);
    float cdx = cxf - cx0, cdy = cyf - cy0;
    const uint8_t* cb00 = cbp + cy0 * c_stride;
    const uint8_t* cb10 = cbp + cy1 * c_stride;
    const uint8_t* cr00 = crp + cy0 * c_stride;
    const uint8_t* cr10 = crp + cy1 * c_stride;
    float Cb = ((float)cb00[cx0 * c_px_stride] * (1 - cdx) +
                (float)cb00[cx1 * c_px_stride] * cdx) * (1 - cdy) +
               ((float)cb10[cx0 * c_px_stride] * (1 - cdx) +
                (float)cb10[cx1 * c_px_stride] * cdx) * cdy;
    float Cr = ((float)cr00[cx0 * c_px_stride] * (1 - cdx) +
                (float)cr00[cx1 * c_px_stride] * cdx) * (1 - cdy) +
               ((float)cr10[cx0 * c_px_stride] * (1 - cdx) +
                (float)cr10[cx1 * c_px_stride] * cdx) * cdy;

    float yv = (Yv - 16.0f) * (255.0f / 219.0f);
    float cb = Cb - 128.0f, cr = Cr - 128.0f;
    float R, G, B;
    if (bt709) {
        R = yv + 1.792741f * cr;
        G = yv - 0.213249f * cb - 0.532909f * cr;
        B = yv + 2.112402f * cb;
    } else {
        R = yv + 1.596027f * cr;
        G = yv - 0.391762f * cb - 0.812968f * cr;
        B = yv + 2.017232f * cb;
    }
    float rgb[3] = {fminf(fmaxf(R / 255.0f, 0.0f), 1.0f), fminf(fmaxf(G / 255.0f, 0.0f), 1.0f),
                    fminf(fmaxf(B / 255.0f, 0.0f), 1.0f)};
    float m[3] = {mean.x, mean.y, mean.z};
    float isd[3] = {invstd.x, invstd.y, invstd.z};
    float* out = dst + (size_t)t * 3 * dstH * dstW;
    for (int c = 0; c < 3; c++) out[(c * dstH + y) * dstW + x] = (rgb[c] - m[c]) * isd[c];
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

// Dedicated stream + output ring for the GPU-resident normalize path (independent of the
// nvJPEG/nvImageCodec paths). Each call returns a different ring slot so prefetched batches stay
// valid while the consumer reads an earlier one.
static cudaStream_t g_rn_stream = nullptr;
static const int RN_RING = 4;
static float* g_rn_pool[RN_RING] = {nullptr};
static size_t g_rn_cap[RN_RING] = {0};
static int g_rn_idx = 0;
static std::mutex g_rn_mutex;

// Video conversion path: own stream + double-buffered output (a returned pointer
// stays valid until the NEXT call — same lifetime contract as the Metal side).
static cudaStream_t g_vid_stream = nullptr;
static float* g_vid_pool[2] = {nullptr, nullptr};
static size_t g_vid_cap[2] = {0, 0};
static int g_vid_idx = 0;
static std::mutex g_vid_mutex;

uintptr_t video_yuv420_batch(const std::vector<uintptr_t>& y_ptrs,
                             const std::vector<uintptr_t>& cb_ptrs,
                             const std::vector<uintptr_t>& cr_ptrs, int y_stride, int c_stride,
                             int c_px_stride, int src_w, int src_h, int dst_h, int dst_w,
                             bool bt709, const float mean[3], const float std_[3]) {
    const size_t n = y_ptrs.size();
    if (!available() || n == 0 || cb_ptrs.size() != n || cr_ptrs.size() != n || src_w <= 0 ||
        src_h <= 0 || dst_h <= 0 || dst_w <= 0 || y_stride < src_w ||
        (c_px_stride != 1 && c_px_stride != 2))
        return 0;
    std::lock_guard<std::mutex> lk(g_vid_mutex);
    if (!g_vid_stream && cudaStreamCreate(&g_vid_stream) != cudaSuccess) return 0;
    const size_t per = (size_t)3 * dst_h * dst_w;
    const size_t out_bytes = n * per * sizeof(float);
    g_vid_idx ^= 1;
    float*& outp = g_vid_pool[g_vid_idx];
    if (out_bytes > g_vid_cap[g_vid_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_vid_cap[g_vid_idx] = 0;
            return 0;
        }
        g_vid_cap[g_vid_idx] = out_bytes;
    }
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    const dim3 block(16, 16);
    const dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (size_t i = 0; i < n; i++) {
        yuv420_resize_normalize_kernel<<<grid, block, 0, g_vid_stream>>>(
            reinterpret_cast<const uint8_t*>(y_ptrs[i]),
            reinterpret_cast<const uint8_t*>(cb_ptrs[i]),
            reinterpret_cast<const uint8_t*>(cr_ptrs[i]), y_stride, c_stride, c_px_stride, src_w,
            src_h, dst_w, dst_h, bt709 ? 1 : 0, outp + i * per, m, isd);
    }
    cudaError_t err = cudaStreamSynchronize(g_vid_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] video yuv420: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return reinterpret_cast<uintptr_t>(outp);
}

// Device-side pointer arrays for the clip kernel (grown on demand, guarded by
// g_vid_mutex like the output pool).
static const uint8_t** g_clip_ptrs = nullptr;  // holds 3*T pointers: y | cb | cr
static size_t g_clip_ptrs_cap = 0;

uintptr_t video_clip_yuv420(const std::vector<uintptr_t>& y_ptrs,
                            const std::vector<uintptr_t>& cb_ptrs,
                            const std::vector<uintptr_t>& cr_ptrs, int y_stride, int c_stride,
                            int c_px_stride, int src_w, int src_h, int dst_h, int dst_w,
                            const float crop[4], bool flip, bool bt709, const float mean[3],
                            const float std_[3]) {
    const size_t t = y_ptrs.size();
    if (!available() || t == 0 || cb_ptrs.size() != t || cr_ptrs.size() != t || src_w <= 0 ||
        src_h <= 0 || dst_h <= 0 || dst_w <= 0 || (c_px_stride != 1 && c_px_stride != 2))
        return 0;
    std::lock_guard<std::mutex> lk(g_vid_mutex);
    if (!g_vid_stream && cudaStreamCreate(&g_vid_stream) != cudaSuccess) return 0;
    const size_t per = (size_t)3 * dst_h * dst_w;
    const size_t out_bytes = t * per * sizeof(float);
    g_vid_idx ^= 1;
    float*& outp = g_vid_pool[g_vid_idx];
    if (out_bytes > g_vid_cap[g_vid_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_vid_cap[g_vid_idx] = 0;
            return 0;
        }
        g_vid_cap[g_vid_idx] = out_bytes;
    }
    const size_t ptr_bytes = 3 * t * sizeof(const uint8_t*);
    if (ptr_bytes > g_clip_ptrs_cap) {
        if (g_clip_ptrs) cudaFree(g_clip_ptrs);
        if (cudaMalloc(&g_clip_ptrs, ptr_bytes) != cudaSuccess) {
            g_clip_ptrs_cap = 0;
            return 0;
        }
        g_clip_ptrs_cap = ptr_bytes;
    }
    std::vector<const uint8_t*> host_ptrs(3 * t);
    for (size_t i = 0; i < t; i++) {
        host_ptrs[i] = reinterpret_cast<const uint8_t*>(y_ptrs[i]);
        host_ptrs[t + i] = reinterpret_cast<const uint8_t*>(cb_ptrs[i]);
        host_ptrs[2 * t + i] = reinterpret_cast<const uint8_t*>(cr_ptrs[i]);
    }
    if (cudaMemcpyAsync(g_clip_ptrs, host_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice,
                        g_vid_stream) != cudaSuccess)
        return 0;
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    const float4 cr4 = make_float4(crop[0], crop[1], crop[2], crop[3]);
    const dim3 block(16, 16);
    const dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16, (unsigned)t);
    yuv420_clip_crop_normalize_kernel<<<grid, block, 0, g_vid_stream>>>(
        g_clip_ptrs, g_clip_ptrs + t, g_clip_ptrs + 2 * t, y_stride, c_stride, c_px_stride,
        src_w, src_h, dst_w, dst_h, cr4, flip ? 1 : 0, bt709 ? 1 : 0, outp, m, isd);
    cudaError_t err = cudaStreamSynchronize(g_vid_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] video clip: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return reinterpret_cast<uintptr_t>(outp);
}

uintptr_t normalize_resident_batch(uintptr_t src_dev, int N, int H, int W, const float mean[3],
                                   const float std_[3]) {
    if (!available() || !src_dev || N <= 0 || H <= 0 || W <= 0) return 0;
    std::lock_guard<std::mutex> lk(g_rn_mutex);
    if (!g_rn_stream && cudaStreamCreate(&g_rn_stream) != cudaSuccess) return 0;
    const size_t per = (size_t)3 * H * W;
    const size_t out_bytes = (size_t)N * per * sizeof(float);
    float*& outp = g_rn_pool[g_rn_idx];
    if (out_bytes > g_rn_cap[g_rn_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_rn_cap[g_rn_idx] = 0;
            return 0;
        }
        g_rn_cap[g_rn_idx] = out_bytes;
    }
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    const long total = (long)N * 3 * H * W;
    const int block = 256;
    const long grid = (total + block - 1) / block;
    normalize_nhwc_to_nchw_kernel<<<(unsigned)grid, block, 0, g_rn_stream>>>(
        reinterpret_cast<const uint8_t*>(src_dev), outp, N, H, W, m, isd);
    cudaError_t err = cudaStreamSynchronize(g_rn_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] resident normalize: %s\n", cudaGetErrorString(err));
        return 0;
    }
    float* result = outp;
    g_rn_idx = (g_rn_idx + 1) % RN_RING;
    return reinterpret_cast<uintptr_t>(result);
}

// GPU-resident normalize with a fused gather: output image n pulls from source image sel[n]
// (device int64 index array). Lets CudaResidentLoader shuffle at full speed (no torch gather).
uintptr_t normalize_resident_gather_batch(uintptr_t src_dev, uintptr_t sel_dev, int N, int H, int W,
                                          const float mean[3], const float std_[3]) {
    if (!available() || !src_dev || !sel_dev || N <= 0 || H <= 0 || W <= 0) return 0;
    std::lock_guard<std::mutex> lk(g_rn_mutex);
    if (!g_rn_stream && cudaStreamCreate(&g_rn_stream) != cudaSuccess) return 0;
    const size_t per = (size_t)3 * H * W;
    const size_t out_bytes = (size_t)N * per * sizeof(float);
    float*& outp = g_rn_pool[g_rn_idx];
    if (out_bytes > g_rn_cap[g_rn_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            g_rn_cap[g_rn_idx] = 0;
            return 0;
        }
        g_rn_cap[g_rn_idx] = out_bytes;
    }
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    const long total = (long)N * 3 * H * W;
    const int block = 256;
    const long grid = (total + block - 1) / block;
    normalize_gather_kernel<<<(unsigned)grid, block, 0, g_rn_stream>>>(
        reinterpret_cast<const uint8_t*>(src_dev), reinterpret_cast<const long long*>(sel_dev),
        outp, N, H, W, m, isd);
    cudaError_t err = cudaStreamSynchronize(g_rn_stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] resident gather: %s\n", cudaGetErrorString(err));
        return 0;
    }
    float* result = outp;
    g_rn_idx = (g_rn_idx + 1) % RN_RING;
    return reinterpret_cast<uintptr_t>(result);
}

// ---- Streaming: K-slot async H2D + normalize (dataset in host RAM, larger than VRAM) ----
// Each slot has its own CUDA stream + device scratch input + output ring; K slots on K threads
// overlap one batch's H2D with another's kernel — the K-slot pattern applied to the H2D copy,
// to beat a streaming loader (FFCV raw) whose per-batch H2D would otherwise serialize.
namespace {
struct StreamSlot {
    cudaStream_t stream = nullptr;
    uint8_t* d_in = nullptr;  // device uint8 scratch (the batch, after H2D)
    size_t in_cap = 0;
    static const int SRING = 8;  // >= loader out_q depth + producing + consuming (per slot)
    float* out_pool[SRING] = {nullptr};
    size_t out_cap[SRING] = {0};
    int out_idx = 0;
    std::mutex mu;
};
static std::vector<std::unique_ptr<StreamSlot>> g_sm_slots;
static std::mutex g_sm_init_mutex;
}  // namespace

int stream_normalize_init(int num_slots) {
    std::lock_guard<std::mutex> lk(g_sm_init_mutex);
    if (!available()) return 0;
    if (num_slots < 1) num_slots = 1;
    while ((int)g_sm_slots.size() < num_slots) {
        auto s = std::make_unique<StreamSlot>();
        if (cudaStreamCreate(&s->stream) != cudaSuccess) break;
        g_sm_slots.push_back(std::move(s));
    }
    return (int)g_sm_slots.size();
}

uintptr_t stream_normalize_batch(uintptr_t host_src, int N, int H, int W, const float mean[3],
                                 const float std_[3], int slot) {
    if (!available() || !host_src || slot < 0 || slot >= (int)g_sm_slots.size() || N <= 0 ||
        H <= 0 || W <= 0)
        return 0;
    StreamSlot& S = *g_sm_slots[slot];
    std::lock_guard<std::mutex> lk(S.mu);
    const size_t in_bytes = (size_t)N * H * W * 3;
    if (in_bytes > S.in_cap) {
        if (S.d_in) cudaFree(S.d_in);
        if (cudaMalloc(&S.d_in, in_bytes) != cudaSuccess) {
            S.in_cap = 0;
            return 0;
        }
        S.in_cap = in_bytes;
    }
    if (cudaMemcpyAsync(S.d_in, reinterpret_cast<const void*>(host_src), in_bytes,
                        cudaMemcpyHostToDevice, S.stream) != cudaSuccess)
        return 0;
    const size_t per = (size_t)3 * H * W;
    const size_t out_bytes = (size_t)N * per * sizeof(float);
    float*& outp = S.out_pool[S.out_idx];
    if (out_bytes > S.out_cap[S.out_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            S.out_cap[S.out_idx] = 0;
            return 0;
        }
        S.out_cap[S.out_idx] = out_bytes;
    }
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    const long total = (long)N * 3 * H * W;
    const int block = 256;
    const long grid = (total + block - 1) / block;
    normalize_nhwc_to_nchw_kernel<<<(unsigned)grid, block, 0, S.stream>>>(S.d_in, outp, N, H, W, m,
                                                                          isd);
    cudaError_t err = cudaStreamSynchronize(S.stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader cuda] stream normalize: %s\n", cudaGetErrorString(err));
        return 0;
    }
    float* result = outp;
    S.out_idx = (S.out_idx + 1) % StreamSlot::SRING;
    return reinterpret_cast<uintptr_t>(result);
}

// ---- Fully-in-C++ streaming loader: persistent worker pool + double-buffered output pool ----
// The whole iteration runs GIL-free: K worker threads each pull a batch, async-H2D it on their
// own stream, normalize into a free output buffer, and enqueue it; next_batch() pops the next
// ready buffer for Python. Only one Python call per batch, so the GIL is not the bottleneck.
struct CudaStreamCore::Impl {
    const uint8_t* host = nullptr;
    int N = 0, H = 0, W = 0, batch = 0;
    float mean[3] = {0, 0, 0}, istd[3] = {0, 0, 0};
    bool drop_last = true;
    int K = 1;
    size_t stride = 0;   // H*W*3 bytes/image
    size_t out_per = 0;  // 3*H*W floats/image
    int total = 0;       // batches/epoch
    int n_bufs = 0;

    std::vector<cudaStream_t> stream;
    std::vector<uint8_t*> d_in;  // K device uint8 scratch (one per worker)
    std::vector<float*> buf;     // n_bufs output device buffers
    std::vector<std::thread> workers;

    std::mutex mu;
    std::condition_variable cv;
    std::queue<int> work;                   // batch indices to process
    std::queue<int> freeb;                  // free output-buffer indices
    std::queue<std::pair<int, int>> ready;  // (buf_idx, n_images)
    bool stop = false;
    int consumed = 0;
    int prev_buf = -1;

    void worker(int wi) {
        const float3 m = make_float3(mean[0], mean[1], mean[2]);
        const float3 isd = make_float3(istd[0], istd[1], istd[2]);
        for (;;) {
            int b, bi;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !work.empty(); });
                if (stop) return;
                b = work.front();
                work.pop();
            }
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !freeb.empty(); });
                if (stop) return;
                bi = freeb.front();
                freeb.pop();
            }
            const int n = std::min(batch, N - b * batch);
            cudaMemcpyAsync(d_in[wi], host + (size_t)b * batch * stride, (size_t)n * stride,
                            cudaMemcpyHostToDevice, stream[wi]);
            const long tot = (long)n * 3 * H * W;
            const int block = 256;
            const long grid = (tot + block - 1) / block;
            normalize_nhwc_to_nchw_kernel<<<(unsigned)grid, block, 0, stream[wi]>>>(d_in[wi], buf[bi],
                                                                                    n, H, W, m, isd);
            cudaStreamSynchronize(stream[wi]);
            {
                std::lock_guard<std::mutex> lk(mu);
                ready.push({bi, n});
            }
            cv.notify_all();
        }
    }
};

CudaStreamCore::CudaStreamCore(uintptr_t host_ptr, int n, int h, int w, int batch,
                               const float mean[3], const float std_[3], int num_slots,
                               bool drop_last) {
    impl_ = new Impl();
    Impl& I = *impl_;
    I.host = reinterpret_cast<const uint8_t*>(host_ptr);
    I.N = n;
    I.H = h;
    I.W = w;
    I.batch = batch;
    I.drop_last = drop_last;
    for (int c = 0; c < 3; c++) {
        I.mean[c] = mean[c];
        I.istd[c] = 1.0f / std_[c];
    }
    I.K = std::max(1, num_slots);
    I.stride = (size_t)h * w * 3;
    I.out_per = (size_t)3 * h * w;
    I.total = drop_last ? (n / batch) : ((n + batch - 1) / batch);
    if (!available() || !host_ptr || n <= 0 || batch <= 0) return;  // no workers -> next_batch()=0
    I.stream.resize(I.K, nullptr);
    I.d_in.resize(I.K, nullptr);
    for (int i = 0; i < I.K; i++) {
        // Non-blocking: worker streams must NOT implicitly serialize with the default (NULL)
        // stream, or the consumer's torch ops (default stream) would barrier against all workers.
        if (cudaStreamCreateWithFlags(&I.stream[i], cudaStreamNonBlocking) != cudaSuccess) return;
        if (cudaMalloc(&I.d_in[i], (size_t)batch * I.stride) != cudaSuccess) return;
    }
    I.n_bufs = I.K + 4;  // K in flight + a few queued + 1 held by the consumer
    I.buf.resize(I.n_bufs, nullptr);
    for (int j = 0; j < I.n_bufs; j++)
        if (cudaMalloc(&I.buf[j], (size_t)batch * I.out_per * sizeof(float)) != cudaSuccess) return;
    for (int i = 0; i < I.K; i++) I.workers.emplace_back([&I, i] { I.worker(i); });
}

CudaStreamCore::~CudaStreamCore() {
    Impl& I = *impl_;
    {
        std::lock_guard<std::mutex> lk(I.mu);
        I.stop = true;
    }
    I.cv.notify_all();
    for (auto& t : I.workers)
        if (t.joinable()) t.join();
    for (auto s : I.stream)
        if (s) cudaStreamDestroy(s);
    for (auto p : I.d_in)
        if (p) cudaFree(p);
    for (auto p : I.buf)
        if (p) cudaFree(p);
    delete impl_;
}

int CudaStreamCore::num_batches() const { return impl_->total; }

void CudaStreamCore::begin_epoch() {
    Impl& I = *impl_;
    if (I.workers.empty()) return;
    std::lock_guard<std::mutex> lk(I.mu);
    std::queue<int>().swap(I.work);
    std::queue<std::pair<int, int>>().swap(I.ready);
    std::queue<int>().swap(I.freeb);
    for (int j = 0; j < I.n_bufs; j++) I.freeb.push(j);
    for (int b = 0; b < I.total; b++) I.work.push(b);
    I.consumed = 0;
    I.prev_buf = -1;
    I.cv.notify_all();
}

uintptr_t CudaStreamCore::next_batch(int* out_n) {
    Impl& I = *impl_;
    if (I.workers.empty()) return 0;
    std::unique_lock<std::mutex> lk(I.mu);
    if (I.prev_buf >= 0) {
        I.freeb.push(I.prev_buf);
        I.prev_buf = -1;
        lk.unlock();
        I.cv.notify_all();
        lk.lock();
    }
    if (I.consumed >= I.total) return 0;
    I.cv.wait(lk, [&] { return I.stop || !I.ready.empty(); });
    if (I.stop || I.ready.empty()) return 0;
    auto pr = I.ready.front();
    I.ready.pop();
    I.consumed++;
    I.prev_buf = pr.first;
    if (out_n) *out_n = pr.second;
    return reinterpret_cast<uintptr_t>(I.buf[pr.first]);
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
static bool g_nv_ready = false;
static std::mutex g_nv_init_mutex;  // guards init only

// One independent pipeline slot: its own decoder + CUDA stream + reused per-image handles +
// persistent device input buffers + output ring. K slots, each driven by ONE Python thread, let
// the host Huffman-decode of one batch overlap the GPU work of another (DALI-style multi-batch-
// in-flight). A slot is single-threaded, so its state needs no per-call locking; its mutex only
// guards against accidental concurrent reuse. Output ring depth must be >= the loader's out_q
// maxsize + 2, so a queued-but-unconsumed output is never overwritten by the slot's later calls.
static const int NV_OUT_RING = 8;
struct NvSlot {
    nvimgcodecDecoder_t decoder = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<nvimgcodecCodeStream_t> cs;   // reused code-stream handles (one per image)
    std::vector<nvimgcodecImage_t> img;       // reused output image handles
    std::vector<uint8_t*> inbuf;              // persistent device decode buffers (grown on demand)
    std::vector<size_t> incap;
    float* out_pool[NV_OUT_RING] = {nullptr};
    size_t out_cap[NV_OUT_RING] = {0};
    int out_idx = 0;
    std::mutex mu;
};
static std::vector<std::unique_ptr<NvSlot>> g_slots;

template <typename T>
static bool load_sym(void* h, T& fn, const char* name) {
    fn = reinterpret_cast<T>(dlsym(h, name));
    if (!fn) fprintf(stderr, "[turboloader nvimgcodec] missing symbol %s\n", name);
    return fn != nullptr;
}

}  // namespace

bool nvimgcodec_pipeline_init(int num_slots, const char* lib_path, const char* ext_path,
                              int device_id) {
    std::lock_guard<std::mutex> lk(g_nv_init_mutex);
    if (num_slots < 1) num_slots = 1;
    if (g_nv_ready) return (int)g_slots.size() >= num_slots;  // already up (idempotent)
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
    // Split host decode threads across the slots so K decoders total ~= the core count rather
    // than K * cores (which would thrash). This mirrors DALI's num_threads budget.
    unsigned hc = std::thread::hardware_concurrency();
    if (hc < 2) hc = 2;
    int per = (int)(hc / (unsigned)num_slots);
    if (per < 2) per = 2;
    for (int s = 0; s < num_slots; s++) {
        auto slot = std::make_unique<NvSlot>();
        nvimgcodecExecutionParams_t ep{};
        ep.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
        ep.struct_size = sizeof(ep);
        ep.device_id = device_id;  // NVIMGCODEC_DEVICE_CURRENT = -1
        ep.max_num_cpu_threads = per;
        ep.pre_init = 1;
        ep.num_backends = 0;  // all backends allowed (HW + GPU-hybrid + CPU fallback)
        ep.backends = nullptr;
        if (nv_DecoderCreate(g_nv_instance, &slot->decoder, &ep, nullptr) !=
            NVIMGCODEC_STATUS_SUCCESS) {
            fprintf(stderr, "[turboloader nvimgcodec] DecoderCreate(slot %d) failed\n", s);
            return false;
        }
        if (cudaStreamCreate(&slot->stream) != cudaSuccess) return false;
        g_slots.push_back(std::move(slot));
    }
    g_nv_ready = true;
    return true;
}

int nvimgcodec_num_slots() { return (int)g_slots.size(); }

uintptr_t nvimgcodec_decode_resize_normalize_slot(int slot,
                                                  const std::vector<const uint8_t*>& jpegs,
                                                  const std::vector<size_t>& sizes, int dst_h,
                                                  int dst_w, const float mean[3],
                                                  const float std_[3]) {
    if (!g_nv_ready || slot < 0 || slot >= (int)g_slots.size() || jpegs.empty() ||
        jpegs.size() != sizes.size() || dst_h <= 0 || dst_w <= 0)
        return 0;
    NvSlot& S = *g_slots[slot];
    std::lock_guard<std::mutex> lk(S.mu);  // single-thread-per-slot; guards accidental reuse
    const int N = (int)jpegs.size();
    if ((int)S.cs.size() < N) {
        S.cs.resize(N, nullptr);
        S.img.resize(N, nullptr);
        S.inbuf.resize(N, nullptr);
        S.incap.resize(N, 0);
    }
    std::vector<int> ws(N), hs(N);

    nvimgcodecDecodeParams_t dp{};
    dp.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    dp.struct_size = sizeof(dp);
    dp.apply_exif_orientation = 1;

    // Per-image setup (serial on this slot's thread): parse header, size device buffer, wrap
    // input+output handles. Decode onto S.stream so the kernel (same stream) is ordered after it.
    for (int i = 0; i < N; i++) {
        if (nv_CodeStreamFromHostMem(g_nv_instance, &S.cs[i], jpegs[i], sizes[i], nullptr) !=
            NVIMGCODEC_STATUS_SUCCESS)
            return 0;
        nvimgcodecImageInfo_t si{};
        si.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        si.struct_size = sizeof(si);
        if (nv_CodeStreamGetImageInfo(S.cs[i], &si) != NVIMGCODEC_STATUS_SUCCESS) return 0;
        const int W = (int)si.plane_info[0].width, H = (int)si.plane_info[0].height;
        if (W <= 0 || H <= 0) return 0;
        ws[i] = W;
        hs[i] = H;
        const size_t need = (size_t)W * H * 3;
        if (need > S.incap[i]) {
            if (S.inbuf[i]) cudaFree(S.inbuf[i]);
            if (cudaMalloc(&S.inbuf[i], need) != cudaSuccess) {
                S.incap[i] = 0;
                return 0;
            }
            S.incap[i] = need;
        }
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
        oi.buffer = S.inbuf[i];
        oi.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        oi.cuda_stream = S.stream;
        if (nv_ImageCreate(g_nv_instance, &S.img[i], &oi) != NVIMGCODEC_STATUS_SUCCESS) return 0;
    }

    nvimgcodecFuture_t fut = nullptr;
    if (nv_DecoderDecode(S.decoder, S.cs.data(), S.img.data(), N, &dp, &fut) !=
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
    float*& outp = S.out_pool[S.out_idx];
    const size_t out_bytes = (size_t)N * per_out * sizeof(float);
    if (out_bytes > S.out_cap[S.out_idx]) {
        if (outp) cudaFree(outp);
        if (cudaMalloc(&outp, out_bytes) != cudaSuccess) {
            S.out_cap[S.out_idx] = 0;
            return 0;
        }
        S.out_cap[S.out_idx] = out_bytes;
    }
    const float3 m = make_float3(mean[0], mean[1], mean[2]);
    const float3 isd = make_float3(1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]);
    dim3 block(16, 16), grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    for (int i = 0; i < N; i++)
        resize_normalize_kernel<<<grid, block, 0, S.stream>>>(
            S.inbuf[i], outp + (size_t)i * per_out, ws[i], hs[i], dst_w, dst_h, m, isd);
    // Sync THIS slot's stream so the returned pointer is fully ready (no async-handoff race).
    // Concurrency comes from K slots overlapping, not from skipping the sync.
    cudaError_t err = cudaStreamSynchronize(S.stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turboloader nvimgcodec] kernel: %s\n", cudaGetErrorString(err));
        return 0;
    }
    float* result = outp;
    S.out_idx = (S.out_idx + 1) % NV_OUT_RING;
    return reinterpret_cast<uintptr_t>(result);
}

uintptr_t nvimgcodec_decode_resize_normalize(const std::vector<const uint8_t*>& jpegs,
                                             const std::vector<size_t>& sizes, int dst_h,
                                             int dst_w, const float mean[3], const float std_[3]) {
    return nvimgcodec_decode_resize_normalize_slot(0, jpegs, sizes, dst_h, dst_w, mean, std_);
}
#else
bool nvimgcodec_pipeline_init(int, const char*, const char*, int) { return false; }
int nvimgcodec_num_slots() { return 0; }
uintptr_t nvimgcodec_decode_resize_normalize_slot(int, const std::vector<const uint8_t*>&,
                                                  const std::vector<size_t>&, int, int,
                                                  const float[3], const float[3]) {
    return 0;  // built without nvImageCodec
}
uintptr_t nvimgcodec_decode_resize_normalize(const std::vector<const uint8_t*>&,
                                             const std::vector<size_t>&, int, int, const float[3],
                                             const float[3]) {
    return 0;  // built without nvImageCodec
}
#endif

}  // namespace cuda
}  // namespace turboloader
