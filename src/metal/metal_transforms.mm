// Metal (Apple GPU) implementation of the GPU transform path. Compiled as Objective-C++
// only on macOS arm64 (see setup.py). The compute kernel is compiled at runtime from
// source (newLibraryWithSource) so no offline `metal` toolchain / full Xcode is required.
//
// Validated bit-exact vs the CPU bilinear (experiments/metal/): the half-pixel sampling
// here mirrors the CPU convention exactly.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_transforms.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Per-image bilinear resize + per-channel normalize. HWC uint8 -> CHW float32.
const char* kKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void resize_normalize(
    device const uchar* src    [[buffer(0)]],
    device float*       dst    [[buffer(1)]],
    constant uint4&     dims   [[buffer(2)]],   // srcW, srcH, dstW, dstH
    constant float3&    mean   [[buffer(3)]],
    constant float3&    invstd [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint srcW = dims.x, srcH = dims.y, dstW = dims.z, dstH = dims.w;
    if (gid.x >= dstW || gid.y >= dstH) return;
    float sx = max(0.0f, (float(gid.x) + 0.5f) * float(srcW) / float(dstW) - 0.5f);
    float sy = max(0.0f, (float(gid.y) + 0.5f) * float(srcH) / float(dstH) - 0.5f);
    uint x0 = uint(sx), y0 = uint(sy);
    uint x1 = min(x0 + 1u, srcW - 1u), y1 = min(y0 + 1u, srcH - 1u);
    float dx = sx - float(x0), dy = sy - float(y0);
    float m[3]   = { mean.x, mean.y, mean.z };
    float isd[3] = { invstd.x, invstd.y, invstd.z };
    for (uint c = 0; c < 3; c++) {
        float p00 = src[(y0 * srcW + x0) * 3 + c], p10 = src[(y0 * srcW + x1) * 3 + c];
        float p01 = src[(y1 * srcW + x0) * 3 + c], p11 = src[(y1 * srcW + x1) * 3 + c];
        float v = mix(mix(p00, p10, dx), mix(p01, p11, dx), dy) / 255.0f;
        dst[(c * dstH + gid.y) * dstW + gid.x] = (v - m[c]) * isd[c];
    }
}

// Fused RandomResizedCrop + horizontal-flip + normalize. The crop window (float pixels)
// and flip are chosen per-image by the caller; this samples that window with half-pixel
// bilinear into the (dstW,dstH) output. The canonical train-time pipeline in one pass.
kernel void crop_resize_normalize(
    device const uchar* src    [[buffer(0)]],
    device float*       dst    [[buffer(1)]],
    constant uint4&     dims   [[buffer(2)]],   // srcW, srcH, dstW, dstH
    constant float4&    crop   [[buffer(3)]],   // cropX, cropY, cropW, cropH (src pixels)
    constant uint&      flip   [[buffer(4)]],   // 1 = horizontal flip
    constant float3&    mean   [[buffer(5)]],
    constant float3&    invstd [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint srcW = dims.x, srcH = dims.y, dstW = dims.z, dstH = dims.w;
    if (gid.x >= dstW || gid.y >= dstH) return;
    uint ox = (flip != 0u) ? (dstW - 1u - gid.x) : gid.x;   // flip the sampled column
    float sx = crop.x + (float(ox) + 0.5f) / float(dstW) * crop.z - 0.5f;
    float sy = crop.y + (float(gid.y) + 0.5f) / float(dstH) * crop.w - 0.5f;
    sx = clamp(sx, 0.0f, float(srcW - 1u));
    sy = clamp(sy, 0.0f, float(srcH - 1u));
    uint x0 = uint(sx), y0 = uint(sy);
    uint x1 = min(x0 + 1u, srcW - 1u), y1 = min(y0 + 1u, srcH - 1u);
    float dx = sx - float(x0), dy = sy - float(y0);
    float m[3]   = { mean.x, mean.y, mean.z };
    float isd[3] = { invstd.x, invstd.y, invstd.z };
    for (uint c = 0; c < 3; c++) {
        float p00 = src[(y0 * srcW + x0) * 3 + c], p10 = src[(y0 * srcW + x1) * 3 + c];
        float p01 = src[(y1 * srcW + x0) * 3 + c], p11 = src[(y1 * srcW + x1) * 3 + c];
        float v = mix(mix(p00, p10, dx), mix(p01, p11, dx), dy) / 255.0f;
        dst[(c * dstH + gid.y) * dstW + gid.x] = (v - m[c]) * isd[c];
    }
}

// Full train pipeline: crop + resize + hflip + color jitter + normalize, one pass.
kernel void train_transform(
    device const uchar* src    [[buffer(0)]],
    device float*       dst    [[buffer(1)]],
    constant uint4&     dims   [[buffer(2)]],
    constant float4&    crop   [[buffer(3)]],
    constant uint&      flip   [[buffer(4)]],
    constant float3&    jit    [[buffer(5)]],   // brightness, contrast, saturation
    constant float3&    mean   [[buffer(6)]],
    constant float3&    invstd [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint srcW = dims.x, srcH = dims.y, dstW = dims.z, dstH = dims.w;
    if (gid.x >= dstW || gid.y >= dstH) return;
    uint ox = (flip != 0u) ? (dstW - 1u - gid.x) : gid.x;
    float sx = crop.x + (float(ox) + 0.5f) / float(dstW) * crop.z - 0.5f;
    float sy = crop.y + (float(gid.y) + 0.5f) / float(dstH) * crop.w - 0.5f;
    sx = clamp(sx, 0.0f, float(srcW - 1u));
    sy = clamp(sy, 0.0f, float(srcH - 1u));
    uint x0 = uint(sx), y0 = uint(sy), x1 = min(x0 + 1u, srcW - 1u), y1 = min(y0 + 1u, srcH - 1u);
    float dx = sx - float(x0), dy = sy - float(y0);
    float3 rgb;
    for (uint c = 0; c < 3; c++) {
        float p00 = src[(y0 * srcW + x0) * 3 + c], p10 = src[(y0 * srcW + x1) * 3 + c];
        float p01 = src[(y1 * srcW + x0) * 3 + c], p11 = src[(y1 * srcW + x1) * 3 + c];
        rgb[c] = mix(mix(p00, p10, dx), mix(p01, p11, dx), dy) / 255.0f;
    }
    rgb *= jit.x;                                              // brightness
    rgb = (rgb - 0.5f) * jit.y + 0.5f;                         // contrast (around mid-gray)
    float lum = dot(rgb, float3(0.299f, 0.587f, 0.114f));      // saturation
    rgb = mix(float3(lum), rgb, jit.z);
    rgb = clamp(rgb, 0.0f, 1.0f);
    float m[3] = {mean.x, mean.y, mean.z}, isd[3] = {invstd.x, invstd.y, invstd.z};
    for (uint c = 0; c < 3; c++) dst[(c * dstH + gid.y) * dstW + gid.x] = (rgb[c] - m[c]) * isd[c];
}

// Resident-dataset epoch kernel: gather sample idx[gid.z] from the NHWC uint8
// dataset and write it normalized as CHW float32 — shuffle + normalize + layout
// conversion for a WHOLE batch in one launch. ulong offsets: datasets > 4 GB.
kernel void gather_normalize_u8(
    device const uchar* src    [[buffer(0)]],   // resident N x H x W x 3
    device const int*   idx    [[buffer(1)]],   // B sample indices
    device float*       dst    [[buffer(2)]],   // B x 3 x H x W
    constant uint4&     dims   [[buffer(3)]],   // W, H, B, unused
    constant float3&    mean   [[buffer(4)]],
    constant float3&    invstd [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint W = dims.x, H = dims.y, B = dims.z;
    if (gid.x >= W || gid.y >= H || gid.z >= B) return;
    ulong pix = (ulong)uint(idx[gid.z]) * H * W + (ulong)gid.y * W + gid.x;
    device const uchar* p = src + pix * 3;
    float m[3]   = { mean.x, mean.y, mean.z };
    float isd[3] = { invstd.x, invstd.y, invstd.z };
    for (uint c = 0; c < 3; c++) {
        float v = float(p[c]) / 255.0f;
        dst[(((ulong)gid.z * 3 + c) * H + gid.y) * W + gid.x] = (v - m[c]) * isd[c];
    }
}

// Dtype-agnostic span gather: dst[b] = src[offs[b] : offs[b]+span] for a whole
// batch in one launch. Byte offsets (not row indices) so overlapping token
// windows and aligned row gathers use the same kernel. 16 bytes per thread.
kernel void gather_spans(
    device const uchar* src  [[buffer(0)]],
    device const ulong* offs [[buffer(1)]],   // B byte offsets into src
    device uchar*       dst  [[buffer(2)]],   // B x span_bytes, packed
    constant uint2&     dims [[buffer(3)]],   // span_bytes, B
    uint2 gid [[thread_position_in_grid]])
{
    uint span = dims.x, B = dims.y;
    uint start = gid.x * 16u;
    if (gid.y >= B || start >= span) return;
    uint n = min(16u, span - start);
    device const uchar* s = src + offs[gid.y] + start;
    device uchar*       d = dst + (ulong)gid.y * span + start;
    for (uint k = 0; k < n; k++) d[k] = s[k];
}
)";

// Process-lifetime singletons (ARC keeps file-scope strong globals retained).
id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pso = nil;         // resize_normalize
id<MTLComputePipelineState> g_pso_crop = nil;    // crop_resize_normalize
id<MTLComputePipelineState> g_pso_train = nil;   // train_transform
id<MTLComputePipelineState> g_pso_gather = nil;  // gather_normalize_u8
id<MTLComputePipelineState> g_pso_spans = nil;   // gather_spans
std::string g_name;
bool g_ok = false;

// Resident-dataset registry. Handles index into nullable slots; a mutex guards
// slot assignment/lookup (the gathers themselves serialize on the MTLQueue).
struct ResidentImages {
    id<MTLBuffer> data;    // N x H x W x 3 uint8
    id<MTLBuffer> idx;     // max_batch int32
    id<MTLBuffer> out[2];  // double-buffered: max_batch x 3 x H x W float
    size_t n = 0, max_batch = 0;
    int h = 0, w = 0, cur = 0;
};
struct ResidentBytes {
    id<MTLBuffer> data;    // total_bytes
    id<MTLBuffer> offs;    // max_batch uint64 byte offsets
    id<MTLBuffer> out[2];  // double-buffered: max_batch x max_span bytes
    size_t total = 0, max_batch = 0, max_span = 0;
    int cur = 0;
};
std::mutex g_res_mu;
std::vector<ResidentImages*> g_res_imgs;
std::vector<ResidentBytes*> g_res_bytes;

template <class T>
int registry_insert(std::vector<T*>& reg, T* r) {
    std::lock_guard<std::mutex> lk(g_res_mu);
    for (size_t i = 0; i < reg.size(); ++i)
        if (!reg[i]) {
            reg[i] = r;
            return (int)i;
        }
    reg.push_back(r);
    return (int)reg.size() - 1;
}

template <class T>
T* registry_get(std::vector<T*>& reg, int handle) {
    std::lock_guard<std::mutex> lk(g_res_mu);
    if (handle < 0 || (size_t)handle >= reg.size()) return nullptr;
    return reg[handle];
}

template <class T>
void registry_erase(std::vector<T*>& reg, int handle) {
    T* r = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_res_mu);
        if (handle < 0 || (size_t)handle >= reg.size()) return;
        r = reg[handle];
        reg[handle] = nullptr;
    }
    delete r;  // ARC releases the MTLBuffers with the struct
}

void init_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        @autoreleasepool {
            g_dev = MTLCreateSystemDefaultDevice();
            if (!g_dev) return;
            NSError* err = nil;
            id<MTLLibrary> lib =
                [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSrc]
                                    options:nil
                                      error:&err];
            if (!lib) return;
            id<MTLFunction> fn = [lib newFunctionWithName:@"resize_normalize"];
            g_pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso) return;
            id<MTLFunction> fn_crop = [lib newFunctionWithName:@"crop_resize_normalize"];
            g_pso_crop = [g_dev newComputePipelineStateWithFunction:fn_crop error:&err];
            if (!g_pso_crop) return;
            id<MTLFunction> fn_train = [lib newFunctionWithName:@"train_transform"];
            g_pso_train = [g_dev newComputePipelineStateWithFunction:fn_train error:&err];
            if (!g_pso_train) return;
            id<MTLFunction> fn_gather = [lib newFunctionWithName:@"gather_normalize_u8"];
            g_pso_gather = [g_dev newComputePipelineStateWithFunction:fn_gather error:&err];
            if (!g_pso_gather) return;
            id<MTLFunction> fn_spans = [lib newFunctionWithName:@"gather_spans"];
            g_pso_spans = [g_dev newComputePipelineStateWithFunction:fn_spans error:&err];
            if (!g_pso_spans) return;
            g_queue = [g_dev newCommandQueue];
            g_name = g_dev.name ? g_dev.name.UTF8String : "Apple GPU";
            g_ok = (g_queue != nil);
        }
    });
}

}  // namespace

namespace turboloader {
namespace metal {

bool available() {
    init_once();
    return g_ok;
}

const char* device_name() {
    init_once();
    return g_ok ? g_name.c_str() : "";
}

bool resize_normalize_batch(const std::vector<ImageRef>& imgs, int dst_h, int dst_w,
                            const float mean[3], const float std_[3], float* out) {
    init_once();
    if (!g_ok || imgs.empty() || dst_h <= 0 || dst_w <= 0) return false;

    @autoreleasepool {
        const size_t N = imgs.size();
        const size_t per_out = (size_t)3 * dst_h * dst_w;  // floats per image

        // Pack all source images into one shared buffer (unified memory: ~free).
        std::vector<size_t> off(N + 1, 0);
        for (size_t i = 0; i < N; i++) {
            if (!imgs[i].data || imgs[i].w <= 0 || imgs[i].h <= 0) return false;
            off[i + 1] = off[i] + (size_t)imgs[i].w * imgs[i].h * 3;
        }
        id<MTLBuffer> bsrc = [g_dev newBufferWithLength:std::max<size_t>(off[N], 1)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bdst = [g_dev newBufferWithLength:N * per_out * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (!bsrc || !bdst) return false;
        for (size_t i = 0; i < N; i++) {
            std::memcpy((uint8_t*)bsrc.contents + off[i], imgs[i].data, off[i + 1] - off[i]);
        }

        const float invstd[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso];
        for (size_t i = 0; i < N; i++) {
            uint32_t dims[4] = {(uint32_t)imgs[i].w, (uint32_t)imgs[i].h, (uint32_t)dst_w,
                                (uint32_t)dst_h};
            [e setBuffer:bsrc offset:off[i] atIndex:0];
            [e setBuffer:bdst offset:i * per_out * sizeof(float) atIndex:1];
            [e setBytes:dims length:sizeof(dims) atIndex:2];
            [e setBytes:mean length:3 * sizeof(float) atIndex:3];
            [e setBytes:invstd length:3 * sizeof(float) atIndex:4];
            [e dispatchThreads:MTLSizeMake(dst_w, dst_h, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return false;
        std::memcpy(out, bdst.contents, N * per_out * sizeof(float));
    }
    return true;
}

bool crop_resize_normalize_batch(const std::vector<ImageRef>& imgs,
                                 const std::vector<CropParams>& crops, int dst_h, int dst_w,
                                 const float mean[3], const float std_[3], float* out) {
    init_once();
    if (!g_ok || imgs.empty() || dst_h <= 0 || dst_w <= 0 || crops.size() != imgs.size())
        return false;

    @autoreleasepool {
        const size_t N = imgs.size();
        const size_t per_out = (size_t)3 * dst_h * dst_w;
        std::vector<size_t> off(N + 1, 0);
        for (size_t i = 0; i < N; i++) {
            if (!imgs[i].data || imgs[i].w <= 0 || imgs[i].h <= 0) return false;
            off[i + 1] = off[i] + (size_t)imgs[i].w * imgs[i].h * 3;
        }
        id<MTLBuffer> bsrc = [g_dev newBufferWithLength:std::max<size_t>(off[N], 1)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bdst = [g_dev newBufferWithLength:N * per_out * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (!bsrc || !bdst) return false;
        for (size_t i = 0; i < N; i++)
            std::memcpy((uint8_t*)bsrc.contents + off[i], imgs[i].data, off[i + 1] - off[i]);

        const float invstd[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso_crop];
        for (size_t i = 0; i < N; i++) {
            uint32_t dims[4] = {(uint32_t)imgs[i].w, (uint32_t)imgs[i].h, (uint32_t)dst_w,
                                (uint32_t)dst_h};
            float crop[4] = {crops[i].x, crops[i].y, crops[i].w, crops[i].h};
            uint32_t flip = crops[i].flip ? 1u : 0u;
            [e setBuffer:bsrc offset:off[i] atIndex:0];
            [e setBuffer:bdst offset:i * per_out * sizeof(float) atIndex:1];
            [e setBytes:dims length:sizeof(dims) atIndex:2];
            [e setBytes:crop length:sizeof(crop) atIndex:3];
            [e setBytes:&flip length:sizeof(flip) atIndex:4];
            [e setBytes:mean length:3 * sizeof(float) atIndex:5];
            [e setBytes:invstd length:3 * sizeof(float) atIndex:6];
            [e dispatchThreads:MTLSizeMake(dst_w, dst_h, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return false;
        std::memcpy(out, bdst.contents, N * per_out * sizeof(float));
    }
    return true;
}

bool train_transform_batch(const std::vector<ImageRef>& imgs,
                           const std::vector<CropParams>& crops,
                           const std::vector<JitterParams>& jitter, int dst_h, int dst_w,
                           const float mean[3], const float std_[3], float* out) {
    init_once();
    if (!g_ok || imgs.empty() || dst_h <= 0 || dst_w <= 0 || crops.size() != imgs.size() ||
        jitter.size() != imgs.size())
        return false;

    @autoreleasepool {
        const size_t N = imgs.size();
        const size_t per_out = (size_t)3 * dst_h * dst_w;
        std::vector<size_t> off(N + 1, 0);
        for (size_t i = 0; i < N; i++) {
            if (!imgs[i].data || imgs[i].w <= 0 || imgs[i].h <= 0) return false;
            off[i + 1] = off[i] + (size_t)imgs[i].w * imgs[i].h * 3;
        }
        id<MTLBuffer> bsrc = [g_dev newBufferWithLength:std::max<size_t>(off[N], 1)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bdst = [g_dev newBufferWithLength:N * per_out * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        if (!bsrc || !bdst) return false;
        for (size_t i = 0; i < N; i++)
            std::memcpy((uint8_t*)bsrc.contents + off[i], imgs[i].data, off[i + 1] - off[i]);

        const float invstd[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso_train];
        for (size_t i = 0; i < N; i++) {
            uint32_t dims[4] = {(uint32_t)imgs[i].w, (uint32_t)imgs[i].h, (uint32_t)dst_w,
                                (uint32_t)dst_h};
            float crop[4] = {crops[i].x, crops[i].y, crops[i].w, crops[i].h};
            uint32_t flip = crops[i].flip ? 1u : 0u;
            float jit[3] = {jitter[i].brightness, jitter[i].contrast, jitter[i].saturation};
            [e setBuffer:bsrc offset:off[i] atIndex:0];
            [e setBuffer:bdst offset:i * per_out * sizeof(float) atIndex:1];
            [e setBytes:dims length:sizeof(dims) atIndex:2];
            [e setBytes:crop length:sizeof(crop) atIndex:3];
            [e setBytes:&flip length:sizeof(flip) atIndex:4];
            [e setBytes:jit length:sizeof(jit) atIndex:5];
            [e setBytes:mean length:3 * sizeof(float) atIndex:6];
            [e setBytes:invstd length:3 * sizeof(float) atIndex:7];
            [e dispatchThreads:MTLSizeMake(dst_w, dst_h, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return false;
        std::memcpy(out, bdst.contents, N * per_out * sizeof(float));
    }
    return true;
}

// --------------------------- resident images -------------------------------

int resident_images_create(size_t n, int h, int w, size_t max_batch) {
    init_once();
    if (!g_ok || n == 0 || h <= 0 || w <= 0 || max_batch == 0) return -1;
    @autoreleasepool {
        auto* r = new ResidentImages();
        const size_t data_bytes = n * (size_t)h * w * 3;
        const size_t out_bytes = max_batch * (size_t)3 * h * w * sizeof(float);
        r->data = [g_dev newBufferWithLength:data_bytes options:MTLResourceStorageModeShared];
        r->idx = [g_dev newBufferWithLength:max_batch * sizeof(int32_t)
                                    options:MTLResourceStorageModeShared];
        r->out[0] = [g_dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        r->out[1] = [g_dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (!r->data || !r->idx || !r->out[0] || !r->out[1]) {
            delete r;
            return -1;
        }
        r->n = n;
        r->h = h;
        r->w = w;
        r->max_batch = max_batch;
        return registry_insert(g_res_imgs, r);
    }
}

uint8_t* resident_images_data(int handle) {
    ResidentImages* r = registry_get(g_res_imgs, handle);
    return r ? (uint8_t*)r->data.contents : nullptr;
}

const float* resident_images_gather(int handle, const int32_t* idx, size_t b,
                                    const float mean[3], const float std_[3]) {
    ResidentImages* r = registry_get(g_res_imgs, handle);
    if (!r || !idx || b == 0 || b > r->max_batch) return nullptr;
    for (size_t i = 0; i < b; i++)  // a bad index would be a silent wild read on GPU
        if (idx[i] < 0 || (size_t)idx[i] >= r->n) return nullptr;
    @autoreleasepool {
        std::memcpy(r->idx.contents, idx, b * sizeof(int32_t));
        const float invstd[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
        r->cur ^= 1;  // double buffer: previous batch stays valid during this one
        id<MTLBuffer> bout = r->out[r->cur];
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso_gather];
        uint32_t dims[4] = {(uint32_t)r->w, (uint32_t)r->h, (uint32_t)b, 0};
        [e setBuffer:r->data offset:0 atIndex:0];
        [e setBuffer:r->idx offset:0 atIndex:1];
        [e setBuffer:bout offset:0 atIndex:2];
        [e setBytes:dims length:sizeof(dims) atIndex:3];
        [e setBytes:mean length:3 * sizeof(float) atIndex:4];
        [e setBytes:invstd length:3 * sizeof(float) atIndex:5];
        [e dispatchThreads:MTLSizeMake(r->w, r->h, b)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return nullptr;
        return (const float*)bout.contents;
    }
}

void resident_images_destroy(int handle) { registry_erase(g_res_imgs, handle); }

// ---------------------------- resident bytes -------------------------------

int resident_bytes_create(size_t total_bytes, size_t max_batch, size_t max_span_bytes) {
    init_once();
    if (!g_ok || total_bytes == 0 || max_batch == 0 || max_span_bytes == 0) return -1;
    @autoreleasepool {
        auto* r = new ResidentBytes();
        r->data = [g_dev newBufferWithLength:total_bytes options:MTLResourceStorageModeShared];
        r->offs = [g_dev newBufferWithLength:max_batch * sizeof(uint64_t)
                                     options:MTLResourceStorageModeShared];
        const size_t out_bytes = max_batch * max_span_bytes;
        r->out[0] = [g_dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        r->out[1] = [g_dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (!r->data || !r->offs || !r->out[0] || !r->out[1]) {
            delete r;
            return -1;
        }
        r->total = total_bytes;
        r->max_batch = max_batch;
        r->max_span = max_span_bytes;
        return registry_insert(g_res_bytes, r);
    }
}

uint8_t* resident_bytes_data(int handle) {
    ResidentBytes* r = registry_get(g_res_bytes, handle);
    return r ? (uint8_t*)r->data.contents : nullptr;
}

const uint8_t* resident_bytes_gather(int handle, const uint64_t* offs_bytes, size_t b,
                                     size_t span_bytes) {
    ResidentBytes* r = registry_get(g_res_bytes, handle);
    if (!r || !offs_bytes || b == 0 || b > r->max_batch || span_bytes == 0 ||
        span_bytes > r->max_span)
        return nullptr;
    for (size_t i = 0; i < b; i++)
        if (offs_bytes[i] + span_bytes > r->total) return nullptr;
    @autoreleasepool {
        std::memcpy(r->offs.contents, offs_bytes, b * sizeof(uint64_t));
        r->cur ^= 1;
        id<MTLBuffer> bout = r->out[r->cur];
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso_spans];
        uint32_t dims[2] = {(uint32_t)span_bytes, (uint32_t)b};
        [e setBuffer:r->data offset:0 atIndex:0];
        [e setBuffer:r->offs offset:0 atIndex:1];
        [e setBuffer:bout offset:0 atIndex:2];
        [e setBytes:dims length:sizeof(dims) atIndex:3];
        const size_t chunks = (span_bytes + 15) / 16;
        [e dispatchThreads:MTLSizeMake(chunks, b, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return nullptr;
        return (const uint8_t*)bout.contents;
    }
}

void resident_bytes_destroy(int handle) { registry_erase(g_res_bytes, handle); }

}  // namespace metal
}  // namespace turboloader
