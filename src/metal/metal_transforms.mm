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
)";

// Process-lifetime singletons (ARC keeps file-scope strong globals retained).
id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pso = nil;
std::string g_name;
bool g_ok = false;

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

}  // namespace metal
}  // namespace turboloader
