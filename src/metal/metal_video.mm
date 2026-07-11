// Metal video path: AVFoundation/VideoToolbox hardware decode feeding a fused
// NV12 -> RGB + bilinear resize + normalize Metal kernel. See metal_video.hpp.
//
// Design notes:
// - AVAssetReaderTrackOutput with kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
//   gives NV12 (Y plane + interleaved CbCr at half resolution) and engages the
//   media engine's hardware decoder automatically.
// - Colorimetry: the YCbCr->RGB matrix (BT.601 vs BT.709) is read from the first
//   pixel buffer's kCVImageBufferYCbCrMatrixKey attachment; when the stream
//   carries no tag, >= 720p defaults to 709 (the standard heuristic). Video
//   range (16..235) is what the requested pixel format yields.
// - Planes are memcpy'd row-by-row (bytesPerRow stride!) into one shared staging
//   MTLBuffer per batch; unified memory makes this cheap and keeps v1 simple and
//   correct (CVMetalTextureCache zero-copy is a later optimization).
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_video.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

const char* kVideoKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

// One thread per output pixel of one frame. Bilinear-samples the NV12 planes at
// half-pixel positions (Y full-res, CbCr half-res), converts video-range YCbCr
// to RGB with the selected matrix, then normalizes into CHW float32.
kernel void nv12_resize_normalize(
    device const uchar* ybuf   [[buffer(0)]],   // srcH x yStride
    device const uchar* cbuf   [[buffer(1)]],   // (srcH/2) x cStride, CbCr interleaved
    device float*       dst    [[buffer(2)]],   // 3 x dstH x dstW (one frame)
    constant uint4&     sdims  [[buffer(3)]],   // srcW, srcH, yStride, cStride
    constant uint4&     ddims  [[buffer(4)]],   // dstW, dstH, bt709 flag, unused
    constant float3&    mean   [[buffer(5)]],
    constant float3&    invstd [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint srcW = sdims.x, srcH = sdims.y, yStride = sdims.z, cStride = sdims.w;
    uint dstW = ddims.x, dstH = ddims.y, bt709 = ddims.z;
    if (gid.x >= dstW || gid.y >= dstH) return;

    float sx = max(0.0f, (float(gid.x) + 0.5f) * float(srcW) / float(dstW) - 0.5f);
    float sy = max(0.0f, (float(gid.y) + 0.5f) * float(srcH) / float(dstH) - 0.5f);

    // Luma: bilinear at full resolution.
    uint x0 = uint(sx), y0 = uint(sy);
    uint x1 = min(x0 + 1u, srcW - 1u), y1 = min(y0 + 1u, srcH - 1u);
    float dx = sx - float(x0), dy = sy - float(y0);
    float Y = mix(mix(float(ybuf[y0 * yStride + x0]), float(ybuf[y0 * yStride + x1]), dx),
                  mix(float(ybuf[y1 * yStride + x0]), float(ybuf[y1 * yStride + x1]), dx), dy);

    // Chroma: bilinear at half resolution (interleaved Cb,Cr pairs). MPEG 4:2:0
    // siting: horizontally co-sited with even luma columns (cx = sx/2, no
    // offset), vertically centered between the two luma rows (cy = sy/2 - 1/4).
    uint cw = (srcW + 1u) / 2u, ch = (srcH + 1u) / 2u;
    float cxf = max(0.0f, sx * 0.5f);
    float cyf = max(0.0f, sy * 0.5f - 0.25f);
    uint cx0 = min(uint(cxf), cw - 1u), cy0 = min(uint(cyf), ch - 1u);
    uint cx1 = min(cx0 + 1u, cw - 1u), cy1 = min(cy0 + 1u, ch - 1u);
    float cdx = cxf - float(cx0), cdy = cyf - float(cy0);
    float Cb = mix(mix(float(cbuf[cy0 * cStride + cx0 * 2u]),
                       float(cbuf[cy0 * cStride + cx1 * 2u]), cdx),
                   mix(float(cbuf[cy1 * cStride + cx0 * 2u]),
                       float(cbuf[cy1 * cStride + cx1 * 2u]), cdx), cdy);
    float Cr = mix(mix(float(cbuf[cy0 * cStride + cx0 * 2u + 1u]),
                       float(cbuf[cy0 * cStride + cx1 * 2u + 1u]), cdx),
                   mix(float(cbuf[cy1 * cStride + cx0 * 2u + 1u]),
                       float(cbuf[cy1 * cStride + cx1 * 2u + 1u]), cdx), cdy);

    // Video-range YCbCr -> RGB.
    float yv = (Y - 16.0f) * (255.0f / 219.0f);
    float cb = Cb - 128.0f, cr = Cr - 128.0f;
    float R, G, B;
    if (bt709 != 0u) {
        R = yv + 1.792741f * cr;
        G = yv - 0.213249f * cb - 0.532909f * cr;
        B = yv + 2.112402f * cb;
    } else {  // BT.601
        R = yv + 1.596027f * cr;
        G = yv - 0.391762f * cb - 0.812968f * cr;
        B = yv + 2.017232f * cb;
    }
    float3 rgb = clamp(float3(R, G, B) / 255.0f, 0.0f, 1.0f);
    float m[3]   = { mean.x, mean.y, mean.z };
    float isd[3] = { invstd.x, invstd.y, invstd.z };
    for (uint c = 0; c < 3; c++)
        dst[(c * dstH + gid.y) * dstW + gid.x] = (rgb[c] - m[c]) * isd[c];
}
)";

id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pso = nil;
bool g_ok = false;

void init_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        @autoreleasepool {
            g_dev = MTLCreateSystemDefaultDevice();
            if (!g_dev) return;
            NSError* err = nil;
            id<MTLLibrary> lib =
                [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kVideoKernelSrc]
                                    options:nil
                                      error:&err];
            if (!lib) return;
            id<MTLFunction> fn = [lib newFunctionWithName:@"nv12_resize_normalize"];
            g_pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso) return;
            g_queue = [g_dev newCommandQueue];
            g_ok = (g_queue != nil);
        }
    });
}

struct VideoCtx {
    AVAsset* asset = nil;
    AVAssetReader* reader = nil;
    AVAssetReaderTrackOutput* output = nil;
    id<MTLBuffer> stage;   // per-batch NV12 staging (Y + CbCr regions per frame)
    id<MTLBuffer> out[2];  // double-buffered (max_batch, 3, dstH, dstW) float
    int src_w = 0, src_h = 0, dst_w = 0, dst_h = 0;
    int frame_step = 1;
    size_t max_batch = 0;
    long next_src_index = 0;  // source index of the next frame read from the stream
    double fps = 0.0;
    long total_kept = -1;
    int cur = 0;
    bool exhausted = false;
    bool failed = false;     // decode/GPU error (distinct from clean end-of-stream)
    bool dims_latched = false;  // src dims/colorimetry confirmed from a real buffer
    uint32_t bt709 = 0;
    // Serializes next_batch against close: next_batch runs GIL-released, so a
    // concurrent close() would otherwise delete the ctx mid-use.
    std::mutex mu;
};

std::mutex g_mu;
std::vector<VideoCtx*> g_ctxs;

VideoCtx* get_ctx(int handle) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (handle < 0 || (size_t)handle >= g_ctxs.size()) return nullptr;
    return g_ctxs[handle];
}

}  // namespace

namespace turboloader {
namespace metal_video {

bool available() {
    init_once();
    return g_ok;
}

int open_video(const char* path, int frame_step, size_t max_batch, int dst_h, int dst_w) {
    init_once();
    if (!g_ok || !path || frame_step < 1 || max_batch == 0 || dst_h <= 0 || dst_w <= 0)
        return -1;
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVAsset* asset = [AVAsset assetWithURL:url];
        NSArray<AVAssetTrack*>* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        if (tracks.count == 0) return -1;
        AVAssetTrack* track = tracks[0];

        NSError* err = nil;
        AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:&err];
        if (!reader) return -1;
        NSDictionary* settings = @{
            (id)kCVPixelBufferPixelFormatTypeKey :
                @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        };
        AVAssetReaderTrackOutput* output =
            [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:settings];
        output.alwaysCopiesSampleData = NO;
        if (![reader canAddOutput:output]) return -1;
        [reader addOutput:output];
        if (![reader startReading]) return -1;

        CGSize sz = track.naturalSize;
        auto* c = new VideoCtx();
        c->asset = asset;
        c->reader = reader;
        c->output = output;
        c->src_w = (int)sz.width;
        c->src_h = (int)sz.height;
        c->dst_w = dst_w;
        c->dst_h = dst_h;
        c->frame_step = frame_step;
        c->max_batch = max_batch;
        c->fps = track.nominalFrameRate;
        if (c->fps > 0.0) {
            double dur = CMTimeGetSeconds(asset.duration);
            long total = (long)(dur * c->fps + 0.5);
            c->total_kept = (total + frame_step - 1) / frame_step;
        }
        // Staging: worst-case strides unknown until the first frame. Rotation
        // metadata can swap effective w/h vs naturalSize, so size by the (rotation
        // invariant) pixel COUNT plus a per-row padding budget of 256 bytes against
        // the larger dimension — covers CoreVideo's alignments either way.
        const size_t pixels = (size_t)c->src_w * c->src_h;
        const size_t maxdim = (size_t)std::max(c->src_w, c->src_h);
        const size_t y_max = pixels + 256 * maxdim;
        const size_t c_max = pixels / 2 + 128 * maxdim + 2;
        const size_t out_bytes = max_batch * (size_t)3 * dst_h * dst_w * sizeof(float);
        c->stage = [g_dev newBufferWithLength:max_batch * (y_max + c_max)
                                      options:MTLResourceStorageModeShared];
        c->out[0] = [g_dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        c->out[1] = [g_dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        if (!c->stage || !c->out[0] || !c->out[1]) {
            [reader cancelReading];
            delete c;
            return -1;
        }
        std::lock_guard<std::mutex> lk(g_mu);
        for (size_t i = 0; i < g_ctxs.size(); ++i)
            if (!g_ctxs[i]) {
                g_ctxs[i] = c;
                return (int)i;
            }
        g_ctxs.push_back(c);
        return (int)g_ctxs.size() - 1;
    }
}

long frame_count(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->total_kept : -1;
}
int src_width(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->src_w : 0;
}
int src_height(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->src_h : 0;
}
double fps(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->fps : 0.0;
}

size_t next_batch(int handle, size_t batch, const float mean[3], const float std_[3],
                  const float** out, long* first_index) {
    VideoCtx* c = get_ctx(handle);
    if (!c || !out || batch == 0 || batch > c->max_batch) return 0;
    std::lock_guard<std::mutex> ctx_lock(c->mu);  // close_video waits on this
    if (c->exhausted) return 0;
    @autoreleasepool {
        struct Slot {
            size_t y_off, c_off;
            uint32_t y_stride, c_stride;
        };
        std::vector<Slot> slots;
        slots.reserve(batch);
        long first = -1;
        uint8_t* stage = (uint8_t*)c->stage.contents;
        size_t stage_used = 0;

        while (slots.size() < batch) {
            CMSampleBufferRef sb = [c->output copyNextSampleBuffer];
            if (!sb) {
                c->exhausted = true;
                break;
            }
            CVImageBufferRef pb = CMSampleBufferGetImageBuffer(sb);
            const long idx = c->next_src_index++;
            const bool keep = pb && (idx % c->frame_step == 0);
            if (keep) {
                CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                if (!c->dims_latched) {
                    // Trust the DECODED buffer, not track.naturalSize: rotation
                    // metadata or codec padding can make them disagree, and the
                    // kernel must index the real planes. Colorimetry likewise
                    // comes from the buffer's attachment when the stream is
                    // tagged; untagged HD defaults to BT.709.
                    c->src_w = (int)CVPixelBufferGetWidthOfPlane(pb, 0);
                    c->src_h = (int)CVPixelBufferGetHeightOfPlane(pb, 0);
                    CFTypeRef matrix =
                        CVBufferGetAttachment(pb, kCVImageBufferYCbCrMatrixKey, nullptr);
                    if (matrix && CFEqual(matrix, kCVImageBufferYCbCrMatrix_ITU_R_709_2)) {
                        c->bt709 = 1;
                    } else if (matrix &&
                               CFEqual(matrix, kCVImageBufferYCbCrMatrix_ITU_R_601_4)) {
                        c->bt709 = 0;
                    } else {
                        c->bt709 = (c->src_h >= 720 || c->src_w >= 1280) ? 1u : 0u;
                    }
                    c->dims_latched = true;
                }
                const uint8_t* ysrc = (const uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pb, 0);
                const uint8_t* csrc = (const uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pb, 1);
                const size_t ystride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
                const size_t cstride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
                const size_t h = CVPixelBufferGetHeightOfPlane(pb, 0);
                const size_t ch = CVPixelBufferGetHeightOfPlane(pb, 1);
                const bool dims_ok = (int)CVPixelBufferGetWidthOfPlane(pb, 0) == c->src_w &&
                                     (int)h == c->src_h;
                if (!dims_ok || stage_used + ystride * h + cstride * ch > c->stage.length) {
                    // Mid-stream geometry change or stride beyond the allocation
                    // budget: refuse rather than overflow / misindex — surfaced
                    // to Python as an error, not a clean end-of-stream.
                    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                    CFRelease(sb);
                    c->exhausted = true;
                    c->failed = true;
                    break;
                }
                Slot s;
                s.y_off = stage_used;
                std::memcpy(stage + stage_used, ysrc, ystride * h);
                stage_used += ystride * h;
                s.c_off = stage_used;
                std::memcpy(stage + stage_used, csrc, cstride * ch);
                stage_used += cstride * ch;
                s.y_stride = (uint32_t)ystride;
                s.c_stride = (uint32_t)cstride;
                CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                if (first < 0) first = idx;
                slots.push_back(s);
            }
            CFRelease(sb);
        }
        if (slots.empty()) {
            // Distinguish clean EOS from a reader failure so Python can raise.
            if (c->reader.status == AVAssetReaderStatusFailed) c->failed = true;
            return 0;
        }
        const uint32_t bt709 = c->bt709;  // latched from the first buffer

        const float invstd[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
        c->cur ^= 1;
        id<MTLBuffer> bout = c->out[c->cur];
        const size_t per_out = (size_t)3 * c->dst_h * c->dst_w;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g_pso];
        for (size_t i = 0; i < slots.size(); ++i) {
            uint32_t sdims[4] = {(uint32_t)c->src_w, (uint32_t)c->src_h, slots[i].y_stride,
                                 slots[i].c_stride};
            uint32_t ddims[4] = {(uint32_t)c->dst_w, (uint32_t)c->dst_h, bt709, 0};
            [e setBuffer:c->stage offset:slots[i].y_off atIndex:0];
            [e setBuffer:c->stage offset:slots[i].c_off atIndex:1];
            [e setBuffer:bout offset:i * per_out * sizeof(float) atIndex:2];
            [e setBytes:sdims length:sizeof(sdims) atIndex:3];
            [e setBytes:ddims length:sizeof(ddims) atIndex:4];
            [e setBytes:mean length:3 * sizeof(float) atIndex:5];
            [e setBytes:invstd length:3 * sizeof(float) atIndex:6];
            [e dispatchThreads:MTLSizeMake(c->dst_w, c->dst_h, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            c->exhausted = true;
            c->failed = true;
            return 0;
        }
        *out = (const float*)bout.contents;
        if (first_index) *first_index = first;
        return slots.size();
    }
}

bool has_failed(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->failed : false;
}

int dst_width(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->dst_w : 0;
}

int dst_height(int handle) {
    VideoCtx* c = get_ctx(handle);
    return c ? c->dst_h : 0;
}

void close_video(int handle) {
    VideoCtx* c = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (handle < 0 || (size_t)handle >= g_ctxs.size()) return;
        c = g_ctxs[handle];
        g_ctxs[handle] = nullptr;
    }
    if (!c) return;
    {
        // Wait for any in-flight next_batch (it runs GIL-released); new calls
        // can no longer reach this ctx — its registry slot is already null.
        std::lock_guard<std::mutex> lk(c->mu);
        @autoreleasepool {
            if (c->reader && c->reader.status == AVAssetReaderStatusReading)
                [c->reader cancelReading];
        }
    }
    delete c;  // .mm units compile with -fobjc-arc: this releases the ObjC members
}

}  // namespace metal_video
}  // namespace turboloader
