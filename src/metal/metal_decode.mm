// Hybrid GPU JPEG decoder (Apple Silicon). CPU (libjpeg) does parse + Huffman entropy
// decode -> quantized DCT coefficients; GPU (Metal) does dequant + 8x8 IDCT (the parallel
// heavy lifting); CPU does chroma upsample + YCbCr->RGB. The nvJPEG hybrid split, on Apple
// GPUs. The GPU IDCT is validated bit-exact vs libjpeg's float IDCT (experiments/metal/).
//
// Compiled only on macOS arm64 with TURBOLOADER_METAL (see setup.py). Own device/pipeline
// so the validated transform path (metal_transforms.mm) is untouched.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_transforms.hpp"

#include <jpeglib.h>

#include <csetjmp>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

// One thread per output pixel: dequantize + direct 8x8 IDCT + level shift. Bit-exact vs
// libjpeg JDCT_FLOAT (proven in experiments/metal/hybrid_jpeg_decode.mm).
const char* kIdctSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void idct_dequant(
    device const short* coefs [[buffer(0)]],   // blocks of 64, NATURAL order
    device const float* quant [[buffer(1)]],   // 64, NATURAL order
    device uchar*       plane [[buffer(2)]],
    constant uint4&     d     [[buffer(3)]],   // blocksPerRow, blocksPerCol, planeW, planeH
    uint2 gid [[thread_position_in_grid]])
{
    uint planeW = d.z, planeH = d.w;
    if (gid.x >= planeW || gid.y >= planeH) return;
    uint bx = gid.x >> 3, by = gid.y >> 3, px = gid.x & 7u, py = gid.y & 7u;
    uint base = (by * d.x + bx) * 64u;
    float sum = 0.0f;
    for (uint v = 0; v < 8; v++) {
        float cy = cos((2.0f * float(py) + 1.0f) * float(v) * 3.14159265358979f / 16.0f);
        float Cv = (v == 0u) ? 0.70710678f : 1.0f;
        for (uint u = 0; u < 8; u++) {
            float cx = cos((2.0f * float(px) + 1.0f) * float(u) * 3.14159265358979f / 16.0f);
            float Cu = (u == 0u) ? 0.70710678f : 1.0f;
            sum += Cu * Cv * float(coefs[base + v * 8u + u]) * quant[v * 8u + u] * cx * cy;
        }
    }
    plane[gid.y * planeW + gid.x] = uchar(clamp(sum * 0.25f + 128.0f, 0.0f, 255.0f));
}
)";

id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_idct = nil;
bool g_ok = false;

void init_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        @autoreleasepool {
            g_dev = MTLCreateSystemDefaultDevice();
            if (!g_dev) return;
            NSError* err = nil;
            id<MTLLibrary> lib =
                [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kIdctSrc]
                                    options:nil
                                      error:&err];
            if (!lib) return;
            g_idct = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"idct_dequant"]
                                                          error:&err];
            if (!g_idct) return;
            g_queue = [g_dev newCommandQueue];
            g_ok = (g_queue != nil);
        }
    });
}

struct Comp {
    int bw, bh, pw, ph, hs, vs;
    float quant[64];
    std::vector<short> coef;
    std::vector<uint8_t> plane;
};

}  // namespace

namespace turboloader {
namespace metal {

bool decode_jpeg(const uint8_t* data, size_t size, std::vector<uint8_t>& out, int& width,
                 int& height) {
    init_once();
    if (!g_ok || !data || size == 0) return false;

    jpeg_decompress_struct cinfo;
    // libjpeg's DEFAULT error_exit calls exit(): a malformed JPEG killed the whole Python
    // process (verified: interpreter exits code 1, no traceback). Trap fatal errors with
    // the standard setjmp/longjmp handler and report failure instead.
    struct ErrorJmp {
        jpeg_error_mgr pub;
        std::jmp_buf jb;
        static void error_exit(j_common_ptr c) {
            std::longjmp(reinterpret_cast<ErrorJmp*>(c->err)->jb, 1);
        }
    } jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = &ErrorJmp::error_exit;
    if (setjmp(jerr.jb)) {
        jpeg_destroy_decompress(&cinfo);
        return false;  // malformed JPEG: fail cleanly, caller falls back / raises
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, const_cast<unsigned char*>(data), size);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    const int W = cinfo.image_width, H = cinfo.image_height, NC = cinfo.num_components;
    if (NC != 1 && NC != 3) {  // only grayscale / YCbCr
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    jvirt_barray_ptr* coef_arrays = jpeg_read_coefficients(&cinfo);
    if (!coef_arrays) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    std::vector<Comp> comps(NC);
    for (int ci = 0; ci < NC; ci++) {
        jpeg_component_info* c = &cinfo.comp_info[ci];
        Comp& C = comps[ci];
        C.bw = c->width_in_blocks;
        C.bh = c->height_in_blocks;
        C.pw = C.bw * 8;
        C.ph = C.bh * 8;
        C.hs = c->h_samp_factor;
        C.vs = c->v_samp_factor;
        // libjpeg quantval is already natural order (matches coefficients) — no de-zigzag.
        for (int k = 0; k < 64; k++) C.quant[k] = (float)c->quant_table->quantval[k];
        C.coef.resize((size_t)C.bw * C.bh * 64);
        for (int by = 0; by < C.bh; by++) {
            JBLOCKARRAY buf = (cinfo.mem->access_virt_barray)((j_common_ptr)&cinfo,
                                                              coef_arrays[ci], by, 1, FALSE);
            for (int bx = 0; bx < C.bw; bx++)
                std::memcpy(&C.coef[((size_t)by * C.bw + bx) * 64], buf[0][bx],
                            64 * sizeof(short));
        }
    }
    const int Hmax = cinfo.max_h_samp_factor, Vmax = cinfo.max_v_samp_factor;
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    // GPU: dequant + IDCT per component
    @autoreleasepool {
        for (int ci = 0; ci < NC; ci++) {
            Comp& C = comps[ci];
            C.plane.resize((size_t)C.pw * C.ph);
            id<MTLBuffer> bc = [g_dev newBufferWithBytes:C.coef.data()
                                                  length:C.coef.size() * sizeof(short)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bq = [g_dev newBufferWithBytes:C.quant
                                                  length:64 * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bp = [g_dev newBufferWithLength:C.plane.size()
                                                  options:MTLResourceStorageModeShared];
            uint32_t d[4] = {(uint32_t)C.bw, (uint32_t)C.bh, (uint32_t)C.pw, (uint32_t)C.ph};
            id<MTLBuffer> bd = [g_dev newBufferWithBytes:d
                                                  length:sizeof(d)
                                                 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:g_idct];
            [e setBuffer:bc offset:0 atIndex:0];
            [e setBuffer:bq offset:0 atIndex:1];
            [e setBuffer:bp offset:0 atIndex:2];
            [e setBuffer:bd offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake(C.pw, C.ph, 1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) return false;
            std::memcpy(C.plane.data(), bp.contents, C.plane.size());
        }
    }

    // CPU: replication upsample + YCbCr->RGB
    out.resize((size_t)W * H * 3);
    auto samp = [&](int ci, int x, int y) -> int {
        const Comp& C = comps[ci];
        int sx = x * C.hs / Hmax, sy = y * C.vs / Vmax;
        if (sx >= C.pw) sx = C.pw - 1;
        if (sy >= C.ph) sy = C.ph - 1;
        return C.plane[(size_t)sy * C.pw + sx];
    };
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float Y = samp(0, x, y);
            float Cb = NC > 1 ? samp(1, x, y) : 128.0f;
            float Cr = NC > 2 ? samp(2, x, y) : 128.0f;
            float r = Y + 1.402f * (Cr - 128.0f);
            float g = Y - 0.344136f * (Cb - 128.0f) - 0.714136f * (Cr - 128.0f);
            float b = Y + 1.772f * (Cb - 128.0f);
            size_t o = ((size_t)y * W + x) * 3;
            out[o] = (uint8_t)std::min(std::max(r, 0.0f), 255.0f);
            out[o + 1] = (uint8_t)std::min(std::max(g, 0.0f), 255.0f);
            out[o + 2] = (uint8_t)std::min(std::max(b, 0.0f), 255.0f);
        }
    }
    width = W;
    height = H;
    return true;
}

}  // namespace metal
}  // namespace turboloader
