// PROOF: hybrid GPU JPEG decode on Apple Silicon.
//   CPU (libjpeg): parse + Huffman entropy decode -> quantized DCT coefficients.
//   GPU (Metal):   dequantize + 8x8 IDCT (the parallel heavy lifting) -> component planes.
//   CPU:           (simple) chroma upsample + YCbCr->RGB.
// Validated against libjpeg's own decode configured with JDCT_FLOAT + no fancy upsampling,
// so the only variable is the GPU IDCT math.
//
// Build:
//   clang++ -std=c++17 -ObjC++ -O3 hybrid_jpeg_decode.mm \
//     -I/opt/homebrew/opt/jpeg-turbo/include -L/opt/homebrew/opt/jpeg-turbo/lib -ljpeg \
//     -framework Metal -framework Foundation -o hybrid && ./hybrid <img.jpg>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <jpeglib.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// zigzag -> natural index map (to de-zigzag the quant table; coefficients are already natural)
static const int NAT[64] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

static const char* IDCT_SRC = R"(
#include <metal_stdlib>
using namespace metal;
// One thread per output pixel. Dequantize + direct 8x8 IDCT (O(64)/pixel), +128 level shift.
kernel void idct_dequant(
    device const short*  coefs [[buffer(0)]],   // all blocks, 64 each, NATURAL order
    device const float*  quant [[buffer(1)]],   // 64, NATURAL order (de-zigzagged)
    device uchar*        plane [[buffer(2)]],   // component plane, planeW x planeH
    constant uint4&      d     [[buffer(3)]],   // blocksPerRow, blocksPerCol, planeW, planeH
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
            float F = float(coefs[base + v * 8u + u]) * quant[v * 8u + u];
            sum += Cu * Cv * F * cx * cy;
        }
    }
    sum = sum * 0.25f + 128.0f;
    plane[gid.y * planeW + gid.x] = uchar(clamp(sum, 0.0f, 255.0f));
}
)";

struct Comp {
    int bw, bh, pw, ph, hs, vs;  // blocks w/h, plane w/h, samp factors
    std::vector<short> coef;     // bw*bh*64
    float quant[64];
};

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "../imagenette2-160/train/n03394916/n03394916_58454.JPEG";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> jpeg(sz); fread(jpeg.data(), 1, sz, f); fclose(f);

    // ---- CPU: parse + entropy decode -> coefficients (libjpeg transcoding API) ----
    jpeg_decompress_struct cinfo; jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, jpeg.data(), jpeg.size());
    jpeg_read_header(&cinfo, TRUE);
    int W = cinfo.image_width, H = cinfo.image_height, NC = cinfo.num_components;
    jvirt_barray_ptr* coef_arrays = jpeg_read_coefficients(&cinfo);

    std::vector<Comp> comps(NC);
    for (int ci = 0; ci < NC; ci++) {
        jpeg_component_info* c = &cinfo.comp_info[ci];
        Comp& C = comps[ci];
        C.bw = c->width_in_blocks; C.bh = c->height_in_blocks;
        C.pw = C.bw * 8; C.ph = C.bh * 8; C.hs = c->h_samp_factor; C.vs = c->v_samp_factor;
        // libjpeg's quantval is ALREADY in natural order (it de-zigzags on read), matching
        // the coefficients — so NO de-zigzag here. (NAT[] is kept only for reference.)
        for (int k = 0; k < 64; k++) C.quant[k] = (float)c->quant_table->quantval[k];
        (void)NAT;
        C.coef.resize((size_t)C.bw * C.bh * 64);
        for (int by = 0; by < C.bh; by++) {
            JBLOCKARRAY buf = (cinfo.mem->access_virt_barray)((j_common_ptr)&cinfo, coef_arrays[ci], by, 1, FALSE);
            for (int bx = 0; bx < C.bw; bx++) {
                short* dst = &C.coef[((size_t)by * C.bw + bx) * 64];
                JCOEF* src = buf[0][bx];
#ifdef ASSUME_ZIGZAG
                for (int k = 0; k < 64; k++) dst[NAT[k]] = src[k];  // de-zigzag
#else
                memcpy(dst, src, 64 * sizeof(short));  // assume already natural
#endif
            }
        }
    }
    int Hmax = cinfo.max_h_samp_factor, Vmax = cinfo.max_v_samp_factor;
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    // ---- GPU: dequant + IDCT per component ----
    std::vector<std::vector<unsigned char>> planes(NC);
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(IDCT_SRC) options:nil error:&err];
        if (!lib) { printf("shader: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"idct_dequant"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        for (int ci = 0; ci < NC; ci++) {
            Comp& C = comps[ci];
            planes[ci].resize((size_t)C.pw * C.ph);
            id<MTLBuffer> bc = [dev newBufferWithBytes:C.coef.data() length:C.coef.size() * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bq = [dev newBufferWithBytes:C.quant length:64 * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bp = [dev newBufferWithLength:planes[ci].size() options:MTLResourceStorageModeShared];
            uint32_t d[4] = {(uint32_t)C.bw, (uint32_t)C.bh, (uint32_t)C.pw, (uint32_t)C.ph};
            id<MTLBuffer> bd = [dev newBufferWithBytes:d length:16 options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:bc offset:0 atIndex:0]; [e setBuffer:bq offset:0 atIndex:1];
            [e setBuffer:bp offset:0 atIndex:2]; [e setBuffer:bd offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake(C.pw, C.ph, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            memcpy(planes[ci].data(), bp.contents, planes[ci].size());
        }
    }

    // ---- CPU: simple (replication) upsample + YCbCr->RGB (BT.601) ----
    std::vector<unsigned char> rgb((size_t)W * H * 3);
    auto sample = [&](int ci, int x, int y) -> int {
        Comp& C = comps[ci];
        int sx = x * C.hs / Hmax, sy = y * C.vs / Vmax;  // replication upsample
        if (sx >= C.pw) sx = C.pw - 1; if (sy >= C.ph) sy = C.ph - 1;
        return planes[ci][(size_t)sy * C.pw + sx];
    };
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        float Y = sample(0, x, y);
        float Cb = NC > 1 ? sample(1, x, y) : 128.0f;
        float Cr = NC > 2 ? sample(2, x, y) : 128.0f;
        float r = Y + 1.402f * (Cr - 128), g = Y - 0.344136f * (Cb - 128) - 0.714136f * (Cr - 128), b = Y + 1.772f * (Cb - 128);
        size_t o = ((size_t)y * W + x) * 3;
        rgb[o]   = (unsigned char)fminf(fmaxf(r, 0), 255);
        rgb[o+1] = (unsigned char)fminf(fmaxf(g, 0), 255);
        rgb[o+2] = (unsigned char)fminf(fmaxf(b, 0), 255);
    }

    // ---- Reference: libjpeg decode with JDCT_FLOAT + no fancy upsampling ----
    jpeg_decompress_struct r2; jpeg_error_mgr e2; r2.err = jpeg_std_error(&e2);
    jpeg_create_decompress(&r2); jpeg_mem_src(&r2, jpeg.data(), jpeg.size());
    jpeg_read_header(&r2, TRUE);
    r2.dct_method = JDCT_FLOAT; r2.do_fancy_upsampling = FALSE; r2.out_color_space = JCS_RGB;
    jpeg_start_decompress(&r2);
    std::vector<unsigned char> ref((size_t)W * H * 3);
    while (r2.output_scanline < r2.output_height) {
        unsigned char* row = ref.data() + (size_t)r2.output_scanline * W * 3;
        jpeg_read_scanlines(&r2, &row, 1);
    }
    jpeg_finish_decompress(&r2); jpeg_destroy_decompress(&r2);

    // ---- Compare ----
    double sumabs = 0; int maxd = 0; long over2 = 0;
    for (size_t i = 0; i < rgb.size(); i++) {
        int dd = abs((int)rgb[i] - (int)ref[i]);
        sumabs += dd; if (dd > maxd) maxd = dd; if (dd > 2) over2++;
    }
    printf("Hybrid GPU JPEG decode: %dx%d, %d components\n", W, H, NC);
    printf("  vs libjpeg(JDCT_FLOAT, no-fancy): mean abs diff = %.4f, max = %d, %% pixels>2 = %.3f%%\n",
           sumabs / rgb.size(), maxd, 100.0 * over2 / rgb.size());
    // diagnostics: channel means + a few pixels mine vs ref
    double mr[3] = {0, 0, 0}, rr[3] = {0, 0, 0};
    for (size_t i = 0; i < rgb.size(); i += 3)
        for (int c = 0; c < 3; c++) { mr[c] += rgb[i + c]; rr[c] += ref[i + c]; }
    size_t np = rgb.size() / 3;
    printf("  channel means  mine=(%.1f,%.1f,%.1f)  ref=(%.1f,%.1f,%.1f)\n",
           mr[0]/np, mr[1]/np, mr[2]/np, rr[0]/np, rr[1]/np, rr[2]/np);
    printf("  pixel(5,5) mine=(%d,%d,%d) ref=(%d,%d,%d)\n",
           rgb[(5*W+5)*3], rgb[(5*W+5)*3+1], rgb[(5*W+5)*3+2],
           ref[(5*W+5)*3], ref[(5*W+5)*3+1], ref[(5*W+5)*3+2]);
    printf("  pixel(100,80) mine=(%d,%d,%d) ref=(%d,%d,%d)\n",
           rgb[(80*W+100)*3], rgb[(80*W+100)*3+1], rgb[(80*W+100)*3+2],
           ref[(80*W+100)*3], ref[(80*W+100)*3+1], ref[(80*W+100)*3+2]);

    // ---- ISOLATE THE IDCT: compare my Y plane (comp 0, full res) vs libjpeg GRAYSCALE ----
    jpeg_decompress_struct g; jpeg_error_mgr ge; g.err = jpeg_std_error(&ge);
    jpeg_create_decompress(&g); jpeg_mem_src(&g, jpeg.data(), jpeg.size());
    jpeg_read_header(&g, TRUE);
    g.dct_method = JDCT_FLOAT; g.out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(&g);
    std::vector<unsigned char> gray((size_t)W * H);
    while (g.output_scanline < g.output_height) {
        unsigned char* row = gray.data() + (size_t)g.output_scanline * W;
        jpeg_read_scanlines(&g, &row, 1);
    }
    jpeg_finish_decompress(&g); jpeg_destroy_decompress(&g);
    double ysum = 0; int ymax = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int dd = abs((int)planes[0][(size_t)y * comps[0].pw + x] - (int)gray[(size_t)y * W + x]);
        ysum += dd; if (dd > ymax) ymax = dd;
    }
    printf("  [IDCT isolation] my Y plane vs libjpeg grayscale: mean abs diff = %.4f, max = %d\n",
           ysum / ((double)W * H), ymax);
    return 0;
}
