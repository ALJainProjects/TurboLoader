// Metal proof-of-concept: GPU bilinear-resize + normalize for a batch of RGB images,
// benchmarked honestly against a straightforward CPU reference. Decode is NOT here
// (Metal can't decode JPEG) — this isolates the *transform* step, which is the only
// part a GPU can accelerate. We measure GPU compute-only AND GPU including the
// host<->device transfers, because for small images the transfer can dominate.
//
// Build: clang++ -std=c++17 -ObjC++ -O3 metal_resize.mm -framework Metal \
//          -framework Foundation -o metal_resize
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

static const char* KERNEL_SRC = R"(
#include <metal_stdlib>
using namespace metal;
kernel void resize_normalize(
    device const uchar*  src     [[buffer(0)]],   // N*srcH*srcW*3 uint8 HWC
    device float*        dst     [[buffer(1)]],   // N*3*dstH*dstW float CHW
    constant uint4&      dims    [[buffer(2)]],   // srcW, srcH, dstW, dstH
    constant uint&       N       [[buffer(3)]],
    constant float3&     mean    [[buffer(4)]],
    constant float3&     invstd  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint srcW=dims.x, srcH=dims.y, dstW=dims.z, dstH=dims.w;
    if (gid.x>=dstW || gid.y>=dstH || gid.z>=N) return;
    uint n = gid.z;
    float scaleX = float(srcW)/float(dstW), scaleY = float(srcH)/float(dstH);
    float sx = max(0.0f,(float(gid.x)+0.5f)*scaleX-0.5f);
    float sy = max(0.0f,(float(gid.y)+0.5f)*scaleY-0.5f);
    uint x0=uint(sx), y0=uint(sy);
    uint x1=min(x0+1u,srcW-1u), y1=min(y0+1u,srcH-1u);
    float dx=sx-float(x0), dy=sy-float(y0);
    device const uchar* base = src + (ulong)n*srcH*srcW*3;
    float m[3]={mean.x,mean.y,mean.z}, isd[3]={invstd.x,invstd.y,invstd.z};
    for (uint c=0;c<3;c++){
        float p00=base[(y0*srcW+x0)*3+c], p10=base[(y0*srcW+x1)*3+c];
        float p01=base[(y1*srcW+x0)*3+c], p11=base[(y1*srcW+x1)*3+c];
        float top=mix(p00,p10,dx), bot=mix(p01,p11,dx);
        float v=mix(top,bot,dy)/255.0f;
        v=(v-m[c])*isd[c];
        dst[(((ulong)n*3+c)*dstH+gid.y)*dstW+gid.x]=v;
    }
}
)";

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// CPU reference: same half-pixel bilinear + normalize.
static void cpu_resize(const uint8_t* src, float* dst, int N, int sW, int sH, int dW, int dH,
                       const float* mean, const float* invstd) {
    float scaleX = (float)sW/dW, scaleY = (float)sH/dH;
    for (int n=0;n<N;n++){
        const uint8_t* base = src + (long)n*sH*sW*3;
        for (int y=0;y<dH;y++){
            float sy = std::max(0.f,(y+0.5f)*scaleY-0.5f);
            int y0=(int)sy, y1=std::min(y0+1,sH-1); float dy=sy-y0;
            for (int x=0;x<dW;x++){
                float sx=std::max(0.f,(x+0.5f)*scaleX-0.5f);
                int x0=(int)sx, x1=std::min(x0+1,sW-1); float dx=sx-x0;
                for (int c=0;c<3;c++){
                    float p00=base[(y0*sW+x0)*3+c], p10=base[(y0*sW+x1)*3+c];
                    float p01=base[(y1*sW+x0)*3+c], p11=base[(y1*sW+x1)*3+c];
                    float top=p00*(1-dx)+p10*dx, bot=p01*(1-dx)+p11*dx;
                    float v=(top*(1-dy)+bot*dy)/255.f;
                    dst[(((long)n*3+c)*dH+y)*dW+x]=(v-mean[c])*invstd[c];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = argc>1?atoi(argv[1]):64;
    int sW = argc>2?atoi(argv[2]):768, sH=sW;
    int dW = argc>3?atoi(argv[3]):160, dH=dW;
    int iters = 30;
    float mean[3]={0.485f,0.456f,0.406f}, std[3]={0.229f,0.224f,0.225f};
    float invstd[3]={1/std[0],1/std[1],1/std[2]};

    // synthetic batch
    std::vector<uint8_t> src((size_t)N*sH*sW*3);
    for (size_t i=0;i<src.size();i++) src[i]=(uint8_t)((i*1103515245u+12345u)>>16);
    std::vector<float> cpu_out((size_t)N*3*dH*dW), gpu_out((size_t)N*3*dH*dW);

    // ---- CPU ----
    cpu_resize(src.data(), cpu_out.data(), N, sW,sH,dW,dH, mean, invstd);   // warm
    double t0=now_ms();
    for (int it=0; it<iters; it++) cpu_resize(src.data(), cpu_out.data(), N, sW,sH,dW,dH, mean, invstd);
    double cpu_ms=(now_ms()-t0)/iters;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev){ printf("no Metal device\n"); return 1; }
        NSError* err=nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:KERNEL_SRC]
                                               options:nil error:&err];
        if (!lib){ printf("shader compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn=[lib newFunctionWithName:@"resize_normalize"];
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
        id<MTLCommandQueue> q=[dev newCommandQueue];

        id<MTLBuffer> bsrc=[dev newBufferWithLength:src.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bdst=[dev newBufferWithLength:gpu_out.size()*4 options:MTLResourceStorageModeShared];
        uint32_t dims[4]={(uint32_t)sW,(uint32_t)sH,(uint32_t)dW,(uint32_t)dH}; uint32_t Nu=N;
        float m3[3]={mean[0],mean[1],mean[2]}, i3[3]={invstd[0],invstd[1],invstd[2]};
        id<MTLBuffer> bdim=[dev newBufferWithBytes:dims length:16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bN  =[dev newBufferWithBytes:&Nu length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bmean=[dev newBufferWithBytes:m3 length:12 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bistd=[dev newBufferWithBytes:i3 length:12 options:MTLResourceStorageModeShared];

        MTLSize grid=MTLSizeMake(dW,dH,N);
        NSUInteger tgw=pso.threadExecutionWidth, tgh=pso.maxTotalThreadsPerThreadgroup/tgw;
        MTLSize tg=MTLSizeMake(tgw, std::min<NSUInteger>(tgh, dH), 1);

        auto dispatch=[&](bool copyIn){
            if (copyIn) memcpy(bsrc.contents, src.data(), src.size());
            id<MTLCommandBuffer> cb=[q commandBuffer];
            id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:bsrc offset:0 atIndex:0]; [e setBuffer:bdst offset:0 atIndex:1];
            [e setBuffer:bdim offset:0 atIndex:2]; [e setBuffer:bN offset:0 atIndex:3];
            [e setBuffer:bmean offset:0 atIndex:4]; [e setBuffer:bistd offset:0 atIndex:5];
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (copyIn) memcpy(gpu_out.data(), bdst.contents, gpu_out.size()*4);
        };
        dispatch(true); // warm

        // GPU compute-only (buffers already resident, shared memory)
        double g0=now_ms();
        for (int it=0; it<iters; it++) dispatch(false);
        double gpu_compute_ms=(now_ms()-g0)/iters;
        // GPU including the copy-in/copy-out each iter (the realistic per-batch cost)
        double g1=now_ms();
        for (int it=0; it<iters; it++) dispatch(true);
        double gpu_total_ms=(now_ms()-g1)/iters;

        // correctness
        double maxerr=0; for (size_t i=0;i<cpu_out.size();i++) maxerr=std::max(maxerr,(double)std::fabs(cpu_out[i]-gpu_out[i]));

        printf("Batch N=%d  %dx%d -> %dx%d  (M4 Max, %d GPU cores)\n", N,sW,sH,dW,dH,40);
        printf("  CPU (scalar ref):        %7.2f ms/batch  = %8.0f img/s\n", cpu_ms, N/(cpu_ms/1000));
        printf("  GPU compute only:        %7.2f ms/batch  = %8.0f img/s  (%.1fx vs CPU)\n",
               gpu_compute_ms, N/(gpu_compute_ms/1000), cpu_ms/gpu_compute_ms);
        printf("  GPU + host<->dev copies: %7.2f ms/batch  = %8.0f img/s  (%.1fx vs CPU)\n",
               gpu_total_ms, N/(gpu_total_ms/1000), cpu_ms/gpu_total_ms);
        printf("  max abs error vs CPU:    %.5f\n", maxerr);
    }
    return 0;
}
