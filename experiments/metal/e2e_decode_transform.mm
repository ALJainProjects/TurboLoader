// End-to-end honesty check: decode REAL Imagenette JPEGs (turbojpeg, CPU) and then
// resize+normalize on CPU vs Metal GPU, to see whether GPU transform offload actually
// moves end-to-end throughput once the unmovable CPU decode cost is in the loop.
//
// Build:
//   clang++ -std=c++17 -ObjC++ -O3 e2e_decode_transform.mm \
//     -I/opt/homebrew/opt/jpeg-turbo/include -L/opt/homebrew/opt/jpeg-turbo/lib -lturbojpeg \
//     -framework Metal -framework Foundation -o e2e
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <turbojpeg.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>

static const char* KERNEL_SRC = R"(
#include <metal_stdlib>
using namespace metal;
kernel void resize_normalize(
    device const uchar* src [[buffer(0)]], device float* dst [[buffer(1)]],
    constant uint4& dims [[buffer(2)]], constant float3& mean [[buffer(3)]],
    constant float3& invstd [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    uint srcW=dims.x, srcH=dims.y, dstW=dims.z, dstH=dims.w;
    if (gid.x>=dstW || gid.y>=dstH) return;
    float sx=max(0.0f,(float(gid.x)+0.5f)*float(srcW)/float(dstW)-0.5f);
    float sy=max(0.0f,(float(gid.y)+0.5f)*float(srcH)/float(dstH)-0.5f);
    uint x0=uint(sx),y0=uint(sy),x1=min(x0+1u,srcW-1u),y1=min(y0+1u,srcH-1u);
    float dx=sx-float(x0),dy=sy-float(y0);
    float m[3]={mean.x,mean.y,mean.z}, isd[3]={invstd.x,invstd.y,invstd.z};
    for(uint c=0;c<3;c++){
        float p00=src[(y0*srcW+x0)*3+c],p10=src[(y0*srcW+x1)*3+c];
        float p01=src[(y1*srcW+x0)*3+c],p11=src[(y1*srcW+x1)*3+c];
        float v=mix(mix(p00,p10,dx),mix(p01,p11,dx),dy)/255.0f;
        dst[(c*dstH+gid.y)*dstW+gid.x]=(v-m[c])*isd[c];
    }
}
)";

static double now_ms(){ using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count(); }

static void walk(const std::string& dir, std::vector<std::string>& out, size_t cap){
    DIR* d=opendir(dir.c_str()); if(!d) return; struct dirent* e;
    while((e=readdir(d)) && out.size()<cap){
        std::string n=e->d_name; if(n=="."||n=="..") continue;
        std::string p=dir+"/"+n;
        if(e->d_type==DT_DIR) walk(p,out,cap);
        else if(n.size()>4 && (n.substr(n.size()-5)==".JPEG"||n.substr(n.size()-4)==".jpg")) out.push_back(p);
    }
    closedir(d);
}

int main(int argc,char**argv){
    size_t M = argc>1?atoi(argv[1]):2000;
    int dW=160,dH=160;
    float mean[3]={0.485f,0.456f,0.406f}, istd[3]={1/0.229f,1/0.224f,1/0.225f};
    const char* root = argc>2?argv[2]:"../imagenette2-160/train";

    std::vector<std::string> paths; walk(root,paths,M);
    if(paths.empty()){ printf("no JPEGs under %s\n",root); return 1; }
    M=paths.size();

    // load file bytes
    std::vector<std::vector<uint8_t>> jpeg(M);
    for(size_t i=0;i<M;i++){ FILE*f=fopen(paths[i].c_str(),"rb"); fseek(f,0,SEEK_END); long sz=ftell(f);
        fseek(f,0,SEEK_SET); jpeg[i].resize(sz); fread(jpeg[i].data(),1,sz,f); fclose(f); }

    // ---- decode (turbojpeg, CPU) ----
    std::vector<std::vector<uint8_t>> rgb(M); std::vector<int> W(M),H(M);
    tjhandle tj=tjInitDecompress();
    auto decode_all=[&](){
        for(size_t i=0;i<M;i++){ int w,h,ss,cs;
            tjDecompressHeader3(tj,jpeg[i].data(),jpeg[i].size(),&w,&h,&ss,&cs);
            W[i]=w;H[i]=h; rgb[i].resize((size_t)w*h*3);
            tjDecompress2(tj,jpeg[i].data(),jpeg[i].size(),rgb[i].data(),w,0,h,TJPF_RGB,TJFLAG_FASTDCT); }
    };
    decode_all(); // warm + populate
    double t0=now_ms(); decode_all(); double decode_ms=now_ms()-t0;

    // ---- CPU transform (scalar ref) ----
    std::vector<float> cpu_out((size_t)M*3*dH*dW);
    auto cpu_xform=[&](){
        for(size_t i=0;i<M;i++){ const uint8_t* s=rgb[i].data(); int sW=W[i],sH=H[i];
            float* o=cpu_out.data()+(size_t)i*3*dH*dW;
            for(int y=0;y<dH;y++){ float sy=std::max(0.f,(y+0.5f)*sH/dH-0.5f); int y0=(int)sy,y1=std::min(y0+1,sH-1); float dy=sy-y0;
              for(int x=0;x<dW;x++){ float sx=std::max(0.f,(x+0.5f)*sW/dW-0.5f); int x0=(int)sx,x1=std::min(x0+1,sW-1); float dx=sx-x0;
                for(int c=0;c<3;c++){ float p00=s[(y0*sW+x0)*3+c],p10=s[(y0*sW+x1)*3+c],p01=s[(y1*sW+x0)*3+c],p11=s[(y1*sW+x1)*3+c];
                  float v=((p00*(1-dx)+p10*dx)*(1-dy)+(p01*(1-dx)+p11*dx)*dy)/255.f; o[(c*dH+y)*dW+x]=(v-mean[c])*istd[c]; } } } }
    };
    cpu_xform(); t0=now_ms(); cpu_xform(); double cpu_xform_ms=now_ms()-t0;

    // ---- GPU transform (Metal, unified memory) ----
    double gpu_xform_ms=0, gpu_xform_upload_ms=0;
    @autoreleasepool{
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError*err=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:KERNEL_SRC] options:nil error:&err];
        if(!lib){printf("shader err %s\n",err.localizedDescription.UTF8String);return 1;}
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"resize_normalize"] error:&err];
        id<MTLCommandQueue> q=[dev newCommandQueue];
        // one shared src buffer (sum of decoded sizes) + offsets; one dst buffer
        std::vector<size_t> off(M+1,0); for(size_t i=0;i<M;i++) off[i+1]=off[i]+rgb[i].size();
        id<MTLBuffer> bsrc=[dev newBufferWithLength:off[M] options:MTLResourceStorageModeShared];
        id<MTLBuffer> bdst=[dev newBufferWithLength:(size_t)M*3*dH*dW*4 options:MTLResourceStorageModeShared];
        auto upload=[&](){ for(size_t i=0;i<M;i++) memcpy((uint8_t*)bsrc.contents+off[i],rgb[i].data(),rgb[i].size()); };
        auto encode=[&](){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            for(size_t i=0;i<M;i++){ uint32_t dims[4]={(uint32_t)W[i],(uint32_t)H[i],(uint32_t)dW,(uint32_t)dH};
              [e setBuffer:bsrc offset:off[i] atIndex:0]; [e setBuffer:bdst offset:(size_t)i*3*dH*dW*4 atIndex:1];
              [e setBytes:dims length:16 atIndex:2]; [e setBytes:mean length:12 atIndex:3]; [e setBytes:istd length:12 atIndex:4];
              [e dispatchThreads:MTLSizeMake(dW,dH,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; }
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; };
        upload(); encode(); // warm
        t0=now_ms(); encode(); gpu_xform_ms=now_ms()-t0;                  // compute only (already resident)
        t0=now_ms(); upload(); encode(); gpu_xform_upload_ms=now_ms()-t0; // + per-batch upload (real cost)
        // correctness vs CPU
        double maxerr=0; const float* g=(const float*)bdst.contents;
        for(size_t i=0;i<cpu_out.size();i++) maxerr=std::max(maxerr,(double)fabs(cpu_out[i]-g[i]));
        printf("decoded %zu real Imagenette JPEGs, resize->%dx%d, max|gpu-cpu|=%.5f\n\n",M,dW,dH,maxerr);
    }

    auto rate=[&](double ms){ return M/(ms/1000.0); };
    double dec=decode_ms/M*1000, cx=cpu_xform_ms/M*1000, gx=gpu_xform_ms/M*1000, gxu=gpu_xform_upload_ms/M*1000;
    printf("PER-IMAGE (us):  decode=%.1f   cpu_xform=%.1f   gpu_xform=%.1f (compute)  %.1f (+upload)\n",dec,cx,gx,gxu);
    printf("STAGE RATES:     decode-only=%.0f img/s   cpu_xform-only=%.0f   gpu_xform-only=%.0f\n",
           rate(decode_ms),rate(cpu_xform_ms),rate(gpu_xform_ms));
    printf("\nEND-TO-END (decode + transform, single thread):\n");
    printf("  CPU path:  %.1f us/img = %.0f img/s\n", dec+cx, 1e6/(dec+cx));
    printf("  GPU path:  %.1f us/img = %.0f img/s   (%.2fx)   [+upload: %.0f img/s]\n",
           dec+gx, 1e6/(dec+gx), (dec+cx)/(dec+gx), 1e6/(dec+gxu));
    printf("\n  decode is %.0f%% of the CPU-path time, %.0f%% of the GPU-path time\n",
           100*dec/(dec+cx), 100*dec/(dec+gx));
    return 0;
}
