# Metal GPU transforms — proof of concept

Validates GPU-accelerated **resize + normalize** on Apple Silicon (Metal compute), since
NVIDIA DALI's GPU advantage has no MPS equivalent and the loader is otherwise CPU-only.

```bash
clang++ -std=c++17 -ObjC++ -O3 metal_resize.mm -framework Metal -framework Foundation -o metal_resize
./metal_resize 64 768 160     # N, srcPx, dstPx
```

## Measured (Apple M4 Max, 40 GPU cores), bilinear resize + ImageNet normalize, bit-exact vs CPU

| Workload        | CPU scalar ref | GPU compute-only | GPU + host<->dev copies |
|-----------------|---------------:|-----------------:|------------------------:|
| 768->160, N=64  |    9,733 img/s |  153,904 (15.8x) |          23,676 (2.4x)  |
| 160->160, N=64  |   11,051 img/s |  198,759 (18x)   |         128,202 (11.6x) |
| 256->224, N=128 |    5,469 img/s |  185,440 (34x)   |          69,433 (12.7x) |

## Honest reading
- GPU compute is hugely faster (15-34x **vs a scalar reference**; ~4-9x once you discount
  for the CPU SIMD path). It is **bit-exact** with the CPU bilinear (max err 0.00000).
- The **host<->device copy dominates** for large source images (the 768px copy-in is
  ~113 MB/batch). The win is much larger when the batch can stay **GPU-resident**.
- **Decode stays on CPU** (Metal has no JPEG decode). So end-to-end benefit depends on the
  decode/transform ratio — this accelerates transforms, not decode.

## Where this is actually worth integrating
A **GPU-resident transform path** that decodes on CPU, uploads each batch once, runs all
transforms on the GPU, and hands back an **MPS tensor** for Apple-Silicon PyTorch training
(no copy-out). That would make TurboLoader the only loader doing DALI-style GPU transforms
on Apple Silicon (DALI is CUDA-only). Build-system + MPS-interop + correctness work
required; gated behind an opt-in flag so CPU wheels are unaffected.
