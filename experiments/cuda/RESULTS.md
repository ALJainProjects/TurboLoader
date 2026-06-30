# CUDA loader benchmark (RTX 3090, CUDA 13.3)

Imagenette 160px, batch 64, 8 workers, real consumption, median of timed epochs.

## Final standings (img/s, higher = better)
| Loader | img/s | notes |
|---|---:|---|
| **DALI** (GPU decode -> GPU tensor) | **~19,000** | NVIDIA flagship: advanced nvJPEG API, pinned staging, multi-stream, async prefetch |
| PyTorch DataLoader (PIL, CPU) | ~5,300 | CPU decode + transform |
| TurboLoader-CUDA (optimized) | ~2,100 loader / ~3,750 fused-op | nvJPEG batched decode + cuda transform |

## Optimization journey (the iteration loop)
1. **v1 naive** — serial single-image nvJPEG -> host -> GPU transform -> host (3 PCIe crossings): **2,587**.
2. **Fused GPU-resident** — decode straight to device, transform in place, one D2H: no faster
   (lost v1's thread-local-decoder parallelism).
3. **Batched nvJPEG + persistent device pools** — nvjpegDecodeBatched, zero per-batch malloc.
4. **Multi-threaded host Huffman** (max_cpu_threads = hw concurrency).
   -> Fused C++ op settles at **~3,750 img/s** (15-17 ms/batch); decode-bound at ~234 us/image.

## Honest verdict
DALI is ~5-9x faster and that gap is NOT the things above — it's NVIDIA's deeply-tuned
nvJPEG pipeline: the advanced host/transfer/device split API, pinned-memory staging,
multiple decode streams, and an async prefetch pipeline overlapping decode with consumption.
Matching it means essentially re-implementing DALI (multi-week, low ROI — DALI is free and
excellent). **On CUDA, use DALI.** TurboLoader's real edges are the CPU path (beats PyTorch/
tf.data), the Metal path (Apple Silicon, where DALI does not exist), and one unified
CPU/Metal/CUDA API. The CUDA path is functional + correct (bit-exact transforms, working
nvJPEG decode) but not competitive with DALI on raw throughput.

## What would close the gap (if ever pursued)
- Advanced nvJPEG API (nvjpegDecodeJpegHost/TransferToDevice/Device) with pinned buffers.
- Multiple decode streams + async prefetch (overlap batch N+1 decode with batch N use).
- Return a GPU tensor (cupy/torch CUDA) instead of CPU numpy — drop the final D2H.
- Single batched transform kernel instead of N launches.
