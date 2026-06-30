# CUDA loader benchmark (RTX 3090, CUDA 13.3)

Imagenette 160px, batch 64, real consumption (every batch adopted into a CUDA tensor and
summed), median of timed epochs. All loaders read JPEGs from disk (page cache warm).

## Final standings (img/s, higher = better)
| Loader | img/s | notes |
|---|---:|---|
| **DALI** (GPU decode -> GPU tensor) | **~20,000** | NVIDIA flagship: C++ pipeline, no GIL, overlapped I/O |
| **TurboLoader-CUDA** (`decode="nvimgcodec"`, prefetch=2) | **~14,500** | **nvImageCodec GPU decode + our resize/normalize kernel, GPU-tensor output** |
| **FFCV** (fixed .beton, gpu) | **~13,800** | numba/turbojpeg CPU decode + GPU transform, packed .beton (memory-mapped I/O) |
| TurboLoader-CUDA (`decode="gpu"`, nvJPEG) | ~9,200 | nvJPEG batched HW-hybrid decode + cuda transform |
| TurboLoader-CUDA (-> cpu numpy) | ~5,400 | with D2H to numpy |
| PyTorch DataLoader (PIL, CPU) | ~5,400 | CPU decode + transform |

**TurboLoader-CUDA now beats FFCV and reaches ~72% of DALI end-to-end** (was ~46% with nvJPEG).

## The breakthrough: nvImageCodec
DALI's speed was never its kernels — it was the decoder. DALI moved off `nvjpegDecodeBatched`
to **nvImageCodec**, NVIDIA's modern codec library. Measured in isolation on the 3090,
nvImageCodec decodes JPEGs at **~21,600 img/s** — faster than DALI's *whole* pipeline. It
outputs GPU-resident HWC uint8 RGB images exposing `__cuda_array_interface__` (contiguous,
`strides: None`), which feed our `resize_normalize` kernel **in place, zero extra copies**
(`cuda_resize_normalize_from_device`). That single swap took us 9,184 -> the numbers below.

## End-to-end (14.5k) vs GPU-pipeline ceiling (17.7k)
- **17,679 img/s** — decode+transform+consume with JPEG bytes already in RAM. This is the pure
  GPU pipeline ceiling: what nvImageCodec + our kernel sustain with no file I/O.
- **14,487 img/s** — full loader reading files from disk. The gap is Python file-I/O overhead.
  Closing it: (1) a C++ `read_files` that reads a batch via the thread pool with the GIL
  *released*, and (2) a **3-stage pipeline** — reader thread -> bytes queue -> decode+transform
  thread -> output queue -> consumer — so disk-read ‖ GPU-decode ‖ consume all overlap. Measured
  best at ~14.9k (standalone) / ~14.5k (integrated) vs ~13.9k single-producer.
- DALI hits ~20k because its file reader, batcher, and prefetch are all C++ (no GIL); our
  remaining gap to DALI is the Python orchestration layer, **not** the GPU work.

## Correctness (vs the validated libjpeg-turbo + kernel path)
nvImageCodec and libjpeg-turbo are different JPEG decoders, so the pipeline is **not** bit-exact
(JPEG IDCT and chroma upsampling are implementation-defined). It is numerically equivalent:
- correlation **0.99955**; normalized-space mean|diff| 0.022, p99 0.12, max 0.87
- raw decoder-level diff: mean 1.27 uint8 levels, p99 8, max 31 — the expected variance between
  any two JPEG decoders (DALI and FFCV differ from CPU decode the same way).

## TurboLoader optimization journey
2,587 -> 9,184 img/s (nvJPEG): batched nvJPEG -> persistent device pools -> hardware decode
backend -> serial file read (threadpool was 16ms/batch!) -> GPU-resident output -> async
prefetch + output ring. **9,184 -> 14,487 (nvJPEG -> nvImageCodec + 3-stage pipeline).**

## Honest verdict
With nvImageCodec, TurboLoader-CUDA **beats FFCV (~13.8k) and PyTorch (~5.4k)** and reaches
**~72% of DALI end-to-end / ~88% of DALI on the pure GPU pipeline**. DALI still leads because
its entire data path is GIL-free C++; matching it would mean moving file reading + batching into
C++ (the `read_files` binding is a first step). TurboLoader's portable edges remain: the CPU
path, the Metal path (Apple Silicon, where neither DALI nor FFCV runs), and one unified
CPU/Metal/CUDA loader API — now genuinely competitive on NVIDIA too.

## Reproduce
```python
import turboloader as t, torch, glob
paths = glob.glob("imagenette2-160/train/*/*.JPEG")
ld = t.CudaImageLoader(paths, batch_size=64, image_size=160,
                       decode="nvimgcodec", prefetch=2, drop_last=True)
for batch in ld:                       # batch: __cuda_array_interface__, GPU-resident
    x = torch.as_tensor(batch, device="cuda")   # (64,3,160,160) float32, zero-copy
```
Build: `TURBOLOADER_ENABLE_CUDA=1 TURBOLOADER_ENABLE_NVJPEG=1 TURBOLOADER_CUDA_ARCH=native pip
install -e .` plus `pip install nvidia-nvimgcodec-cu12`.
