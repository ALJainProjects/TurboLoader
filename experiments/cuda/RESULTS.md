# CUDA loader benchmark (RTX 3090, CUDA 13.3)

Imagenette 160px, batch 64, 8 workers, real consumption, median of timed epochs.

## Final standings (img/s, higher = better)
| Loader | img/s | notes |
|---|---:|---|
| **DALI** (GPU decode -> GPU tensor) | **~20,000** | NVIDIA flagship: advanced nvJPEG + async prefetch |
| **FFCV** (fixed .beton, gpu) | **~13,800** | numba/turbojpeg CPU decode + GPU transform, tuned .beton format |
| **TurboLoader-CUDA** (gpu-resident, prefetch=1) | **~9,200** | nvJPEG batched HW-hybrid decode + cuda transform, GPU-tensor output |
| TurboLoader-CUDA (-> cpu numpy) | ~5,400 | with D2H to numpy |
| PyTorch DataLoader (PIL, CPU) | ~5,400 | CPU decode + transform |

## TurboLoader optimization journey: 2,587 -> 9,184 img/s (3.55x)
batched nvJPEG -> persistent device pools -> hardware decode backend -> serial file read
(threadpool was 16ms/batch!) -> GPU-resident output (__cuda_array_interface__) -> async
prefetch + output ring buffer.

## Honest verdict
TurboLoader-CUDA beats PyTorch and is in the 8-12k range, but **both DALI (~20k) and FFCV
(~13.8k) are faster**. The remaining gap is decoder pipelining: DALI uses the nvJPEG ADVANCED
API (nvjpegDecodeJpegHost on CPU threads overlapped with nvjpegDecodeJpegDevice on the GPU),
which I haven't matched (I use nvjpegDecodeBatched). The 3090 (GA102) has no hardware JPEG
decoder, so this is all GPU-hybrid; the difference is host/device overlap. TurboLoader's real
edges remain the CPU path, the Metal path (Apple Silicon, where neither DALI nor FFCV runs),
and one unified CPU/Metal/CUDA API.
