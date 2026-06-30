# CUDA loader benchmark (RTX 3090, CUDA 13.3)

Imagenette 160px, batch 64, 8 workers, real consumption, median of timed epochs.

## Final standings (img/s, higher = better)
| Loader | img/s | notes |
|---|---:|---|
| **DALI** (GPU decode -> GPU tensor) | **~19,900** | NVIDIA flagship: tuned nvJPEG + async prefetch pipeline |
| **TurboLoader-CUDA (gpu-resident)** | **~8,100** | nvJPEG batched decode + cuda transform, GPU-tensor output |
| TurboLoader-CUDA (-> cpu numpy) | ~5,400 | same, but D2H to numpy |
| PyTorch DataLoader (PIL, CPU) | ~5,400 | CPU decode + transform |

## Optimization journey: 2,587 -> 8,079 img/s (3.1x)
1. **v1 naive** — serial single-image nvJPEG -> host -> GPU transform -> host (3 PCIe crossings): 2,587.
2. **Batched nvJPEG + persistent device pools** — nvjpegDecodeBatched, zero per-batch malloc.
3. **Hardware decode backend** — nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE/GPU_HYBRID): fused op 3,750 -> 5,316.
4. **Serial file read in the loader** — ThreadPoolExecutor over 64 tiny cached reads cost ~16ms/batch
   (future+GIL overhead) vs ~0.6ms serial: loader 2,371 -> 6,332.
5. **GPU-resident output** — return the device pointer (__cuda_array_interface__), zero-copy into a
   torch CUDA tensor; no D2H + GPU-side consumption: 6,332 -> **8,079**.

## Honest verdict
TurboLoader-CUDA now **beats PyTorch DataLoader** and reaches the low end of the 8-12k target,
at ~40% of DALI. The remaining ~2.5x gap to DALI is its async prefetch pipeline (overlapping
batch N+1 decode with batch N consumption) + deeper nvJPEG tuning. Async prefetch mostly pays
off in REAL training (hiding decode behind the model forward/backward), which this sum-only
benchmark does not reward. DALI remains the throughput king on CUDA; TurboLoader-CUDA is now
genuinely competitive (not a toy) and its real edges remain the CPU path, the Metal path
(Apple Silicon, where DALI does not exist), and one unified CPU/Metal/CUDA API.
