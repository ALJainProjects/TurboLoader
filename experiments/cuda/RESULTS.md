# CUDA loader benchmark (RTX 3090, CUDA 13.3)

Imagenette 160px, batch 64, 8 workers, real consumption, median of timed epochs.

| Loader | img/s | notes |
|---|---:|---|
| **DALI** (GPU decode -> GPU tensor) | **19,237** | NVIDIA flagship, fully GPU-resident, batched nvJPEG |
| PyTorch DataLoader (PIL, CPU) | 5,395 | CPU decode + transform |
| **TurboLoader-CUDA v1** (nvjpeg -> cpu numpy) | 2,587 | UNOPTIMIZED baseline |

## Why v1 is slow (the optimization target)
v1 does, per image: serial single-image nvJPEG (mutex-locked) -> copy GPU->host ->
cuda_resize_normalize copies host->GPU -> transform -> copy GPU->host to numpy. Three
PCIe crossings per image and no decode batching. The kernels themselves are bit-exact and
fine; the architecture is the bottleneck.

## Optimization plan (task #19)
1. Batched nvJPEG decode (nvjpegDecodeBatched) instead of serial per-image.
2. GPU-resident pipeline: decode into a GPU buffer, transform in place, no host round-trips.
3. Return a torch CUDA tensor (no final D2H copy).
4. Single batched transform kernel + texture-memory hardware bilinear.

## Honest takeaway
On CUDA, DALI is the king (NVIDIA's own). TurboLoader's realistic goal here is competitive,
not beating it. TurboLoader's real edges are the CPU path (beats PyTorch/tf.data), the Metal
path (Apple Silicon, where DALI doesn't exist), and one unified API across CPU/Metal/CUDA.
