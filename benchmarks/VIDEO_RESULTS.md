# Video loading — results across CPU, Metal, and CUDA

Every number: real 1080p H.264 clip (450 frames, crf 23) through the FULL
training-input pipeline — decode → RGB → bilinear resize to 224px → ImageNet
normalize → float32, every frame consumed. Warmup excluded, median of 3 passes.
`benchmarks/benchmark_video_standards.py` (per-library subprocess isolation —
decord's teardown segfaults later in-process CUDA users) and
`benchmarks/benchmark_metal_video.py`.

## Apple Silicon (M4 Max)

| Pipeline | frames/s |
|---|---:|
| **TurboLoader `MetalVideoLoader`** (VideoToolbox hw decode + fused NV12 kernel) | **2,556** |
| OpenCV (VideoCapture + resize) | 657 |
| PyAV (reformat/swscale) | 535 |
| torchcodec (Meta) | 173 |
| decord | no macOS arm64 wheel |

**3.9× the best industry standard.** The loader runs at 97–99% of the media
engine's decode rate (decode-only measures 2,637 f/s; conversion adds ~2%), and
multi-stream decode does NOT scale (1→4 concurrent streams all plateau at
~2.6k aggregate) — this is the hardware ceiling, not software headroom.
Resolution is the lever: the same pipeline does ~4,970 f/s on 720p sources.

## NVIDIA RTX 3090 (WSL2; 6-core desktop CPU)

| Pipeline | frames/s |
|---|---:|
| decord (fully-C++ decode-time resize) | 604 (high run-to-run variance: 274–619) |
| **TurboLoader `CudaVideoLoader` (decode="cpu")** | **371** |
| OpenCV (VideoCapture + resize) | 312 |
| PyAV (reformat/swscale) | 131 |
| TurboLoader `CudaVideoLoader` (decode="nvdec") | 96 |
| torchcodec | not installable (needs system FFmpeg libs) |

Honest reading, in order:
- **decord leads pure throughput on this weak-CPU box** — its decode+resize loop
  is entirely C++ with zero per-frame Python. Our cpu backend pays ~2 ms/frame
  of Python-side plane copies (already halved from 5.5 ms by plane-direct pinned
  staging); closing the rest means moving the frame loop into C++ (roadmap).
- What decord does NOT give you: our output is **GPU-resident, normalized, CHW
  float32** ready for the model (decord's pip build hands back CPU numpy that
  training still has to normalize correctly and upload); and our conversion
  matrix/siting is the numpy-validated kernel, identical on Metal and CUDA.
- **NVDEC under WSL2 is virtualization-throttled**: 130 f/s raw decode at ~99%
  reported engine utilization vs 1,453 f/s CPU FFmpeg decode on the same box —
  hence `decode="cpu"` is the default and `decode="nvdec"` (96 f/s end-to-end
  here) is the right choice only on native Linux where NVDEC runs at spec.
  Measure both on your machine; the flag is one word.

## The novel kernel: fused clip assembly (`iter_clips`)

`yuv420_clip_crop_normalize_kernel` builds a whole `(T, 3, H, W)` training clip
in ONE launch: the SAME RandomResizedCrop window + horizontal flip applied to
every frame (the video-augmentation contract — spatial aug must be consistent
across time), fused with YUV→RGB + resize + normalize. Standard stacks do this
as 4–5 separate passes. Exposed as
`CudaVideoLoader.iter_clips(clip_len, train_aug=True)` with torchvision-parity
crop sampling; measured 356 f/s (22 clips/s of 16×224² augmented clips) on the
3090 box with cpu decode.

Correctness pins (`tests/test_cuda_video.py`, `tests/test_metal_video.py`):
- `train_aug=False` clips equal the plain batch path at 1e-6 (the crop math
  reduces exactly to the resize path);
- random crops verified per-frame against numpy references using the reported
  rect/flip; conversion matches raw-YUV numpy references at mean < 0.004.

## Bugs the real hardware found (fixed, with regression tests)

- **NVDEC surface-pool recycling**: retaining PyNvVideoCodec frame objects does
  NOT keep their device pointers valid past the next `Decode()` call — a
  retained batch became N copies of the last surface (caught by the
  cpu-vs-nvdec agreement test: mean diff 0.24). Frames now copy D2D into
  loader-owned staging immediately.
- **decord + CUDA teardown**: in-process sequencing of decord before a CUDA
  user segfaults; the standards benchmark isolates every library in its own
  subprocess.
