# GPU acceleration

TurboLoader's core is CPU (SIMD) by design. Two GPU transform paths exist, at very
different maturity levels. We are deliberate about which is real.

## Metal (Apple GPU) — shipped & validated

A real GPU **resize + normalize** path on Apple Silicon. JPEG **decode stays on the CPU**
(Metal has no JPEG decoder); this accelerates the *transform* stage.

```python
import turboloader as t
t.metal_available()        # True on a macOS arm64 build with a Metal device
t.metal_device_name()      # e.g. 'Apple M4 Max'

# --- transforms (list of HWC uint8 RGB images -> (N, 3, dst_h, dst_w) CHW float32) ---
t.metal_resize_normalize(imgs, 160, 160, mean=(...), std=(...))             # eval
t.metal_crop_resize_normalize(imgs, crops, flips, 160, 160)                 # RandomResizedCrop+flip
t.metal_train_transform(imgs, crops, flips, jitter, 160, 160)              # + color jitter, ONE pass
#   crops:(N,4) x,y,w,h src px | flips:(N,) 0/1 | jitter:(N,3) brightness,contrast,saturation

# --- hybrid GPU JPEG decode (CPU Huffman -> GPU IDCT -> CPU upsample/colorconvert) ---
rgb = t.metal_decode_jpeg(open('x.jpg','rb').read())   # HxWx3 uint8

# --- end-to-end GPU loader (parallel CPU decode + Metal transforms) ---
for batch in t.GpuImageLoader(paths, batch_size=64, image_size=160, train_aug=True):
    ...   # (N,3,160,160) float32
```

The fused `metal_train_transform` runs the whole ImageNet train pipeline
(RandomResizedCrop + horizontal flip + brightness/contrast/saturation jitter + normalize)
in a **single** GPU pass; `metal_decode_jpeg` is the hybrid decoder whose GPU IDCT is
proven bit-exact vs libjpeg (see `experiments/metal/`).

- **Validated:** bit-accurate vs the CPU bilinear (max err ~7e-5); see `experiments/metal/`.
- **Why it helps:** the GPU transform is ~free (≈3 µs vs ≈67 µs CPU decode), so offloading
  it makes the pipeline **decode-bound** — the DALI architecture. Apple's **unified memory**
  keeps the per-image upload to ~4 µs (a same-RAM memcpy, not a PCIe transfer), so the win
  isn't eaten by transfers the way it often is on a discrete GPU. End-to-end single-thread
  measured ~1.3–2.1× once the transform is offloaded.
- **Build:** compiled automatically on macOS arm64 (kernel compiled at runtime, so no full
  Xcode needed). Opt out with `TURBOLOADER_ENABLE_METAL=0`. Linux/Intel wheels are
  unaffected — `metal_available()` simply returns `False` there.
- **Roadmap:** GPU-resident output (hand back an MPS tensor with no copy-out) for zero-copy
  Apple-Silicon PyTorch training. Not done yet.

## CUDA (NVIDIA) — VALIDATED end-to-end; **beats DALI** on a 3090

The CUDA path is real and validated on actual hardware (two Jetson AGX Orins + an RTX 3090).
The transform kernels are a faithful port of the bit-exact Metal math: `cuda_resize_normalize`
matches the numpy bilinear reference to **3.2e-05**. On top of that, the end-to-end image loader
(`CudaImageLoader`) reaches **DALI-class throughput — and edges past DALI** on a 3090. It builds
from source with the flags below (it does **not** ship in the PyPI wheels — those are portable
CPU/Metal; CUDA needs a toolkit + GPU at build time).

### The fast path: in-C++ nvImageCodec, multi-batch-in-flight

```python
import turboloader as t, torch, glob
paths = glob.glob("imagenette2-160/train/*/*.JPEG")

ld = t.CudaImageLoader(paths, batch_size=64, image_size=160,
                       decode="nvimgcodec",   # NVIDIA nvImageCodec (what DALI uses)
                       nvimgcodec_slots=3,     # K async decode slots — default 3
                       drop_last=True)
for batch in ld:                              # GPU-resident, yielded as-completed
    x = torch.as_tensor(batch, device="cuda") # (64,3,160,160) float32, zero-copy
    train_step(x)
```

`decode="nvimgcodec"` runs the **whole** read → decode → resize → normalize → batch in
GIL-released C++. **nvImageCodec** (NVIDIA's modern codec library, what DALI moved to) decodes
each JPEG straight into a device buffer; the resize/normalize kernel reads it in place on the
same CUDA stream. With `nvimgcodec_slots=K`, K **independent** slots (each its own decoder +
stream + buffers + output ring) run on K worker threads, so one batch's host Huffman-decode
overlaps another's GPU work — the multi-batch-in-flight that DALI's `num_threads` /
`prefetch_queue_depth` buys. Each slot still synchronizes its own stream before returning, so
every batch pointer is fully ready (**no async-handoff race** — concurrency comes from the K
slots, not from skipping the sync). Batches are yielded **as completed** (out of index order
when K>1 — correct for training, which sees every batch per epoch regardless of order). nvJPEG
is `dlopen`'d at runtime; nvImageCodec too — TurboLoader links neither, so the build just needs
the header.

### Benchmark (RTX 3090, Imagenette-160, batch 64, real consumption, interleaved)

Compared **interleaved** (each loader timed adjacently every round) because the WSL/3090 host
drifts ~40% run-to-run (CPU-side, not the GPU) — only within-run relative numbers are reliable.

**On-the-fly loaders** (read a JPEG folder, decode+resize+normalize every epoch):

| Loader | vs TurboLoader |
|---|---:|
| **TurboLoader** `decode="nvimgcodec"`, `nvimgcodec_slots=3` | **1.0× (fastest)** |
| NVIDIA **DALI** (`num_threads=8/12`, best-tuned) | ~0.9× |
| PyTorch `DataLoader` (PIL, CPU) | ~0.25× |

Cleanest run (FFCV-free 2-way): TurboLoader median **28,527** (min 25,676) vs DALI **25,479**
(max 26,743) — **+12%**, TurboLoader's median above DALI's max; TurboLoader is ≥ DALI in every
run (+6–28% by host load). Journey: **9.2k → 14.5k → 22.4k → 28.5k** (nvJPEG → Python nvImageCodec
→ single-slot C++ → K-slot async).

**FFCV is faster than both — but it's not on-the-fly.** It needs a one-time offline conversion to
its `.beton` format (here pre-resized to 160), then never decodes/resizes on the fly:
**FFCV with a JPEG `.beton` ≈ 2.6× TurboLoader; with a raw `.beton` ≈ 5.9×** (the raw form does no
decode at all — just mmap-load + GPU normalize). It trades preprocessing + disk + format lock-in
for per-epoch speed. TurboLoader beats DALI, not FFCV. See `experiments/cuda/RESULTS.md`.

**Correctness.** (1) vs the libjpeg-turbo + kernel path: correlation **0.99986**, max|diff| 0.095
— not bit-exact (JPEG IDCT + chroma upsampling are implementation-defined, so any two decoders
differ; DALI/FFCV differ the same way) but numerically equivalent. (2) The K-slot async pipeline
vs the single-slot synced path: a **bijective** harness (96 batches, 3 concurrent slots, slow
consumer stressing the output ring) — every async batch matched a distinct reference batch
**exactly, 96/96**, no corruption, no ring-overwrite duplication.

### Build (on a real CUDA box; needs `nvcc` + toolkit, gcc 10+ for C++20)

> **Prebuilt CUDA wheel (no compile):** Linux x86_64 / Python 3.10 / CUDA 13.x runtime —
> attached to the [GitHub Release](https://github.com/ALJainProjects/TurboLoader/releases/tag/v2.34.1):
> ```bash
> pip install https://github.com/ALJainProjects/TurboLoader/releases/download/v2.34.1/turboloader-2.34.1+cu13-cp310-cp310-linux_x86_64.whl
> pip install nvidia-nvimgcodec-cu12
> ```
> Other Pythons/CUDA versions: build from source below.


```bash
pip install nvidia-nvimgcodec-cu12        # the nvImageCodec runtime + header

CUDA_HOME=/usr/local/cuda \
TURBOLOADER_ENABLE_CUDA=1 \               # transform kernels (cuda_transforms.cu) + cudart
TURBOLOADER_ENABLE_NVJPEG=1 \             # nvJPEG decoder (decode="gpu")
TURBOLOADER_ENABLE_NVIMGCODEC=1 \         # nvImageCodec pipeline (decode="nvimgcodec")
TURBOLOADER_CUDA_ARCH=native \            # required on CUDA 13+; or sm_86 (3090), sm_87 (Orin)
  pip install -e . --no-build-isolation
```

- `TURBOLOADER_ENABLE_NVIMGCODEC=1` **auto-discovers** the `nvidia-nvimgcodec-cu12` header from
  the installed wheel — set `TURBOLOADER_NVIMGCODEC_INCLUDE=<dir>` only to override.
- Jetson / non-standard CUDA layout: `TURBOLOADER_CUDA_INCLUDE=<dir>` (headers) and
  `TURBOLOADER_CUDA_LIB=<dir>` (libs). On Jetson, nvJPEG is the tegra variant — leave
  `TURBOLOADER_ENABLE_NVJPEG` off there.

### Decode backends (`CudaImageLoader(decode=...)`), fastest first
- `"nvimgcodec"` — in-C++ nvImageCodec, K async slots. **Fastest on-the-fly (~28.5k, beats DALI).**
  Falls back to a Python `nvimgcodec.Decoder` path (~14.5k) if the C++ pipeline isn't compiled in.
- `"gpu"` — nvJPEG batched HW-hybrid decode + fused resize/normalize (~9.2k).
- `"cpu"` — libjpeg-turbo decode + GPU transform.

### Pre-processed loaders — beat FFCV for fits-in-VRAM
Like FFCV's `.beton`, these decode+resize **once**; then TurboLoader keeps the uint8 on the GPU:
- **`CudaResidentLoader`** — upload the pre-resized uint8 dataset to the GPU once, normalize per
  epoch with a **custom single-launch kernel** (`cuda_normalize_resident`), **zero per-epoch H2D**.
  **~280k img/s on a 3090 = ~3.5× FFCV-raw (~79k)**, ~20× DALI; shuffles at ~257k via a fused
  gather+normalize kernel. Needs the uint8 dataset to fit in VRAM (`N*H*W*3` bytes). Kernel
  correct to 4e-07.
- **`CudaStreamLoader`** — for datasets **larger than VRAM**: pinned host uint8 streamed via
  `CudaStreamCore`, a **fully-in-C++** loop (persistent worker pool + async H2D on non-blocking
  streams + double-buffered prefetch; Python calls `next_batch()` once/batch, GIL released).
  **~140k img/s = ~1.6× FFCV-raw (~85k)**, near the PCIe H2D ceiling. (The earlier Python-thread
  version was GIL-capped at ~55k; the C++ core removed that wall.) Correctness bijective 0.0.

So: TurboLoader beats **DALI** on-the-fly and beats **FFCV** on pre-processed data — both
fits-in-VRAM (`CudaResidentLoader`) and streaming > VRAM (`CudaStreamLoader`).
See `experiments/cuda/RESULTS.md`.

### Lower-level CUDA API (analogues of the Metal API)
```python
turboloader.cuda_available(), turboloader.cuda_device_name()
turboloader.cuda_resize_normalize(imgs, dst_h, dst_w, mean, std)   # mirror of metal_*
turboloader.cuda_decode_jpeg(jpeg_bytes)                            # nvJPEG full-GPU decode
turboloader.cuda_nvimgcodec_init(lib, ext, device_id, num_slots)   # init the slot pipeline
turboloader.cuda_nvimgcodec_decode_resize_normalize(jpegs, h, w, mean, std, slot)  # one batch
```

Caveats: measured on one RTX 3090 (GPU-hybrid JPEG decode, no hardware JPEG unit) at 160px — a
GPU with a hardware JPEG decoder, or different sizes, could shift the balance. `nvimgcodec_slots`
trades GPU memory (~K× the input/output buffers) for throughput; 3 is a good default on a 24 GB
card. The older `__global__` kernels in `gpu_pipeline_integration.hpp` are a separate unwired
path; this build uses the clean `cuda_transforms.cu`.

## Resident (pre-processed) loaders — decode once, GPU-serve every epoch

FFCV's trick without the `.beton`: decode + resize the dataset to uint8 ONCE, keep it
GPU/unified-memory resident, then serve each epoch with one fused
gather(+shuffle)+normalize kernel launch per batch.

```python
# NVIDIA (CUDA build): ~280k img/s on a 3090 — beats FFCV-raw ~3.5x
dl = turboloader.CudaResidentLoader(paths, image_size=160, batch_size=64, shuffle=True)

# Apple Silicon (in the pip wheel): 433-757k img/s on an M4 Max (unified memory:
# "upload" is one memcpy, every GPU-written batch is a zero-copy numpy view)
dl = turboloader.MetalResidentLoader(paths, image_size=160, batch_size=256, shuffle=True)

# Any-dtype rows (embedding tables, tabular): ~5x numpy fancy-indexing on M4
ra = turboloader.MetalResidentArrays(embedding_table)   # .gather(idx) -> zero-copy view
```

Honest note: `MetalTokenGather` (token windows on GPU) measured a TIE with the CPU
memmap path (0.87–1.08x) — keep using `TokenDataLoader` for tokens. Numbers and
caveats: `benchmarks/METAL_RESIDENT_RESULTS.md`.

## Video loaders — hardware decode to training batches

```python
# Apple Silicon (in the pip wheel; no FFmpeg needed): VideoToolbox HARDWARE decode ->
# fused NV12->RGB + resize + normalize Metal kernel. 2,556 f/s on 1080p H.264 -> 224px
# (M4 Max) = 3.9x the best industry standard (OpenCV/PyAV/torchcodec), at 97-99% of the
# media engine's decode ceiling.
for batch in turboloader.MetalVideoLoader("clip.mp4", image_size=224, batch_size=32):
    ...  # (B, 3, 224, 224) float32 zero-copy view, valid until the next batch

# NVIDIA (CUDA build): GPU-resident batches with a dual decode backend —
# decode="cpu" (PyAV threaded, DEFAULT: NVDEC measured virtualization-throttled
# under WSL2) or decode="nvdec" (PyNvVideoCodec, zero H2D; right on native Linux).
for batch in turboloader.CudaVideoLoader("clip.mp4", image_size=224, decode="cpu"):
    x = torch.as_tensor(batch, device="cuda")  # zero-copy adopt

# Novel fused clip assembly: ONE kernel launch builds a (T,3,H,W) clip with the SAME
# RandomResizedCrop + flip across every frame (the video-augmentation contract),
# fused with YUV->RGB + resize + normalize:
for clip in turboloader.CudaVideoLoader("clip.mp4", image_size=224).iter_clips(
    16, train_aug=True
):
    ...
```

Cross-platform scorecard vs the industry standards (PyAV, OpenCV, decord, torchcodec),
including where decord still wins on weak-CPU hosts and the WSL2 NVDEC measurement:
`benchmarks/VIDEO_RESULTS.md`.
