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

## CUDA — transforms VALIDATED on a Jetson Orin; nvJPEG decode still pending

The CUDA transform path is **real**: the kernels (a faithful port of the bit-exact Metal
math) were compiled with nvcc (CUDA 11.4, aarch64) and run on an actual **Jetson Orin**
GPU — `cuda_resize_normalize` matches the numpy bilinear reference to **3.2e-05** (bit-exact,
same as Metal). It still ships nowhere by default (off-CUDA `cuda_available()` is `False`);
it builds with the flags below.

The **nvJPEG decode** path (`cuda_decode_jpeg`) is the one still unproven — Jetson ships
only a *tegra* `libnvjpeg` (no `nvjpeg.h`, different API), so it's gated behind a separate
`TURBOLOADER_ENABLE_NVJPEG=1` and needs a box with the **standard** CUDA nvJPEG (e.g. a
desktop/server with the full CUDA toolkit).

Enable on a real CUDA box (needs `nvcc` + the CUDA toolkit; use gcc 10+ for C++20):

```bash
# CUDA transforms only (validated on Jetson Orin / CUDA 11.4):
CC=gcc-10 CXX=g++-10 CUDA_HOME=/usr/local/cuda TURBOLOADER_ENABLE_CUDA=1 \
  pip install -e . --no-build-isolation

# + nvJPEG decode, on a box with the STANDARD CUDA nvJPEG (not Jetson tegra):
#   add TURBOLOADER_ENABLE_NVJPEG=1
# Jetson: nvJPEG/other CUDA libs in a non-standard dir? add TURBOLOADER_CUDA_LIB=<dir>
```

This:
- compiles **`src/cuda/cuda_transforms.cu`** with `nvcc` (the transform kernels — a
  line-for-line port of the bit-exact Metal `resize_normalize` / `crop_resize_normalize`),
  defining `TURBOLOADER_CUDA_TRANSFORMS` and linking `cudart`;
- with `TURBOLOADER_ENABLE_NVJPEG=1`, also defines `HAVE_NVJPEG` and links `nvjpeg`,
  activating the **nvJPEG full-GPU decoder** (`src/decode/nvjpeg_decoder.hpp`).

Then these become available (CUDA analogues of the Metal API):

```python
turboloader.cuda_available(), turboloader.cuda_device_name()
turboloader.cuda_resize_normalize(imgs, dst_h, dst_w, mean, std)   # mirror of metal_*
turboloader.cuda_decode_jpeg(jpeg_bytes)                            # nvJPEG full-GPU decode
```

### Validation checklist (on the GPU box)
1. `pip install` with the flags above; fix any nvcc/include issues (expect some).
2. `assert turboloader.cuda_available()` and check `cuda_device_name()`.
3. **Correctness:** `cuda_resize_normalize` should match `metal_resize_normalize` /
   numpy bit-close (same kernel math); `cuda_decode_jpeg` should match a CPU `decode_jpeg`
   the way `metal_decode_jpeg` does (~1–2 levels, JPEG-lossy range).
4. **Then** run the real FFCV / NVIDIA DALI comparison there — their home turf — to see how
   TurboLoader's CUDA path stacks up (DALI's GPU decode may well win; that's worth knowing).

Note: the older `__global__` kernels in `gpu_pipeline_integration.hpp` (`TURBOLOADER_HAS_CUDA`)
are a separate, still-unwired path; this flag uses the clean `cuda_transforms.cu` instead.
