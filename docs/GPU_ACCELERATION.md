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

## CUDA / nvJPEG — experimental, gated, UNVALIDATED here

Dormant CUDA code exists (`src/decode/nvjpeg_decoder.hpp`, `src/transforms/gpu/`,
`src/gpu/`, `src/pipeline/gpu_pipeline_integration.hpp`). **It is not built into any
released wheel, and we cannot validate it** — there is no NVIDIA GPU on the dev/CI
machines. Treat it as a starting point, not a working feature.

To attempt it on a real CUDA box:

```bash
TURBOLOADER_ENABLE_CUDA=1 CUDA_HOME=/usr/local/cuda pip install -e . --no-build-isolation
```

This defines `HAVE_NVJPEG` and links `cudart` + `nvjpeg` for the **host-API nvJPEG decode
path** (which compiles with the ordinary C++ compiler). What this flag does **not** do:

- It does **not** compile the `__global__` CUDA transform kernels in
  `gpu_pipeline_integration.hpp` — those require `nvcc` and a separate build rule
  (`TURBOLOADER_HAS_CUDA`), which is not wired up.
- Nothing here has been compiled or run by the maintainers. Expect to fix build issues.

If you have a CUDA box and want this finished and benchmarked against FFCV/DALI on their
home turf, that's the right place to do it.
