# Benchmarks — full methodology and results

The complete, honest scorecard (corrections included). Headlines live in the
[README](../../README.md); raw scripts in [`benchmarks/`](../../benchmarks/);
GPU details in [`experiments/cuda/RESULTS.md`](../../experiments/cuda/RESULTS.md),
[`benchmarks/METAL_RESIDENT_RESULTS.md`](../../benchmarks/METAL_RESIDENT_RESULTS.md),
[`benchmarks/VIDEO_RESULTS.md`](../../benchmarks/VIDEO_RESULTS.md), and
[`benchmarks/E2E_TRAINING_RESULTS.md`](../../benchmarks/E2E_TRAINING_RESULTS.md).


Measured on **Apple Silicon** over **Imagenette-160** (9,469 real ImageNet JPEGs →
resize 160×160 → ImageNet-normalize → batched CHW float32, batch 64). To control for
thermal throttling, every loader is built once, warmed up one epoch, then timed over
**5 interleaved rounds** (each loader runs once per round); the table reports the
median. Output is verified correct against torchvision (mean abs diff ≈ 0.04, bilinear
antialiasing only).

**Image — on-the-fly decode** (re-decode every epoch; for datasets too large to cache
or with per-epoch random augmentation):

| Loader | img/s (median) | vs tf.data |
|---|---:|---:|
| **TurboLoader `DataLoader`** (`output_format='pytorch'`, nw=6) | **~55,000** | **2.0×** |
| TensorFlow `tf.data` (AUTOTUNE) | ~27,300 | 1.00× |
| PyTorch `DataLoader` (PIL, 8 persistent workers) | ~20,500 | 0.75× |

**Image — cached** (decoded tensors held in RAM; both sides consume identically via
`np.sum`, i.e. delivered as numpy/torch-ready batches — the PyTorch use case):

| Loader | img/s (median) | vs tf.data.cache |
|---|---:|---:|
| **TurboLoader** (`cache_decoded=True`, prefetch) | **~67,000** | **1.9×** |
| TensorFlow `tf.data.cache()` (+ `.numpy()` materialize) | ~35,100 | 1.00× |

(For *TF-native* consumption that stays in tf tensors, `tf.data.cache()` is faster —
TurboLoader's cache win is for delivering numpy/torch batches.)

**Pre-processed (TBL-RAW)** — decode once (`preprocess_to_tbl`: 9,469 images in
6 s), then serve every epoch from an mmap through one fused parallel SIMD pass
(`normalize_u8_gather`), output bit-identical to the TAR pipeline. M4 Max,
Imagenette, 160px, every configuration in its OWN subprocess (cross-stage
thread-pool/allocator state measurably contaminates in-process comparisons —
a lesson re-learned here), reported at TWO consumption levels
(`benchmark_tbl_raw.py`) so under-consumption artifacts are ruled out:

| Pipeline | produce img/s | np.sum-consumed | peak RSS |
|---|---:|---:|---:|
| on-the-fly TAR (decode every epoch) | 32,417 | 31,130 | 519 MB |
| **TBL-RAW, `prefetch_batches` default (training)** | 144k† | **98,560** | 1,008 MB (file-backed, evictable) |
| **TBL-RAW, `prefetch_batches=0` (raw serve)** | **531,367** | 88,945 | 931 MB (file-backed, evictable) |
| `cache_decoded=True` (float32 in RAM) | 62,493 | 59,744 | **8,796 MB** (anonymous) |

† prefetch's produce figure is thread-scheduling noise (a no-op consumer makes
the producer thread thrash); its stable, honest number is the consumed one —
which prefetch IMPROVES (98.6k vs 88.9k sync) because production overlaps the
consumer, which is the whole point for training loops.

TBL-RAW beats the float32 RAM cache at BOTH consumption levels in BOTH modes
while the "memory" it uses is clean page cache the kernel can drop at any
time. Honest notes: LZ4 on decoded photos = **1.06x** (why RAW defaults to
`compression=False` — the real compression is uint8-not-float32, 4x); the .tbl
is bigger than the JPEG TAR (727 vs 263 MB — disk traded for decode); no
per-epoch random crop (serve-time hflip only) — full-aug training stays on the
TAR pipeline. The GPU-resident loaders ingest the same file and skip their
decode-all pass. Details: [tbl_v2_format.md](../tbl_v2_format.md).

**LLM tokens** (real text, 55M-token memory-mapped corpus, `seq_len=1024`, next-token):

| Loader | sequences/s (median) |
|---|---:|
| **TurboLoader `TokenDataLoader`** | **~467,000** |
| numpy memmap idiom (nanoGPT `get_batch`) | ~251,000 |

On CUDA, measured **delivered to device** at GPT-2 shape (RTX 3090, 32×1024,
`benchmarks/benchmark_token_loader.py`): TokenDataLoader **168M tok/s vs 88M** for
the exact nanoGPT idiom incl. `.pin_memory().to(non_blocking=True)` — **1.9×** —
and `TokenDataLoader(device="cuda")` (zero-alloc pinned ring, side-stream H2D,
CUDA-event-guarded reuse) matches that while yielding ready GPU tensors. In a full
GPT training loop all paths are within ~2% (the model hides the pipeline —
e2e details in [E2E_TRAINING_RESULTS.md](../../benchmarks/E2E_TRAINING_RESULTS.md)).

**Transforms** (per-image throughput vs torchvision): Resize **2.7×**, ImageNetNormalize
**3.3×**, HFlip ~1.0×. For CenterCrop, torchvision returns a **lazy strided view** (moves
zero bytes); compared against TurboLoader's real contiguous crop that looks like 0.45×,
but when torchvision actually materializes the crop (`.contiguous()`, required before
batching/most ops) it drops to ~23k img/s and **TurboLoader's contiguous crop is ~6.8×
faster** (155k vs 23k). Like the cache, this is a lazy-vs-eager comparison; for the
realistic crop→batch path TurboLoader wins.

> Earlier drafts quoted single-run figures (~42k, "1.4×") and a "cached epoch" in the
> tens-of-millions img/s. Those were artifacts (thermal noise; a no-op loop over aliased
> cached arrays) and were replaced with the interleaved, identical-consumption medians
> above. Numbers are hardware-dependent — run `benchmarks/` yourself.

The fast path runs decode + resize + normalize + batch assembly in C++ across a thread
pool with zero Python per-sample work. Use it like this:

```python
loader = turboloader.DataLoader(
    'imagenet.tar', batch_size=64, num_workers=6,
    output_format='pytorch',          # (N, C, H, W) float32 array per batch
    image_size=160,                   # exact resize, done in C++
    transform=turboloader.ImageNetNormalize())
for epoch in range(epochs):           # re-iterable
    for images, meta in loader:       # images.shape == (64, 3, 160, 160)
        train_step(images)
```

Honest caveats:
- **Run it yourself** (`benchmarks/`) — results depend heavily on hardware, image size,
  and pipeline; Linux `fork`-based PyTorch workers shift the PyTorch numbers a lot.
- **Decode backend differs**: TurboLoader uses libjpeg-turbo; the PyTorch baseline uses PIL.
- The `output_format='dict'` path returns per-sample dicts and stacks in Python
  (GIL-bound), so it is much slower — use it only when you need per-sample metadata.

For **large source images**, the default path also wins: on 768×768 JPEGs resized to
160 it runs ~15,000 img/s — faster than even an expertly-tuned `tf.data` pipeline using
manual `decode_jpeg(ratio=...)` (~14,400) — because it picks the libjpeg-turbo DCT
scaled-decode factor automatically (you don't have to know to set `ratio`).

### GPU loaders (NVIDIA & Apple)

On **NVIDIA**, `CudaImageLoader(decode="nvimgcodec")` runs the whole decode + resize + normalize
+ batch in GIL-released C++ via **nvImageCodec** (the codec DALI uses), with K independent decode
slots overlapping batches (multi-batch-in-flight). Among **on-the-fly** loaders (read a JPEG
folder, decode+resize every epoch) on an **RTX 3090** (Imagenette-160, batch 64, real consumption,
interleaved rounds to control for ~40% host drift):

| On-the-fly loader | vs TurboLoader |
|---|---:|
| **TurboLoader** `decode="nvimgcodec"`, `nvimgcodec_slots=3` | **1.0× (fastest)** |
| NVIDIA **DALI** (`num_threads=8`, best-tuned) | ~0.9× (TurboLoader **+12%** cleanest run) |
| PyTorch `DataLoader` (PIL, CPU) | ~0.25× |

**TurboLoader beats DALI** (median above DALI's max in the cleanest run), output bijectively
verified correct. For **on-the-fly** loading FFCV is faster (~2.6–5.9×) — but it requires an
offline conversion to its `.beton` format.

**Pre-processed loaders** (decode+resize once, like FFCV's `.beton`) — here TurboLoader turns the
tables:

| Pre-processed loader | img/s | |
|---|---:|---|
| **TurboLoader `MetalResidentLoader`** (Apple M4 Max, unified memory: no H2D exists) | **~757,000 produced / ~433,000 consumed** | ships in the pip wheel |
| **TurboLoader `CudaResidentLoader`** (fits-in-VRAM: upload uint8 once, GPU-resident) | **~280,000** | **beats FFCV ~3.5×** |
| **TurboLoader `CudaStreamLoader`** (streaming, dataset > VRAM; fully-C++ loop) | **~140,000** | **beats FFCV ~1.6×** |
| FFCV, raw `.beton` (streams mmap→H2D each epoch, worker processes) | ~85,000 | |

On **Apple Silicon** the resident trick is even better than on NVIDIA: memory is unified, so
"upload" is one memcpy and every GPU-written batch is a **zero-copy numpy view**.
`MetalResidentLoader` serves each epoch as one fused gather+shuffle+normalize kernel launch per
batch; `MetalResidentArrays` does the same for any-dtype rows (embedding tables: ~5× numpy
fancy-indexing). Honest null result included: `MetalTokenGather` ties the CPU memmap path
(0.87–1.08×) — keep using `TokenDataLoader` for tokens.

**Video**: `MetalVideoLoader` (macOS arm64, in the pip wheel — no FFmpeg needed) drives
VideoToolbox **hardware** H.264/HEVC decode into a fused NV12→RGB+resize+normalize Metal
kernel: real 1080p → 224px training batches at **~2,550 frames/s** on an M4 Max —
**3.9× the best industry standard** (OpenCV 657, PyAV 535, torchcodec 173) and 97–99% of
the media engine's hardware decode ceiling. On NVIDIA, `CudaVideoLoader` (CUDA build)
lands GPU-resident batches via a dual decode backend (threaded CPU decode by default;
NVDEC opt-in — measured virtualization-throttled under WSL2) plus a novel **fused
clip-assembly kernel** (`iter_clips`: consistent RandomResizedCrop+flip across a whole
clip + YUV→RGB + resize + normalize in ONE launch). For **training on a labeled video
dataset**, `VideoDatasetLoader` (`root/class_x/*.mp4` → `(clips, labels, meta)` CUDA
batches; threaded decode, deterministic per-(seed, epoch) sampling) trains a real
r3d_18 classifier **1.16× faster end-to-end** than the PyTorch DataLoader + PyAV
recipe — the first e2e video training benchmark, honest caveat included (both
pipelines are decode-bound; see
[E2E_TRAINING_RESULTS.md](../../benchmarks/E2E_TRAINING_RESULTS.md)). Single-file
scorecard incl. where decord
still wins on weak-CPU hosts: [benchmarks/VIDEO_RESULTS.md](../../benchmarks/VIDEO_RESULTS.md).

`CudaResidentLoader` uses a custom single-launch normalize kernel + fused gather (shuffles at
~257k) and **beats FFCV ~3.5×** when the pre-processed uint8 dataset fits in VRAM (very common:
fine-tuning, per-GPU shards, small/medium sets). For datasets **larger than VRAM**,
`CudaStreamLoader` runs the whole iteration GIL-free in C++ (`CudaStreamCore`: worker pool + async
H2D on non-blocking streams + prefetch) and **beats FFCV's streaming ~1.6×** (~140k vs ~85k, near
the PCIe transfer ceiling). So TurboLoader beats **DALI** on-the-fly and **FFCV** on pre-processed
data — both fits-in-VRAM and streaming. On **Apple Silicon**, `GpuImageLoader` offloads
resize+normalize (and a hybrid GPU JPEG decode) to Metal — where neither DALI nor FFCV runs at
all. CUDA is a build-from-source path (not
in the PyPI wheels); see [GPU acceleration](../GPU_ACCELERATION.md) for flags, usage, and the
full write-up (`experiments/cuda/RESULTS.md`).

### Implementation notes
- **Direct-batch path** (`src/pipeline/direct_batch_loader.hpp`): the default fast path
  is FFCV/`tf.data`-style — a persistent thread pool reads JPEG bytes by index and
  decodes → resizes → normalizes **directly into the output batch buffer** in one
  parallel pass (no worker queue, no per-sample heap copy, no serial collection).
  Verified memory-safe and race-free (disjoint slot writes, const mmap reads, atomic
  cursor, per-thread decoders).
- **Automatic DCT scaled decode**: large JPEGs are decoded at the nearest libjpeg-turbo
  scale ≥ target, then finely resized — much faster than full-decode + resize.
- **Resize convention**: half-pixel centers (`align_corners=False`), matching
  PIL/OpenCV/PyTorch/TF (agrees with torchvision plain bilinear to ~0.4/255; the only
  remaining difference vs torchvision's default is its antialiasing low-pass filter).
- SIMD transforms (AVX2/AVX-512/NEON), libjpeg-turbo decode, lock-free SPSC queues
  (legacy/dict + remote path), persistent `std::thread` pool (`src/core/parallel_for.hpp`).
- The GIL is released during C++ processing.
- **OpenMP is opt-in** (`TURBOLOADER_ENABLE_OPENMP=1`); off by default because linking a
  second OpenMP runtime crashes alongside PyTorch on macOS — the thread pool replaces it.
