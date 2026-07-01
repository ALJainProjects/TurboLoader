# CUDA loader benchmark (RTX 3090, CUDA 13.3)

Imagenette 160px, batch 64, real consumption (every batch adopted into a CUDA tensor and
summed), median of timed epochs. Loaders are compared INTERLEAVED (each timed adjacently every
round) because this WSL/3090 host drifts ~40% run-to-run (CPU/host-side, not the GPU — GPU sits
at 67 C, no thermal throttle). **Only within-run relative comparisons are trustworthy; absolute
img/s are not stable across sessions.**

## Two categories — on-the-fly vs pre-processed

**On-the-fly loaders** (read a JPEG folder, decode + resize + normalize *every epoch* — no
preprocessing):

| Loader | relative | notes |
|---|---:|---|
| **TurboLoader-CUDA** (`decode="nvimgcodec"`, 3 async slots) | **fastest on-the-fly** | K-slot multi-stream in-C++ pipeline |
| NVIDIA DALI (`num_threads=8/12`, best-tuned) | ~0.9–0.95× TBL | GPU decode + resize |
| TurboLoader-CUDA (`decode="gpu"`, nvJPEG) | ~0.4× TBL | nvJPEG batched HW-hybrid |
| PyTorch DataLoader (PIL, CPU) | ~0.25× TBL | CPU decode + transform |

Cleanest measurement (FFCV-free 2-way interleaved, least contention): **TBL slots=3 median
28,527** (min 25,676) vs **DALI nt=8 median 25,479** (max 26,743) — **+12%, TBL median above DALI
max**. In noisier runs with more loaders in-process the margin shrinks to ~+6% (overlapping
ranges); TurboLoader is consistently **≥ DALI**, by ~6–28% depending on host load. TBL slots=3 =
3 decoders × ~5 host threads ≈ 15; DALI nt=12 = 12 — comparable budget, TBL wins both.

**Pre-processed loader (FFCV)** — requires a one-time offline conversion to its `.beton` format
(here pre-resized to 160×160), so it does **less per-epoch work** than the on-the-fly loaders:

| FFCV mode | img/s (median) | vs TBL | what it does per epoch |
|---|---:|---:|---|
| FFCV, **raw** `.beton` (797 MB) | ~80,000 | ~5.9× | mmap load + GPU normalize — **no decode, no resize** |
| FFCV, **JPEG** `.beton` (143 MB) | ~35,000 | ~2.6× | CPU-decode tiny pre-resized JPEGs + normalize — **no resize** |

**FFCV is substantially faster than TurboLoader and DALI.** It trades a one-time preprocessing
step + a format lock-in + (for raw) 5× the disk for per-epoch speed, and never re-decodes/resizes.
TurboLoader does **not** beat FFCV; it beats DALI among loaders that read a JPEG folder directly.

> **Correction:** an earlier version of this file claimed "FFCV ~13,800, TurboLoader ahead." That
> was wrong — a doubly-flawed measurement: the FFCV `Loader` was recreated every epoch (worker-
> spawn + pipeline-JIT overhead dominated 3 short epochs) *and* the `.beton` stored raw uint8 (so
> FFCV wasn't even decoding). Reused-loader, steady-state, with both raw and JPEG `.beton`s, FFCV
> is 35k–80k — well ahead of TurboLoader.

## The journey: 9.2k -> 14.5k -> 22.4k -> 28.5k
1. **nvImageCodec, the breakthrough.** DALI's speed was never its kernels — it was the decoder.
   DALI moved off `nvjpegDecodeBatched` to **nvImageCodec** (~21.6k img/s decode in isolation on
   a 3090, faster than DALI's whole pipeline). Feed its GPU-decoded HWC uint8 images (contiguous,
   `strides: None`) straight into our `resize_normalize` kernel, zero extra copies. 9.2k -> 14.5k.
2. **In-C++ pipeline, chasing DALI's GIL-free data path.** The 14.5k path still paid per-batch
   Python: 64x `__cuda_array_interface__` dict extraction + `ptrs/ws/hs` list comprehensions,
   under the GIL. Move the WHOLE read->decode->resize->normalize->batch into one GIL-released C++
   call: nvImageCodec's C API (dlopen'd at runtime, no link dependency) decodes each JPEG into a
   persistent device buffer, then the kernel reads it in place on the SAME CUDA stream. Python
   only reads bytes (GIL-released C++ `read_files`) and wraps the output pointer. 14.5k -> 22.4k.
3. **K-slot async multi-stream pipeline — beats DALI.** The single-slot path decodes ONE batch at
   a time, so the GPU idles between batches while one thread does host prep. Add K INDEPENDENT
   slots (each its own decoder + CUDA stream + device buffers + output ring), driven by K worker
   threads: one batch's host Huffman-decode overlaps another's GPU work — exactly the multi-batch-
   in-flight that DALI's `num_threads`/`prefetch_queue_depth` buys. Crucially, each slot STILL
   `cudaStreamSynchronize`s before returning, so every pointer is fully ready — **no async-handoff
   race; concurrency comes from K slots, not from skipping the sync.** Decoder host threads split
   ~hw/K to avoid oversubscription. Batches yield as-completed (out of order, fine for training).
   22.4k -> 28.5k (slots=3). This is what overtook DALI.

## Correctness
1. **vs libjpeg-turbo + kernel** (single batch): nvImageCodec and libjpeg-turbo are different JPEG
   decoders, so not bit-exact (IDCT + chroma upsampling are implementation-defined) but numerically
   equivalent — correlation **0.99986**, max|diff| 0.095 (same variance DALI/FFCV have vs CPU).
2. **K-slot async vs single-slot synced** (bijective harness): 96 batches through the real 3-slot
   loader, out of order, with a deliberately slow consumer stressing the output ring, each matched
   a DISTINCT single-slot reference batch EXACTLY — **96/96, zero corruption, zero ring-overwrite
   duplication**. Bijection (every reference used once) is what catches a ring overwrite copying
   one batch over another. The async pipeline is provably not racing.

## What did NOT work (measured, reverted)
- **Parallelizing the per-image setup loop** (create-codestream + get-info + create-image, x64)
  across a thread pool — the host-side work DALI fans out across `num_threads`. It REGRESSED
  throughput (20.9k vs 22.4k serial): those 64 ops are too cheap to amortize pool dispatch, and
  the batched nvImageCodec decode already parallelizes internally across its own threads. (It
  also first exposed a real bug: the global `ThreadPool` is single-caller — shared cursor/remaining
  state — so running it concurrently with the reader thread's `read_files` segfaulted. A dedicated
  pool fixed the crash but the op was still a net loss, so it was reverted to serial.)

## What did NOT work (measured, reverted)
- **Parallelizing the per-image setup loop** across a thread pool REGRESSED throughput (20.9k vs
  22.4k serial): those 64 header-parse/handle-create ops are too cheap to amortize pool dispatch,
  and the batched decode already parallelizes internally. (It also exposed a real bug — the global
  `ThreadPool` is single-caller, so running it concurrently with `read_files` segfaulted.) The win
  came instead from parallelizing across BATCHES (K slots), not within a batch.

## Honest verdict
With the K-slot async multi-stream in-C++ nvImageCodec pipeline, **TurboLoader-CUDA beats NVIDIA
DALI among on-the-fly loaders** — the ones that read a JPEG folder and decode+resize every epoch
(+12% in the cleanest interleaved run, TBL median above DALI max; ≥ DALI in every run), output
bijectively verified correct — and far ahead of PyTorch (~5.4×). It does **not** beat **FFCV**:
FFCV is ~2.6× (JPEG `.beton`) to ~5.9× (raw `.beton`) faster, because it front-loads decode/resize
into a one-time offline conversion and never repeats them per epoch — a different trade-off (fast
epochs at the cost of a preprocessing step, disk, and format lock-in). TurboLoader also runs the
*same* unified loader API on the CPU and on Apple Metal, where neither DALI nor FFCV runs at all.
Caveats: one RTX 3090 (GPU-hybrid JPEG decode) at 160px; the host drifts ~40% run-to-run so only
within-run relative numbers hold; a GPU with a hardware JPEG unit or different sizes could shift
the DALI margin.

## Beating FFCV at its own (pre-processed) game — GPU-resident

FFCV's speed comes from an offline `.beton`: pre-resized, decode-once. TurboLoader can do the same
trade — and go further by keeping the pre-processed data **on the GPU**. `CudaResidentLoader`
decodes+resizes the dataset to uint8 once, uploads it to the GPU, then normalizes per epoch with a
**single-launch custom kernel** (`normalize_nhwc_to_nchw`, `cuda_normalize_resident`) and **zero
per-epoch H2D**. Isolated (each loader alone — the real single-loader-feeds-training case):

| Loader (pre-processed) | img/s | vs FFCV-raw |
|---|---:|---:|
| **TurboLoader `CudaResidentLoader`** (GPU-resident uint8, custom kernel) | **~280,000** | **3.5×** |
| FFCV, raw `.beton` (mmap + H2D per epoch) | ~79,000 | 1.0× |
| FFCV, JPEG `.beton` | ~35,000 | 0.44× |

Correctness of the kernel: max|diff| **4e-07** vs the numpy reference (bit-exact bar FP rounding).
**TurboLoader beats FFCV ~3.5× for datasets that fit in VRAM** (uint8: `N*H*W*3` bytes —
Imagenette-160 = 727 MB; a 24 GB card holds ~33 M such images / most fine-tuning + per-GPU shards).
`CudaResidentLoader` **shuffles at full speed** too — a fused gather+normalize kernel
(`cuda_normalize_resident_gather`, output n reads `src[perm[n]]`) does shuffled epochs at
**~257k img/s** (vs ~31k with a torch gather copy), gather output correct to 4e-07.

Two honest limits:
1. **Interleaving is invalid here.** Interleaved with FFCV, `CudaResidentLoader` measures ~47k —
   but that's contamination: FFCV's 8–16 persistent worker *processes* starve the GPU-resident
   loader's light Python loop for CPU. Isolated (one loader at a time = real use) is the fair
   measure; there it's ~280k. (Interleaving was right for TBL-vs-DALI — like in-process footprints.)
2. **Streaming (dataset > VRAM) is still FFCV's.** `CudaStreamLoader` (K async-H2D slots: pinned
   host uint8 → per-slot stream H2D → normalize kernel, K threads) reaches **~55k** — better than
   the naive single-stream (~40k) and than on-the-fly (~28k), but still **below FFCV-raw's 79k**.
   More slots don't help (plateaus ~55–58k); the wall is the **GIL**, not PCIe — FFCV uses GIL-free
   worker *processes* while `CudaStreamLoader` uses threads, so per-batch Python + the GIL cap it.
   Beating FFCV streaming would need a fully-C++ iteration/prefetch loop (no Python per batch) or
   multiprocessing with a device-pointer handoff. GPU Direct Storage (NVMe→GPU DMA) would be the
   ideal streaming answer but isn't available under WSL2.

### Amortized (including FFCV's conversion)
FFCV's `.beton` conversion for Imagenette (9,469 imgs, 8 workers) is only **~1.6 s**. Break-even
vs on-the-fly TurboLoader ≈ **1.6 / (t_onthefly − t_ffcv)/epoch ≈ 3 epochs**. So amortization does
**not** save on-the-fly loaders: past ~3 epochs FFCV wins total wall-clock too. The way to beat
FFCV is per-epoch throughput — which `CudaResidentLoader` does (fits-in-VRAM), by also
pre-processing (a comparably cheap one-time step) and then caching on the GPU.

## Reproduce
```python
import turboloader as t, torch, glob
paths = glob.glob("imagenette2-160/train/*/*.JPEG")

# On-the-fly (any dataset size, no preprocessing): beats DALI
ld = t.CudaImageLoader(paths, batch_size=64, image_size=160,
                       decode="nvimgcodec", nvimgcodec_slots=3, drop_last=True)  # K async slots
for batch in ld:                                   # GPU-resident, yielded as-completed
    x = torch.as_tensor(batch, device="cuda")      # (64,3,160,160) float32, zero-copy

# Pre-processed, fits-in-VRAM (~280k img/s, beats FFCV ~3.5x): decode+upload once, normalize on GPU
rld = t.CudaResidentLoader(paths, image_size=160, batch_size=64, shuffle=True, drop_last=True)
for batch in rld:
    x = torch.as_tensor(batch, device="cuda")      # (64,3,160,160) float32, zero H2D per epoch
```
Build: `TURBOLOADER_ENABLE_CUDA=1 TURBOLOADER_ENABLE_NVJPEG=1 TURBOLOADER_ENABLE_NVIMGCODEC=1
TURBOLOADER_NVIMGCODEC_INCLUDE=<wheel>/nvidia/nvimgcodec/include TURBOLOADER_CUDA_ARCH=native pip
install -e .` plus `pip install nvidia-nvimgcodec-cu12`.
