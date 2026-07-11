# Metal resident loaders — Apple Silicon results

Machine: Apple M4 Max (unified memory), macOS. Real data: Imagenette-160 train
split (9,469 JPEGs). Methodology: warmup epoch excluded, median of 5 epochs,
`benchmarks/benchmark_metal_resident.py` (rerun with `--consume` for the
consumed variant). Correctness: every kernel output is checked against a plain
numpy reference in `tests/test_metal_resident.py` (atol 1e-5; unaligned-tail and
double-buffer-lifetime tests included).

## Images — MetalResidentLoader (fused gather+shuffle+normalize per batch)

| Pipeline | img/s |
|---|---:|
| **MetalResidentLoader, produced** (kernel complete, batch CPU-visible) | **756,599** |
| **MetalResidentLoader, consumed** (CPU touches every batch) | **432,567** |
| numpy resident baseline (same uint8 data, fancy-index + normalize + transpose) | 3,664 |
| TurboLoader on-the-fly DirectBatch (decodes JPEGs every epoch), for context | ~55–58k |

Notes, honestly stated:
- "Produced" matches the CUDA-resident contract (280k img/s on an RTX 3090):
  the GPU has written the full normalized batch and the pointer is valid. On
  unified memory that batch is CPU-visible with zero copies, hence the higher
  ceiling — the epoch is pure memory bandwidth (~3.6 GB in+out in ~12 ms ≈
  300 GB/s of the M4 Max's ~546 GB/s).
- The numpy baseline is single-threaded (that is what `array[idx]` + arithmetic
  gives you); it is the honest "what you'd write without the kernel" number,
  not a claim that numpy can't be parallelized.
- Same trade as CudaResidentLoader: dataset must fit in memory as decoded uint8
  (Imagenette-160 ≈ 727 MB), decode+resize happens once at build time.
- Lifetime contract: a yielded batch aliases a double-buffered output and is
  valid until the NEXT batch (DALI-style). `.copy()` it to keep it.

## Tokens — MetalTokenGather vs TokenDataLoader (CPU memmap)

| Pipeline | tok/s |
|---|---:|
| MetalTokenGather (resident, GPU window gather) | 528–534M |
| TokenDataLoader (CPU, numpy memmap fancy-index) | ~494–530M |

**Verdict: a tie (0.87–1.08x across runs).** The CPU memmap path is already
memory-bandwidth-bound and excellent; the GPU adds a kernel launch + int64 cast
for no net win. Use `TokenDataLoader`. `MetalTokenGather` stays in the tree as
the measured evidence, not as a recommendation.

## Arrays — MetalResidentArrays vs numpy fancy-index

(500k × 256) float32 embedding table, 4,096-row gathers:

| Pipeline | rows/s |
|---|---:|
| **MetalResidentArrays.gather** | **28–30M** |
| numpy fancy-index (`ascontiguousarray(emb[idx])`) | 5.8–5.9M |

**~4.8–5.1x** — the win case for non-image data: large resident tables with
random row access (embedding shuffles, tabular epochs, negative sampling).

## Video — MetalVideoLoader (VideoToolbox hardware decode + fused NV12 kernel)

AVFoundation/VideoToolbox decodes H.264/HEVC on the media engine (system
frameworks only — no FFmpeg dependency, ships in the ordinary macOS wheel); a
fused Metal kernel converts NV12 → RGB (MPEG chroma siting, BT.709 for HD /
BT.601 for SD, video range), bilinear-resizes and normalizes into training-ready
`(B, 3, H, W)` float32 batches.

Real 1080p H.264 clip (450 frames, crf 23) → 224px batches, warmup excluded,
median of 3 passes, every batch consumed
(`benchmarks/benchmark_metal_video.py`):

| Pipeline | frames/s |
|---|---:|
| **MetalVideoLoader** (VideoToolbox + fused kernel) | **2,544** |
| PyAV `reformat`/swscale — the STRONG CPU baseline | 534 |
| PyAV + PIL — the common real-world pattern | 166 |

**4.8× the strong baseline, 15× the common pattern** (≈85× realtime for 30fps
video). Correctness: output matches a numpy reference computed from the raw YUV
planes with the kernel's exact siting/matrix math to mean < 0.004, max < 0.03
(`tests/test_metal_video.py`); frame order/step verified via content-encoded
frame identities.

Honest notes:
- Sequential streaming only (`frame_step` keeps every Nth frame; skipped frames
  are still decoded — inter-frame codecs require it). No random access yet.
- At sharp synthetic chroma edges, different decoders' 4:2:0 upsampling choices
  legitimately differ by whole chroma steps; the kernel uses correct MPEG siting
  (horizontally co-sited, vertically centered) with bilinear interpolation.
- macOS arm64 only, by construction.
