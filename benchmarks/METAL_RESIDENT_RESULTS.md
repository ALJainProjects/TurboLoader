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

## Video

Scoped, not built: the right shape is VideoToolbox hardware H.264/HEVC decode
feeding the existing Metal color/resize kernels. Tracked as roadmap; nothing in
the tree claims video support today.
