# Benchmark Overview

Performance analysis of TurboLoader 2.26.2.

## Executive Summary

TurboLoader's direct-batch image loader (one parallel pass: decode -> resize ->
normalize straight into the output batch buffer) reaches **~39,100 img/s on the fly**
and **~65,499 img/s with the decoded cache enabled** (`cache_decoded=True`) on Apple
Silicon over Imagenette-160. Measured against the same dataset and the same forced,
real consumption, that makes it:

- **1.3x faster** than TensorFlow `tf.data` (AUTOTUNE): ~30,154 img/s
- **2.1x faster** than PyTorch `DataLoader` (8 persistent workers): ~18,991 img/s
- For LLM token streams, `TokenDataLoader` sustains **~441M tokens/s** vs **~163M
  tokens/s** for the NumPy `memmap` idiom (**2.7x**)

All image numbers use `output_format='pytorch'` (CHW), batch size 64, with a warmup
epoch and the median of 3 timed epochs under real consumption that forces
materialization of every batch.

## Latest Results (2.26.2)

### Image Throughput (Imagenette-160)

| Loader | Throughput (img/s) | Relative |
|--------|--------------------|----------|
| **TurboLoader (cached, `cache_decoded=True`)** | **~65,499** | 1.7x vs on-the-fly |
| **TurboLoader (on-the-fly)** | **~39,100** | 1.0x (reference) |
| TensorFlow `tf.data` (AUTOTUNE) | ~30,154 | TurboLoader is 1.3x |
| PyTorch `DataLoader` (8 persistent workers) | ~18,991 | TurboLoader is 2.1x |

The on-the-fly path decodes, resizes, and normalizes each image in a single parallel
pass into the output batch buffer, with automatic libjpeg-turbo DCT scaled decode for
large images. The cached path stores decoded tensors so subsequent epochs skip JPEG
decoding entirely.

### LLM Token Streams

| Loader | Throughput (tokens/s) | Relative |
|--------|-----------------------|----------|
| **TurboLoader `TokenDataLoader`** | **~441M** | TurboLoader is 2.7x |
| NumPy `memmap` idiom | ~163M | 1.0x (reference) |

### Test Configuration

- **Hardware:** Apple Silicon
- **Dataset:** Imagenette-160 — 9,469 real ImageNet JPEGs resized to 160 px
- **Output format:** `pytorch` (CHW)
- **Batch size:** 64
- **Measurement:** real consumption forcing materialization of each batch; one warmup
  epoch followed by the median of 3 timed epochs

### A Note on Worker Scaling

`num_workers` does not mean the same thing across loaders, so it is not a fair single
knob to sweep:

- **PyTorch** scales with `num_workers` because each worker is a separate OS process.
- **TurboLoader's** fast path is a single process-wide C++ thread pool that is already
  saturated at one "worker" — adding workers does not change its throughput.
- **TensorFlow** uses `AUTOTUNE` and picks its own parallelism.

The numbers above use each loader's recommended/best configuration: PyTorch with 8
persistent workers, `tf.data` with AUTOTUNE, and TurboLoader's single saturated thread
pool.

## Multi-Modality

The same engine drives more than images:

- **Images** packed in WebDataset TAR shards via the direct-batch loader.
- **LLM token streams** via `TokenDataLoader` (see the token numbers above).
- **Generic `(N, ...)` arrays** via `ArrayDataLoader`.

Output can be emitted as NumPy, PyTorch CHW, or TensorFlow HWC. Distributed training is
supported with DDP-safe equal/disjoint sharding.

## Transform Performance

Transforms run on SIMD-vectorized kernels (NEON on Apple Silicon / ARM, AVX2 and
AVX-512 on x86). Resize uses half-pixel sampling that matches PIL/PyTorch/TF, with
optional antialiasing. Because transforms are fused into the same parallel pass that
produces the output batch, their cost is already included in the end-to-end throughput
numbers reported above rather than measured in isolation.

## Memory Usage

The direct-batch loader writes decoded, resized, normalized samples into a single
reusable output batch buffer instead of allocating per-sample intermediates, which
keeps the steady-state working set small. Enabling `cache_decoded=True` trades memory
for speed by retaining decoded tensors across epochs.

## Methodology

See the [Benchmark Setup Guide](../benchmark_setup.md) for:

- Installing the frameworks compared (TurboLoader, PyTorch, TensorFlow)
- Dataset preparation (Imagenette-160)
- Measurement technique (real consumption, warmup + median of timed epochs)

## Why no FFCV / NVIDIA DALI numbers (yet)

These are the most relevant high-performance comparisons, but they can't be measured
fairly on the hardware these numbers were collected on (Apple Silicon CPU):

- **NVIDIA DALI** is CUDA-only — there is no Apple Silicon / CPU-only build. A fair
  comparison needs a Linux + NVIDIA GPU box. Note that DALI's **GPU** JPEG decode could
  well *out-throughput* TurboLoader's CPU pipeline — that's a result worth having, not
  one to fear.
- **FFCV** is Linux/x86-oriented and its wheel does not build on macOS arm64; it also
  uses its own `.beton` format (a conversion step).

So the honest statement is: TurboLoader is **measured faster than PyTorch DataLoader and
tf.data on CPU** (above), and FFCV/DALI are a deliberately open comparison — they belong
on Linux + GPU and will be added when run there.

## Reproducing Results

```bash
# Clone repository
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader

# Install (prebuilt manylinux wheels on Linux x86_64 / aarch64)
pip install turboloader            # torch is optional: pip install turboloader[torch]

# Run the image benchmark against Imagenette-160 (batch 64)
cd benchmarks
python benchmark_comparison.py --dataset /path/to/imagenette-160 --batch-size 64
```

## Questions?

- [Benchmark Setup](../benchmark_setup.md) - How to reproduce these numbers
- [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
