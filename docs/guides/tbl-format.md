<!-- Generated from the project wiki page https://github.com/ALJainProjects/TurboLoader/wiki/TBL-RAW-Preprocessed-Pipeline — edit there. -->
> Canonical, always-current version: **[TBL RAW Preprocessed Pipeline](https://github.com/ALJainProjects/TurboLoader/wiki/TBL-RAW-Preprocessed-Pipeline)** on the wiki.

# TBL-RAW — decode once, mmap-serve every epoch

FFCV's core insight, portably and without the `.beton` lock-in: JPEG decode dominates input-pipeline cost, and for many-epoch training you only need to pay it once.

```python
import turboloader as tl

# ONE-TIME: parallel decode + resize through the C++ fast path, RGB uint8 samples into TBL v2
tl.preprocess_to_tbl("imagenet.tar", "imagenet_160.tbl", image_size=160)     # 9,469 Imagenette images: ~6 s on an M4 Max

# EVERY RUN: serve from a memory map — no decode, instant startup
loader = tl.DataLoader("imagenet_160.tbl", batch_size=64, transform=tl.ImageNetNormalize(), shuffle=True)
for batch, meta in loader:            # (64, 3, 160, 160) float32; meta['indices'] aligns labels
    train_step(batch, labels[meta["indices"]])
```

## Why this is the efficiency frontier

- **Zero decode per epoch.** Serving is one fused, parallel, GIL-released SIMD pass (`normalize_u8_gather`): rows are gathered straight from the mmap and written as normalized CHW float32.
- **Bit-identical output** to `DataLoader(tar, transform=ImageNetNormalize())` — the same uint8 pixels (exact `rint` recovery from the fast path's `[0,1]` floats) through the same fused kernel (`deinterleave_hwc_to_chw_f32`). Tested with `np.array_equal`, not `allclose`.
- **~Zero owned memory.** The OS page cache holds the working set in file-backed pages — shared between processes, clean, evicted under pressure. Peak RSS ~1 GB for Imagenette-160 vs 3.3 GB anonymous RAM for `cache_decoded=True` (which also re-decodes at every startup).
- **Insulated from source resolution.** On-the-fly throughput drops with larger source JPEGs; TBL-RAW serve speed depends only on the target size.

## Measured (M4 Max, Imagenette 160px, each configuration in its own subprocess)

| Pipeline | produce img/s | np.sum-consumed | peak RSS |
|---|---:|---:|---:|
| on-the-fly TAR (decode every epoch) | 33,493 | 33,157 | 516 MB |
| **TBL-RAW, `prefetch_batches` default** | 127k† | **98,780** | 1,006 MB (file-backed, evictable) |
| **TBL-RAW, `prefetch_batches=0` (raw serve)** | **585,992** | 91,160 | 931 MB (file-backed, evictable) |
| `cache_decoded=True` (float32 in RAM, v2.37) | 137,494 | 101,358 | 3,343 MB (anonymous) + decode-all startup |

† with a no-op consumer the prefetch thread thrashes; its honest number is the consumed one, which prefetch *improves* (98.8k vs 91.2k) because production overlaps the consumer — the point for training loops.

**End-to-end** (RTX 3090, ResNet-18, Imagenette-160): TBL-RAW **3.64 s/epoch** vs TAR 3.76 s vs PyTorch 3.92 s, pure-GPU floor 3.39 s — the fastest input pipeline this benchmark has measured, with an honesty log: the first run was *slower* (4.51 s) until background prefetch moved serving off the training thread. See [End to End Training Results](https://github.com/ALJainProjects/TurboLoader/wiki/End-to-End-Training-Results).

## Serve-time augmentation (v2.38)

Store a little larger than you train at, then let the fused crop kernel do torchvision-parity RandomResizedCrop + hflip **per epoch, straight from the mmap**:

```python
tl.preprocess_to_tbl("imagenet.tar", "imagenet_192.tbl", image_size=192)      # once
loader = tl.DataLoader("imagenet_192.tbl", batch_size=128, image_size=160,
                       transform=tl.ImageNetNormalize(), train_aug=True, shuffle=True)
```

`crop_resize_normalize_u8_gather` gathers the rows, crops, bilinear-resizes, flips and normalizes in ONE parallel SIMD pass — the same sampling math (half-pixel centers, mirrored-x flip, clamp, bilinear) as the Metal/CUDA crop kernels, driven by the shared `pick_crop` sampler.

| Full-aug recipe (M4 Max, per-stage subprocesses) | produce img/s | np.sum-consumed | peak RSS |
|---|---:|---:|---:|
| on-the-fly TAR `train_aug=True` | 32,473 | 32,089 | 520 MB |
| **TBL-RAW serve-time aug** (192px file → 160px batches) | **86,913** | **78,200** | 1,274 MB (evictable) |

**2.7× (2.4× consumed)** with no decode. `meta['crops']` / `meta['flips']` report what was applied; crops are deterministic per `(seed, epoch, rank)`. Not served: color jitter and other photometric aug (bake them or use the TAR path); the kernel is bilinear, no antialias.

## API

### `preprocess_to_tbl(source, dst, image_size=160, *, batch_size=64, num_workers=8, compression=False)`
Runs `DataLoader(source, output_format='pytorch', image_size=..., shuffle=False)` and writes each sample as `SampleFormat.RAW_U8` with its dims. Asserts in-order delivery. Returns the sample count. `compression=True` is allowed but pointless for photos (LZ4 measured **1.06×** on decoded images) and disables the mmap fast path.

### `TblRawImageLoader(path, batch_size=64, *, mean=IMAGENET, std=IMAGENET, shuffle=True, seed=42, drop_last=False, pin_memory=False, ring=4, hflip_prob=0.0, prefetch_batches=2, train_aug=False, scale=(0.08, 1), ratio=(3/4, 4/3), image_size=None, dtype='float32', world_rank=0, world_size=1)`
- `mean=None, std=None` → plain `[0,1]` floats.
- `image_size` — output size; default the file's sample size; a different size is a serve-time bilinear resize.
- `train_aug` — RandomResizedCrop (`scale`, `ratio`) + hflip (`hflip_prob`) through the fused crop kernel; `hflip_prob` alone (no crop, same size, float32) uses the exact uint8-mirror path.
- `dtype='float16'` — half-precision output (portable round-to-nearest-even converter; bit-equal to numpy's own conversion), halves pinned-ring bytes and H2D.
- `world_rank`/`world_size` — disjoint, equal-size per-rank slice of the global `(seed, epoch)` permutation (`num_samples // world_size` per rank).
- `pin_memory=True` → torch page-locked ring of `ring` buffers (needs CUDA); a yielded batch's buffer is overwritten `ring` batches later. With prefetch the effective depth is clamped to `ring - 2` so nothing is overwritten while you or the queue hold it.
- `prefetch_batches` — background producer thread (stop-aware; winds down on early exit); `0` = synchronous. Output is identical either way (tested, including the flip and aug paths).
- `set_epoch`, `state_dict`/`load_state_dict`, `__len__`, `close()`, context manager — the family contract.

`DataLoader('x.tbl', ...)` forwards `batch_size`, `shuffle`, `seed`, `drop_last`, `pin_memory`, `prefetch_batches`, `image_size` (or a `Resize` in `transform`), `train_aug`/`hflip_prob`, and `enable_distributed`/`world_rank`/`world_size`; `transform` may contain `Resize` and `ImageNetNormalize` only.

### `turboloader.tbl.open_raw_view(path) -> (view, H, W)`
Validates the file (all `RAW_U8`, uniform dims, uncompressed, `size == W*H*3`, contiguous arithmetic offsets) and returns a zero-copy `(N, H, W, 3)` uint8 memmap view. This is the ingestion primitive the resident loaders use.

### GPU-resident ingestion (skips their decode-all pass)

```python
tl.MetalResidentLoader("imagenet_160.tbl", batch_size=256)          # Apple: "upload" = one memcpy into unified memory
tl.CudaResidentLoader.from_tbl("imagenet_160.tbl", batch_size=64)   # NVIDIA: ~64 MB chunked upload through the mmap
```

### The ops, exported standalone
- `normalize_u8_batch(input (N,H,W,3) u8, output (N,3,H,W) f32, mean=None, std=None, scale01=True)`
- `normalize_u8_gather(dataset (N,H,W,3) u8, indices int64 (B,), output (B,3,H,W) f32, mean=None, std=None, scale01=True)`
- `crop_resize_normalize_u8_gather(dataset (N,H,W,3) u8, indices int64 (B,), crops (B,4) f32 x,y,w,h, flips (B,) u8, output (B,3,dh,dw) f32|f16, mean=None, std=None, scale01=True)`

All are parallel, GIL-released, write into caller-provided (optionally pinned) arrays, and reject wrong shapes/dtypes/out-of-range indices instead of copying silently.

## Honest limits

- Photometric aug (color jitter etc.) is not served — bake it or use the TAR pipeline; the crop kernel is bilinear without antialias.
- The `.tbl` is larger than the source TAR (727 MB at 160px, 1.05 GB at 192px, vs 263 MB): disk traded for decode.
- On a bandwidth-poor host (the WSL2 3090 box) a GIL-holding pure-CPU consumer favors the RAM cache (27.6k vs 26.1k consumed); a real GPU step releases the GIL and TBL-RAW wins e2e. Both numbers are printed by `benchmarks/benchmark_tbl_raw.py`.
- The e2e proof of the serve-time-aug path on the 3090 is pending that machine's return (loader-only it is 2.7× the TAR `train_aug` path).

Format details: [TBL v2 Format](https://github.com/ALJainProjects/TurboLoader/wiki/TBL-v2-Format).
