<!-- Generated from the project wiki page https://github.com/ALJainProjects/TurboLoader/wiki/Image-DataLoader-API — edit there. -->
> Canonical, always-current version: **[Image DataLoader API](https://github.com/ALJainProjects/TurboLoader/wiki/Image-DataLoader-API)** on the wiki.

# `turboloader.DataLoader` — API reference

The single entry point for images (TAR and `.tbl`), and — via `modality=` — tokens, arrays, and map-style datasets. Signature as of v2.37.0:

```python
turboloader.DataLoader(
    data_path=None, batch_size=32, num_workers=4, shuffle=False, transform=None,
    image_size=None, output_format='dict', modality='image',
    seq_len=None, token_dtype='uint16', arrays=None, dataset=None, collate_fn=None,
    seed=42,
    enable_distributed=False, world_rank=0, world_size=1, drop_last=False, distributed_seed=42,
    enable_cache=False, cache_l1_mb=512, cache_l2_gb=0, cache_dir='/tmp/turboloader_cache',
    auto_smart_batching=False, enable_smart_batching=False,
    prefetch_batches=4, cache_decoded=False, antialias=False, pin_memory=False,
    train_aug=False, hflip_prob=0.5,
)
```

## Routing — what you get depends on three arguments

| `data_path` / `modality` / `output_format` | Path taken | Yields |
|---|---|---|
| `*.tar`, `modality='image'`, `output_format` in `pytorch` · `numpy_chw` · `tensorflow` · `numpy` | **Fast path** (`_DirectFast`): one parallel C++ pass, decode → resize → (train_aug) → normalize straight into the batch buffer | `(batch, meta)`; batch `(N,3,H,W)` float32 (CHW formats) or `(N,H,W,3)` (`tensorflow`/`numpy`) |
| `*.tar`, `output_format='dict'` (default) | Per-sample pipeline | list of dicts: `image` (H,W,3) uint8, `filename`, `index` |
| `*.tbl` (RAW_U8) | `TblRawImageLoader` via mmap (see [TBL RAW Preprocessed Pipeline](https://github.com/ALJainProjects/TurboLoader/wiki/TBL-RAW-Preprocessed-Pipeline)) | `(batch, meta)` |
| `modality='tokens'` | `TokenDataLoader(data_path, seq_len=..., dtype=token_dtype)` | `(x, y)` int64 |
| `modality='array'`, `arrays=[...]` | `ArrayDataLoader(*arrays)` | array or tuple |
| `dataset=...` or `modality='map'` | `MapDataLoader(dataset)` | collated batches |

The fast path **requires a fixed size**: pass `image_size=N` or `(H, W)`, or a `Resize` in `transform` (anywhere in a `Compose`/pipe chain). Passing both with different sizes raises `ValueError("Conflicting sizes: ...")`.

## Parameters

### Data and batching
- **`data_path`** — path to a TAR (local; http/s3/gcs readers are compiled in per `features()`), or a RAW `.tbl`.
- **`batch_size`** — samples per batch. **`drop_last`** — drop the final partial batch (required for equal DDP batches).
- **`num_workers`** — worker threads for the per-sample path. The fast path runs on one process-wide C++ thread pool that parallelizes internally; raising this does not multiply throughput.
- **`shuffle`**, **`seed`** — epoch order is a deterministic function of `(seed, epoch)`; call `set_epoch(e)` each epoch (same contract as `DistributedSampler`).

### Output
- **`output_format`** — `'dict'` (default, per-sample), `'pytorch'` (CHW float32 — returns a `torch.Tensor` when torch is installed, numpy otherwise), `'numpy_chw'`, `'numpy'` / `'tensorflow'` (HWC float32).
- **`image_size`** — int or `(H, W)`; enables the fast path. Values are `[0,1]` floats unless a normalize transform is present.
- **`transform`** — a transform or composition: `Resize(w, h) | ImageNetNormalize()` (pipe operator) or `Compose([...])`. On the fast path `Resize` and `Normalize`/`ImageNetNormalize` are fused into the C++ pass; see [Transforms](https://github.com/ALJainProjects/TurboLoader/wiki/Transforms) for the full library and the per-sample `.apply()` interface.
- **`antialias`** — antialiased downscale (torchvision `antialias=True` parity); default False is plain bilinear with half-pixel centers (PIL/OpenCV/TF convention).

### Training-loop features
- **`train_aug`** — fused RandomResizedCrop (torchvision-parity distribution incl. the aspect-clamped central-crop fallback) + horizontal flip with probability **`hflip_prob`** (default 0.5), inside the C++ pass (~3% overhead). Per-epoch randomness follows `(seed, epoch)`.
- **`pin_memory`** — stream batches through a **ring of recycled page-locked torch buffers** for async `.to(device, non_blocking=True)`. LIFETIME: the batch you hold is never overwritten, but the previous one may be recycled as soon as you take the next batch (the ring is `prefetch_batches + 2` buffers: yours, the queued ones, and one being filled). Consume (`.to(device)`) or `.clone()` before calling `next()` again. Requires torch + CUDA.
- **`prefetch_batches`** — decode-ahead depth (default 4). The loader keeps working while your step runs.
- **`cache_decoded`** — decode the whole dataset once into a contiguous float32 cache (index-ordered, one allocation as of v2.37), then serve every epoch from RAM with a parallel C++ row gather. Costs 4× the uint8 bytes as anonymous RAM and a decode-all pass at every process start — for many-epoch training prefer TBL-RAW, which keeps uint8 in evictable page cache. Numbers: [Benchmarks](https://github.com/ALJainProjects/TurboLoader/wiki/Benchmarks).

### Distributed
- **`enable_distributed`**, **`world_rank`**, **`world_size`**, **`distributed_seed`** — DDP-safe disjoint, equal-size sharding across ranks; combine with `drop_last=True`. Caching (`cache_decoded`) under sharding caches only this rank's shard (a warning says so). See [Determinism Resume and Distributed](https://github.com/ALJainProjects/TurboLoader/wiki/Determinism-Resume-and-Distributed).

### Other
- **`enable_cache`**, `cache_l1_mb`, `cache_l2_gb`, `cache_dir` — tiered (memory + disk) byte cache for remote sources.
- **`auto_smart_batching`**, **`enable_smart_batching`** — size-grouped batching for the per-sample path (both default False in the signature; the docstring's "default True" is stale).
- **`modality`**, `seq_len`, `token_dtype`, `arrays`, `dataset`, `collate_fn` — routing to the non-image loaders (above).

## Methods

| Method | Behavior |
|---|---|
| `__iter__` / `next_batch()` | iterate; `next_batch()` advances an internal epoch iterator |
| `set_epoch(epoch)` | reproducible per-epoch shuffle/aug order |
| `__len__` | batches per epoch (`ceil(n / batch_size)`, or `floor` with `drop_last`) |
| `state_dict()` / `load_state_dict(sd)` | `{'version', 'epoch', 'batches_served'}`; resume skips decode-free to the exact batch |
| `close()` / `stop()` / context manager / `__del__` | releases producer threads and file handles; idempotent. Abandoned loaders are also shut down at interpreter exit. |

## `meta`

Fast-path batches come with `meta = {"indices": [...], "batch_size": n}` — `indices` are the sample indices in the TAR's member order, which is how you align labels (`labels[np.asarray(meta["indices"])]`).

## Related classes

- `TblRawImageLoader` — the `.tbl` serve class with every knob (`train_aug`, `scale`/`ratio`, `hflip_prob`, `image_size`, `dtype`, `world_rank`/`world_size`, `ring`, `prefetch_batches`, `mean/std`); `DataLoader('x.tbl')` forwards `batch_size`, `shuffle`, `seed`, `drop_last`, `pin_memory`, `prefetch_batches`, `image_size` (or a `Resize` in `transform` — a serve-time resize), `train_aug`/`hflip_prob`, and `enable_distributed`/`world_rank`/`world_size`; `transform` may contain `Resize` and `ImageNetNormalize` only (other transforms raise with guidance).
- `FastDataLoader`, `Loader()`, `create_loader()` — legacy; see [Which Loader Do I Use](https://github.com/ALJainProjects/TurboLoader/wiki/Which-Loader-Do-I-Use).
- `PyTorchCompatibleLoader(data_path, label_extractor=..., transform=..., device=...)` — `(images, labels)` tuples with `FolderLabelExtractor`, `FilenamePatternExtractor`, `JSONSidecarExtractor`, `MetadataLabelExtractor`, `CallableLabelExtractor`; `TransformAdapter` converts torchvision transforms; `convert_imagefolder` / `ImageFolderConverter` write an ImageFolder tree to a TAR.
