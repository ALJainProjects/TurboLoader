# TBL v2 Binary Format & the TBL-RAW Pre-processed Pipeline

TurboLoader's custom binary format, and — the reason you'd actually use it —
the **pre-processed training pipeline** built on it: decode your dataset once,
then serve every epoch from a memory map with **zero JPEG decode**.

## The pre-processed pipeline (TBL-RAW)

FFCV's core insight, portably: JPEG decode dominates input-pipeline cost, and
for many-epoch training you only need to pay it once.

```python
import turboloader as tl

# ONE-TIME: parallel decode + resize the TAR, store RGB uint8 samples
# (9,469 Imagenette images -> 6s on an M4 Max; 727 MB at 160px)
tl.preprocess_to_tbl('imagenet.tar', 'imagenet_160.tbl', image_size=160)

# EVERY RUN: serve from an mmap — no decode, instant startup
loader = tl.DataLoader('imagenet_160.tbl', batch_size=64,
                       transform=tl.ImageNetNormalize(), shuffle=True)
for batch, meta in loader:          # (64, 3, 160, 160) float32
    train_step(batch, labels[meta['indices']])
```

Why this is the efficiency frontier (M4 Max, Imagenette, 160px — full numbers
incl. an np.sum-consumed variant in [docs/benchmarks](benchmarks/index.md)):

- **Zero decode per epoch.** Serving is one fused SIMD pass
  (`normalize_u8_gather`): rows are gathered straight from the mmap and written
  as normalized CHW float32, in parallel, GIL-released. Output is
  **bit-identical** to `DataLoader(tar, transform=ImageNetNormalize())` —
  tested with `np.array_equal`, not `allclose`.
- **~Zero owned memory.** The OS page cache holds the working set in
  file-backed pages — shared, clean, evicted under pressure.
  `cache_decoded=True` owns the whole decoded dataset as float32 anonymous RAM
  (4x the bytes of the uint8 file, unevictable) and still serves slower.
- **Insulated from source resolution.** On-the-fly throughput drops when the
  source JPEGs are large (more decode work); TBL-RAW serve speed depends only
  on the target size.
- **Instant startup.** No decode-all pass on re-runs; an mmap is O(1).

More paths consume the same file:

```python
# GPU-resident, skipping their decode-all pass entirely:
tl.MetalResidentLoader('imagenet_160.tbl')            # Apple: upload = 1 memcpy
tl.CudaResidentLoader.from_tbl('imagenet_160.tbl')    # NVIDIA: chunked mmap upload

# Direct class (adds hflip, the one aug this path supports):
tl.TblRawImageLoader('imagenet_160.tbl', hflip_prob=0.5, pin_memory=True)
```

**Honest limits.** Samples are stored post-resize, so per-epoch
`RandomResizedCrop`/color aug is impossible — bake it or use the TAR pipeline
(`train_aug=True`) when aug matters; random hflip IS supported at serve time.
LZ4 on decoded photos measured **1.06x** (high-entropy data) — RAW defaults to
`compression=False`; the real "compression" is uint8-instead-of-float32 (4x)
and resized-instead-of-full-size. The .tbl is larger than the source TAR
(727 MB vs 263 MB for Imagenette-160) — you trade disk for decode.
Numbers: `benchmarks/benchmark_tbl_raw.py`.

## Format specification

- LZ4 compression (optional, per-file)
- Memory-mapped access, O(1) random access via indexed structure
- CRC checksums; cached per-sample dimensions
- Sample formats: JPEG, PNG, WebP, BMP, TIFF, MP4, AVI, and **RAW_U8**
  (decoded RGB uint8 HWC — the pre-processed training format; payload size
  must equal `width * height * 3`)

Layout: 64-byte header (`TBL\x02` magic) → index table (24 B/sample: offset,
size, uncompressed size, width, height, format, flags, CRC16) → concatenated
sample payloads → optional metadata section. Uncompressed uniform RAW_U8
payloads are contiguous, which is what makes the zero-copy
`(N, H, W, 3)` mmap view (`turboloader.tbl.open_raw_view`) possible.

## Low-level API (any sample format)

```python
import tarfile
import turboloader

writer = turboloader.TblWriterV2("/data/imagenet.tbl", enable_compression=True)

# The TAR archive is read with Python's stdlib (TurboLoader does not expose a
# standalone Python TarReader; the DataLoader reads TAR directly for training).
with tarfile.open("/data/imagenet.tar") as tar:
    for member in tar.getmembers():
        if not member.name.lower().endswith((".jpg", ".jpeg")):
            continue
        data = tar.extractfile(member).read()
        writer.add_sample(data=data, format=turboloader.SampleFormat.JPEG)

writer.finalize()

reader = turboloader.TblReaderV2("/data/imagenet.tbl")
data = reader.read_sample(0)              # bytes (LZ4-decompressed if needed)
info = reader.get_sample_info(0)          # offset/size/dims/format/flags
```

> For bulk conversion of encoded samples there is also a C++ CLI tool,
> `tools/tar_to_tbl_v2.cpp`. For training data, prefer `preprocess_to_tbl`.
