"""TBL-RAW: the pre-processed training pipeline (decode once, mmap forever).

FFCV's core insight, portably: JPEG decode is ~all of the input-pipeline cost, and
for many-epoch training you only need to pay it ONCE. ``preprocess_to_tbl`` runs
the C++ fast path over a TAR (parallel decode + resize) and writes the resulting
RGB uint8 samples into a ``.tbl`` file (TBL v2, ``SampleFormat.RAW_U8``).
``TblRawImageLoader`` then serves training batches straight from a **memory map**:

  * zero decode per epoch — one fused SIMD op (u8 HWC -> normalized f32 CHW)
    away from a ready batch, bit-identical to the TAR pipeline's output;
  * ~zero owned RAM — the OS page cache holds (and evicts) the working set,
    unlike ``cache_decoded=True`` which owns the whole decoded dataset as
    float32 (4x the bytes) in process memory;
  * instant startup on re-runs — no decode-all pass, just an mmap.

Honest notes: the file stores uint8 (like FFCV / CudaResidentLoader), so
augmentation baked at preprocess time is fixed — this pipeline fits the
resize+normalize recipe, NOT per-epoch RandomResizedCrop (use the TAR path for
that). LZ4 on decoded photos compresses poorly (~1.0-1.2x, measured in
benchmarks) — compression defaults OFF for RAW; the real "compression" is
storing uint8 instead of float32 (4x) and resized instead of full-size.
"""

import os

import numpy as np

__all__ = ["preprocess_to_tbl", "TblRawImageLoader", "open_raw_view"]


def open_raw_view(path):
    """Memory-map an uncompressed, uniform-dims RAW_U8 ``.tbl`` as a zero-copy
    ``(N, H, W, 3)`` uint8 array view. Returns ``(view, H, W)``.

    This is the ingestion primitive shared by TblRawImageLoader and the
    GPU-resident loaders (their one-time upload reads straight through it —
    no decode pass)."""
    import turboloader as t

    path = os.fspath(path)
    reader = t.TblReaderV2(path, verify_checksums=False)
    n = reader.num_samples()
    if n == 0:
        raise ValueError(f"{path} contains no samples")
    infos = [reader.get_sample_info(i) for i in range(n)]
    raw = int(t.SampleFormat.RAW_U8)
    if {int(i["format"]) for i in infos} != {raw}:
        fmts = sorted({str(i["format"]) for i in infos})
        raise ValueError(
            f"{path} holds {fmts} samples; the training loader serves RAW_U8 "
            "files — create one with turboloader.preprocess_to_tbl(tar, tbl, "
            "image_size=N)"
        )
    dims = {(i["height"], i["width"]) for i in infos}
    if len(dims) != 1:
        raise ValueError(
            f"samples have mixed sizes {sorted(dims)}; batching needs uniform "
            "dims — preprocess with a fixed image_size"
        )
    H, W = dims.pop()
    sz = H * W * 3
    if any(i["is_compressed"] for i in infos):
        raise ValueError(
            "this .tbl is LZ4-compressed; the mmap fast path needs uncompressed "
            "RAW (preprocess_to_tbl(..., compression=False) — measured, LZ4 on "
            "decoded photos saves ~nothing anyway)"
        )
    if any(i["size"] != sz for i in infos):
        raise ValueError("corrupt RAW_U8 file: sample size != W*H*3")
    offs = np.array([i["offset"] for i in infos], dtype=np.int64)
    if not np.array_equal(offs, offs[0] + np.arange(n, dtype=np.int64) * sz):
        raise ValueError("non-contiguous payload layout; refusing mmap view")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return mm[offs[0] : offs[0] + n * sz].reshape(n, H, W, 3), H, W


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_to_tbl(
    source,
    dst,
    image_size=160,
    *,
    batch_size=64,
    num_workers=8,
    compression=False,
):
    """Decode + resize every image in ``source`` (TAR of JPEGs) once, writing
    RGB uint8 samples to ``dst`` (a ``.tbl`` file). Returns the sample count.

    The uint8 quantization is the same storage semantic as CudaResidentLoader
    and FFCV; serving then normalizes with the same fused SIMD math as the TAR
    fast path, so batches are bit-identical to
    ``DataLoader(tar, transform=ImageNetNormalize())``.
    """
    import turboloader as t

    loader = t.DataLoader(
        source,
        batch_size=batch_size,
        output_format="pytorch",
        image_size=image_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    writer = t.TblWriterV2(str(dst), enable_compression=bool(compression))
    expect = 0
    n = 0
    try:
        for batch, meta in loader:
            x = np.asarray(batch)  # (B, 3, H, W) float32 in [0, 1]
            idx = np.asarray(meta["indices"])
            if not np.array_equal(idx, np.arange(expect, expect + len(idx))):
                raise RuntimeError(
                    "preprocess requires in-order delivery; got indices "
                    f"{idx[:4]}... at position {expect}"
                )
            expect += len(idx)
            u8 = np.rint(x * 255.0).clip(0, 255).astype(np.uint8)  # exact u8 recovery
            hwc = np.ascontiguousarray(u8.transpose(0, 2, 3, 1))
            H, W = hwc.shape[1], hwc.shape[2]
            for row in hwc:
                writer.add_sample(row.tobytes(), t.SampleFormat.RAW_U8, width=W, height=H)
                n += 1
    finally:
        loader.close()
    writer.finalize()
    return n


class TblRawImageLoader:
    """Training batches from a RAW_U8 ``.tbl`` via memory map — zero decode.

    Yields ``(batch, meta)`` like the image DataLoader: ``batch`` is
    ``(B, 3, H, W)`` float32 (ImageNet-normalized by default), ``meta['indices']``
    aligns external labels. Deterministic per ``(seed, epoch)`` via ``set_epoch``;
    ``state_dict()``/``load_state_dict()`` resume mid-epoch.

    Args:
        path: RAW_U8 .tbl file (from ``preprocess_to_tbl``).
        mean/std: normalization (default ImageNet; pass ``mean=None, std=None``
            for plain [0,1] output).
        pin_memory: yield torch tensors backed by a reused ring of ``ring``
            page-locked buffers (CUDA hosts). LIFETIME: a yielded batch's buffer
            is overwritten ``ring`` batches later. Default (False) yields fresh
            numpy arrays with no reuse contract.
    """

    def __init__(
        self,
        path,
        batch_size=64,
        *,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        shuffle=True,
        seed=42,
        drop_last=False,
        pin_memory=False,
        ring=3,
        hflip_prob=0.0,
    ):
        import turboloader as t

        self._t = t
        self.path = os.fspath(path)
        self.batch_size = int(batch_size)
        if (mean is None) != (std is None):
            raise ValueError("pass both mean and std, or neither")
        self.mean = None if mean is None else list(mean)
        self.std = None if std is None else list(std)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._pin = bool(pin_memory)
        self._ring = int(ring)
        self.hflip_prob = float(hflip_prob)
        self._epoch = 0
        self._served = 0
        self._resume_batches = 0

        self._view, self._h, self._w = open_raw_view(self.path)
        self.num_samples = self._view.shape[0]

    # ------------------------------------------------------------------ api
    def __len__(self):
        n = self.num_samples // self.batch_size
        return n if self.drop_last else -(-self.num_samples // self.batch_size)

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def state_dict(self):
        return {"version": 1, "epoch": self._epoch, "batches_served": self._served}

    def load_state_dict(self, sd):
        self._epoch = int(sd["epoch"])
        self._resume_batches = int(sd["batches_served"])

    def _order(self):
        if not self.shuffle:
            return np.arange(self.num_samples, dtype=np.int64)
        rng = np.random.default_rng(self.seed + self._epoch)
        return rng.permutation(self.num_samples).astype(np.int64)

    def __iter__(self):
        t = self._t
        order = self._order()
        bs = self.batch_size
        n_batches = len(self)
        resume = self._resume_batches
        self._resume_batches = 0
        self._served = resume

        if self._pin:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("pin_memory=True needs CUDA (page-locked memory)")
            ring = [
                torch.empty((bs, 3, self._h, self._w), dtype=torch.float32, pin_memory=True)
                for _ in range(self._ring)
            ]
            ring_np = [r.numpy() for r in ring]
        stage = np.empty((bs, self._h, self._w, 3), dtype=np.uint8)

        flip_rng = (
            np.random.default_rng((self.seed, self._epoch, 1)) if self.hflip_prob > 0 else None
        )
        for b in range(resume, n_batches):
            idx = order[b * bs : (b + 1) * bs]
            k = len(idx)
            if self._pin:
                out_t = ring[b % self._ring]
                out = ring_np[b % self._ring]
            else:
                out = np.empty((k, 3, self._h, self._w), dtype=np.float32)
            if flip_rng is None:
                # ONE parallel pass: gather rows straight from the mmap and
                # write normalized CHW float32 — no decode, no staging copy
                t.normalize_u8_gather(self._view, idx, out[:k], mean=self.mean, std=self.std)
            else:
                # flip path stages in uint8 first (flipping the small u8 rows,
                # not the 4x-larger float output). Crop/color aug must be baked
                # at preprocess time — use the TAR path for those.
                np.take(self._view, idx, axis=0, out=stage[:k])
                sel = np.nonzero(flip_rng.random(k) < self.hflip_prob)[0]
                if sel.size:
                    stage[sel] = stage[sel, :, ::-1]
                t.normalize_u8_batch(stage[:k], out[:k], mean=self.mean, std=self.std)
            self._served += 1
            meta = {"indices": idx.copy()}
            if self._pin:
                yield (out_t[:k], meta)
            else:
                yield (out, meta)

    def close(self):  # symmetry with the other loaders; nothing owned
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
