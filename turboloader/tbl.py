"""TBL-RAW: the pre-processed training pipeline (decode once, mmap forever).

FFCV's core insight, portably: JPEG decode is ~all of the input-pipeline cost, and
for many-epoch training you only need to pay it ONCE. ``preprocess_to_tbl`` runs
the C++ fast path over a TAR (parallel decode + resize) and writes the resulting
RGB uint8 samples into a ``.tbl`` file (TBL v2, ``SampleFormat.RAW_U8``).
``TblRawImageLoader`` then serves training batches straight from a **memory map**:

  * zero decode per epoch — one fused SIMD op (u8 HWC -> normalized f32 CHW)
    away from a ready batch, bit-identical to the TAR pipeline's output;
  * serve-time augmentation — ``train_aug=True`` applies torchvision-parity
    RandomResizedCrop + hflip through a fused crop+resize+normalize SIMD kernel
    (store samples a little larger than the training size, e.g. 192 for 160);
  * ~zero owned RAM — the OS page cache holds (and evicts) the working set,
    unlike ``cache_decoded=True`` which owns the whole decoded dataset as
    float32 (4x the bytes) in process memory;
  * ``dtype='float16'`` halves output bytes (and H2D) for AMP training;
  * ``world_rank``/``world_size`` give disjoint, equal shards per rank;
  * instant startup on re-runs — no decode-all pass, just an mmap.

Honest notes: LZ4 on decoded photos compresses poorly (~1.06x, measured) —
compression defaults OFF for RAW; the real "compression" is storing uint8
instead of float32 (4x) and resized instead of full-size.
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
_DONE = object()  # prefetch-queue end sentinel
_DTYPES = {"float32": np.float32, "float16": np.float16}


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
    ``DataLoader(tar, transform=ImageNetNormalize())``. For serve-time
    RandomResizedCrop, store a little larger than you train at (e.g. 192 -> 160).
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
    ``(B, 3, H, W)`` float32 (or float16), ImageNet-normalized by default;
    ``meta['indices']`` aligns external labels. Deterministic per
    ``(seed, epoch)`` via ``set_epoch``; ``state_dict()``/``load_state_dict()``
    resume mid-epoch.

    Args:
        path: RAW_U8 .tbl file (from ``preprocess_to_tbl``).
        mean/std: normalization (default ImageNet; ``mean=None, std=None`` for
            plain [0,1] output).
        image_size: output size (int or (H, W)); default = the file's sample
            size. A different size is a serve-time bilinear resize.
        train_aug: torchvision-parity RandomResizedCrop (``scale``, ``ratio``,
            the shared ``pick_crop`` sampler) + hflip with ``hflip_prob``, fused
            into one crop+resize+normalize SIMD pass per sample. Store samples
            larger than ``image_size`` to give the crop room.
        hflip_prob: horizontal-flip probability (applies with or without
            ``train_aug``; default 0).
        dtype: ``'float32'`` (default) or ``'float16'`` output.
        pin_memory: yield torch tensors backed by a reused ring of ``ring``
            page-locked buffers (CUDA hosts). LIFETIME: a yielded batch's buffer
            is overwritten ``ring`` batches later. Default (False) yields fresh
            numpy arrays with no reuse contract.
        prefetch_batches: background-produce this many batches ahead (the SIMD
            serve releases the GIL, so production overlaps your training step —
            without it the serve cost sits on the training thread). 0 disables.
            With ``pin_memory`` the effective depth is clamped to ``ring - 2``.
        world_rank/world_size: disjoint, equal-size shard of every epoch's
            order for this rank (DDP); the per-rank epoch has
            ``num_samples // world_size`` samples.
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
        ring=4,
        hflip_prob=0.0,
        prefetch_batches=2,
        train_aug=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        image_size=None,
        dtype="float32",
        world_rank=0,
        world_size=1,
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
        if self._pin and self._ring < 3:
            raise ValueError("ring must be >= 3 with pin_memory (consumer + queue + producer)")
        self.hflip_prob = float(hflip_prob)
        self._prefetch = max(0, int(prefetch_batches))
        self.train_aug = bool(train_aug)
        self.scale, self.ratio = tuple(scale), tuple(ratio)
        if dtype not in _DTYPES:
            raise ValueError("dtype must be 'float32' or 'float16'")
        self.dtype = dtype
        self._np_dtype = _DTYPES[dtype]
        self.world_rank, self.world_size = int(world_rank), int(world_size)
        if self.world_size < 1 or not 0 <= self.world_rank < self.world_size:
            raise ValueError("need 0 <= world_rank < world_size")
        self._epoch = 0
        self._served = 0
        self._resume_batches = 0

        self._view, self._h, self._w = open_raw_view(self.path)
        self.num_samples = self._view.shape[0]
        self.samples_per_rank = self.num_samples // self.world_size
        if self.samples_per_rank == 0:
            raise ValueError("fewer samples than ranks")
        if image_size is None:
            self._oh, self._ow = self._h, self._w
        else:
            oh, ow = (image_size, image_size) if isinstance(image_size, int) else image_size
            self._oh, self._ow = int(oh), int(ow)
        # The exact gather path is only for the identity case (no crop/resize/flip,
        # float32) — it is bit-identical to the TAR pipeline. Everything else goes
        # through the fused crop kernel (same math as the Metal / CUDA crop kernels).
        self._resample = (
            self.train_aug or (self._oh, self._ow) != (self._h, self._w) or dtype == "float16"
        )

    # ------------------------------------------------------------------ api
    def __len__(self):
        n = self.samples_per_rank
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def state_dict(self):
        return {"version": 1, "epoch": self._epoch, "batches_served": self._served}

    def load_state_dict(self, sd):
        self._epoch = int(sd["epoch"])
        self._resume_batches = int(sd["batches_served"])

    def _order(self):
        """This rank's epoch order: a disjoint, equal-size slice of the global
        (seed, epoch) permutation, so all ranks agree on the epoch length."""
        if self.shuffle:
            full = np.random.default_rng(self.seed + self._epoch).permutation(self.num_samples)
        else:
            full = np.arange(self.num_samples)
        return full[self.world_rank :: self.world_size][: self.samples_per_rank].astype(np.int64)

    def _epoch_aug(self, n):
        """Per-sample crop windows + flips for this epoch, drawn up front in
        sample order so prefetch/resume cannot change them."""
        crops = np.empty((n, 4), dtype=np.float32)
        if self.train_aug:
            from turboloader._augment import pick_crop

            rng = np.random.default_rng((self.seed, self._epoch, 2, self.world_rank))
            for i in range(n):
                crops[i] = pick_crop(self._w, self._h, rng, scale=self.scale, ratio=self.ratio)
        else:
            crops[:] = (0.0, 0.0, float(self._w), float(self._h))
        flips = np.zeros(n, dtype=np.uint8)
        if self.hflip_prob > 0:
            frng = np.random.default_rng((self.seed, self._epoch, 1, self.world_rank))
            flips[:] = frng.random(n) < self.hflip_prob
        return crops, flips

    def __iter__(self):
        t = self._t
        order = self._order()
        bs = self.batch_size
        n_batches = len(self)
        resume = self._resume_batches
        self._resume_batches = 0
        self._served = resume
        oh, ow = self._oh, self._ow
        crops, flips = self._epoch_aug(len(order))
        flip_only = (not self._resample) and self.hflip_prob > 0

        if self._pin:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("pin_memory=True needs CUDA (page-locked memory)")
            tdt = torch.float16 if self.dtype == "float16" else torch.float32
            ring = [
                torch.empty((bs, 3, oh, ow), dtype=tdt, pin_memory=True) for _ in range(self._ring)
            ]
            ring_np = [r.numpy() for r in ring]
        stage = np.empty((bs, self._h, self._w, 3), dtype=np.uint8) if flip_only else None

        def make(b):
            idx = order[b * bs : (b + 1) * bs]
            k = len(idx)
            if self._pin:
                out_t = ring[b % self._ring]
                out = ring_np[b % self._ring]
            else:
                out = np.empty((k, 3, oh, ow), dtype=self._np_dtype)
            sl = slice(b * bs, b * bs + k)
            if self._resample:
                # ONE fused pass: gather + RandomResizedCrop/resize + hflip + normalize
                t.crop_resize_normalize_u8_gather(
                    self._view, idx, crops[sl], flips[sl], out[:k], mean=self.mean, std=self.std
                )
            elif flip_only:
                # exact path with flips: mirror the small u8 rows, then the
                # bit-identical normalize (float sampling would add ~1e-5)
                np.take(self._view, idx, axis=0, out=stage[:k])
                sel = np.nonzero(flips[sl])[0]
                if sel.size:
                    stage[sel] = stage[sel, :, ::-1]
                t.normalize_u8_batch(stage[:k], out[:k], mean=self.mean, std=self.std)
            else:
                # identity: gather rows straight from the mmap and write normalized
                # CHW float32 — bit-identical to DataLoader(tar, ImageNetNormalize())
                t.normalize_u8_gather(self._view, idx, out[:k], mean=self.mean, std=self.std)
            meta = {"indices": idx.copy()}
            if self.train_aug:
                meta["crops"] = crops[sl].copy()
                meta["flips"] = flips[sl].copy()
            return (out_t[:k], meta) if self._pin else (out, meta)

        # Background prefetch: batch b+1 is produced (SIMD ops release the GIL)
        # while the consumer trains on batch b — without this the whole serve
        # cost sits on the training thread and e2e is SLOWER than the TAR
        # pipeline's threaded prefetch (measured on the 3090: 4.51s vs 3.73s
        # epochs before this thread existed). Depth is clamped so the pinned
        # ring can never be overwritten while the consumer (or queue) holds it.
        depth = self._prefetch if not self._pin else min(self._prefetch, self._ring - 2)
        if depth <= 0:
            for b in range(resume, n_batches):
                self._served += 1
                yield make(b)
            return

        import queue as _queue
        import threading

        stop = threading.Event()
        q = _queue.Queue(maxsize=depth)

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.25)
                    return True
                except _queue.Full:
                    continue
            return False

        def producer():
            try:
                for b in range(resume, n_batches):
                    if not put(make(b)):
                        return
                put(_DONE)
            except Exception as e:  # surfaced on the consumer thread
                put(("__tblraw_err__", repr(e)))

        th = threading.Thread(target=producer, daemon=True, name="tblraw-prefetch")
        th.start()
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    break
                if isinstance(item[0], str) and item[0] == "__tblraw_err__":
                    raise RuntimeError(f"TBL-RAW prefetch failed: {item[1]}")
                self._served += 1
                yield item
        finally:
            stop.set()
            th.join(timeout=5)

    def close(self):  # symmetry with the other loaders; nothing owned
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
