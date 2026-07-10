"""Metal resident loaders (Apple Silicon) — pre-processed epochs on unified memory.

The CudaResidentLoader trick, Apple-style: decode + resize the dataset to uint8
ONCE, keep it resident in a shared ``MTLBuffer``, then serve every epoch with a
single fused gather(+shuffle)+normalize GPU kernel launch per batch. On M-series
chips the memory is unified, so there is no host->device upload at all — "resident"
costs one memcpy at build time, and every batch the GPU writes is immediately
CPU-visible as a zero-copy numpy view.

Lifetime contract (DALI-style, same as the pinned-memory ring): a yielded batch
aliases a double-buffered output and stays valid until the NEXT batch is drawn.
Consume it (``torch.from_numpy(batch)`` is zero-copy CPU; ``.to('mps')`` for GPU
training) before advancing, or ``.copy()`` it.

Datasets must fit in RAM (unified memory) — for Imagenette-160 that is ~727 MB.
"""

import numpy as np

__all__ = ["MetalResidentLoader", "MetalTokenGather", "MetalResidentArrays"]


def _require_metal():
    import turboloader as t

    if not getattr(t, "metal_available", lambda: False)() or not hasattr(
        t, "metal_resident_images_create"
    ):
        raise RuntimeError(
            "Metal resident loaders need a macOS arm64 build with Metal support "
            "(pip wheel includes it; check turboloader.metal_available())."
        )
    return t


class MetalResidentLoader:
    """Resident pre-processed image loader for Apple Silicon.

    Mirrors :class:`turboloader.cuda_loader.CudaResidentLoader`: build once from
    image paths (decode+resize in parallel) or adopt a pre-built ``(N, H, W, 3)``
    uint8 array, then every epoch is one fused gather+shuffle+normalize kernel
    launch per batch. Yields ``(B, 3, H, W)`` float32 zero-copy views (valid
    until the next batch; see module docstring), optionally with sample indices.
    """

    def __init__(
        self,
        source,
        image_size=160,
        batch_size=64,
        *,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        drop_last=True,
        num_workers=8,
        shuffle=False,
        seed=42,
        return_indices=False,
    ):
        t = _require_metal()
        self._t = t
        self.batch_size = int(batch_size)
        self.mean = list(mean)
        self.std = list(std)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.return_indices = bool(return_indices)
        self._epoch = 0
        self._handle = None

        if isinstance(source, np.ndarray):
            if source.ndim != 4 or source.shape[3] != 3 or source.dtype != np.uint8:
                raise ValueError("array source must be (N, H, W, 3) uint8")
            n, H, W = source.shape[0], source.shape[1], source.shape[2]
        else:
            paths = list(source)
            n = len(paths)
            H = W = int(image_size)
        if n == 0:
            raise ValueError("empty dataset")
        self._n, self._H, self._W = n, H, W

        self._handle = t.metal_resident_images_create(n, H, W, max(self.batch_size, 1))
        view = t.metal_resident_images_view(self._handle, n, H, W)
        if isinstance(source, np.ndarray):
            view[:] = source  # one memcpy into unified memory — that's the whole "upload"
        else:
            from concurrent.futures import ThreadPoolExecutor

            from PIL import Image

            def _load(i):
                with Image.open(paths[i]) as im:
                    view[i] = np.asarray(im.convert("RGB").resize((W, H)))

            with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
                list(ex.map(_load, range(n)))

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        n, bs = self._n, self.batch_size
        return n // bs if self.drop_last else -(-n // bs)

    def __iter__(self):
        t, bs = self._t, self.batch_size
        end = (self._n // bs) * bs if self.drop_last else self._n
        order = (
            np.random.default_rng(self.seed + self._epoch).permutation(self._n)
            if self.shuffle
            else np.arange(self._n)
        ).astype(np.int32)
        for b in range(0, end, bs):
            idx = order[b : b + min(bs, self._n - b)]
            batch = t.metal_resident_images_gather(
                self._handle, idx, self._H, self._W, mean=self.mean, std=self.std
            )
            if self.return_indices:
                yield batch, idx
            else:
                yield batch

    def close(self):
        """Free the resident buffers (invalidates all views). Idempotent."""
        h, self._handle = self._handle, None
        if h is not None:
            self._t.metal_resident_images_destroy(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class MetalResidentArrays:
    """Resident row store for arrays (embeddings, labels, tabular; any dtype).

    ``gather(idx)`` returns rows re-packed in ``idx`` order as a zero-copy view
    shaped ``(len(idx), *row_shape)`` — one GPU kernel launch, valid until the
    next gather.
    """

    def __init__(self, array, *, max_batch=4096):
        t = _require_metal()
        a = np.ascontiguousarray(array)
        if a.ndim < 1 or a.shape[0] == 0:
            raise ValueError("array must be (N, ...) with N >= 1")
        self._t = t
        self._dtype, self._row_shape = a.dtype, a.shape[1:]
        self._row_bytes = int(a.itemsize * np.prod(a.shape[1:], dtype=np.int64)) or a.itemsize
        self._n = a.shape[0]
        self._max_batch = int(max_batch)
        self._handle = t.metal_resident_bytes_create(
            self._n * self._row_bytes, self._max_batch, self._row_bytes
        )
        view = t.metal_resident_bytes_view(self._handle, self._n * self._row_bytes)
        view[:] = a.reshape(-1).view(np.uint8)

    def __len__(self):
        return self._n

    def gather(self, idx):
        idx = np.asarray(idx, dtype=np.uint64)
        if idx.ndim != 1 or idx.size == 0 or idx.size > self._max_batch:
            raise ValueError(f"idx must be 1-D with 1..{self._max_batch} entries")
        offs = idx * np.uint64(self._row_bytes)
        raw = self._t.metal_resident_bytes_gather(self._handle, offs, self._row_bytes)
        return raw.view(self._dtype).reshape((idx.size,) + self._row_shape)

    def close(self):
        h, self._handle = getattr(self, "_handle", None), None
        if h is not None:
            self._t.metal_resident_bytes_destroy(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MetalTokenGather:
    """Resident token windows for LLM batches — TokenDataLoader semantics on GPU.

    Tokens live once in unified memory; each batch gathers ``batch_size``
    overlapping ``seq_len + 1`` windows in one kernel launch and yields
    ``(x, y)`` int64 arrays (``y`` shifted by one). Exists primarily so the
    honest comparison against the CPU memmap path can be measured — see
    benchmarks/benchmark_metal_resident.py for which one actually wins.
    """

    def __init__(self, tokens, seq_len, batch_size, *, seed=42):
        t = _require_metal()
        a = np.ascontiguousarray(tokens)
        if a.ndim != 1 or a.size <= seq_len + 1:
            raise ValueError("tokens must be 1-D with more than seq_len+1 entries")
        self._t = t
        self._dtype = a.dtype
        self._itemsize = a.itemsize
        self._n = a.size
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(seed)
        self._span = (self.seq_len + 1) * self._itemsize
        self._handle = t.metal_resident_bytes_create(
            self._n * self._itemsize, self.batch_size, self._span
        )
        view = t.metal_resident_bytes_view(self._handle, self._n * self._itemsize)
        view[:] = a.view(np.uint8)

    def next_batch(self):
        starts = self._rng.integers(
            0, self._n - self.seq_len - 1, size=self.batch_size, dtype=np.uint64
        )
        offs = starts * np.uint64(self._itemsize)
        raw = self._t.metal_resident_bytes_gather(self._handle, offs, self._span)
        win = raw.view(self._dtype).reshape(self.batch_size, self.seq_len + 1)
        return win[:, :-1].astype(np.int64), win[:, 1:].astype(np.int64)

    def close(self):
        h, self._handle = getattr(self, "_handle", None), None
        if h is not None:
            self._t.metal_resident_bytes_destroy(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
