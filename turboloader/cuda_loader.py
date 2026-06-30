"""End-to-end CUDA GPU image loader (NVIDIA). The CUDA analogue of GpuImageLoader.

Decodes JPEGs with nvJPEG on the GPU (``cuda_decode_jpeg``) and runs resize+normalize on
the GPU (``cuda_resize_normalize``), yielding ``(N, 3, H, W)`` float32 batches. Lets
TurboLoader be compared fairly against DALI/FFCV on NVIDIA hardware.

Honest v1 note: decode and transform each round-trip through host memory (nvJPEG decode ->
host -> GPU transform), and the single-image nvJPEG path is mutex-serialized. The fused,
GPU-resident pipeline (no round-trips, batched nvJPEG) is the optimization tracked
separately. Requires a CUDA build (``turboloader.cuda_available()``).
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["CudaImageLoader"]


class _CudaArray:
    """Wraps a CUDA device pointer as a ``__cuda_array_interface__`` array so torch/cupy can
    adopt it zero-copy (``torch.as_tensor(x, device='cuda')``). Valid until the loader yields
    the next batch (the device pool is reused), exactly like a DALI pipeline output."""

    def __init__(self, ptr, shape):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": tuple(shape),
            "typestr": "<f4",
            "version": 3,
        }


class CudaImageLoader:
    """Parallel decode (nvJPEG or CPU) + cuda_resize_normalize image loader.

    Args:
        paths: image file paths.
        batch_size, image_size, num_workers, shuffle, mean, std, seed, drop_last: as usual.
        decode: "nvimgcodec" (NVIDIA nvImageCodec, fastest — ~17.7k img/s on a 3090, needs the
            `nvidia-nvimgcodec-cu12` wheel), "gpu" (nvJPEG, default) or "cpu" (libjpeg-turbo).
            "nvimgcodec" yields GPU-resident `__cuda_array_interface__` batches.
    """

    def __init__(
        self,
        paths,
        batch_size=64,
        image_size=160,
        *,
        num_workers=8,
        shuffle=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        seed=42,
        drop_last=False,
        decode="gpu",
        gpu_output=False,
        prefetch=0,
    ):
        import turboloader as t

        if not getattr(t, "cuda_available", lambda: False)():
            raise RuntimeError("CudaImageLoader needs a CUDA build; cuda_available() is False.")
        self._t = t
        gpu_decode = decode == "gpu" and hasattr(t, "cuda_decode_jpeg")
        self._decode = t.cuda_decode_jpeg if gpu_decode else t.decode_jpeg
        self.decode_mode = "nvjpeg" if gpu_decode else "cpu"
        # Prefer the fused GPU-resident decode+transform op when available (decode="gpu").
        self._fused = decode == "gpu" and hasattr(t, "cuda_decode_resize_normalize")
        # gpu_output=True yields a __cuda_array_interface__ batch kept on the GPU (no D2H).
        self._gpu_output = bool(gpu_output) and hasattr(t, "cuda_decode_resize_normalize_gpu")
        # Async prefetch depth (background thread decodes ahead). Capped at 2 to stay within
        # the C++ output ring (OUT_RING=4: producing + queued + consuming slots).
        self._prefetch = max(0, min(int(prefetch), 2))
        # nvImageCodec backend (decode="nvimgcodec"): NVIDIA's modern codec, ~2x nvJPEG here.
        # It decodes on the GPU; our transform reads its device images in place (zero copies),
        # reaching ~17.7k img/s on a 3090 (vs ~9k with nvJPEG) — DALI-competitive.
        self._nvimgcodec = None
        if decode == "nvimgcodec" and hasattr(t, "cuda_resize_normalize_from_device"):
            try:
                from nvidia import nvimgcodec

                self._nvimgcodec = nvimgcodec.Decoder()
            except Exception:
                self._nvimgcodec = None
        self.paths = list(paths)
        self.batch_size = int(batch_size)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.num_workers = max(1, int(num_workers))
        self.shuffle = bool(shuffle)
        self.mean = list(mean)
        self.std = list(std)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        n = len(self.paths)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        idx = np.arange(len(self.paths))
        if self.shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(idx)
        bs = self.batch_size
        end = (len(idx) // bs) * bs if self.drop_last else len(idx)
        paths = self.paths
        dh, dw = self.image_size
        # Fused GPU-resident path: read raw JPEG bytes and let the C++ pipeline decode +
        # transform on the GPU with one D2H (no per-image host round-trips). Falls back to
        # the v1 decode->host->transform path if the fused op isn't compiled in.
        fused = getattr(self._t, "cuda_decode_resize_normalize", None) if self._fused else None
        decode = self._decode

        def _read(i):
            with open(paths[int(i)], "rb") as f:
                return f.read()

        def _load(i):
            return decode(_read(i))

        # nvImageCodec path (decode="nvimgcodec"): its GPU decoder produces HWC uint8 RGB
        # device images; we hand their device pointers straight to the transform kernel (zero
        # copies) and yield a GPU-resident batch. The fastest path — DALI-competitive.
        if self._nvimgcodec is not None:
            dec = self._nvimgcodec
            xform = self._t.cuda_resize_normalize_from_device

            def _nvimg_batch(bidx):
                imgs = dec.decode([_read(i) for i in bidx])  # held alive across the transform
                cai = [im.__cuda_array_interface__ for im in imgs]
                ptr = xform(
                    [c["data"][0] for c in cai], [c["shape"][1] for c in cai],
                    [c["shape"][0] for c in cai], dh, dw, mean=self.mean, std=self.std,
                )
                return _CudaArray(ptr, (len(imgs), 3, dh, dw))

            if self._prefetch > 0:
                q = queue.Queue(maxsize=self._prefetch)
                sentinel = object()

                def _producer():
                    try:
                        for start in range(0, end, bs):
                            q.put(_nvimg_batch(idx[start : start + bs]))
                    finally:
                        q.put(sentinel)

                th = threading.Thread(target=_producer, daemon=True)
                th.start()
                while True:
                    item = q.get()
                    if item is sentinel:
                        break
                    yield item
                th.join(timeout=0.5)
            else:
                for start in range(0, end, bs):
                    yield _nvimg_batch(idx[start : start + bs])
            return

        gpu_out = (
            getattr(self._t, "cuda_decode_resize_normalize_gpu", None) if self._gpu_output else None
        )
        if gpu_out is not None and self._prefetch > 0:
            # Async prefetch: a background thread decodes batches ahead into a bounded queue
            # (depth fits the C++ output ring), overlapping decode with the consumer's work.
            starts = list(range(0, end, bs))
            q = queue.Queue(maxsize=self._prefetch)
            sentinel = object()

            def _producer():
                try:
                    for start in starts:
                        bidx = idx[start : start + bs]
                        jpegs = [_read(i) for i in bidx]
                        ptr = gpu_out(jpegs, dh, dw, mean=self.mean, std=self.std)
                        q.put(_CudaArray(ptr, (len(jpegs), 3, dh, dw)))
                finally:
                    q.put(sentinel)

            th = threading.Thread(target=_producer, daemon=True)
            th.start()
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item
            th.join(timeout=0.5)
        elif fused is not None:
            # Fused path: the C++ op decodes the whole batch on the GPU, so Python only
            # reads bytes. Read SERIALLY — a ThreadPoolExecutor over 64 tiny (page-cached)
            # reads costs ~16 ms of future/GIL overhead vs ~0.6 ms serial.
            for start in range(0, end, bs):
                batch_idx = idx[start : start + bs]
                jpegs = [_read(i) for i in batch_idx]
                if gpu_out is not None:
                    # GPU-resident output: no D2H; wrap the device pointer (consume before
                    # the next batch — the device pool is reused).
                    ptr = gpu_out(jpegs, dh, dw, mean=self.mean, std=self.std)
                    yield _CudaArray(ptr, (len(jpegs), 3, dh, dw))
                else:
                    yield fused(jpegs, dh, dw, mean=self.mean, std=self.std)
        else:
            # v1 path: decode per image; the thread pool parallelizes the GIL-releasing
            # nvJPEG/libjpeg decode calls.
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for start in range(0, end, bs):
                    batch_idx = idx[start : start + bs]
                    imgs = list(ex.map(_load, batch_idx))
                    yield self._t.cuda_resize_normalize(imgs, dh, dw, mean=self.mean, std=self.std)
