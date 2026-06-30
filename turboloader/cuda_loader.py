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

from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["CudaImageLoader"]


class CudaImageLoader:
    """Parallel decode (nvJPEG or CPU) + cuda_resize_normalize image loader.

    Args:
        paths: image file paths.
        batch_size, image_size, num_workers, shuffle, mean, std, seed, drop_last: as usual.
        decode: "gpu" (nvJPEG, default) or "cpu" (libjpeg-turbo).
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

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for start in range(0, end, bs):
                batch_idx = idx[start : start + bs]
                if fused is not None:
                    jpegs = list(ex.map(_read, batch_idx))  # raw bytes; parallel file I/O
                    yield fused(jpegs, dh, dw, mean=self.mean, std=self.std)
                else:
                    imgs = list(ex.map(_load, batch_idx))
                    yield self._t.cuda_resize_normalize(imgs, dh, dw, mean=self.mean, std=self.std)
