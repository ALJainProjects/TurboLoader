"""GPU-accelerated image loader for Apple Silicon (Metal).

Decodes JPEGs on the CPU in parallel (``decode_jpeg`` releases the GIL) and runs
resize/crop/flip/normalize on the GPU via the Metal kernels — the DALI architecture, but
on Apple GPUs (DALI is CUDA-only). Yields ``(N, 3, H, W)`` float32 batches.

Honest scope: this accelerates the *transform* stage; JPEG decode stays on the CPU (Metal
has no JPEG decoder — see the experimental hybrid decoder for that). Requires a macOS
arm64 build with the Metal path (``turboloader.metal_available()``).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["GpuImageLoader"]


class GpuImageLoader:
    """Parallel CPU decode + Metal GPU transform image loader.

    Args:
        paths: sequence of image file paths.
        batch_size: images per batch.
        image_size: int or (H, W) output size.
        num_workers: CPU threads for parallel decode.
        shuffle: shuffle order each epoch (reproducible via ``set_epoch``).
        train_aug: if True, apply RandomResizedCrop + RandomHorizontalFlip on the GPU
            (the canonical ImageNet train pipeline); else a plain resize.
        scale, ratio: RandomResizedCrop area-scale and aspect-ratio ranges.
        hflip_prob: horizontal-flip probability when ``train_aug``.
        mean, std: per-channel normalization.
        seed, drop_last: as in a standard loader.
    """

    def __init__(
        self,
        paths,
        batch_size=64,
        image_size=160,
        *,
        num_workers=8,
        shuffle=False,
        train_aug=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        seed=42,
        drop_last=False,
    ):
        import turboloader as t

        if not getattr(t, "metal_available", lambda: False)():
            raise RuntimeError(
                "GpuImageLoader needs the Metal GPU transform path (macOS arm64 build). "
                "metal_available() is False here."
            )
        self._t = t
        self.paths = list(paths)
        self.batch_size = int(batch_size)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.num_workers = max(1, int(num_workers))
        self.shuffle = bool(shuffle)
        self.train_aug = bool(train_aug)
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = float(hflip_prob)
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

    def _rrc_params(self, h, w, rng):
        """torchvision-style RandomResizedCrop window (x, y, cw, ch) in source pixels."""
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target = rng.uniform(self.scale[0], self.scale[1]) * area
            ar = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))
            cw = int(round(math.sqrt(target * ar)))
            ch = int(round(math.sqrt(target / ar)))
            if 0 < cw <= w and 0 < ch <= h:
                x = int(rng.integers(0, w - cw + 1))
                y = int(rng.integers(0, h - ch + 1))
                return float(x), float(y), float(cw), float(ch)
        s = float(min(h, w))  # fallback: center crop
        return float((w - s) // 2), float((h - s) // 2), s, s

    def __iter__(self):
        idx = np.arange(len(self.paths))
        if self.shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(idx)
        bs = self.batch_size
        end = (len(idx) // bs) * bs if self.drop_last else len(idx)
        aug_rng = np.random.default_rng(self.seed * 7919 + self._epoch)
        decode = self._t.decode_jpeg
        paths = self.paths
        dh, dw = self.image_size

        def _load(i):
            with open(paths[int(i)], "rb") as f:
                return decode(f.read())

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for start in range(0, end, bs):
                batch_idx = idx[start : start + bs]
                imgs = list(ex.map(_load, batch_idx))  # parallel CPU decode (GIL released)
                if self.train_aug:
                    crops = np.empty((len(imgs), 4), np.float32)
                    flips = np.empty(len(imgs), np.int32)
                    for j, im in enumerate(imgs):
                        crops[j] = self._rrc_params(im.shape[0], im.shape[1], aug_rng)
                        flips[j] = 1 if aug_rng.random() < self.hflip_prob else 0
                    yield self._t.metal_crop_resize_normalize(
                        imgs, crops, flips, dh, dw, mean=self.mean, std=self.std
                    )
                else:
                    yield self._t.metal_resize_normalize(imgs, dh, dw, mean=self.mean, std=self.std)
