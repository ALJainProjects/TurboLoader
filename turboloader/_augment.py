"""Shared augmentation sampling — the ONE Python copy of RandomResizedCrop.

This must stay in exact agreement with the C++ `pick_crop` in
src/pipeline/direct_batch_loader.hpp (both implement torchvision's
RandomResizedCrop parity, INCLUDING the fallback: an aspect-ratio-CLAMPED
central crop, not a center square). Before this module existed there were three
divergent copies whose fallbacks produced different augmentation distributions
across the DirectBatch / Metal / CUDA paths whenever the 10-attempt loop failed.
"""

import math

__all__ = ["pick_crop"]


def pick_crop(w, h, rng, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    """torchvision RandomResizedCrop window in source pixels.

    Returns (x, y, cw, ch) floats. ``rng`` is a numpy Generator.
    """
    area = float(w) * float(h)
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(10):
        target = area * rng.uniform(scale[0], scale[1])
        ar = math.exp(rng.uniform(log_lo, log_hi))
        cw = int(round(math.sqrt(target * ar)))
        ch = int(round(math.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            x = int(rng.integers(0, w - cw + 1))
            y = int(rng.integers(0, h - ch + 1))
            return float(x), float(y), float(cw), float(ch)
    # Fallback (torchvision-exact, matches the C++ pick_crop): central crop at
    # the CLAMPED aspect ratio — NOT a center square.
    in_ratio = float(w) / float(h)
    cw, ch = w, h
    if in_ratio < ratio[0]:
        cw = w
        ch = min(h, int(round(w / ratio[0])))
    elif in_ratio > ratio[1]:
        ch = h
        cw = min(w, int(round(h * ratio[1])))
    return float((w - cw) // 2), float((h - ch) // 2), float(cw), float(ch)
