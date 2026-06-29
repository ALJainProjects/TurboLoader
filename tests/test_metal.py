"""Tests for the Metal GPU transform path (macOS arm64). Skipped where unavailable."""

import numpy as np
import pytest

tl = pytest.importorskip("turboloader")

pytestmark = pytest.mark.skipif(
    not getattr(tl, "metal_available", lambda: False)(),
    reason="Metal GPU transform path not available (not macOS arm64, or not compiled in)",
)

MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])


def _numpy_ref(img, dst_h, dst_w):
    """Half-pixel bilinear resize + normalize, matching the kernel's convention."""
    h, w, _ = img.shape
    out = np.empty((3, dst_h, dst_w), np.float32)
    for y in range(dst_h):
        sy = max(0.0, (y + 0.5) * h / dst_h - 0.5)
        y0 = int(sy)
        y1 = min(y0 + 1, h - 1)
        dy = sy - y0
        for x in range(dst_w):
            sx = max(0.0, (x + 0.5) * w / dst_w - 0.5)
            x0 = int(sx)
            x1 = min(x0 + 1, w - 1)
            dx = sx - x0
            p = (
                img[y0, x0] * (1 - dx) * (1 - dy)
                + img[y0, x1] * dx * (1 - dy)
                + img[y1, x0] * (1 - dx) * dy
                + img[y1, x1] * dx * dy
            ) / 255.0
            out[:, y, x] = (p - MEAN) / STD
    return out


def test_device_name_nonempty():
    assert tl.metal_device_name() != ""


def test_shape_and_dtype():
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 255, (160, 213, 3), dtype=np.uint8) for _ in range(5)]
    out = tl.metal_resize_normalize(imgs, 160, 160)
    assert out.shape == (5, 3, 160, 160)
    assert out.dtype == np.float32


def test_bit_accurate_vs_numpy_bilinear():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (137, 211, 3), dtype=np.uint8)
    out = tl.metal_resize_normalize([img], 160, 160)[0]
    ref = _numpy_ref(img, 160, 160)
    assert np.abs(out - ref).max() < 1e-3


def test_variable_sizes_in_one_batch():
    rng = np.random.default_rng(3)
    imgs = [
        rng.integers(
            0, 255, (int(rng.integers(120, 220)), int(rng.integers(120, 220)), 3), dtype=np.uint8
        )
        for _ in range(6)
    ]
    out = tl.metal_resize_normalize(imgs, 96, 96)
    assert out.shape == (6, 3, 96, 96)
    # each row should match its own CPU reference
    for i in range(6):
        assert np.abs(out[i] - _numpy_ref(imgs[i], 96, 96)).max() < 1e-3


def test_custom_mean_std():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    out = tl.metal_resize_normalize([img], 160, 160, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])[0]
    # with mean=0,std=1 the output is just the resized image in [0,1]
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_empty_list_raises():
    with pytest.raises(Exception):
        tl.metal_resize_normalize([], 160, 160)
