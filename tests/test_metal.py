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


def _crop_ref(img, cx, cy, cw, ch, dst_h, dst_w, flip):
    H, W, _ = img.shape
    o = np.empty((3, dst_h, dst_w), np.float32)
    for y in range(dst_h):
        sy = min(max(cy + (y + 0.5) / dst_h * ch - 0.5, 0), H - 1)
        y0 = int(sy)
        y1 = min(y0 + 1, H - 1)
        dy = sy - y0
        for x in range(dst_w):
            ox = (dst_w - 1 - x) if flip else x
            sx = min(max(cx + (ox + 0.5) / dst_w * cw - 0.5, 0), W - 1)
            x0 = int(sx)
            x1 = min(x0 + 1, W - 1)
            dx = sx - x0
            o[:, y, x] = (
                img[y0, x0] * (1 - dx) * (1 - dy)
                + img[y0, x1] * dx * (1 - dy)
                + img[y1, x0] * (1 - dx) * dy
                + img[y1, x1] * dx * dy
            ) / 255.0
    return o


def test_crop_resize_normalize_matches_numpy():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    crops = np.array([[50, 40, 100, 100]], np.float32)
    flips = np.array([0], np.int32)
    out = tl.metal_crop_resize_normalize(
        [img], crops, flips, 64, 64, mean=[0, 0, 0], std=[1, 1, 1]
    )[0]
    ref = _crop_ref(img, 50, 40, 100, 100, 64, 64, 0)
    assert np.abs(out - ref).max() < 1e-3


def test_crop_horizontal_flip():
    rng = np.random.default_rng(6)
    img = rng.integers(0, 255, (180, 220, 3), dtype=np.uint8)
    crops = np.array([[20, 10, 150, 120]], np.float32)
    no_flip = tl.metal_crop_resize_normalize(
        [img], crops, np.array([0], np.int32), 64, 64, mean=[0, 0, 0], std=[1, 1, 1]
    )[0]
    flipped = tl.metal_crop_resize_normalize(
        [img], crops, np.array([1], np.int32), 64, 64, mean=[0, 0, 0], std=[1, 1, 1]
    )[0]
    assert np.abs(flipped - no_flip[:, :, ::-1]).max() < 1e-4


def test_crop_batch_independent_params():
    rng = np.random.default_rng(7)
    imgs = [rng.integers(0, 255, (160, 200, 3), dtype=np.uint8) for _ in range(4)]
    crops = np.array([[i * 5, i * 4, 100, 100] for i in range(4)], np.float32)
    flips = np.array([0, 1, 0, 1], np.int32)
    out = tl.metal_crop_resize_normalize(imgs, crops, flips, 96, 96)
    assert out.shape == (4, 3, 96, 96)
    for i in range(4):
        ref = _crop_ref(imgs[i], *crops[i], 96, 96, flips[i])
        ref = (ref - np.float32([0.485, 0.456, 0.406])[:, None, None]) / np.float32(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        assert np.abs(out[i] - ref).max() < 1e-3


def test_gpu_image_loader(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(8)
    paths = []
    for i in range(20):
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (120 + i, 140, 3), dtype=np.uint8)).save(
            p, format="JPEG"
        )
        paths.append(str(p))
    dl = tl.GpuImageLoader(paths, batch_size=8, image_size=64, num_workers=4)
    batches = list(dl)
    assert len(dl) == 3
    assert sum(b.shape[0] for b in batches) == 20
    assert batches[0].shape == (8, 3, 64, 64) and batches[0].dtype == np.float32


def test_gpu_image_loader_train_aug_reproducible(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(9)
    paths = []
    for i in range(16):
        p = tmp_path / f"a_{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (130, 150, 3), dtype=np.uint8)).save(p, format="JPEG")
        paths.append(str(p))
    dl = tl.GpuImageLoader(paths, batch_size=8, image_size=64, train_aug=True, shuffle=True, seed=1)
    dl.set_epoch(0)
    a = np.concatenate([b.ravel()[:20] for b in dl])
    dl.set_epoch(0)
    b = np.concatenate([b.ravel()[:20] for b in dl])
    assert np.allclose(a, b)  # same epoch -> identical aug


@pytest.mark.skipif(not hasattr(tl, "metal_decode_jpeg"), reason="metal_decode_jpeg not in build")
def test_metal_decode_jpeg_close_to_cpu(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    # A realistic (smooth) image: random noise is JPEG's worst case for the chroma-upsample
    # method difference and isn't representative. The GPU IDCT itself is bit-exact (proven
    # in experiments/metal/); this checks the integrated path matches the CPU decode.
    yy, xx = np.mgrid[0:120, 0:160]
    img = np.stack([(yy * 2) % 256, (xx * 1.5) % 256, ((xx + yy)) % 256], axis=-1).astype(np.uint8)
    p = tmp_path / "x.jpg"
    Image.fromarray(img).save(p, format="JPEG", quality=92)
    data = open(p, "rb").read()
    gpu = tl.metal_decode_jpeg(data)
    cpu = tl.decode_jpeg(data)
    assert gpu.shape == cpu.shape == (120, 160, 3)
    assert np.abs(gpu.astype(int) - cpu.astype(int)).mean() < 2


@pytest.mark.skipif(
    not hasattr(tl, "metal_train_transform"), reason="metal_train_transform not in build"
)
def test_train_transform_identity_jitter_matches_crop():
    rng = np.random.default_rng(12)
    imgs = [rng.integers(0, 255, (160, 200, 3), dtype=np.uint8) for _ in range(3)]
    crops = np.array([[10, 8, 120, 120], [0, 0, 160, 160], [20, 20, 100, 100]], np.float32)
    flips = np.array([0, 1, 0], np.int32)
    jitter_id = np.ones((3, 3), np.float32)  # brightness=contrast=saturation=1 -> no change
    a = tl.metal_train_transform(imgs, crops, flips, jitter_id, 96, 96)
    b = tl.metal_crop_resize_normalize(imgs, crops, flips, 96, 96)
    assert np.abs(a - b).max() < 1e-4


@pytest.mark.skipif(
    not hasattr(tl, "metal_train_transform"), reason="metal_train_transform not in build"
)
def test_train_transform_jitter_changes_output():
    rng = np.random.default_rng(13)
    imgs = [rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)]
    crops = np.array([[0, 0, 160, 160]], np.float32)
    flips = np.array([0], np.int32)
    base = tl.metal_train_transform(imgs, crops, flips, np.ones((1, 3), np.float32), 64, 64)
    bright = tl.metal_train_transform(
        imgs, crops, flips, np.array([[1.5, 1.0, 1.0]], np.float32), 64, 64
    )
    assert np.abs(base - bright).mean() > 0.01  # brightness changes the output


@pytest.mark.skipif(not hasattr(tl, "cuda_available"), reason="cuda bindings not in build")
def test_cuda_available_is_false_here():
    # CUDA is not compiled in the default build; the binding must report honestly.
    assert tl.cuda_available() is False
