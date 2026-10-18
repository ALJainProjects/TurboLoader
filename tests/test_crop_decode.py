"""Crop-window JPEG decode, the train_aug fast path's decode step.

``begin_scaled`` reports the DCT-scaled size without decoding pixels; ``read_region``
decodes only rows [y, y+h) of the iMCU-aligned column span covering [x, x+w)
(jpeg_crop_scanline + jpeg_skip_scanlines) and aborts the rest. Contract pinned
here: the window is pixel-identical to the same window of a full scaled decode,
the decoder stays reusable afterwards, and the loader path stays deterministic.
"""

import io
import tarfile

import numpy as np
import pytest

import turboloader as tl

Image = pytest.importorskip("PIL.Image")

TARGET = 160  # 320x240 at target 160 picks 3/4 -> 240x180


def _jpeg(w=320, h=240, q=90, ss=2, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // (w - 1), y * 255 // (h - 1), rng.random((h, w)) * 255], -1)
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, "JPEG", quality=q, subsampling=ss)
    return buf.getvalue()


@pytest.mark.parametrize("ss", [2, 1, 0])  # 4:2:0, 4:2:2, 4:4:4
def test_region_is_pixel_identical_to_full_decode(ss):
    b = _jpeg(ss=ss)
    full = tl.decode_jpeg_scaled(b, TARGET, TARGET)
    H, W = full.shape[:2]
    assert (H, W) == (180, 240)
    rng = np.random.default_rng(1)
    for _ in range(25):
        cw, ch = int(rng.integers(1, W + 1)), int(rng.integers(1, H + 1))
        cx, cy = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
        strip, sx, sw, sh = tl.decode_jpeg_region(b, TARGET, TARGET, cx, cy, cw, ch)
        assert (sw, sh) == (W, H)
        assert sx <= cx and sx + strip.shape[1] >= cx + cw and strip.shape[0] == ch
        np.testing.assert_array_equal(
            strip[:, cx - sx : cx - sx + cw], full[cy : cy + ch, cx : cx + cw]
        )


def test_region_spans_align_to_mcu_boundaries():
    b = _jpeg(ss=2)
    strip, sx, _, _ = tl.decode_jpeg_region(b, TARGET, TARGET, 37, 29, 101, 77)
    assert sx <= 37 and sx + strip.shape[1] >= 138
    # the span never widens by more than one iMCU on each side (16 px at 4:2:0 x 3/4 = 12)
    assert 37 - sx < 16 and (sx + strip.shape[1]) - 138 < 16


def test_region_rejects_windows_outside_the_scaled_image():
    b = _jpeg()
    with pytest.raises(ValueError):
        tl.decode_jpeg_region(b, TARGET, TARGET, 200, 0, 100, 10)
    with pytest.raises(ValueError):
        tl.decode_jpeg_region(b, TARGET, TARGET, 0, 170, 10, 20)


def test_decoder_is_reusable_after_a_region_decode():
    # The decoder object is thread_local and reused: an aborted (partial) decode
    # must leave it able to decode the next image fully.
    b = _jpeg(seed=3)
    fresh = tl.decode_jpeg_scaled(b, TARGET, TARGET)
    tl.decode_jpeg_region(b, TARGET, TARGET, 10, 10, 20, 20)
    with pytest.raises(ValueError):
        tl.decode_jpeg_region(b, TARGET, TARGET, 230, 0, 20, 20)
    np.testing.assert_array_equal(tl.decode_jpeg_scaled(b, TARGET, TARGET), fresh)


def test_scaled_decode_picks_the_smallest_covering_ratio():
    b = _jpeg()
    assert tl.decode_jpeg_scaled(b, 100, 100).shape[:2] == (120, 160)  # 1/2
    assert tl.decode_jpeg_scaled(b, 240, 240).shape[:2] == (240, 320)  # 1/1
    assert tl.decode_jpeg_scaled(b, 300, 300).shape[:2] == (300, 400)  # 5/4: upscaled in the IDCT
    assert tl.decode_jpeg_scaled(b, 40, 40).shape[:2] == (60, 80)  # 1/4: 1/8 would give 30 rows
    assert tl.decode_jpeg_scaled(b, 160, 160).shape[:2] == (180, 240)  # 3/4: the height decides


@pytest.fixture(scope="module")
def small_tar(tmp_path_factory):
    root = tmp_path_factory.mktemp("cropdecode")
    p = root / "imgs.tar"
    with tarfile.open(p, "w") as tf:
        for i in range(16):
            data = _jpeg(seed=i, w=200 + 8 * i, h=150 + 5 * i)
            info = tarfile.TarInfo(f"{i:03d}.jpg")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        bad = tarfile.TarInfo("016.jpg")  # not a JPEG at all: must zero-fill and be counted
        payload = b"definitely not a jpeg " * 64
        bad.size = len(payload)
        tf.addfile(bad, io.BytesIO(payload))
    return str(p)


def _aug_batches(tar, seed):
    loader = tl.DataLoader(
        tar,
        batch_size=6,
        output_format="numpy_chw",
        image_size=64,
        train_aug=True,
        hflip_prob=0.5,
        shuffle=False,
        seed=seed,
        num_workers=2,
    )
    try:
        return [(np.asarray(x).copy(), dict(m)) for x, m in loader]
    finally:
        loader.close()


def test_train_aug_loader_is_deterministic_and_counts_the_bad_file(small_tar):
    a = _aug_batches(small_tar, seed=5)
    b = _aug_batches(small_tar, seed=5)
    assert len(a) == 3 and all(np.array_equal(x, y) for (x, _), (y, _) in zip(a, b))
    c = _aug_batches(small_tar, seed=6)
    assert not all(np.array_equal(x, y) for (x, _), (y, _) in zip(a, c))
    allx = np.concatenate([x for x, _ in a])
    assert allx.shape[1:] == (3, 64, 64) and np.isfinite(allx).all()
    assert allx[:16].min() >= 0.0 and allx[:16].max() <= 1.0 and allx[:16].max() > 0.5
    assert np.all(allx[16] == 0.0)  # the non-JPEG zero-fills instead of crashing
    assert int(a[-1][1]["decode_failures"]) == 1  # cumulative count on the last batch
