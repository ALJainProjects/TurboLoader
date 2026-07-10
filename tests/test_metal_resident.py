"""Correctness tests for the Metal resident (pre-processed) loaders.

Skipped entirely off-macOS / without a Metal device. Every numeric check is
against a plain numpy reference — same discipline as the CUDA kernels (which
match numpy to ~4e-07)."""

import numpy as np
import pytest

import turboloader as tl

pytestmark = pytest.mark.skipif(
    not getattr(tl, "metal_available", lambda: False)()
    or not hasattr(tl, "metal_resident_images_create"),
    reason="Metal resident path not available",
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ref_normalize(imgs_u8):  # (B,H,W,3) uint8 -> (B,3,H,W) float32
    x = imgs_u8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (0, 3, 1, 2)).copy()


def test_resident_images_gather_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(37, 24, 24, 3), dtype=np.uint8)
    h = tl.metal_resident_images_create(37, 24, 24, 16)
    try:
        view = tl.metal_resident_images_view(h, 37, 24, 24)
        view[:] = data
        idx = rng.permutation(37)[:16].astype(np.int32)
        out = tl.metal_resident_images_gather(h, idx, 24, 24, mean=MEAN.tolist(), std=STD.tolist())
        assert out.shape == (16, 3, 24, 24) and out.dtype == np.float32
        np.testing.assert_allclose(out, _ref_normalize(data[idx]), atol=1e-5)
    finally:
        tl.metal_resident_images_destroy(h)


def test_resident_images_double_buffer_lifetime():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, size=(8, 16, 16, 3), dtype=np.uint8)
    h = tl.metal_resident_images_create(8, 16, 16, 4)
    try:
        tl.metal_resident_images_view(h, 8, 16, 16)[:] = data
        i0 = np.array([0, 1, 2, 3], dtype=np.int32)
        i1 = np.array([4, 5, 6, 7], dtype=np.int32)
        b0 = tl.metal_resident_images_gather(h, i0, 16, 16)
        b0_copy = b0.copy()
        b1 = tl.metal_resident_images_gather(h, i1, 16, 16)  # b0 must survive ONE more gather
        np.testing.assert_array_equal(b0, b0_copy)
        np.testing.assert_allclose(b1, _ref_normalize(data[i1]), atol=1e-5)
    finally:
        tl.metal_resident_images_destroy(h)


def test_resident_images_rejects_bad_index():
    h = tl.metal_resident_images_create(4, 8, 8, 4)
    try:
        tl.metal_resident_images_view(h, 4, 8, 8)[:] = 0
        with pytest.raises(RuntimeError):
            tl.metal_resident_images_gather(h, np.array([4], dtype=np.int32), 8, 8)
        with pytest.raises(RuntimeError):
            tl.metal_resident_images_gather(h, np.array([-1], dtype=np.int32), 8, 8)
    finally:
        tl.metal_resident_images_destroy(h)


@pytest.mark.parametrize(
    "row_shape,dtype", [((7,), np.float32), ((5, 3), np.uint16), ((13,), np.int64)]
)
def test_resident_arrays_gather_matches_fancy_index(row_shape, dtype):
    rng = np.random.default_rng(2)
    a = (rng.integers(0, 1000, size=(50,) + row_shape)).astype(dtype)
    ra = tl.MetalResidentArrays(a, max_batch=32)
    try:
        idx = rng.permutation(50)[:32]
        np.testing.assert_array_equal(ra.gather(idx), a[idx])
    finally:
        ra.close()


def test_resident_bytes_unaligned_span():
    # span_bytes not a multiple of the kernel's 16-byte chunk: tail must be exact.
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, size=(20, 21), dtype=np.uint8)  # 21-byte rows
    ra = tl.MetalResidentArrays(a, max_batch=8)
    try:
        idx = np.array([3, 19, 0, 7], dtype=np.int64)
        np.testing.assert_array_equal(ra.gather(idx), a[idx])
    finally:
        ra.close()


def test_token_gather_windows_shift_by_one():
    rng = np.random.default_rng(4)
    toks = rng.integers(0, 50257, size=100_000).astype(np.uint16)
    tg = tl.MetalTokenGather(toks, seq_len=64, batch_size=8, seed=7)
    try:
        x, y = tg.next_batch()
        assert x.shape == y.shape == (8, 64) and x.dtype == y.dtype == np.int64
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])  # y is x shifted by one
        # every window must be a verbatim slice of the source stream
        for r in range(8):
            starts = np.flatnonzero(
                np.all(
                    np.lib.stride_tricks.sliding_window_view(toks.astype(np.int64), 64) == x[r],
                    axis=1,
                )
            )
            assert starts.size >= 1
    finally:
        tg.close()


def test_metal_resident_loader_end_to_end():
    rng = np.random.default_rng(5)
    data = rng.integers(0, 256, size=(40, 16, 16, 3), dtype=np.uint8)
    with tl.MetalResidentLoader(
        data, batch_size=16, shuffle=True, seed=9, return_indices=True
    ) as dl:
        assert len(dl) == 2  # drop_last: 40 // 16
        dl.set_epoch(0)
        seen = []
        for batch, idx in dl:
            assert batch.shape == (16, 3, 16, 16)
            np.testing.assert_allclose(batch, _ref_normalize(data[idx]), atol=1e-5)
            seen.extend(idx.tolist())
        assert len(set(seen)) == 32  # shuffled, no repeats
        # epoch determinism: same epoch -> same order; different epoch -> different
        dl.set_epoch(0)
        order0 = [i for _, idx in dl for i in idx.tolist()]
        dl.set_epoch(1)
        order1 = [i for _, idx in dl for i in idx.tolist()]
        assert order0 == seen and order0 != order1
