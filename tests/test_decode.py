"""Tests for the decode_jpeg primitive (CPU, libjpeg-turbo)."""

import io

import numpy as np
import pytest

tl = pytest.importorskip("turboloader")

pytestmark = pytest.mark.skipif(
    not hasattr(tl, "decode_jpeg"), reason="decode_jpeg not in this build"
)


def _jpeg_bytes(arr, quality=95):
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def test_decode_jpeg_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)
    out = tl.decode_jpeg(_jpeg_bytes(img))
    assert out.shape == (64, 80, 3)
    assert out.dtype == np.uint8


def test_decode_jpeg_content_close():
    # A smooth gradient survives JPEG well; decoded should be close to the original.
    y = np.linspace(0, 255, 96, dtype=np.uint8)
    img = np.broadcast_to(y[:, None, None], (96, 96, 3)).copy()
    out = tl.decode_jpeg(_jpeg_bytes(img, quality=98))
    assert out.shape == (96, 96, 3)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 5


def test_decode_jpeg_bad_data_raises():
    with pytest.raises(Exception):
        tl.decode_jpeg(b"not a jpeg")
