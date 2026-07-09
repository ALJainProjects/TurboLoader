"""Regression: corrupt samples must be OBSERVABLE, not silent black images.

Bug (<= 2.31.0): DirectBatchCore::process_one caught every decode failure with
``catch(...) { memset(out, 0, ...); }`` — a corrupt/truncated JPEG (or CMYK/grayscale
channel mismatch) trained the model on an all-black image with zero indication anywhere.
Now: a per-loader counter, a rate-limited stderr warning, ``decode_failures`` in every
batch's metadata, and a one-time Python RuntimeWarning.
"""

import tarfile
import warnings

import numpy as np
import pytest

import turboloader

Image = pytest.importorskip("PIL.Image")


@pytest.fixture()
def tar_with_corrupt(tmp_path):
    tar_path = str(tmp_path / "data.tar")
    with tarfile.open(tar_path, "w") as tf:
        for i in range(8):
            img = np.full((32, 32, 3), 100 + i, dtype=np.uint8)
            p = str(tmp_path / f"{i:03d}.jpg")
            Image.fromarray(img).save(p, quality=95)
            tf.add(p, arcname=f"{i:03d}.jpg")
        # two corrupt entries: truncated JPEG and pure garbage
        good = open(str(tmp_path / "000.jpg"), "rb").read()
        for name, payload in (
            ("bad_truncated.jpg", good[: len(good) // 3]),
            ("bad_garbage.jpg", b"\xde\xad\xbe\xef" * 64),
        ):
            p = str(tmp_path / name)
            with open(p, "wb") as f:
                f.write(payload)
            tf.add(p, arcname=name)
    return tar_path


def test_decode_failures_counted_and_warned(tar_with_corrupt):
    loader = turboloader.DataLoader(
        tar_with_corrupt, batch_size=10, output_format="pytorch", image_size=32, shuffle=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batches = [(np.asarray(im), meta) for im, meta in loader]
    total = sum(b[0].shape[0] for b in batches)
    assert total == 10, "corrupt samples should still be served (zero-filled), not dropped"

    # counter exposed on the loader and in metadata
    assert loader._impl.decode_failures >= 1, "decode_failures counter must be nonzero"
    assert batches[-1][1].get("decode_failures", 0) >= 1, "metadata must carry the count"

    # one-time Python warning fired
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("failed to decode" in m for m in msgs), f"expected RuntimeWarning, got: {msgs}"


def test_no_failures_no_warning(tmp_path):
    tar_path = str(tmp_path / "clean.tar")
    with tarfile.open(tar_path, "w") as tf:
        for i in range(4):
            img = np.full((32, 32, 3), 50 + i, dtype=np.uint8)
            p = str(tmp_path / f"{i}.jpg")
            Image.fromarray(img).save(p, quality=95)
            tf.add(p, arcname=f"{i}.jpg")
    loader = turboloader.DataLoader(
        tar_path, batch_size=4, output_format="pytorch", image_size=32, shuffle=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _im, meta in loader:
            assert meta.get("decode_failures", 0) == 0
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert loader._impl.decode_failures == 0
