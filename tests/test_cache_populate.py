"""cache_decoded=True: preallocated single-copy populate + parallel serve gather.

Pins the behavior contract of the decoded cache after the memory rewrite
(one allocation, batches scattered into place — previously three full copies
at peak): served batches equal the on-the-fly pipeline sample-for-sample,
rows sit in original-index order, the array is one contiguous float32 block,
and the shuffled serve (now a parallel C++ row gather) matches numpy exactly.
"""

import io
import tarfile

import numpy as np
import pytest
from PIL import Image

import turboloader as tl

SIZE = 48
N_IMGS = 29  # not a multiple of the batch size


@pytest.fixture(scope="module")
def tar_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("cachepop") / "imgs.tar"
    rng = np.random.default_rng(3)
    with tarfile.open(p, "w") as tf:
        for i in range(N_IMGS):
            arr = rng.integers(0, 256, size=(64, 72, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=95)
            info = tarfile.TarInfo(f"{i:04d}.jpg")
            info.size = len(buf.getvalue())
            tf.addfile(info, io.BytesIO(buf.getvalue()))
    return str(p)


def _loader(tar_path, **kw):
    return tl.DataLoader(
        tar_path,
        batch_size=8,
        output_format="pytorch",
        image_size=SIZE,
        transform=tl.ImageNetNormalize(),
        num_workers=2,
        **kw,
    )


def _by_index(loader):
    out = {}
    for batch, meta in loader:
        for row, i in zip(np.asarray(batch), np.asarray(meta["indices"])):
            out[int(i)] = row.copy()
    return out


class TestPopulate:
    def test_cache_matches_on_the_fly_and_is_index_ordered(self, tar_path):
        fly = _loader(tar_path, shuffle=False)
        try:
            ref = _by_index(fly)
        finally:
            fly.close()
        cached = _loader(tar_path, shuffle=False, cache_decoded=True)
        try:
            got = _by_index(cached)
            impl = cached._impl if hasattr(cached, "_impl") else cached
            X, I = impl._cache_X, impl._cache_indices
            assert X.dtype == np.float32 and X.flags["C_CONTIGUOUS"]
            assert X.shape[0] == N_IMGS
            assert np.array_equal(I, np.arange(N_IMGS))  # original-index order
            for i in range(N_IMGS):
                assert np.array_equal(got[i], ref[i]), f"sample {i}"
                assert np.array_equal(X[i], ref[i])
        finally:
            cached.close()

    def test_cached_shuffle_is_deterministic_and_complete(self, tar_path):
        a = _loader(tar_path, shuffle=True, seed=5, cache_decoded=True)
        b = _loader(tar_path, shuffle=True, seed=5, cache_decoded=True)
        try:
            a.set_epoch(2)
            b.set_epoch(2)
            ia = [m["indices"] for _, m in a]
            ib = [m["indices"] for _, m in b]
            assert ia == ib
            flat = [i for bt in ia for i in bt]
            assert sorted(flat) == list(range(N_IMGS))
            # shuffled batches carry the right pixels for their indices
            ref = _by_index(_loader(tar_path, shuffle=False))
            for batch, meta in a:
                for row, i in zip(np.asarray(batch), meta["indices"]):
                    assert np.array_equal(row, ref[int(i)])
        finally:
            a.close()
            b.close()


class TestGatherRowsOp:
    def test_matches_numpy_fancy_index(self):
        rng = np.random.default_rng(1)
        ds = rng.random((37, 3, 9, 11), dtype=np.float32)
        idx = rng.permutation(37)[:13].astype(np.int64)
        out = np.empty((13, 3, 9, 11), dtype=np.float32)
        tl._gather_rows_f32(ds, idx, out)
        assert np.array_equal(out, ds[idx])

    def test_generic_row_shapes(self):
        ds = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
        out = np.empty((2, 5), dtype=np.float32)
        tl._gather_rows_f32(ds, np.array([5, 0], dtype=np.int64), out)
        assert np.array_equal(out, ds[[5, 0]])

    def test_rejects_bad_shapes_and_indices(self):
        ds = np.zeros((4, 2, 2), dtype=np.float32)
        with pytest.raises(Exception, match="range"):
            tl._gather_rows_f32(ds, np.array([4], dtype=np.int64), np.zeros((1, 2, 2), np.float32))
        with pytest.raises(Exception, match="row shape"):
            tl._gather_rows_f32(ds, np.array([0], dtype=np.int64), np.zeros((1, 2, 3), np.float32))
        with pytest.raises(Exception, match="rows"):
            tl._gather_rows_f32(ds, np.array([0], dtype=np.int64), np.zeros((2, 2, 2), np.float32))
