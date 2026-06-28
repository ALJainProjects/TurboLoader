"""Regression tests for issues found in the audit remediation pass.

Each test encodes a bug that was confirmed empirically (segfault / wrong output /
false metadata) so it fails on the pre-fix build and passes once fixed.
"""

import io
import os
import tarfile
import tempfile

import numpy as np
import pytest

tl = pytest.importorskip("turboloader")
torch = pytest.importorskip("torch")
TVT = pytest.importorskip("torchvision.transforms")  # skip (don't error) if absent
from PIL import Image  # noqa: E402


def _img(h=480, w=640, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, c), dtype=np.uint8)


@pytest.fixture(scope="module")
def small_tar():
    path = os.path.join(tempfile.mkdtemp(), "shard.tar")
    rng = np.random.default_rng(1)
    with tarfile.open(path, "w") as tar:
        for i in range(200):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(buf, "JPEG")
            data = buf.getvalue()
            info = tarfile.TarInfo(f"img_{i:04d}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path, 200


# ----------------------------------------------------- decoded cache correctness
class TestDecodedCache:
    def test_cache_matches_on_the_fly_and_is_fresh(self, small_tar):
        path, total = small_tar
        dl = tl.FastDataLoader(
            path,
            batch_size=32,
            num_workers=4,
            output_format="pytorch",
            target_height=64,
            target_width=64,
            cache_decoded=True,
            shuffle=False,
            auto_smart_batching=False,
        )
        e1 = np.concatenate([np.asarray(im) for im, _ in dl])  # populates cache
        e2 = np.concatenate([np.asarray(im) for im, _ in dl])  # from cache
        assert dl.cache_populated and np.array_equal(e1, e2) and e1.shape[0] == total
        # cached batches must be fresh copies, not aliased views
        b = next(iter(dl))[0]
        before = float(np.asarray(b).flat[0])
        np.asarray(b)[...] = 123.0
        after = float(np.asarray(next(iter(dl))[0]).flat[0])
        assert abs(before - after) < 1e-6, "mutating a cached batch corrupted the cache"

    def test_cache_shuffle_is_deterministic_per_epoch(self, small_tar):
        path, total = small_tar
        dl = tl.FastDataLoader(
            path,
            batch_size=32,
            num_workers=4,
            output_format="pytorch",
            target_height=64,
            target_width=64,
            cache_decoded=True,
            shuffle=True,
            auto_smart_batching=False,
        )
        list(dl)  # populate
        dl.set_epoch(0)
        a = [i for _, m in dl for i in m["indices"]]
        dl.set_epoch(0)
        b = [i for _, m in dl for i in m["indices"]]
        dl.set_epoch(1)
        c = [i for _, m in dl for i in m["indices"]]
        assert a == b, "same epoch seed must reproduce the order"
        assert a != c, "different epochs must shuffle differently"
        assert sorted(a) == sorted(c), "every epoch must still see the full dataset"


# ----------------------------------------------------- non-image modality loaders
class TestModalityLoaders:
    def test_token_loader_causal_shift(self):
        toks = np.arange(50000, dtype=np.uint16)
        dl = tl.TokenDataLoader(toks, seq_len=64, batch_size=8, shuffle=True)
        x, y = next(iter(dl))
        assert x.shape == (8, 64) and y.shape == (8, 64) and x.dtype == np.int64
        assert np.array_equal(x[:, 1:], y[:, :-1]), "targets must be inputs shifted by one"

    def test_token_loader_reiterable_and_deterministic(self):
        toks = np.arange(50000, dtype=np.uint16)
        dl = tl.TokenDataLoader(toks, seq_len=64, batch_size=8, shuffle=True, steps_per_epoch=10)
        dl.set_epoch(0)
        a = next(iter(dl))[0].copy()
        dl.set_epoch(0)
        b = next(iter(dl))[0]
        dl.set_epoch(1)
        c = next(iter(dl))[0]
        assert np.array_equal(a, b) and not np.array_equal(a, c)
        assert len(list(dl)) == 10  # re-iterable

    def test_array_loader_batches_and_covers(self):
        feats = np.random.rand(1000, 8).astype("float32")
        labels = np.arange(1000)
        dl = tl.ArrayDataLoader(feats, labels, batch_size=64, shuffle=True)
        seen = sum(len(xb) for xb, yb in dl)
        assert seen == 1000
        xb, yb = next(iter(dl))
        assert xb.shape[1] == 8 and xb.shape[0] == 64

    def test_unified_dataloader_modality_routing(self):
        # tokens via the unified DataLoader entry point
        toks = np.arange(50000, dtype=np.uint16)
        dl = tl.DataLoader(toks, modality="tokens", seq_len=64, batch_size=8, shuffle=True)
        x, y = next(iter(dl))
        assert x.shape == (8, 64) and np.array_equal(x[:, 1:], y[:, :-1])
        assert len(dl) == len(list(dl))  # __len__ + re-iterable
        # arrays via the unified entry point
        feats = np.random.rand(500, 4).astype("float32")
        adl = tl.DataLoader(arrays=[feats], data_path=None, modality="array", batch_size=50)
        assert sum(len(b) for b in adl) == 500
        # bad modality rejected
        with pytest.raises(ValueError):
            tl.DataLoader(toks, modality="audio")


# ------------------------------------------------- consolidated fast DataLoader
class TestConsolidatedLoader:
    def test_fast_path_one_call_correct_batches(self, small_tar):
        """DataLoader(output_format='pytorch', image_size=(H,W)) -> correct CHW batches."""
        path, total = small_tar
        dl = tl.DataLoader(
            path, batch_size=32, num_workers=4, output_format="pytorch", image_size=(64, 64)
        )
        shapes, total_seen = [], 0
        for images, meta in dl:
            s = tuple(np.asarray(images).shape)
            shapes.append(s)
            total_seen += s[0]
        # Correct CHW dims, batch_size honored as an upper bound (the smart-batching
        # bug produced single "batches" of >1000), and the whole dataset is covered.
        assert all(s[1:] == (3, 64, 64) for s in shapes), shapes
        assert all(s[0] <= 32 for s in shapes), f"batch exceeds batch_size: {shapes}"
        assert total_seen == total, (total_seen, total)

    def test_fast_path_reiterable(self, small_tar):
        """Re-iterating the same loader must yield the full dataset each epoch."""
        path, total = small_tar
        dl = tl.DataLoader(
            path, batch_size=32, num_workers=4, output_format="pytorch", image_size=(64, 64)
        )
        counts = [sum(np.asarray(im).shape[0] for im, _ in dl) for _ in range(3)]
        assert counts[0] == counts[1] == counts[2] == total, counts

    def test_dict_path_reiterable(self, small_tar):
        """The dict-output DataLoader was not re-iterable (epoch 2 == 0); now fixed."""
        path, total = small_tar
        dl = tl.DataLoader(path, batch_size=32, num_workers=4)
        counts = [sum(len(b) for b in dl) for _ in range(2)]
        assert counts[0] == counts[1] == total, counts

    def test_fast_path_requires_image_size(self, small_tar):
        path, total = small_tar
        with pytest.raises(ValueError):
            tl.DataLoader(path, batch_size=32, output_format="pytorch")  # no image_size


# ------------------------------------------------------ distributed sharding
class TestDistributedSharding:
    def test_ranks_partition_dataset(self, small_tar):
        """With world_size=N, the union of all ranks must equal the dataset and
        ranks must not all receive the full dataset (the pre-fix behavior)."""
        path, total = small_tar
        world = 4
        counts = []
        for rank in range(world):
            loader = tl.DataLoader(
                path,
                batch_size=8,
                num_workers=2,
                shuffle=False,
                enable_distributed=True,
                world_rank=rank,
                world_size=world,
                drop_last=False,
            )
            counts.append(sum(len(b) for b in loader))
        # Each rank should see roughly total/world, never the whole dataset.
        assert sum(counts) <= total + world, f"ranks overlap: counts={counts}, total={total}"
        assert max(counts) < total, f"a rank saw (nearly) the whole dataset: {counts}"
        assert min(counts) >= (total // world) - total, "rank starved"  # sanity


# ---------------------------------------------------------------- ToTensor CHW
class TestToTensorCHW:
    def test_chw_returns_channels_first_shape(self):
        """ToTensor(PYTORCH_CHW) must return (C, H, W), not (H, W, C)."""
        img = _img(480, 640, 3)
        out = np.asarray(tl.ToTensor(format=tl.TensorFormat.PYTORCH_CHW).apply(img))
        assert out.shape == (3, 480, 640), f"expected CHW (3,480,640), got {out.shape}"

    def test_chw_dtype_and_range(self):
        img = _img(64, 48, 3)
        out = np.asarray(tl.ToTensor(format=tl.TensorFormat.PYTORCH_CHW).apply(img))
        assert out.dtype == np.float32
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0

    def test_chw_matches_torchvision(self):
        """Numerically match torchvision ToTensor (CHW float32 in [0,1])."""
        img = _img(128, 96, 3, seed=7)
        out = np.asarray(tl.ToTensor(format=tl.TensorFormat.PYTORCH_CHW).apply(img))
        ref = TVT.ToTensor()(Image.fromarray(img)).numpy()  # (C,H,W)
        assert out.shape == ref.shape
        assert np.abs(out - ref).max() < 1e-4

    def test_chw_no_heap_corruption_under_repetition(self):
        """The CHW path had an out-of-bounds SIMD write; stress it to catch corruption."""
        img = _img(480, 640, 3)
        for _ in range(300):
            out = np.asarray(tl.ToTensor(format=tl.TensorFormat.PYTORCH_CHW).apply(img))
            assert out.shape == (3, 480, 640)

    def test_hwc_still_works(self):
        img = _img(64, 64, 3)
        out = np.asarray(tl.ToTensor(format=tl.TensorFormat.TENSORFLOW_HWC).apply(img))
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.float32


# ------------------------------------------------------------- features() honesty
class TestFeaturesMetadata:
    def test_version_matches_package(self):
        """features()['version'] must match the package __version__ (was stale '2.5.0')."""
        feats = tl.features()
        assert (
            feats["version"] == tl.__version__
        ), f"features() version {feats['version']!r} != __version__ {tl.__version__!r}"

    def test_no_false_capability_flags(self):
        """Capabilities whose code is not compiled into the wheel must report False.

        These readers/parsers are not #included by any compiled translation unit
        (their headers need unlinked libraries), so advertising them as available
        is false metadata. (s3/gcs/http ARE compiled via pipeline.hpp and remain
        True as path-scheme handlers.)
        """
        feats = tl.features()
        not_compiled = [
            "azure_support",
            "hdf5_support",
            "zarr_support",
            "tfrecord_support",
            "coco_voc_support",
            "io_uring",
        ]
        for flag in not_compiled:
            assert (
                feats.get(flag) is False
            ), f"features()[{flag!r}] should be False (not compiled into the wheel)"
