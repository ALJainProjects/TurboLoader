"""VideoDatasetLoader: discovery on any platform; clip batches under CUDA.

The CUDA tests build a tiny REAL 2-class dataset (PyAV-encoded mpeg4: class 0 =
dark footage, class 1 = bright) and verify label alignment through pixel means,
determinism of the (seed, epoch) plan, and the batch contract. Kernel math is
already pinned down by test_cuda_video.py."""

import os

import numpy as np
import pytest

from turboloader.video_dataset import _discover

try:
    import av
except ImportError:
    av = None

try:
    import torch

    _cuda_ok = torch.cuda.is_available()
except ImportError:
    _cuda_ok = False

import turboloader as tl

_loader_ok = (
    av is not None
    and _cuda_ok
    and getattr(tl, "cuda_available", lambda: False)()
    and hasattr(tl, "cuda_video_clip_yuv420")
)
cuda_video = pytest.mark.skipif(not _loader_ok, reason="CUDA video path not available")


class TestDiscover:
    def test_classes_sorted_and_labeled(self, tmp_path):
        for cls in ("zebra", "ant", "moth"):
            d = tmp_path / cls
            d.mkdir()
            (d / "a.mp4").touch()
            (d / "b.MP4").touch()
            (d / "notes.txt").touch()
        items, classes = _discover(str(tmp_path))
        assert classes == ["ant", "moth", "zebra"]
        assert len(items) == 6
        assert all(p.lower().endswith(".mp4") for p, _ in items)
        labels = {os.path.basename(os.path.dirname(p)): y for p, y in items}
        assert labels == {"ant": 0, "moth": 1, "zebra": 2}

    def test_empty_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="no videos"):
            _discover(str(tmp_path))


def _encode(path, frames_rgb, rate=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = frames_rgb[0].shape[1]
        stream.height = frames_rgb[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 8_000_000
        for arr in frames_rgb:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


@pytest.fixture(scope="module")
def two_class_root(tmp_path_factory):
    if av is None:
        pytest.skip("PyAV required")
    root = tmp_path_factory.mktemp("vids")
    rng = np.random.default_rng(0)
    for ci, (cls, lo, hi) in enumerate([("dark", 0, 60), ("bright", 180, 250)]):
        d = root / cls
        d.mkdir()
        for v in range(3):
            frames = [rng.integers(lo, hi, size=(96, 128, 3)).astype(np.uint8) for _ in range(24)]
            _encode(d / f"{cls}_{v}.mp4", frames)
    return str(root)


@cuda_video
class TestLoader:
    def test_batch_contract_and_labels(self, two_class_root):
        dl = tl.VideoDatasetLoader(
            two_class_root,
            clip_len=4,
            batch_size=4,
            image_size=64,
            workers=2,
            seed=1,
            steps_per_epoch=3,
        )
        assert dl.classes == ["bright", "dark"]
        seen = 0
        for clips, labels, meta in dl:
            assert clips.shape == (4, 4, 3, 64, 64) and clips.is_cuda
            assert clips.dtype == torch.float32
            assert labels.shape == (4,) and labels.is_cuda
            # label alignment through the pixels: bright class must denormalize
            # brighter than dark class, per clip
            means = clips.mean(dim=(1, 2, 3, 4)).cpu().numpy()
            for m, y, p in zip(means, labels.cpu().numpy(), meta["paths"]):
                assert ("bright" in p) == (y == 0)
                assert (m > 0) == (y == 0), (m, y, p)
            seen += 1
        assert seen == 3

    def test_deterministic_per_epoch(self, two_class_root):
        kw = dict(
            clip_len=4,
            batch_size=4,
            image_size=64,
            workers=3,
            seed=7,
            steps_per_epoch=2,
            train_aug=True,
        )
        a = tl.VideoDatasetLoader(two_class_root, **kw)
        b = tl.VideoDatasetLoader(two_class_root, **kw)
        a.set_epoch(5)
        b.set_epoch(5)
        for (ca, la, ma), (cb, lb, mb) in zip(a, b):
            assert ma == mb  # same files, starts, crops, flips despite 3 workers
            assert torch.equal(la, lb)
            assert torch.allclose(ca, cb)

    def test_epoch_changes_plan(self, two_class_root):
        dl = tl.VideoDatasetLoader(
            two_class_root,
            clip_len=4,
            batch_size=4,
            image_size=64,
            workers=2,
            seed=7,
            steps_per_epoch=2,
            train_aug=True,
        )
        dl.set_epoch(0)
        m0 = [m for _, _, m in dl]
        dl.set_epoch(1)
        m1 = [m for _, _, m in dl]
        assert m0 != m1

    def test_early_exit_no_hang(self, two_class_root):
        dl = tl.VideoDatasetLoader(
            two_class_root,
            clip_len=4,
            batch_size=4,
            image_size=64,
            workers=2,
            seed=1,
            steps_per_epoch=50,
        )
        for i, _ in enumerate(dl):
            if i == 1:
                break  # generator close must wind down feeder + workers
