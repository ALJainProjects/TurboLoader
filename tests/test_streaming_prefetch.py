"""Regression: prefetch_batches on the DirectBatch STREAMING path.

Before 2.33.0, ``prefetch_batches`` was honored only by the cached-epoch path; the
streaming path decoded synchronously (the trainer waited for every decode, the loader sat
idle during every training step). Now a bounded producer thread decodes ahead —
``next_batch`` releases the GIL for the whole C++ fill, so decode genuinely overlaps the
consumer's work. Invariants pinned here: byte-identical output/order vs the serial path,
and clean shutdown when the consumer abandons the generator mid-epoch.
"""

import tarfile

import numpy as np
import pytest

import turboloader

Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tar24(tmp_path_factory):
    root = tmp_path_factory.mktemp("prefetch")
    tar_path = str(root / "data.tar")
    rng = np.random.default_rng(3)
    with tarfile.open(tar_path, "w") as tf:
        for i in range(24):
            img = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
            p = str(root / f"{i:03d}.jpg")
            Image.fromarray(img.astype(np.uint8)).save(p, quality=92)
            tf.add(p, arcname=f"{i:03d}.jpg")
    return tar_path


def _loader(tar_path, prefetch):
    return turboloader.DataLoader(
        tar_path,
        batch_size=8,
        output_format="pytorch",
        image_size=48,
        shuffle=False,
        prefetch_batches=prefetch,
    )


def test_prefetch_output_identical_to_serial(tar24):
    serial = [(np.asarray(x).copy(), list(m["indices"])) for x, m in _loader(tar24, 0)]
    pref = [(np.asarray(x).copy(), list(m["indices"])) for x, m in _loader(tar24, 4)]
    assert len(serial) == len(pref) == 3
    for (xs, isx), (xp, ipx) in zip(serial, pref):
        assert isx == ipx
        assert np.array_equal(xs, xp)


def test_prefetch_abandon_midepoch_no_hang(tar24):
    loader = _loader(tar24, 4)
    it = iter(loader)
    next(it)
    del it  # abandon: producer must unblock and exit (drain + stop event)
    # a fresh epoch must still work and be complete
    assert sum(1 for _ in loader) == 3


def test_prefetch_shuffled_epochs_still_seeded(tar24):
    loader = turboloader.DataLoader(
        tar24,
        batch_size=8,
        output_format="pytorch",
        image_size=48,
        shuffle=True,
        prefetch_batches=2,
    )
    loader.set_epoch(5)
    o1 = [i for _x, m in loader for i in m["indices"]]
    loader.set_epoch(5)
    o2 = [i for _x, m in loader for i in m["indices"]]
    assert o1 == o2, "prefetch must not break seeded shuffle reproducibility"
