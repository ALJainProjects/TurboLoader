"""Mid-epoch checkpoint/resume: state_dict()/load_state_dict().

Batch order on the fast paths is deterministic given (seed, epoch, shard), so
{epoch, batches_served} is sufficient state: a resumed loader rebuilds the identical
order and skips the served prefix WITHOUT decoding it. The invariant pinned here is
EXACT continuation: reference tail == interrupted run + resumed run, byte-for-byte.
"""

import tarfile

import numpy as np
import pytest

import turboloader

Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tar48(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    tar_path = str(root / "data.tar")
    rng = np.random.default_rng(5)
    with tarfile.open(tar_path, "w") as tf:
        for i in range(48):
            img = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
            p = str(root / f"{i:03d}.jpg")
            Image.fromarray(img.astype(np.uint8)).save(p, quality=92)
            tf.add(p, arcname=f"{i:03d}.jpg")
    return tar_path


def _mk(tar, **kw):
    kw.setdefault("prefetch_batches", 0)
    return turboloader.DataLoader(
        tar, batch_size=8, output_format="pytorch", image_size=40, shuffle=True, seed=9, **kw
    )


def _epoch(loader, epoch):
    loader.set_epoch(epoch)
    return [(np.asarray(x).copy(), list(m["indices"])) for x, m in loader]


@pytest.mark.parametrize("kw", [{}, {"prefetch_batches": 3}, {"train_aug": True}])
def test_exact_midepoch_resume(tar48, kw):
    ref = _epoch(_mk(tar48, **kw), epoch=2)
    # interrupted run: serve 2 of 6 batches, checkpoint
    src = _mk(tar48, **kw)
    src.set_epoch(2)
    it = iter(src)
    got = [next(it), next(it)]
    sd = src.state_dict()
    assert sd["epoch"] == 2 and sd["batches_served"] == 2
    del it
    # fresh loader (as after a crash/restart) resumes exactly
    dst = _mk(tar48, **kw)
    dst.load_state_dict(sd)
    tail = [(np.asarray(x).copy(), list(m["indices"])) for x, m in dst]
    assert len(got) + len(tail) == len(ref)
    for (xr, ir), (xt, it_) in zip(ref[2:], tail):
        assert ir == it_, "resumed order diverged"
        assert np.array_equal(xr, xt), "resumed pixels diverged"


def test_resume_cached_path(tar48):
    ref = _epoch(_mk(tar48, cache_decoded=True), epoch=1)
    src = _mk(tar48, cache_decoded=True)
    src.set_epoch(1)
    it = iter(src)
    next(it)
    sd = src.state_dict()
    del it
    dst = _mk(tar48, cache_decoded=True)
    dst.load_state_dict(sd)
    tail = [(np.asarray(x).copy(), list(m["indices"])) for x, m in dst]
    assert len(tail) == len(ref) - 1
    for (xr, ir), (xt, it_) in zip(ref[1:], tail):
        assert ir == it_ and np.array_equal(xr, xt)


def test_token_loader_resume():
    tokens = np.arange(100_000, dtype=np.uint16)
    mk = lambda: turboloader.TokenDataLoader(tokens, seq_len=64, batch_size=32, seed=4)
    snap = lambda b: tuple(np.asarray(t).copy() for t in (b if isinstance(b, tuple) else (b,)))
    a = mk()
    a.set_epoch(3)
    ref = [snap(b) for b in a]
    b_ = mk()
    b_.set_epoch(3)
    it = iter(b_)
    next(it)
    next(it)
    sd = b_.state_dict()
    del it
    c = mk()
    c.load_state_dict(sd)
    tail = [snap(b) for b in c]
    assert len(tail) == len(ref) - 2
    for xr, xt in zip(ref[2:], tail):
        assert all(np.array_equal(r, t) for r, t in zip(xr, xt))


def test_unsupported_config_raises(tar48):
    loader = turboloader.DataLoader(tar48, batch_size=8, output_format="dict")
    with pytest.raises(NotImplementedError):
        loader.state_dict()
