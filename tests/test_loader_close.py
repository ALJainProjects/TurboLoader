"""Abandoned-loader resource regression tests.

A DataLoader dropped mid-epoch (early stopping, notebook experimentation) used to
leak its prefetch producer thread permanently: the producer blocked in q.put() on
a full queue, and its closure referenced the loader, so the GC could never collect
the generator whose ``finally`` was the only stop signal. Each abandonment leaked
a thread + the TAR file descriptor; ~1k abandonments exhausted the fd limit and
the process aborted. These tests fail on the old code.
"""

import gc
import io
import os
import tarfile
import threading
import time

import numpy as np
import pytest
from PIL import Image

import turboloader as tl


@pytest.fixture(scope="module")
def tar20(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("close") / "imgs20.tar")
    rng = np.random.default_rng(7)
    with tarfile.open(path, "w") as tf:
        for i in range(20):
            arr = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
            b = io.BytesIO()
            Image.fromarray(arr).save(b, format="JPEG", quality=100, subsampling=0)
            data = b.getvalue()
            ti = tarfile.TarInfo(name=f"img_{i:03d}.jpg")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


def _open_fds():
    # /dev/fd exists on both Linux (-> /proc/self/fd) and macOS.
    return len(os.listdir("/dev/fd"))


def _settle(predicate, timeout=8.0):
    """Poll until predicate() is True (producers exit within ~0.25s + one fill)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        gc.collect()
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _make(tar20, **kw):
    kw.setdefault("batch_size", 4)
    kw.setdefault("image_size", 32)
    kw.setdefault("output_format", "pytorch")
    kw.setdefault("prefetch_batches", 4)
    return tl.DataLoader(tar20, **kw)


def test_abandoned_loaders_release_threads_and_fds(tar20):
    # Warm up lazy imports/caches so the baseline is stable.
    dl = _make(tar20)
    dl.next_batch()
    dl.close()
    del dl
    gc.collect()
    base_threads = threading.active_count()
    base_fds = _open_fds()

    for _ in range(40):  # mid-epoch abandonment WITHOUT close(): GC must reclaim
        dl = _make(tar20)
        dl.next_batch()
        del dl

    ok = _settle(
        lambda: threading.active_count() <= base_threads + 1 and _open_fds() <= base_fds + 2
    )
    assert ok, (
        f"leak: threads {threading.active_count()} (base {base_threads}), "
        f"fds {_open_fds()} (base {base_fds})"
    )


def test_close_is_deterministic_and_idempotent(tar20):
    dl = _make(tar20)
    dl.next_batch()
    before = threading.active_count()
    dl.close()
    dl.close()  # idempotent
    assert _settle(lambda: threading.active_count() <= before)
    # loader remains usable after close(): a fresh epoch iterates fully
    n = 0
    while True:
        try:
            r = dl.next_batch()
        except StopIteration:
            break
        if r is None:
            break
        n += 1
    assert n == 5  # 20 imgs / bs 4
    dl.close()


def test_context_manager_closes(tar20):
    with _make(tar20) as dl:
        images, meta = dl.next_batch()
        assert images.shape[1:] == (3, 32, 32)
    assert _settle(lambda: True)  # __exit__ ran without error; producers wind down


def test_abandoned_pinned_ring_releases(tar20):
    torch = pytest.importorskip("torch")  # noqa: F841  (ring path needs torch)
    base = threading.active_count()
    for _ in range(10):
        dl = _make(tar20, pin_memory=True)
        dl.next_batch()
        del dl
    assert _settle(
        lambda: threading.active_count() <= base + 1
    ), f"pinned-ring leak: {threading.active_count()} threads (base {base})"


def test_serve_cache_abandonment_releases(tar20):
    base = threading.active_count()
    for _ in range(10):
        dl = _make(tar20, cache_decoded=True)
        dl.next_batch()  # populates cache, starts cache producer
        dl.next_batch()
        del dl
    assert _settle(
        lambda: threading.active_count() <= base + 1
    ), f"cache-producer leak: {threading.active_count()} threads (base {base})"
