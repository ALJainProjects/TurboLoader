"""Coverage / regression tests for previously-untested DataLoader runtime paths
in ``turboloader/__init__.py``.

Targets:
  * the fast ``_DirectFast`` path (numpy / numpy_chw / pytorch tensor batches)
  * cache_decoded population + deterministic re-serving
  * enable_distributed sharding
  * drop_last
  * transforms / normalize params (ImageNetNormalize, explicit mean/std)
  * DataLoader.next_batch() delegation (fast impl + array delegate)
  * set_epoch reproducible shuffling
  * __len__, is_finished(), stop(), smart_batching_enabled() toggles
  * the module-level helpers ``_serve_cache``, ``_DirectFast``,
    ``_normalize_params`` and ``_resize_target_from_transform``

Everything is self-contained: TAR archives of JPEGs are built in ``tmp_path`` with
fixed seeds so the tests are deterministic. Batch sizes / worker counts are kept
small.
"""

import io
import os
import tarfile
import warnings

import numpy as np
import pytest

import turboloader as tl

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _build_tar(path, n, size=32, seed=0):
    """Write ``n`` deterministic JPEG images of ``size`` x ``size`` into a TAR."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=95)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"img_{i:04d}.jpg")
            info.size = len(buf.getvalue())
            tar.addfile(info, buf)
    return str(path)


@pytest.fixture(scope="module")
def tar20(tmp_path_factory):
    """20 JPEGs (deliberately not a multiple of common batch sizes)."""
    d = tmp_path_factory.mktemp("tl_tar20")
    return _build_tar(os.path.join(str(d), "data20.tar"), n=20, size=32, seed=0)


@pytest.fixture(scope="module")
def tar12(tmp_path_factory):
    d = tmp_path_factory.mktemp("tl_tar12")
    return _build_tar(os.path.join(str(d), "data12.tar"), n=12, size=16, seed=1)


def _drain_indices(loader):
    out = []
    for _imgs, meta in loader:
        out.extend(meta["indices"])
    return out


# --------------------------------------------------------------------------- #
# Fast path: shapes / dtypes / metadata / __len__
# --------------------------------------------------------------------------- #
def test_fast_numpy_path_shapes_and_range(tar20):
    """numpy output is HWC float32 in [0, 1] with correct shapes + metadata."""
    dl = tl.DataLoader(tar20, batch_size=8, num_workers=2, image_size=32, output_format="numpy")
    assert type(dl._impl).__name__ == "_DirectFast"
    imgs, meta = dl.next_batch()
    assert imgs.shape == (8, 32, 32, 3)  # HWC
    assert imgs.dtype == np.float32
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    # normalize_01 actually ran (not raw uint8 cast left in 0..255)
    assert imgs.max() <= 1.0 + 1e-6 and imgs.max() > 0.1
    assert set(meta.keys()) >= {"indices", "batch_size"}
    assert meta["batch_size"] == 8
    assert len(meta["indices"]) == 8


def test_fast_pytorch_path_is_chw(tar20):
    dl = tl.DataLoader(tar20, batch_size=8, image_size=32, output_format="pytorch")
    imgs, _ = dl.next_batch()
    assert imgs.shape == (8, 3, 32, 32)  # CHW
    assert imgs.dtype == np.float32


def test_len_fast_path_is_ceil(tar20):
    """__len__ on the fast path == ceil(num_samples / batch_size)."""
    dl = tl.DataLoader(tar20, batch_size=8, image_size=32, output_format="numpy")
    assert dl._impl.num_samples() == 20
    assert len(dl) == 3  # ceil(20 / 8)
    dl2 = tl.DataLoader(tar20, batch_size=5, image_size=32, output_format="numpy")
    assert len(dl2) == 4  # ceil(20 / 5)


def test_iteration_covers_every_sample_once(tar20):
    dl = tl.DataLoader(tar20, batch_size=8, image_size=32, output_format="numpy")
    idx = _drain_indices(dl)
    assert sorted(idx) == list(range(20))


# --------------------------------------------------------------------------- #
# next_batch() delegation + StopIteration semantics
# --------------------------------------------------------------------------- #
def test_next_batch_advances_and_raises_stopiteration(tar20):
    """DataLoader.next_batch() delegates to the fast impl, advances each call,
    and raises StopIteration exactly once the dataset is exhausted."""
    dl = tl.DataLoader(tar20, batch_size=8, image_size=32, output_format="numpy")
    seen = 0
    batches = 0
    while True:
        try:
            imgs, _ = dl.next_batch()
        except StopIteration:
            break
        seen += imgs.shape[0]
        batches += 1
    assert batches == len(dl) == 3
    assert seen == 20


# --------------------------------------------------------------------------- #
# drop_last
# --------------------------------------------------------------------------- #
def test_drop_last_drops_incomplete_final_batch(tar20):
    keep = [
        im.shape[0]
        for im, _ in tl.DataLoader(
            tar20, batch_size=8, image_size=32, output_format="numpy", drop_last=False
        )
    ]
    drop = [
        im.shape[0]
        for im, _ in tl.DataLoader(
            tar20, batch_size=8, image_size=32, output_format="numpy", drop_last=True
        )
    ]
    assert keep == [8, 8, 4] and sum(keep) == 20
    assert drop == [8, 8] and sum(drop) == 16  # the partial trailing 4 is dropped


# --------------------------------------------------------------------------- #
# Distributed sharding
# --------------------------------------------------------------------------- #
def test_distributed_sharding_is_disjoint_and_complete(tar20):
    r0 = tl.DataLoader(
        tar20,
        batch_size=4,
        image_size=32,
        output_format="numpy",
        enable_distributed=True,
        world_rank=0,
        world_size=2,
    )
    r1 = tl.DataLoader(
        tar20,
        batch_size=4,
        image_size=32,
        output_format="numpy",
        enable_distributed=True,
        world_rank=1,
        world_size=2,
    )
    idx0 = set(_drain_indices(r0))
    idx1 = set(_drain_indices(r1))
    # each rank only sees its shard
    assert r0._impl.num_samples() == 10
    assert r1._impl.num_samples() == 10
    # shards are disjoint and together cover the whole dataset
    assert idx0.isdisjoint(idx1)
    assert idx0 | idx1 == set(range(20))


def test_cache_decoded_with_distributed_warns(tar20):
    """cache_decoded + multi-rank sharding emits a RuntimeWarning (shard freezes)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.DataLoader(
            tar20,
            batch_size=4,
            image_size=32,
            output_format="numpy",
            cache_decoded=True,
            enable_distributed=True,
            world_rank=0,
            world_size=2,
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("cache_decoded" in m and "distributed" in m for m in msgs)


# --------------------------------------------------------------------------- #
# cache_decoded
# --------------------------------------------------------------------------- #
def test_cache_decoded_populates_and_replays_identically(tar20):
    dl = tl.DataLoader(
        tar20, batch_size=8, image_size=32, output_format="numpy", cache_decoded=True
    )
    assert dl._impl.cache_populated is False
    epoch1 = [im.copy() for im, _ in dl]
    assert dl._impl.cache_populated is True
    epoch2 = [im.copy() for im, _ in dl]
    assert len(epoch1) == len(epoch2) == 3
    for a, b in zip(epoch1, epoch2):
        np.testing.assert_array_equal(a, b)


def test_cache_clear_resets_population_flag(tar20):
    dl = tl.DataLoader(
        tar20, batch_size=8, image_size=32, output_format="numpy", cache_decoded=True
    )
    for _ in dl:
        pass
    assert dl._impl.cache_populated is True
    dl._impl.clear_cache()
    assert dl._impl.cache_populated is False


# --------------------------------------------------------------------------- #
# set_epoch + reproducible shuffling (cached path)
# --------------------------------------------------------------------------- #
def test_set_epoch_shuffle_is_reproducible_and_epoch_dependent(tar20):
    dl = tl.DataLoader(
        tar20,
        batch_size=4,
        image_size=32,
        output_format="numpy",
        shuffle=True,
        seed=123,
        cache_decoded=True,
    )
    dl.set_epoch(0)
    order0 = _drain_indices(dl)
    dl.set_epoch(1)
    order1 = _drain_indices(dl)
    dl.set_epoch(0)
    order0_again = _drain_indices(dl)

    # full coverage every epoch
    assert sorted(order0) == sorted(order1) == list(range(20))
    # same epoch+seed => identical order; different epoch => different order
    assert order0 == order0_again
    assert order0 != order1
    # matches the documented seed+epoch RNG contract used by _serve_cache
    assert order0 == np.random.default_rng(123 + 0).permutation(20).tolist()


# --------------------------------------------------------------------------- #
# Transforms / normalize params
# --------------------------------------------------------------------------- #
def test_imagenet_normalize_matches_formula(tar20):
    """ImageNetNormalize output == (x/255 - mean) / std, channel-wise, vs the
    plain [0, 1] pytorch output decoded from the same archive/order."""
    plain = tl.DataLoader(tar20, batch_size=8, image_size=32, output_format="pytorch")
    norm = tl.DataLoader(
        tar20,
        batch_size=8,
        image_size=32,
        output_format="pytorch",
        transform=tl.ImageNetNormalize(),
    )
    p, _ = plain.next_batch()
    q, _ = norm.next_batch()
    # Diagnostic guard: one CI run (2026-07-09, py3.14) saw NaN/garbage in q with a
    # per-row pattern (most rows fine, one uninitialized, two holding [0,1] data).
    # Never reproduced across 223 full-suite reruns + 24M pool dispatches (TSan
    # clean on two platforms); if it ever recurs, fail with the row map instead of
    # a bare q.min() assert so the pattern is captured.
    if np.isnan(q).any() or np.abs(q).max() > 100:
        rows = [
            (i, int(np.isnan(q[i]).sum()), float(np.nanmax(np.abs(q[i]))))
            for i in range(q.shape[0])
        ]
        pytest.fail(f"corrupted normalized batch; per-row (idx, nan_count, absmax): {rows}")
    assert not np.allclose(p, q)  # normalization actually changed the values
    assert q.min() < 0.0  # ImageNet normalization produces negatives
    for c in range(3):
        expected = (p[:, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        np.testing.assert_allclose(q[:, c], expected, atol=1e-4)


def test_explicit_mean_std_normalizes_to_unit_range(tar20):
    """Explicit mean/std=0.5 on _DirectFast maps [0,1] -> [-1,1]."""
    df = tl._DirectFast(
        tar20,
        (32, 32),
        8,
        "pytorch",
        None,
        False,
        0,
        False,
        0,
        1,
        False,
        2,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    imgs, _ = df.next_batch()
    assert imgs.min() >= -1.0 - 1e-5 and imgs.max() <= 1.0 + 1e-5
    # the shift/scale genuinely happened (not still in [0,1])
    assert imgs.min() < -0.3


def test_normalize_params_helper():
    # explicit mean/std passed through verbatim
    assert tl._normalize_params(None, [1, 2, 3], [4, 5, 6]) == ([1, 2, 3], [4, 5, 6])
    # ImageNetNormalize resolves to the canonical constants
    m, s = tl._normalize_params(tl.ImageNetNormalize())
    # values now come from the C++ .mean/.std properties (float32), so compare approx
    assert [round(x, 4) for x in m] == [0.485, 0.456, 0.406]
    assert [round(x, 4) for x in s] == [0.229, 0.224, 0.225]
    # no transform => empty (C++ default)
    assert tl._normalize_params(None) == ([], [])
    # generic Normalize now exposes .mean/.std, so its params ARE extracted (the old
    # ([], []) behavior silently skipped normalization on the fast path — a bug).
    m, s = tl._normalize_params(tl.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]))
    assert [round(x, 4) for x in m] == [0.5, 0.5, 0.5]
    assert [round(x, 4) for x in s] == [0.25, 0.25, 0.25]
    # composed pipelines are walked too (this was the headline silent-drop bug)
    m, s = tl._normalize_params(tl.Resize(32, 32) | tl.ImageNetNormalize())
    assert [round(x, 4) for x in m] == [0.485, 0.456, 0.406]
    assert [round(x, 4) for x in s] == [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# is_finished / stop / smart_batching toggles (fast path is a no-op)
# --------------------------------------------------------------------------- #
def test_fast_path_is_finished_stop_and_smart_batching_are_inert(tar20):
    dl = tl.DataLoader(tar20, batch_size=8, image_size=32, output_format="numpy")
    # fast path keeps no C++ pipeline handle => these report inert defaults
    assert dl.is_finished() is False
    assert dl.smart_batching_enabled() is False
    dl.stop()  # must not raise even though there is no underlying _loader
    assert dl.is_finished() is False


# --------------------------------------------------------------------------- #
# Dict (non-fast) output path: smart_batching / is_finished / __len__ / stop
# --------------------------------------------------------------------------- #
def test_dict_path_len_raises_typeerror(tar12):
    dl = tl.DataLoader(tar12, batch_size=4, num_workers=2, output_format="dict")
    with pytest.raises(TypeError):
        len(dl)
    dl.stop()


def test_dict_path_smart_batching_toggle(tar12):
    off = tl.DataLoader(tar12, batch_size=4, num_workers=2, output_format="dict")
    assert off.smart_batching_enabled() is False
    off.stop()
    on = tl.DataLoader(
        tar12, batch_size=4, num_workers=2, output_format="dict", enable_smart_batching=True
    )
    assert on.smart_batching_enabled() is True
    on.stop()


def test_dict_path_iterates_and_finishes(tar12):
    dl = tl.DataLoader(tar12, batch_size=4, num_workers=2, output_format="dict")
    total = 0
    first = None
    for batch in dl:
        if first is None and batch:
            first = batch[0]
        total += len(batch)
    assert total == 12
    assert {"index", "filename", "image"} <= set(first.keys())
    assert dl.is_finished() is True
    dl.stop()


# --------------------------------------------------------------------------- #
# Missing-image-size error paths + _resize_target_from_transform
# --------------------------------------------------------------------------- #
def test_fast_format_without_image_size_raises(tar20):
    with pytest.raises(ValueError, match="needs a fixed image size"):
        tl.DataLoader(tar20, batch_size=8, output_format="pytorch")


def test_resize_transform_supplies_image_size(tar20):
    """FIXED: the bound Resize now exposes .height/.width, so a Resize-only
    transform satisfies the fast-format size requirement (as the error message
    always advertised), and a CONFLICTING explicit image_size is refused
    instead of silently ignored."""
    assert tl._resize_target_from_transform(None) is None
    assert tl._resize_target_from_transform(tl.Resize(64, 48)) == (48, 64)
    dl = tl.DataLoader(tar20, batch_size=8, output_format="numpy", transform=tl.Resize(16, 16))
    images, _ = dl.next_batch()
    assert images.shape[1:3] == (16, 16)  # numpy format is HWC
    dl.close()
    with pytest.raises(ValueError, match="Conflicting sizes"):
        tl.DataLoader(
            tar20,
            batch_size=8,
            image_size=32,
            output_format="numpy",
            transform=tl.Resize(16, 16),
        )
    # agreeing sizes are fine
    dl = tl.DataLoader(
        tar20, batch_size=8, image_size=16, output_format="numpy", transform=tl.Resize(16, 16)
    )
    dl.close()


# --------------------------------------------------------------------------- #
# _serve_cache helper (direct)
# --------------------------------------------------------------------------- #
def _toy_cache(n=10):
    X = np.arange(n * 2 * 2 * 3, dtype=np.float32).reshape(n, 2, 2, 3)
    I = np.arange(n)
    return X, I


def test_serve_cache_no_shuffle_batches_and_indices():
    X, I = _toy_cache(10)
    batches = list(tl._serve_cache(X, I, batch_size=3, shuffle=False, epoch=0, prefetch=2))
    assert [im.shape[0] for im, _ in batches] == [3, 3, 3, 1]
    assert [m["indices"] for _, m in batches] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert [m["batch_size"] for _, m in batches] == [3, 3, 3, 1]


def test_serve_cache_drop_last():
    X, I = _toy_cache(10)
    batches = list(tl._serve_cache(X, I, 3, False, 0, 2, drop_last=True))
    assert [im.shape[0] for im, _ in batches] == [3, 3, 3]  # trailing remainder dropped


def test_serve_cache_shuffle_matches_seeded_rng():
    X, I = _toy_cache(10)
    batches = list(tl._serve_cache(X, I, 4, True, epoch=5, prefetch=2, seed=7))
    flat = [i for _, m in batches for i in m["indices"]]
    assert flat == np.random.default_rng(7 + 5).permutation(10).tolist()


def test_serve_cache_yields_independent_copies():
    X, I = _toy_cache(10)
    batches = list(tl._serve_cache(X, I, 3, False, 0, 2))
    batches[0][0][:] = -999.0  # mutate the yielded batch
    assert X[0, 0, 0, 0] == 0.0  # source cache must be untouched


def test_serve_cache_none_cache_is_empty():
    assert list(tl._serve_cache(None, None, 3, False, 0, 2)) == []


# --------------------------------------------------------------------------- #
# _DirectFast helper (direct) + delegate (array modality)
# --------------------------------------------------------------------------- #
def test_directfast_num_samples_len_and_drain(tar12):
    df = tl._DirectFast(tar12, (16, 16), 4, "numpy", None, False, 0, False, 0, 1, False, 2)
    assert df.num_samples() == 12
    assert len(df) == 3  # ceil(12 / 4)
    count = 0
    while True:
        try:
            df.next_batch()
        except StopIteration:
            break
        count += 1
    assert count == 3


def test_array_modality_delegates_next_batch(tar12):
    feats = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    labels = np.arange(12)
    dl = tl.DataLoader(None, batch_size=4, modality="array", arrays=[feats, labels])
    assert dl._delegate is not None
    assert len(dl) == 3
    total = 0
    batches = 0
    while True:
        try:
            xb, yb = dl.next_batch()
        except StopIteration:
            break
        assert xb.shape[0] == yb.shape[0]
        total += xb.shape[0]
        batches += 1
    assert total == 12 and batches == 3


def test_tokens_modality_requires_seq_len():
    with pytest.raises(ValueError, match="seq_len"):
        tl.DataLoader("nonexistent.bin", modality="tokens")


def test_invalid_modality_raises():
    with pytest.raises(ValueError, match="modality must be one of"):
        tl.DataLoader("whatever", modality="banana")
