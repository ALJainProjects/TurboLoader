"""Regression tests: the fused C++ fast paths must never silently drop transforms.

Historically (`<= 2.31.0`) two paired bugs made the advertised idiom silently wrong:

1. ``DataLoader('x.tar', output_format='pytorch', transform=Resize(...) | ImageNetNormalize())``
   took the DirectBatch fast path but ``_normalize_params`` could not extract mean/std from a
   ``ComposedTransforms`` (or a bare ``Normalize``), so images were served UN-NORMALIZED with
   no error. Augmentations in the pipeline were silently ignored the same way.
2. ``FastDataLoader``'s ``_can_use_cpp_float32_path`` approved pipelines that its
   ``_extract_normalize_params`` returned ``([], [])`` for — same silent skip on the
   C++ float32 path (and, because ``get_transforms`` was not bound, the gate vacuously
   approved ANY composed pipeline, augmentations included).

These tests pin the fix: mean/std are extracted from ``.mean``/``.std`` (including inside
composed pipelines), and pipelines containing transforms the C++ pass cannot express fall
back to a path that actually applies them.
"""

import os
import tarfile

import numpy as np
import pytest

import turboloader

Image = pytest.importorskip("PIL.Image")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Real-data e2e: point TURBOLOADER_TEST_TAR at a TAR of real JPEGs (e.g. Imagenette) to run
# every test here against it in addition to the synthetic fixture.
REAL_TAR = os.environ.get("TURBOLOADER_TEST_TAR")


@pytest.fixture(scope="module")
def jpeg_tar(tmp_path_factory):
    """A small TAR of real JPEG-encoded images (gradients, not flat colors, so that
    resize/normalize/flip errors actually change pixel values)."""
    root = tmp_path_factory.mktemp("fastpath")
    tar_path = str(root / "data.tar")
    rng = np.random.default_rng(0)
    with tarfile.open(tar_path, "w") as tf:
        for i in range(16):
            # Smooth gradient + noise: JPEG-friendly but spatially asymmetric.
            y, x = np.mgrid[0:96, 0:96]
            img = np.stack(
                [
                    (x * 2 + i * 5) % 256,
                    (y * 2 + i * 11) % 256,
                    ((x + y) + i * 23) % 256,
                ],
                axis=-1,
            ).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + rng.integers(-8, 8, img.shape), 0, 255)
            p = str(root / f"{i:03d}.jpg")
            Image.fromarray(img.astype(np.uint8)).save(p, quality=95)
            tf.add(p, arcname=f"{i:03d}.jpg")
    return tar_path


def _first_batch(tar_path, transform=None, batch_size=8, size=64, **kw):
    loader = turboloader.DataLoader(
        tar_path,
        batch_size=batch_size,
        output_format="pytorch",
        image_size=size,
        transform=transform,
        shuffle=False,
        num_workers=2,
        **kw,
    )
    for images, _meta in loader:
        return np.asarray(images), loader
    raise AssertionError("loader yielded no batches")


def test_fastpath_analysis_flags():
    t = turboloader
    supported = [
        None,
        t.Resize(64, 64),
        t.ImageNetNormalize(),
        t.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        t.Resize(64, 64) | t.ImageNetNormalize(),
    ]
    unsupported = [
        t.RandomHorizontalFlip(1.0),
        t.Resize(64, 64) | t.RandomHorizontalFlip(1.0) | t.ImageNetNormalize(),
        t.Resize(64, 64) | t.ColorJitter(0.4, 0.4, 0.4),
    ]
    from turboloader import _fastpath_analysis

    for tr in supported:
        ok, _, _ = _fastpath_analysis(tr)
        assert ok, f"{type(tr).__name__} should be fast-path expressible"
    for tr in unsupported:
        ok, _, _ = _fastpath_analysis(tr)
        assert not ok, f"{type(tr).__name__} must NOT be fast-path expressible"


def test_normalize_property_extraction():
    n = turboloader.Normalize([0.5, 0.4, 0.3], [0.2, 0.2, 0.2])
    assert np.allclose(list(n.mean), [0.5, 0.4, 0.3], atol=1e-6)
    assert np.allclose(list(n.std), [0.2, 0.2, 0.2], atol=1e-6)
    from turboloader import _fastpath_analysis

    ok, m, s = _fastpath_analysis(turboloader.Resize(32, 32) | n)
    assert ok and np.allclose(m, [0.5, 0.4, 0.3], atol=1e-6) and np.allclose(s, 0.2, atol=1e-6)


def _check_composed_normalize_applied(tar_path):
    plain, _ = _first_batch(tar_path, transform=None)
    normed, loader = _first_batch(
        tar_path, transform=turboloader.Resize(64, 64) | turboloader.ImageNetNormalize()
    )
    # Composed Resize|ImageNetNormalize must still take the fused fast path...
    assert type(loader._impl).__name__ == "_DirectFast"
    # ...and must actually normalize: normed == (plain - mean) / std elementwise.
    expected = (plain - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    err = float(np.abs(normed - expected).max())
    assert err < 1e-4, (
        f"composed ImageNetNormalize was not applied on the fast path (max err {err:.4f}); "
        "images are being served un-normalized"
    )


def test_composed_normalize_applied_on_fast_path(jpeg_tar):
    _check_composed_normalize_applied(jpeg_tar)


def test_bare_normalize_applied_on_fast_path(jpeg_tar):
    plain, _ = _first_batch(jpeg_tar, transform=None)
    mean, std = [0.5, 0.45, 0.4], [0.25, 0.25, 0.3]
    normed, loader = _first_batch(jpeg_tar, transform=turboloader.Normalize(mean, std))
    assert type(loader._impl).__name__ == "_DirectFast"
    m = np.array(mean, dtype=np.float32)[None, :, None, None]
    s = np.array(std, dtype=np.float32)[None, :, None, None]
    err = float(np.abs(normed - (plain - m) / s).max())
    assert err < 1e-4, f"bare Normalize params were not extracted (max err {err:.4f})"


def test_unsupported_transform_routes_off_fast_path(jpeg_tar):
    """DataLoader must NOT pick the fused path for a pipeline with an augmentation."""
    flip = turboloader.Resize(64, 64) | turboloader.RandomHorizontalFlip(1.0)
    _, loader = _first_batch(jpeg_tar, transform=flip)
    assert (
        type(loader._impl).__name__ != "_DirectFast"
    ), "pipeline with RandomHorizontalFlip must not take the fused fast path"


def _fdl_first_batch(tar_path, transform):
    fl = turboloader.FastDataLoader(
        tar_path,
        batch_size=8,
        num_workers=1,
        output_format="pytorch",
        target_height=64,
        target_width=64,
        shuffle=False,
        transform=transform,
    )
    images, meta = fl.next_batch()
    return np.asarray(images, dtype=np.float32), meta


def test_unsupported_transform_actually_applied(jpeg_tar):
    """The fallback must APPLY the augmentation, not drop it. Same loader type +
    num_workers=1 on both sides so the sample order is identical."""
    plain, meta_p = _fdl_first_batch(jpeg_tar, transform=None)
    flip = turboloader.Resize(64, 64) | turboloader.RandomHorizontalFlip(1.0)
    flipped, meta_f = _fdl_first_batch(jpeg_tar, transform=flip)
    if meta_p.get("indices") and meta_f.get("indices"):
        assert list(meta_p["indices"]) == list(meta_f["indices"]), "sample order differs"
    if plain.max() > 1.5:
        plain = plain / 255.0
    if flipped.max() > 1.5:  # fallback may return [0,255]-scaled output
        flipped = flipped / 255.0
    expected = plain[..., ::-1]  # p=1.0 -> every image flipped along W
    err = float(np.abs(flipped - expected).max())
    assert err < 0.02, (
        f"RandomHorizontalFlip(p=1.0) output does not match flipped reference "
        f"(max err {err:.4f}) — transform was dropped or misapplied"
    )


def test_fastdataloader_float32_path_normalizes(jpeg_tar):
    """FastDataLoader's C++ float32 path must apply extracted mean/std (pair-bug #2)."""
    fl_plain = turboloader.FastDataLoader(
        jpeg_tar,
        batch_size=8,
        num_workers=1,
        output_format="pytorch",
        target_height=64,
        target_width=64,
        shuffle=False,
    )
    plain, _ = fl_plain.next_batch()
    fl_norm = turboloader.FastDataLoader(
        jpeg_tar,
        batch_size=8,
        num_workers=1,
        output_format="pytorch",
        target_height=64,
        target_width=64,
        shuffle=False,
        transform=turboloader.Resize(64, 64) | turboloader.ImageNetNormalize(),
    )
    assert fl_norm._can_use_cpp_float32_path()
    m, s = fl_norm._extract_normalize_params()
    assert np.allclose(m, IMAGENET_MEAN, atol=1e-6), "extractor must match the gate"
    normed, _ = fl_norm.next_batch()
    plain = np.asarray(plain, dtype=np.float32)
    if plain.max() > 1.5:
        plain = plain / 255.0
    expected = (plain - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    err = float(np.abs(np.asarray(normed) - expected).max())
    assert err < 1e-3, f"FastDataLoader float32 path skipped normalization (max err {err:.4f})"


@pytest.mark.skipif(
    not REAL_TAR or not os.path.exists(REAL_TAR or ""), reason="real-data TAR not provided"
)
def test_composed_normalize_applied_real_data():
    """Same invariant on a real dataset (e.g. Imagenette) — set TURBOLOADER_TEST_TAR."""
    _check_composed_normalize_applied(REAL_TAR)
