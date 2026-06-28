"""Coverage/regression tests for the ``turboloader.DataLoader`` entry point.

Focus: the modality-routing and validation logic in ``turboloader/__init__.py``
(``DataLoader.__init__`` plus the ``_resize_target_from_transform`` helper).

Covered:
  * modality routing -> image (dict + fast paths), tokens, array (incl. aliases)
  * every ValueError branch: invalid modality, tokens missing seq_len,
    fast output_format with no fixed image size
  * unknown output_format falling through to the dict path (no whitelist)
  * missing-data RuntimeError (fast and dict)
  * cache_decoded + distributed conflicting-args RuntimeWarning
  * len()/TypeError semantics per mode

Inputs are tiny and deterministic: numpy arrays for tokens/array, and a small
TAR of solid-colour JPEGs built with tarfile + PIL for the image path.
"""

import io
import tarfile

import numpy as np
import pytest

import turboloader as tl

# Solid RGB colours used to synthesize the test TAR (order == sample index order).
_COLORS = [
    (10, 20, 30),
    (200, 100, 50),
    (0, 255, 0),
    (123, 45, 67),
    (255, 255, 255),
    (8, 8, 8),
]
_N_IMAGES = len(_COLORS)
_SRC_HW = 64  # source image height/width in the TAR


def _build_image_tar(path, colors=_COLORS, hw=_SRC_HW):
    from PIL import Image

    with tarfile.open(path, "w") as tf:
        for i, c in enumerate(colors):
            im = Image.new("RGB", (hw, hw), c)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=95)
            data = buf.getvalue()
            info = tarfile.TarInfo(name=f"img_{i:03d}.jpg")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture(scope="module")
def image_tar(tmp_path_factory):
    """A small TAR of solid-colour JPEGs, built once per module."""
    p = tmp_path_factory.mktemp("imgtar") / "tiny.tar"
    _build_image_tar(str(p))
    return str(p)


# --------------------------------------------------------------------------- #
# Image modality: dict (default) path
# --------------------------------------------------------------------------- #


def test_dict_mode_routing_and_batch(image_tar):
    """Default output_format='dict' takes the slow per-sample C++ path: it wires
    up ``_loader`` (not the fast ``_impl``) and yields a list of sample dicts."""
    dl = tl.DataLoader(image_tar, batch_size=2, output_format="dict")
    try:
        assert dl._fast is False
        assert dl._impl is None
        assert dl._loader is not None

        batch = dl.next_batch()
        assert isinstance(batch, list)
        assert len(batch) == 2

        sample = batch[0]
        assert set(sample.keys()) >= {
            "image",
            "index",
            "filename",
            "width",
            "height",
            "channels",
        }
        img = sample["image"]
        assert isinstance(img, np.ndarray)
        assert img.shape == (_SRC_HW, _SRC_HW, 3)  # no resize in dict mode
        assert img.dtype == np.uint8
        assert sample["index"] == 0
        assert sample["filename"] == "img_000.jpg"
    finally:
        dl.stop()


def test_dict_mode_len_raises_typeerror(image_tar):
    """len() is intentionally undefined for the dict-output image loader."""
    dl = tl.DataLoader(image_tar, batch_size=2, output_format="dict")
    try:
        with pytest.raises(TypeError, match="len\\(\\) is not defined"):
            len(dl)
    finally:
        dl.stop()


# --------------------------------------------------------------------------- #
# Image modality: fast (contiguous-array) path
# --------------------------------------------------------------------------- #


def test_fast_pytorch_routing_shapes(image_tar):
    """output_format='pytorch' with a scalar image_size routes to the fast impl
    and yields one normalized (N, C, H, W) float32 tensor + metadata."""
    bs = 4
    dl = tl.DataLoader(image_tar, batch_size=bs, image_size=32, output_format="pytorch")
    assert dl._fast is True
    assert dl._impl is not None
    # ceil(6 / 4) == 2 batches.
    assert len(dl) == -(-_N_IMAGES // bs) == 2

    images, meta = dl.next_batch()
    assert isinstance(images, np.ndarray)
    assert images.shape == (bs, 3, 32, 32)  # CHW for pytorch
    assert images.dtype == np.float32
    # normalize_01=True -> values in [0, 1].
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0
    assert set(meta.keys()) >= {"indices", "batch_size"}
    assert meta["batch_size"] == bs


def test_fast_numpy_hwc_tuple_image_size(image_tar):
    """output_format='numpy' keeps HWC layout, and a (H, W) tuple image_size is
    honoured (non-square allowed)."""
    dl = tl.DataLoader(image_tar, batch_size=3, image_size=(16, 24), output_format="numpy")
    images, _ = dl.next_batch()
    assert images.shape == (3, 16, 24, 3)  # HWC, H=16, W=24
    assert images.dtype == np.float32


@pytest.mark.parametrize("fmt", ["pytorch", "numpy_chw", "tensorflow", "numpy"])
def test_all_fast_formats_take_fast_path(image_tar, fmt):
    """Every format in DataLoader._FAST_FORMATS routes through the fast impl."""
    assert fmt in tl.DataLoader._FAST_FORMATS
    dl = tl.DataLoader(image_tar, batch_size=2, image_size=8, output_format=fmt)
    assert dl._fast is True
    assert dl._impl is not None
    images, _ = dl.next_batch()
    assert images.shape[0] == 2
    chw = fmt in ("pytorch", "numpy_chw")
    # CHW formats -> channel axis at 1; HWC -> channel axis last.
    assert images.shape[1 if chw else 3] == 3


def test_unknown_output_format_falls_back_to_dict(image_tar):
    """There is no output_format whitelist: an unrecognized value is simply not
    in _FAST_FORMATS, so the loader takes the dict path rather than raising."""
    dl = tl.DataLoader(image_tar, batch_size=2, output_format="banana")
    try:
        assert dl._fast is False
        assert dl._loader is not None
        assert dl._impl is None
    finally:
        dl.stop()


# --------------------------------------------------------------------------- #
# Tokens modality
# --------------------------------------------------------------------------- #


def test_tokens_routing_and_causal_shift():
    """modality='tokens' delegates to TokenDataLoader and yields (inputs, targets)
    int64 batches obeying the causal-LM shift-by-one contract."""
    from turboloader.sequence import TokenDataLoader

    toks = np.arange(1000, dtype=np.uint16)
    seq_len, bs = 8, 4
    dl = tl.DataLoader(toks, modality="tokens", seq_len=seq_len, batch_size=bs, shuffle=False)

    assert dl._fast is False
    assert isinstance(dl._delegate, TokenDataLoader)
    assert len(dl) == dl._delegate.steps_per_epoch

    x, y = dl.next_batch()
    assert x.shape == (bs, seq_len)
    assert y.shape == (bs, seq_len)
    assert x.dtype == np.int64 and y.dtype == np.int64
    # targets are inputs shifted left by one position.
    assert np.array_equal(y[:, :-1], x[:, 1:])
    # shuffle=False -> first window starts at token 0 (== arange(seq_len)).
    assert np.array_equal(x[0], np.arange(seq_len))


def test_tokens_missing_seq_len_raises():
    toks = np.arange(500, dtype=np.uint16)
    with pytest.raises(ValueError, match="requires seq_len"):
        tl.DataLoader(toks, modality="tokens", batch_size=2)


@pytest.mark.parametrize("alias", ["tokens", "token", "text"])
def test_tokens_modality_aliases(alias):
    from turboloader.sequence import TokenDataLoader

    toks = np.arange(500, dtype=np.uint16)
    dl = tl.DataLoader(toks, modality=alias, seq_len=8, batch_size=2)
    assert isinstance(dl._delegate, TokenDataLoader)


# --------------------------------------------------------------------------- #
# Array modality
# --------------------------------------------------------------------------- #


def test_array_routing_via_arrays_kwarg():
    """modality='array' with arrays=[X, Y] delegates to ArrayDataLoader and yields
    a tuple of aligned, contiguous batches."""
    from turboloader.sequence import ArrayDataLoader

    feats = np.arange(40, dtype=np.float32).reshape(10, 4)
    labels = np.arange(10, dtype=np.int64)
    dl = tl.DataLoader(None, modality="array", arrays=[feats, labels], batch_size=4, shuffle=False)

    assert isinstance(dl._delegate, ArrayDataLoader)
    assert len(dl) == 3  # ceil(10 / 4)

    xb, yb = dl.next_batch()
    assert xb.shape == (4, 4)
    assert yb.shape == (4,)
    assert np.array_equal(xb, feats[:4])
    assert np.array_equal(yb, labels[:4])


def test_array_routing_via_data_path_single():
    """With no arrays= kwarg, a single ndarray passed as data_path is wrapped into
    a one-array dataset, which yields a bare ndarray (not a 1-tuple)."""
    feats = np.arange(40, dtype=np.float32).reshape(10, 4)
    dl = tl.DataLoader(feats, modality="array", batch_size=4, shuffle=False)
    out = dl.next_batch()
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 4)
    assert np.array_equal(out, feats[:4])


@pytest.mark.parametrize("alias", ["array", "arrays", "tensor", "tabular"])
def test_array_modality_aliases(alias):
    from turboloader.sequence import ArrayDataLoader

    feats = np.arange(40, dtype=np.float32).reshape(10, 4)
    dl = tl.DataLoader(feats, modality=alias, batch_size=2)
    assert isinstance(dl._delegate, ArrayDataLoader)


def test_array_shuffle_is_seed_and_epoch_deterministic():
    """Routing preserves seed/shuffle: same (seed, epoch) -> identical order,
    different epoch -> different order."""
    feats = np.arange(100, dtype=np.float32).reshape(10, 10)

    def first_batch(epoch):
        dl = tl.DataLoader(feats, modality="array", batch_size=10, shuffle=True, seed=7)
        dl.set_epoch(epoch)
        return next(iter(dl)).copy()

    a = first_batch(3)
    b = first_batch(3)
    c = first_batch(4)
    assert np.array_equal(a, b)  # reproducible
    assert not np.array_equal(a, c)  # epoch changes the permutation


# --------------------------------------------------------------------------- #
# Validation / error paths
# --------------------------------------------------------------------------- #


def test_invalid_modality_raises():
    with pytest.raises(ValueError, match="modality must be one of"):
        tl.DataLoader("whatever.tar", modality="audio")


@pytest.mark.parametrize("fmt", ["pytorch", "numpy_chw", "tensorflow", "numpy"])
def test_fast_format_missing_image_size_raises(fmt):
    """A fast output_format needs a fixed image size; with neither image_size nor a
    size-bearing transform the loader must reject construction up front."""
    with pytest.raises(ValueError, match="needs a fixed image size"):
        tl.DataLoader("nonexistent.tar", output_format=fmt)


def test_missing_data_fast_path_raises():
    """Fast path opens the archive eagerly in __init__, so a missing TAR surfaces
    a RuntimeError at construction (not a silent empty loader)."""
    with pytest.raises(RuntimeError, match="Failed to open TAR"):
        tl.DataLoader("/no/such/file_xyz.tar", batch_size=2, image_size=16, output_format="pytorch")


def test_missing_data_dict_path_raises():
    with pytest.raises(RuntimeError, match="Failed to open TAR"):
        tl.DataLoader("/no/such/file_xyz.tar", batch_size=2, output_format="dict")


def test_cache_decoded_with_distributed_warns(image_tar):
    """cache_decoded + distributed sharding is a conflicting combination (epoch-0
    shard gets frozen): the loader warns rather than silently misbehaving."""
    with pytest.warns(RuntimeWarning, match="cache_decoded with distributed"):
        tl.DataLoader(
            image_tar,
            batch_size=2,
            image_size=16,
            output_format="pytorch",
            cache_decoded=True,
            enable_distributed=True,
            world_size=2,
            world_rank=0,
        )


# --------------------------------------------------------------------------- #
# _resize_target_from_transform helper (drives the fast-path image-size inference)
# --------------------------------------------------------------------------- #


def test_resize_target_helper_extraction():
    f = tl._resize_target_from_transform
    assert f(None) is None

    class DuckResize:
        target_height = 20
        target_width = 30

    assert f(DuckResize()) == (20, 30)


def test_resize_target_helper_returns_none_for_real_resize():
    """The real C++ Resize object exposes none of the (target_)height/width attrs
    the helper looks for, so size inference yields None for it. This pins the
    current behavior that motivates the xfail below."""
    assert tl._resize_target_from_transform(tl.Resize(20, 20)) is None


@pytest.mark.xfail(
    strict=False,
    reason="BUG: docstring/error message advertise that a Resize(H,W) in the "
    "transform supplies image_size for the fast path, but the C++ Resize exposes "
    "no height/width attrs, so _resize_target_from_transform always returns None "
    "and construction still raises 'needs a fixed image size'.",
)
def test_resize_transform_supplies_fast_image_size(image_tar):
    # Per the documented contract this should construct a working fast loader
    # using the Resize's (20, 20) target; today it raises ValueError instead.
    dl = tl.DataLoader(
        image_tar,
        batch_size=2,
        output_format="pytorch",
        transform=tl.Resize(20, 20),
    )
    images, _ = dl.next_batch()
    assert images.shape == (2, 3, 20, 20)
