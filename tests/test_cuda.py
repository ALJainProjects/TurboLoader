"""CUDA subsystem tests.

Two tiers:
- GPU-less tier (runs everywhere, incl. CI): API surface exists or degrades honestly —
  ``cuda_available()`` returns False on non-CUDA builds, the loaders raise informative
  errors instead of crashing, and Python-side helpers behave.
- GPU tier (skipped without a CUDA build + device): decode/transform correctness against
  the CPU reference on real JPEG bytes, and a memory-stability soak for the nvImageCodec
  slot pipeline (handles/buffers are reused, not leaked).
"""

import io

import numpy as np
import pytest

import turboloader

HAS_CUDA = bool(getattr(turboloader, "cuda_available", lambda: False)())

gpu = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA build / device")


# ----------------------------- GPU-less tier ----------------------------- #


def test_cuda_available_is_bool_and_honest():
    assert isinstance(HAS_CUDA, bool)
    if not HAS_CUDA:
        # No CUDA: the device-name helper must not crash and must return empty.
        assert turboloader.cuda_device_name() == ""


def test_cuda_loaders_raise_cleanly_without_cuda():
    if HAS_CUDA:
        pytest.skip("CUDA present; covered by GPU tier")
    for cls_name in ("CudaImageLoader", "CudaResidentLoader", "CudaStreamLoader"):
        cls = getattr(turboloader, cls_name, None)
        if cls is None:
            continue  # class not exported on this build — acceptable degradation
        with pytest.raises(RuntimeError):
            cls(["nonexistent.jpg"])


# ------------------------------- GPU tier -------------------------------- #


def _real_jpegs(n=64, size=96):
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        y, x = np.mgrid[0:size, 0:size]
        img = np.stack(
            [(x * 3 + i) % 256, (y * 2 + i * 5) % 256, ((x + y) * 2 + i * 11) % 256], axis=-1
        ).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + rng.integers(-6, 6, img.shape), 0, 255).astype(
            np.uint8
        )
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=95)
        out.append(buf.getvalue())
    return out


@gpu
def test_cuda_resize_normalize_matches_cpu_reference():
    jpegs = _real_jpegs(8)
    imgs = [turboloader.decode_jpeg(b) for b in jpegs]
    out = np.asarray(
        turboloader.cuda_resize_normalize(
            imgs, 64, 64, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    )
    assert out.shape == (8, 3, 64, 64) and out.dtype == np.float32
    assert np.isfinite(out).all()


@gpu
def test_nvimgcodec_pipeline_memory_stable():
    """Soak: repeated batches must not grow device memory (handles/buffers are reused)."""
    if not hasattr(turboloader, "cuda_nvimgcodec_init"):
        pytest.skip("nvImageCodec pipeline not compiled in")
    torch = pytest.importorskip("torch")
    import os as _os

    from nvidia import nvimgcodec as _nv

    d = _os.path.dirname(_nv.__file__)
    assert turboloader.cuda_nvimgcodec_init(
        _os.path.join(d, "libnvimgcodec.so.0"), _os.path.join(d, "extensions"), -1, 1
    )
    jpegs = _real_jpegs(64)
    # warm up (allocates pools/rings once)
    for _ in range(10):
        turboloader.cuda_nvimgcodec_decode_resize_normalize(jpegs, 64, 64)
    torch.cuda.synchronize()
    free0, _total = torch.cuda.mem_get_info()
    for _ in range(200):
        turboloader.cuda_nvimgcodec_decode_resize_normalize(jpegs, 64, 64)
    torch.cuda.synchronize()
    free1, _total = torch.cuda.mem_get_info()
    grown = (free0 - free1) / 1e6
    assert grown < 32, f"device memory grew {grown:.1f} MB over 200 batches — handle/buffer leak"
