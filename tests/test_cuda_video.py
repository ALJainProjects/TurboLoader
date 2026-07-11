"""Correctness tests for the CUDA video path (fused yuv420 kernel + CudaVideoLoader).

Clips are ENCODED with PyAV (mpeg4 at a high bitrate — the LGPL PyAV wheels ship no
x264 encoder), then decoded by the loader and compared against a numpy reference
computed from PyAV's own decoded I420 planes with the kernel's exact sampling math.
Decode is shared (both read PyAV's planes), so the comparison isolates OUR kernel
and tolerances are tight. Skipped without a CUDA build + torch.cuda + PyAV.
"""

import numpy as np
import pytest

import turboloader as tl

av = pytest.importorskip("av")

try:
    import torch

    _cuda_ok = torch.cuda.is_available()
except Exception:
    _cuda_ok = False

pytestmark = pytest.mark.skipif(
    not _cuda_ok
    or not getattr(tl, "cuda_available", lambda: False)()
    or not hasattr(tl, "cuda_video_yuv420_batch"),
    reason="CUDA video path not available",
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_numpy(cuda_batch):  # _CudaArray -> host float32 (B,3,H,W)
    return torch.as_tensor(cuda_batch, device="cuda").cpu().numpy()


def _denorm(batch):
    return batch * STD[None, :, None, None] + MEAN[None, :, None, None]


def _encode(path, frames_rgb, rate=30, bit_rate=12_000_000):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width = frames_rgb[0].shape[1]
        stream.height = frames_rgb[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = bit_rate
        for arr in frames_rgb:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def _decode_i420(path):
    """Ground-truth planes from PyAV: list of (y, cb, cr) uint8 arrays."""
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for frame in c.decode(s):
            i420 = frame.to_ndarray()  # (H*3/2, W)
            H = i420.shape[0] * 2 // 3
            W = i420.shape[1]
            y = i420[:H]
            cb = i420[H : H + H // 4].reshape(H // 2, W // 2)
            cr = i420[H + H // 4 :].reshape(H // 2, W // 2)
            out.append((y, cb, cr))
    return out


def _gather_bilinear(plane, yf, xf):
    y0 = np.floor(yf).astype(np.int64)
    x0 = np.floor(xf).astype(np.int64)
    y1 = np.minimum(y0 + 1, plane.shape[0] - 1)
    x1 = np.minimum(x0 + 1, plane.shape[1] - 1)
    dy = (yf - y0).astype(np.float32)[:, None]
    dx = (xf - x0).astype(np.float32)[None, :]
    p = plane.astype(np.float32)
    top = p[y0][:, x0] * (1 - dx) + p[y0][:, x1] * dx
    bot = p[y1][:, x0] * (1 - dx) + p[y1][:, x1] * dx
    return top * (1 - dy) + bot * dy


def _ref_convert(y, cb, cr, dh, dw, bt709):
    H, W = y.shape
    sx = np.maximum(0.0, (np.arange(dw, dtype=np.float32) + 0.5) * W / dw - 0.5)
    sy = np.maximum(0.0, (np.arange(dh, dtype=np.float32) + 0.5) * H / dh - 0.5)
    cxf = np.clip(sx * 0.5, 0, cb.shape[1] - 1)
    cyf = np.clip(sy * 0.5 - 0.25, 0, cb.shape[0] - 1)
    Y = _gather_bilinear(y, sy, sx)
    Cb = _gather_bilinear(cb, cyf, cxf) - 128.0
    Cr = _gather_bilinear(cr, cyf, cxf) - 128.0
    yv = (Y - 16.0) * (255.0 / 219.0)
    if bt709:
        r, g, b = (
            yv + 1.792741 * Cr,
            yv - 0.213249 * Cb - 0.532909 * Cr,
            yv + 2.112402 * Cb,
        )
    else:
        r, g, b = (
            yv + 1.596027 * Cr,
            yv - 0.391762 * Cb - 0.812968 * Cr,
            yv + 2.017232 * Cb,
        )
    return np.clip(np.stack([r, g, b], axis=0) / 255.0, 0.0, 1.0)  # (3, dh, dw)


@pytest.fixture(scope="module")
def content_clip(tmp_path_factory):
    """Smooth varied real content (gaussian blobs drifting) at 316x242? — no:
    yuv420p requires even dims; 320x240 with smooth gradients."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:240, 0:320].astype(np.float32)
    frames = []
    for i in range(48):
        r = 127 + 90 * np.sin(xx / 40 + i / 6) * np.cos(yy / 60)
        g = 127 + 90 * np.cos(xx / 55 - i / 8)
        b = 127 + 90 * np.sin((xx + yy) / 70 + i / 5)
        frames.append(np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8))
    d = tmp_path_factory.mktemp("cvid")
    path = d / "content.mp4"
    _encode(path, frames)
    return str(path)


@pytest.mark.parametrize("dst", [(240, 320), (97, 111)], ids=["noresize", "resize"])
def test_cpu_backend_matches_numpy_reference(content_clip, dst):
    dh, dw = dst
    planes = _decode_i420(content_clip)
    dl = tl.CudaVideoLoader(content_clip, image_size=(dh, dw), batch_size=16, decode="cpu")
    got = np.concatenate([_denorm(_to_numpy(b)) for b in dl], axis=0)
    assert got.shape == (len(planes), 3, dh, dw)
    for k in (0, 21, 47):
        y, cb, cr = planes[k]
        ref = _ref_convert(y, cb, cr, dh, dw, bt709=False)  # SD -> 601
        diff = np.abs(got[k] - ref)
        assert diff.mean() < 0.004, f"frame {k}: mean {diff.mean():.5f}"
        assert diff.max() < 0.03, f"frame {k}: max {diff.max():.5f}"


def test_frame_step_and_indices(content_clip):
    dl = tl.CudaVideoLoader(
        content_clip, image_size=64, batch_size=8, frame_step=5, return_indices=True
    )
    idxs = []
    for batch, idx in dl:
        idxs.extend(idx.tolist())
    assert idxs == list(range(0, 48, 5))


def test_double_buffer_lifetime(content_clip):
    dl = tl.CudaVideoLoader(content_clip, image_size=64, batch_size=8)
    it = iter(dl)
    b0 = _to_numpy(next(it))
    b0_again = _to_numpy(next(it))  # advancing must not corrupt a HELD copy...
    assert b0.shape == b0_again.shape
    # ...and the two batches contain different frames (content drifts each frame)
    assert not np.allclose(b0, b0_again)


def test_nvdec_backend_agrees_with_cpu_backend(content_clip):
    pytest.importorskip("PyNvVideoCodec")
    dl_cpu = tl.CudaVideoLoader(content_clip, image_size=(240, 320), batch_size=48, decode="cpu")
    dl_nv = tl.CudaVideoLoader(content_clip, image_size=(240, 320), batch_size=48, decode="nvdec")
    a = np.concatenate([_denorm(_to_numpy(b)) for b in dl_cpu], axis=0)
    b = np.concatenate([_denorm(_to_numpy(x)) for x in dl_nv], axis=0)
    assert a.shape == b.shape
    # FFmpeg and NVDEC mpeg4 IDCTs may differ slightly; the conversion is ours on
    # both sides, so agreement should still be close.
    diff = np.abs(a - b)
    assert diff.mean() < 0.01, f"backend disagreement: mean {diff.mean():.5f}"
