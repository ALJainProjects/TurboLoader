"""Correctness tests for the Metal video path (VideoToolbox decode + fused NV12
kernel). Real H.264 videos are generated with ffmpeg; decoded output is compared
against an ffmpeg-PNG + numpy reference. Skipped without macOS Metal or ffmpeg.

Tolerances are codec-aware: H.264 at crf 10 plus YUV 4:2:0 chroma subsampling and
matrix/siting differences between decoders legitimately move pixels a few /255.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

import turboloader as tl

pytestmark = pytest.mark.skipif(
    not getattr(tl, "metal_video_available", lambda: False)() or shutil.which("ffmpeg") is None,
    reason="Metal video path or ffmpeg not available",
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(batch):  # (B,3,H,W) normalized -> [0,1] RGB
    return batch * STD[None, :, None, None] + MEAN[None, :, None, None]


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def ramp_video(tmp_path_factory):
    """40 solid-gray frames, gray value = 20 + 5*frame_index — frame identity is
    encoded in the pixel values, so ordering and conversion are both checkable."""
    d = tmp_path_factory.mktemp("vid")
    from PIL import Image

    for i in range(40):
        v = 20 + 5 * i
        Image.new("RGB", (320, 240), (v, v, v)).save(d / f"f{i:03d}.png")
    out = str(d / "ramp.mp4")
    _ffmpeg(
        "-framerate",
        "30",
        "-i",
        str(d / "f%03d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "10",
        out,
    )
    return out


@pytest.fixture(scope="module")
def detail_video(tmp_path_factory):
    """Real varied content for per-pixel reference checks. testsrc2 is gaussian-
    blurred: at razor-sharp synthetic color edges, 4:2:0 chroma siting choices
    legitimately differ between decoders by whole chroma steps, so a per-pixel
    reference is only well-defined on smooth-chroma content. (Order/conversion
    on hard content is covered by the ramp tests; the mean-diff check below
    still runs on this detailed, moving footage.)"""
    d = tmp_path_factory.mktemp("vid2")
    out = str(d / "detail.mp4")
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30:duration=2",
        "-vf",
        "gblur=sigma=1.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "10",
        out,
    )
    return out, d


def test_frame_order_and_color_conversion(ramp_video):
    dl = tl.MetalVideoLoader(ramp_video, image_size=(240, 320), batch_size=16)
    frames = []
    for batch in dl:
        frames.append(_denorm(batch).mean(axis=(1, 2, 3)).copy())
    dl.close()
    got = np.concatenate(frames)
    assert got.shape[0] == 40
    expected = (20 + 5 * np.arange(40)) / 255.0
    # Tolerance MUST be smaller than one ramp step (5/255 = 0.0196), or an
    # off-by-one in frame order would pass. Solid frames at crf 10 survive
    # nearly losslessly, so 0.008 leaves real margin.
    np.testing.assert_allclose(got, expected, atol=0.008)


def test_frame_step(ramp_video):
    dl = tl.MetalVideoLoader(
        ramp_video, image_size=(240, 320), batch_size=8, frame_step=3, return_indices=True
    )
    idxs, means = [], []
    for batch, idx in dl:
        idxs.extend(idx.tolist())
        means.append(_denorm(batch).mean(axis=(1, 2, 3)).copy())
    dl.close()
    assert idxs == list(range(0, 40, 3))  # ceil(40/3) = 14 frames: 0,3,...,39
    expected = (20 + 5 * np.asarray(idxs)) / 255.0
    np.testing.assert_allclose(np.concatenate(means), expected, atol=0.008)


def _gather_bilinear(plane, yf, xf):
    """Bilinear-sample `plane` (2D) at float grids yf (rows) x xf (cols)."""
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
    """Numpy reference matching the kernel exactly for ANY output size: half-pixel
    luma resampling, MPEG chroma siting (horizontal co-sited cx = sx/2; vertical
    centered cy = sy/2 - 1/4), bilinear everywhere, selected video-range matrix,
    clip to [0,1]. Returns (dh, dw, 3)."""
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
        r = yv + 1.792741 * Cr
        g = yv - 0.213249 * Cb - 0.532909 * Cr
        b = yv + 2.112402 * Cb
    else:
        r = yv + 1.596027 * Cr
        g = yv - 0.391762 * Cb - 0.812968 * Cr
        b = yv + 2.017232 * Cb
    return np.clip(np.stack([r, g, b], axis=-1) / 255.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "src,dst",
    [
        ((320, 240), (240, 320)),  # SD no-resize: pure conversion, BT.601 branch
        ((320, 240), (97, 111)),  # SD with a non-integral resize ratio
        ((300, 180), (64, 64)),  # width NOT a multiple of 64: real stride padding
        ((1280, 720), (128, 128)),  # HD: exercises the BT.709 branch
    ],
    ids=["sd-noresize-601", "sd-resize", "odd-stride", "hd-709"],
)
def test_matches_numpy_yuv_reference(tmp_path, src, dst):
    """Decode must match a numpy reference computed from the video's RAW YUV
    planes with the kernel's exact sampling/siting/matrix math. H.264 decode is
    deterministic, so this isolates OUR kernel — tolerances are tight."""
    W, H = src
    dh, dw = dst
    path = str(tmp_path / "clip.mp4")
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={W}x{H}:rate=30:duration=1",
        "-vf",
        "gblur=sigma=1.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "10",
        path,
    )
    raw = tmp_path / "frames.yuv"
    _ffmpeg("-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", str(raw))
    data = np.fromfile(raw, dtype=np.uint8)
    fsz = W * H * 3 // 2
    n_frames = data.size // fsz
    assert n_frames == 30

    bt709 = H >= 720 or W >= 1280  # same rule the loader applies to untagged streams
    dl = tl.MetalVideoLoader(path, image_size=(dh, dw), batch_size=8)
    got = np.concatenate([_denorm(b).copy() for b in dl], axis=0)
    dl.close()
    assert got.shape == (n_frames, 3, dh, dw)

    for k in (0, 13, 29):  # spot-check across the stream
        f = data[k * fsz : (k + 1) * fsz]
        y = f[: W * H].reshape(H, W)
        cb = f[W * H : W * H + W * H // 4].reshape(H // 2, W // 2)
        cr = f[W * H + W * H // 4 :].reshape(H // 2, W // 2)
        ref = _ref_convert(y, cb, cr, dh, dw, bt709)
        ours = np.transpose(got[k], (1, 2, 0))
        diff = np.abs(ours - ref)
        assert diff.mean() < 0.004, f"frame {k}: mean abs diff {diff.mean():.5f}"
        assert diff.max() < 0.03, f"frame {k}: max diff {diff.max():.5f}"


def test_close_releases_native_memory(ramp_video):
    """Regression for the ARC/MRC leak: without -fobjc-arc on the .mm units,
    close_video()'s `delete ctx` released nothing and every open/close cycle
    leaked the staging + output MTLBuffers (~tens of MB each). Metal's own
    allocation counter must return to ~baseline after repeated cycles."""
    if not hasattr(tl, "metal_allocated_bytes"):
        pytest.skip("metal_allocated_bytes not built")

    def cycle():
        dl = tl.MetalVideoLoader(ramp_video, image_size=64, batch_size=256)
        next(iter(dl))
        dl.close()

    cycle()  # warm allocator/pipeline caches
    base = tl.metal_allocated_bytes()
    for _ in range(6):
        cycle()
    grown = tl.metal_allocated_bytes() - base
    # One cycle allocates ~65 MB of MTLBuffers (staging + 2 outputs); leaking six
    # would grow the counter by ~400 MB. Allow one cycle's worth of slack for
    # allocator reuse/heap pooling.
    assert grown < 70e6, f"native Metal memory leak: +{grown / 1e6:.0f} MB after 6 cycles"


def test_resize_output_shape_and_finite(detail_video):
    path, _ = detail_video
    dl = tl.MetalVideoLoader(path, image_size=96, batch_size=32)
    n = 0
    for batch in dl:
        assert batch.shape[1:] == (3, 96, 96) and batch.dtype == np.float32
        assert np.isfinite(batch).all()
        n += batch.shape[0]
    dl.close()
    assert n == 60


def test_double_buffer_lifetime(ramp_video):
    dl = tl.MetalVideoLoader(ramp_video, image_size=64, batch_size=8)
    it = iter(dl)
    b0 = next(it)
    b0_copy = b0.copy()
    next(it)  # b0 must survive exactly one more batch (double buffer)
    np.testing.assert_array_equal(b0, b0_copy)
    dl.close()


def test_reiteration_and_close_idempotent(ramp_video):
    dl = tl.MetalVideoLoader(ramp_video, image_size=64, batch_size=16)
    first = [_denorm(b).mean() for b in dl]
    second = [_denorm(b).mean() for b in dl]  # re-iteration reopens from frame 0
    np.testing.assert_allclose(first, second, atol=1e-6)
    dl.close()
    dl.close()
    with pytest.raises(RuntimeError):
        iter(dl).__next__()


def test_open_missing_file_raises():
    with pytest.raises(RuntimeError):
        tl.MetalVideoLoader("/nonexistent/video.mp4", image_size=64, batch_size=4)
