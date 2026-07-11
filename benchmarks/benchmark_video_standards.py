"""Industry-standard video decode comparison — every library runs the SAME full
training-input pipeline on the SAME real H.264 clip:

    decode -> RGB -> bilinear resize to N px -> ImageNet normalize -> float32,
    every frame materialized/consumed.

Libraries (the standards in video-ML data loading), each used idiomatically fast:
  - PyAV        (FFmpeg bindings; frame.reformat does convert+resize in swscale C)
  - OpenCV      (cv2.VideoCapture + cv2.resize)
  - decord      (VideoReader with native-resolution decode-time resize)
  - torchcodec  (Meta's PyTorch decoder; resize via torch.nn.functional)
  - TurboLoader (MetalVideoLoader on Apple Silicon / CudaVideoLoader on NVIDIA)

Libraries that aren't installed are reported as unavailable — nothing is faked.
Methodology: warmup pass excluded, median of --passes timed passes.

Usage: python benchmarks/benchmark_video_standards.py --video clip.mp4
"""

import argparse
import statistics
import time

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def median_passes(fn, passes):
    fn()
    times = []
    for _ in range(passes):
        t0 = time.perf_counter()
        n = fn()
        times.append((time.perf_counter() - t0, n))
    med = statistics.median(t for t, _ in times)
    return med, times[0][1]


def _norm_hwc(x):  # HWC uint8 -> CHW float32 normalized, consumed
    f = x.astype(np.float32) / 255.0
    f = (f - MEAN) / STD
    f = np.transpose(f, (2, 0, 1))
    f[0, ::16, ::16].sum()
    return f


def bench_pyav(video, size, passes):
    import av

    def one():
        n = 0
        with av.open(video) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for frame in c.decode(s):
                _norm_hwc(frame.reformat(size, size, format="rgb24").to_ndarray())
                n += 1
        return n

    return median_passes(one, passes)


def bench_opencv(video, size, passes):
    import cv2

    def one():
        cap = cv2.VideoCapture(video)
        n = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
            )
            _norm_hwc(rgb)
            n += 1
        cap.release()
        return n

    return median_passes(one, passes)


def bench_decord(video, size, passes, batch=32):
    import decord

    decord.bridge.set_bridge("native")

    def one():
        vr = decord.VideoReader(video, width=size, height=size)  # decode-time resize
        n = 0
        for s in range(0, len(vr), batch):
            arr = vr.get_batch(range(s, min(s + batch, len(vr)))).asnumpy()  # (B,H,W,3)
            f = arr.astype(np.float32) / 255.0
            f = (f - MEAN) / STD
            np.transpose(f, (0, 3, 1, 2))[:, 0, ::16, ::16].sum()
            n += arr.shape[0]
        return n

    return median_passes(one, passes)


def bench_torchcodec(video, size, passes, batch=32):
    import torch
    import torch.nn.functional as F
    from torchcodec.decoders import VideoDecoder

    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)

    def one():
        dec = VideoDecoder(video)
        n = 0
        total = dec.metadata.num_frames
        for s in range(0, total, batch):
            fb = dec.get_frames_in_range(s, min(s + batch, total))
            x = fb.data.float() / 255.0  # (B,3,H,W) uint8 -> float
            x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            float(x[:, 0, ::16, ::16].sum())
            n += x.shape[0]
        return n

    return median_passes(one, passes)


def bench_turboloader(video, size, passes, batch=32):
    import turboloader as tl

    if getattr(tl, "metal_video_available", lambda: False)():

        def one():
            n = 0
            with tl.MetalVideoLoader(video, image_size=size, batch_size=batch) as dl:
                for b in dl:
                    b[:, 0, ::16, ::16].sum()
                    n += b.shape[0]
            return n

        return "TurboLoader MetalVideoLoader", median_passes(one, passes)
    if hasattr(tl, "CudaVideoLoader") and getattr(tl, "cuda_available", lambda: False)():
        import torch

        def one():
            n = 0
            for b in tl.CudaVideoLoader(video, image_size=size, batch_size=batch, decode="cpu"):
                float(torch.as_tensor(b, device="cuda")[:, 0, ::16, ::16].sum())
                n += b.shape[0]
            return n

        return "TurboLoader CudaVideoLoader (cpu-decode)", median_passes(one, passes)
    raise ImportError("no TurboLoader GPU video path on this machine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    results = []
    for name, fn in [
        ("PyAV (reformat/swscale)", lambda: bench_pyav(args.video, args.image_size, args.passes)),
        ("OpenCV (VideoCapture)", lambda: bench_opencv(args.video, args.image_size, args.passes)),
        (
            "decord (decode-time resize)",
            lambda: bench_decord(args.video, args.image_size, args.passes),
        ),
        (
            "torchcodec (get_frames_in_range)",
            lambda: bench_torchcodec(args.video, args.image_size, args.passes),
        ),
        ("TurboLoader", lambda: bench_turboloader(args.video, args.image_size, args.passes)),
    ]:
        try:
            r = fn()
            if isinstance(r[0], str):  # turboloader returns its own label
                name, r = r
            med, n = r
            results.append((name, n / med))
            print(f"{name:44s} {n / med:9,.0f} frames/s")
        except ImportError as e:
            print(f"{name:44s} unavailable ({str(e)[:50]})")
        except Exception as e:
            print(f"{name:44s} FAILED: {str(e)[:70]}")

    if results:
        best_std = max(r for n, r in results if not n.startswith("TurboLoader"))
        ours = [r for n, r in results if n.startswith("TurboLoader")]
        if ours:
            print(f"\nTurboLoader vs best industry standard: {ours[0] / best_std:.2f}x")
