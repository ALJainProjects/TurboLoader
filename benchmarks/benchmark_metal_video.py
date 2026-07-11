"""Metal video loader benchmark (Apple Silicon) — hardware decode to training batches.

Measures frames/s for the FULL pipeline (decode -> RGB -> resize -> normalize ->
(B,3,H,W) float32), on a real H.264 clip, against the standard CPU baseline
(PyAV decode + numpy resize/normalize — what most PyTorch video code does).

Methodology: one warmup pass excluded, median of --passes timed full passes,
every batch materialized. Generate a test clip with:
  ffmpeg -f lavfi -i "testsrc2=size=1920x1080:rate=30:duration=15" \
         -c:v libx264 -pix_fmt yuv420p -crf 23 clip.mp4

Usage: python benchmarks/benchmark_metal_video.py --video clip.mp4
"""

import argparse
import statistics
import time

import numpy as np

import turboloader as tl

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def median_passes(fn, passes):
    fn()  # warmup (excluded)
    times = []
    for _ in range(passes):
        t0 = time.perf_counter()
        n = fn()
        times.append((time.perf_counter() - t0, n))
    med = statistics.median(t for t, _ in times)
    return med, times[0][1]


def bench_metal(video, size, batch, passes):
    def one_pass():
        n = 0
        with tl.MetalVideoLoader(video, image_size=size, batch_size=batch) as dl:
            for b in dl:
                b[:, 0, ::16, ::16].sum()  # consume
                n += b.shape[0]
        return n

    med, n = median_passes(one_pass, passes)
    print(f"MetalVideoLoader (VideoToolbox + fused kernel): {n / med:,.0f} frames/s ({n} frames)")
    return n / med


def bench_pyav_strong(video, size, passes):
    """The STRONG CPU baseline: threaded FFmpeg decode + swscale convert+resize
    in C (frame.reformat) — the fastest idiomatic PyAV pipeline."""
    import av

    def one_pass():
        n = 0
        with av.open(video) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in c.decode(stream):
                rf = frame.reformat(size, size, format="rgb24")
                x = rf.to_ndarray().astype(np.float32) / 255.0
                x = (x - MEAN) / STD
                x = np.transpose(x, (2, 0, 1))
                x[0, ::16, ::16].sum()
                n += 1
        return n

    med, n = median_passes(one_pass, passes)
    print(f"PyAV reformat/swscale (strong CPU baseline):    {n / med:,.0f} frames/s ({n} frames)")
    return n / med


def bench_pyav_common(video, size, passes):
    """The COMMON pattern seen in real training code: to_image() + PIL resize."""
    import av
    from PIL import Image

    def one_pass():
        n = 0
        with av.open(video) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in c.decode(stream):
                img = frame.to_image().resize((size, size), Image.BILINEAR)
                x = np.asarray(img, dtype=np.float32) / 255.0
                x = (x - MEAN) / STD
                x = np.transpose(x, (2, 0, 1))
                x[0, ::16, ::16].sum()
                n += 1
        return n

    med, n = median_passes(one_pass, passes)
    print(f"PyAV + PIL (common real-world pattern):         {n / med:,.0f} frames/s ({n} frames)")
    return n / med


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    if not tl.metal_video_available():
        raise SystemExit("Metal video path not available")
    print(f"device: {tl.metal_device_name()}  |  clip: {args.video} -> {args.image_size}px")

    m = bench_metal(args.video, args.image_size, args.batch_size, args.passes)
    try:
        strong = bench_pyav_strong(args.video, args.image_size, args.passes)
        common = bench_pyav_common(args.video, args.image_size, args.passes)
        print(
            f"speedup vs strong baseline: {m / strong:.2f}x   (vs common pattern: {m / common:.1f}x)"
        )
    except ImportError:
        print("PyAV not installed — baselines skipped")
