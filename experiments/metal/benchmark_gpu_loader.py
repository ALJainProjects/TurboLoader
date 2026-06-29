"""End-to-end benchmark: the Metal GPU loader vs an identical CPU-transform loader.

Both share the SAME parallel CPU JPEG decode (turboloader.decode_jpeg), so the comparison
isolates the transform stage: Metal GPU vs CPU. Real consumption (every batch is summed to
force materialization), warmup + median of timed epochs.

    python benchmark_gpu_loader.py [num_images]
"""
import glob
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import turboloader as t

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
paths = sorted(glob.glob("../imagenette2-160/train/*/*.JPEG"))[:N]
if not paths:
    paths = sorted(glob.glob("imagenette2-160/train/*/*.JPEG"))[:N]
assert paths, "no Imagenette JPEGs found"
print(f"{len(paths)} images | Metal: {t.metal_available()} ({t.metal_device_name()})\n")

BS, SZ, NW = 64, 160, 8
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def timeit(make_loader, epochs=3, warmup=1):
    rates = []
    for e in range(warmup + epochs):
        n = 0
        t0 = time.perf_counter()
        for batch in make_loader():
            n += batch.shape[0]
            float(batch.sum())  # force materialization (real consumption)
        dt = time.perf_counter() - t0
        if e >= warmup:
            rates.append(n / dt)
    return float(np.median(rates))


def gpu_loader():
    return t.GpuImageLoader(paths, batch_size=BS, image_size=SZ, num_workers=NW, mean=MEAN, std=STD)


def cpu_loader():
    """Same parallel decode, but resize+normalize on the CPU (numpy)."""
    end = len(paths) // BS * BS
    mean = np.float32(MEAN)[:, None, None]
    std = np.float32(STD)[:, None, None]

    def load(p):
        return t.decode_jpeg(open(p, "rb").read())

    with ThreadPoolExecutor(max_workers=NW) as ex:
        for s in range(0, end, BS):
            imgs = list(ex.map(load, paths[s : s + BS]))
            out = np.empty((len(imgs), 3, SZ, SZ), np.float32)
            for j, im in enumerate(imgs):
                r = t.Resize(SZ, SZ).apply(im)  # SIMD resize -> HWC uint8
                chw = r.transpose(2, 0, 1).astype(np.float32) / 255.0
                out[j] = (chw - mean) / std
            yield out


gpu = timeit(gpu_loader)
cpu = timeit(cpu_loader)
print(f"CPU loader  (decode CPU + transform CPU): {cpu:8.0f} img/s")
print(f"GPU loader  (decode CPU + transform GPU): {gpu:8.0f} img/s   ({gpu/cpu:.2f}x)")
print("\n(Both share the same parallel CPU decode; the delta is GPU vs CPU transform.)")
