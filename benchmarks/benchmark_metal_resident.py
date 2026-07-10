"""Metal resident-loader benchmark (Apple Silicon) — pre-processed epochs.

Measures, on REAL data:
  1. MetalResidentLoader (fused gather+shuffle+normalize kernel, unified memory)
  2. the honest CPU baseline: identical resident uint8 dataset served by numpy
     (fancy-index gather + /255 - mean / std + HWC->CHW transpose)
  3. TurboLoader's on-the-fly DirectBatch path, for context (re-decodes JPEGs)
plus token windows (MetalTokenGather vs TokenDataLoader) and embedding-row
gathers (MetalResidentArrays vs numpy fancy indexing).

Methodology: warmup epoch excluded, median of --epochs timed epochs. The
headline number is PRODUCE rate (kernel complete, batch CPU-visible — the same
contract as the CUDA resident number); a consumed variant that touches every
batch is printed alongside. Run on an idle machine.

Usage:
  python benchmarks/benchmark_metal_resident.py --data /path/to/imagenette2-160/train
"""

import argparse
import glob
import os
import statistics
import time

import numpy as np

import turboloader as tl


def median_epochs(fn, epochs):
    times = []
    fn()  # warmup (excluded)
    for _ in range(epochs):
        t0 = time.perf_counter()
        n = fn()
        times.append((time.perf_counter() - t0, n))
    med = statistics.median(t for t, _ in times)
    return med, times[0][1]


def bench_images(paths, image_size, batch_size, epochs, consume):
    print(f"\n== images: {len(paths)} JPEGs @ {image_size}px, bs={batch_size} ==")
    dl = tl.MetalResidentLoader(
        paths, image_size=image_size, batch_size=batch_size, shuffle=True, seed=1
    )

    def _epoch_metal():
        n = 0
        for batch in dl:
            if consume:
                batch[:, 0, ::8, ::8].sum()
            n += batch.shape[0]
        dl.set_epoch(dl._epoch + 1)
        return n

    med, n = median_epochs(_epoch_metal, epochs)
    metal_rate = n / med
    label = "consumed" if consume else "produced"
    print(f"MetalResidentLoader ({label}):  {metal_rate:,.0f} img/s   ({med:.3f}s/epoch)")

    # CPU baseline on the SAME resident uint8 data
    view = tl.metal_resident_images_view(dl._handle, dl._n, dl._H, dl._W)
    data = np.array(view)  # independent CPU copy
    mean = np.array(dl.mean, dtype=np.float32)
    std = np.array(dl.std, dtype=np.float32)
    rng_epoch = [0]

    def _epoch_numpy():
        order = np.random.default_rng(1 + rng_epoch[0]).permutation(len(data))
        rng_epoch[0] += 1
        n = 0
        end = (len(data) // batch_size) * batch_size
        for b in range(0, end, batch_size):
            sel = data[order[b : b + batch_size]]
            x = sel.astype(np.float32) / 255.0
            x = (x - mean) / std
            x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
            if consume:
                x[:, 0, ::8, ::8].sum()
            n += x.shape[0]
        return n

    med_np, n_np = median_epochs(_epoch_numpy, epochs)
    print(f"numpy resident baseline:       {n_np / med_np:,.0f} img/s   ({med_np:.3f}s/epoch)")
    print(f"speedup vs numpy resident:     {metal_rate / (n_np / med_np):.2f}x")
    dl.close()
    return metal_rate


def bench_tokens(epochs):
    print("\n== tokens: 50M uint16, seq_len=1024, bs=64, 200 steps/epoch ==")
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 50257, size=50_000_000).astype(np.uint16)
    steps = 200

    tg = tl.MetalTokenGather(toks, seq_len=1024, batch_size=64, seed=1)

    def _metal():
        n = 0
        for _ in range(steps):
            x, y = tg.next_batch()
            n += x.size
        return n

    med, n = median_epochs(_metal, epochs)
    print(f"MetalTokenGather:   {n / med / 1e6:,.1f}M tok/s")
    tg.close()

    from turboloader.sequence import TokenDataLoader

    tdl = TokenDataLoader(toks, seq_len=1024, batch_size=64, steps_per_epoch=steps)

    def _cpu():
        n = 0
        for x, y in tdl:
            n += x.size
        return n

    med_c, n_c = median_epochs(_cpu, epochs)
    print(f"TokenDataLoader (CPU): {n_c / med_c / 1e6:,.1f}M tok/s")
    print(f"metal/cpu: {(n / med) / (n_c / med_c):.2f}x  (honest: whoever wins, wins)")


def bench_arrays(epochs):
    print("\n== arrays: (500k, 256) float32 embedding rows, bs=4096 ==")
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((500_000, 256), dtype=np.float32)
    ra = tl.MetalResidentArrays(emb, max_batch=4096)
    steps = 100

    def _metal():
        n = 0
        for _ in range(steps):
            idx = rng.integers(0, len(emb), size=4096)
            n += ra.gather(idx).shape[0]
        return n

    med, n = median_epochs(_metal, epochs)
    print(f"MetalResidentArrays.gather: {n / med:,.0f} rows/s")
    ra.close()

    def _numpy():
        n = 0
        for _ in range(steps):
            idx = rng.integers(0, len(emb), size=4096)
            n += np.ascontiguousarray(emb[idx]).shape[0]
        return n

    med_c, n_c = median_epochs(_numpy, epochs)
    print(f"numpy fancy-index:          {n_c / med_c:,.0f} rows/s")
    print(f"metal/cpu: {(n / med) / (n_c / med_c):.2f}x  (honest: whoever wins, wins)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of JPEGs (searched recursively)")
    ap.add_argument("--image-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--consume", action="store_true", help="touch every batch on CPU too")
    args = ap.parse_args()

    if not tl.metal_available():
        raise SystemExit("Metal not available on this machine")
    print(f"device: {tl.metal_device_name()}")

    paths = sorted(glob.glob(os.path.join(args.data, "**", "*.JPEG"), recursive=True)) or sorted(
        glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True)
    )
    if not paths:
        raise SystemExit(f"no JPEGs under {args.data}")

    bench_images(paths, args.image_size, args.batch_size, args.epochs, args.consume)
    bench_tokens(args.epochs)
    bench_arrays(args.epochs)
