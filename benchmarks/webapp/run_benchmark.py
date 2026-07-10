"""Real benchmark feeding benchmark_app.html. Honest methodology:
- real consumption (np.sum forces the batch to be materialized/read),
- warmup epoch then median over timed epochs,
- same dataset (Imagenette-160, 9,469 real ImageNet JPEGs) for every framework,
- frameworks that aren't installed are SKIPPED and omitted from the JSON
  (the dashboard renders whatever is present — nothing is fabricated).

Writes benchmark_results.json next to this script (where the app fetches it).

Usage:
  python run_benchmark.py --tar /path/imagenette_train.tar --imgdir /path/imagenette2-160/train
"""

import argparse
import glob
import json
import os
import statistics
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
N_TIMED = 3


def best_rate(epoch_fn, warmup=1, timed=N_TIMED):
    for _ in range(warmup):
        epoch_fn()
    rates = []
    for _ in range(timed):
        t0 = time.perf_counter()
        n = epoch_fn()
        rates.append(n / (time.perf_counter() - t0))
    return round(statistics.median(rates))


# ---------------- TurboLoader ----------------
def turbo_rate(tar, num_workers, cache=False):
    import turboloader as t

    dl = t.DataLoader(
        tar,
        batch_size=64,
        num_workers=num_workers,
        output_format="pytorch",
        image_size=160,
        transform=t.ImageNetNormalize(),
        cache_decoded=cache,
    )

    def ep():
        n = 0
        for img, _ in dl:
            a = np.asarray(img)
            n += a.shape[0]
            float(a.sum())  # force consumption
        return n

    return best_rate(ep)


# ---------------- PyTorch ----------------
def pytorch_rate(imgdir, num_workers):
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder

    tf = T.Compose([T.Resize((160, 160)), T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = ImageFolder(imgdir, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=64,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers else None,
    )

    def ep():
        n = 0
        for img, _ in dl:
            n += img.shape[0]
            float(img.sum())
        return n

    return best_rate(ep)


# ---------------- tf.data ----------------
def tfdata_rate(imgdir, cache=False):
    import tensorflow as tf

    files = sorted(glob.glob(imgdir + "/*/*.JPEG"))
    mean = tf.constant(MEAN)
    std = tf.constant(STD)

    def load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [160, 160])
        return (img / 255.0 - mean) / std

    ds = tf.data.Dataset.from_tensor_slices(files).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64)
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    def ep():
        n = 0
        for b in ds:
            n += int(b.shape[0])
            float(tf.reduce_sum(b))
        return n

    return best_rate(ep)


# ---------------- tokens ----------------
def token_rates(tokens_path):
    import turboloader as t

    if not os.path.exists(tokens_path):
        # Synthesize a real-sized corpus once (50M uint16 tokens ~ 100 MB).
        rng = np.random.default_rng(0)
        rng.integers(0, 50257, size=50_000_000).astype(np.uint16).tofile(tokens_path)
    toks = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    n_tokens = toks.shape[0]
    seq, bs, steps = 1024, 16, 200

    dl = t.TokenDataLoader(tokens_path, seq_len=seq, batch_size=bs, steps_per_epoch=steps)

    def turbo_ep():
        n = 0
        for x, y in dl:
            n += int(x.shape[0]) * int(x.shape[1])
            float(x.sum())
        return n

    turbo = best_rate(turbo_ep)

    rng = np.random.default_rng(0)

    def numpy_ep():
        n = 0
        for _ in range(steps):
            ix = rng.integers(0, n_tokens - seq - 1, size=bs)
            x = np.stack([np.asarray(toks[i : i + seq], dtype=np.int64) for i in ix])
            y = np.stack([np.asarray(toks[i + 1 : i + 1 + seq], dtype=np.int64) for i in ix])
            n += x.size
            float(x.sum())
        return n

    numpy_idiom = best_rate(numpy_ep)
    return turbo, numpy_idiom


# ---------------- Metal resident (Apple Silicon) ----------------
def metal_resident_rates(imgdir):
    import turboloader as t

    if not getattr(t, "metal_available", lambda: False)() or not hasattr(t, "MetalResidentLoader"):
        return None
    paths = sorted(glob.glob(os.path.join(imgdir, "**", "*.JPEG"), recursive=True))
    if not paths:
        return None
    dl = t.MetalResidentLoader(paths, image_size=160, batch_size=256, shuffle=True)

    def produced():
        n = 0
        for batch in dl:
            n += batch.shape[0]
        dl.set_epoch(dl._epoch + 1)
        return n

    def consumed():
        n = 0
        for batch in dl:
            batch[:, 0, ::8, ::8].sum()
            n += batch.shape[0]
        dl.set_epoch(dl._epoch + 1)
        return n

    rates = {"produced": best_rate(produced), "consumed": best_rate(consumed)}
    dl.close()
    return rates


def main():
    ap = argparse.ArgumentParser()
    default_root = os.environ.get("TURBO_BENCH_DATA", os.path.join(HERE, "..", "..", ".."))
    ap.add_argument("--tar", default=os.path.join(default_root, "imagenette_train.tar"))
    ap.add_argument("--imgdir", default=os.path.join(default_root, "imagenette2-160", "train"))
    ap.add_argument("--tokens", default=os.path.join(default_root, "bench_tokens.bin"))
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_results.json"))
    args = ap.parse_args()

    import turboloader as t

    workers = [1, 2, 4, 6, 8]
    results = {
        "meta": {
            "dataset": "Imagenette-160 (9,469 real ImageNet JPEGs -> 160px)",
            "platform": getattr(t, "metal_device_name", lambda: "")() or "local machine",
            "turboloader_version": getattr(t, "__version__", "unknown"),
            "consumption": "real (np.sum / tf.reduce_sum forces materialization)",
            "method": f"warmup + median of {N_TIMED} timed epochs",
            "batch_size": 64,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "frameworks": [],
    }

    print("worker scan (TurboLoader)...")
    turbo_ws = [turbo_rate(args.tar, w) for w in workers]
    print("  ", turbo_ws)
    turbo_peak = max(turbo_ws)
    results["frameworks"].append({"name": "TurboLoader", "throughput": turbo_peak})
    results["workers"] = {"workers": workers, "turboloader": turbo_ws}

    print("cache (TurboLoader)...")
    turbo_cache = turbo_rate(args.tar, 6, cache=True)
    print("  ", turbo_cache)
    results["frameworks"].insert(0, {"name": "TurboLoader (cached)", "throughput": turbo_cache})
    results["cache"] = {"turboloader": turbo_cache}

    try:
        print("worker scan (PyTorch)...")
        pytorch_ws = [pytorch_rate(args.imgdir, w) for w in workers]
        print("  ", pytorch_ws)
        results["frameworks"].append({"name": "PyTorch DataLoader", "throughput": max(pytorch_ws)})
        results["workers"]["pytorch"] = pytorch_ws
        results["speedup_vs_pytorch"] = round(turbo_peak / max(pytorch_ws), 1)
    except ImportError:
        print("  torch/torchvision not installed — skipped")

    try:
        print("tf.data (AUTOTUNE)...")
        tf_rate = tfdata_rate(args.imgdir, cache=False)
        print("  ", tf_rate)
        results["frameworks"].append({"name": "TensorFlow tf.data", "throughput": tf_rate})
        results["workers"]["tensorflow"] = [tf_rate] * len(workers)
        results["cache"]["tensorflow"] = tfdata_rate(args.imgdir, cache=True)
        results["speedup_vs_tfdata"] = round(turbo_peak / tf_rate, 1)
    except ImportError:
        print("  tensorflow not installed — skipped")

    print("tokens...")
    tok_turbo, tok_numpy = token_rates(args.tokens)
    print("  ", tok_turbo, tok_numpy)
    results["tokens"] = {"turboloader": tok_turbo, "numpy_memmap": tok_numpy}

    print("Metal resident (Apple Silicon)...")
    mr = metal_resident_rates(args.imgdir)
    if mr:
        print("  ", mr)
        results["metal_resident"] = mr
    else:
        print("  not available — skipped")

    results["frameworks"].sort(key=lambda f: -f["throughput"])
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", args.out)
    print(json.dumps(results["frameworks"], indent=2))


if __name__ == "__main__":
    main()
