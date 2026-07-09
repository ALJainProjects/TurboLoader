"""Real benchmark feeding benchmark_app.html. Honest methodology:
- real consumption (np.sum forces the batch to be materialized/read),
- warmup epoch then best-of-N median over timed epochs,
- same dataset (Imagenette-160, 9,469 real ImageNet JPEGs) for every framework.
Outputs TurboLoader/benchmark_results.json.
"""

import json, time, os, glob, statistics
import numpy as np

S = "/private/tmp/claude-501/-Users-arnavjain/72f04b64-0199-4f2c-8108-4800fa3e6e79/scratchpad/"
TAR = S + "imagenette_train.tar"
IMGDIR = S + "imagenette2-160/train"
TOKENS = S + "shakespeare_tokens.bin"
OUT = S + "TurboLoader/benchmark_results.json"
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
def turbo_rate(num_workers, cache=False):
    import turboloader as t

    dl = t.DataLoader(
        TAR,
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
def pytorch_rate(num_workers):
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder

    tf = T.Compose([T.Resize((160, 160)), T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = ImageFolder(IMGDIR, transform=tf)
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
def tfdata_rate(cache=False):
    import tensorflow as tf

    files = sorted(glob.glob(IMGDIR + "/*/*.JPEG"))
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
def token_rates():
    import turboloader as t

    toks = np.memmap(TOKENS, dtype=np.uint16, mode="r")
    n_tokens = toks.shape[0]
    seq, bs, steps = 1024, 16, 200

    dl = t.TokenDataLoader(TOKENS, seq_len=seq, batch_size=bs, steps_per_epoch=steps)

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


def main():
    workers = [1, 2, 4, 6, 8]
    print("worker scan (TurboLoader)...")
    turbo_ws = [turbo_rate(w) for w in workers]
    print("  ", turbo_ws)
    print("worker scan (PyTorch)...")
    pytorch_ws = [pytorch_rate(w) for w in workers]
    print("  ", pytorch_ws)
    print("tf.data (AUTOTUNE)...")
    tf_rate = tfdata_rate(cache=False)
    print("  ", tf_rate)
    print("caches...")
    turbo_cache = turbo_rate(6, cache=True)
    tf_cache = tfdata_rate(cache=True)
    print("  turbo_cache", turbo_cache, "tf_cache", tf_cache)
    print("tokens...")
    tok_turbo, tok_numpy = token_rates()
    print("  ", tok_turbo, tok_numpy)

    turbo_peak = max(turbo_ws)
    results = {
        "meta": {
            "dataset": "Imagenette-160 (9,469 real ImageNet JPEGs -> 160px)",
            "platform": "Apple Silicon",
            "consumption": "real (np.sum / tf.reduce_sum forces materialization)",
            "method": f"warmup + median of {N_TIMED} timed epochs",
            "batch_size": 64,
        },
        "frameworks": [
            {"name": "TurboLoader (cached)", "throughput": turbo_cache},
            {"name": "TurboLoader", "throughput": turbo_peak},
            {"name": "TensorFlow tf.data", "throughput": tf_rate},
            {"name": "PyTorch DataLoader", "throughput": max(pytorch_ws)},
        ],
        "workers": {
            "workers": workers,
            "turboloader": turbo_ws,
            "pytorch": pytorch_ws,
            "tensorflow": [tf_rate] * len(workers),
        },
        "cache": {"turboloader": turbo_cache, "tensorflow": tf_cache},
        "tokens": {"turboloader": tok_turbo, "numpy_memmap": tok_numpy},
        "speedup_vs_pytorch": round(turbo_peak / max(pytorch_ws), 1),
        "speedup_vs_tfdata": round(turbo_peak / tf_rate, 1),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", OUT)
    print(json.dumps(results["frameworks"], indent=2))


if __name__ == "__main__":
    main()
