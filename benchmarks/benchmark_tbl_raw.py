"""TBL-RAW pre-processed pipeline vs on-the-fly vs RAM cache — speed AND memory.

Methodology matches docs/benchmarks/index.md: every loader built once, warmed one
epoch, timed medians, identical consumption for every loader — reported at TWO
consumption levels so nobody has to trust us about under-consumption artifacts:
"produce" (full batch is genuinely written by the loader; consumer touches one
element) and "np.sum" (a full extra read pass over every batch). Also reports
peak-RSS growth per stage, file sizes, one-time costs, and the honest
LZ4-on-raw ratio.

Usage:
    python benchmarks/benchmark_tbl_raw.py --imagenette-dir ~/data/imagenette2-320
"""

import argparse
import glob
import os
import resource
import sys
import tarfile
import time

import numpy as np

import turboloader as tl

ROUNDS = 5


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024 if sys.platform == "darwin" else 1024)


def build_tar(imagenette_dir, tar_path):
    if os.path.exists(tar_path):
        return
    paths = sorted(glob.glob(os.path.join(imagenette_dir, "train", "*", "*.JPEG")))
    with tarfile.open(tar_path, "w") as tf:
        for i, p in enumerate(paths):
            tf.add(p, arcname=f"{i:06d}.jpg")
    print(f"built {tar_path}: {len(paths)} images")


def consume(loader, full):
    n, sink = 0, 0.0
    for batch, _meta in loader:
        x = np.asarray(batch)
        # "produce": the loader already wrote every byte of x this epoch;
        # touching one element guards against lazy/aliased-buffer artifacts.
        sink += float(np.sum(x)) if full else float(x[0, 0, 0, 0])
        n += x.shape[0]
    return n, sink


def bench(name, make, epochs=ROUNDS):
    dl = make()
    consume(dl, full=True)  # warmup epoch (page cache, pools)
    rates = {}
    for full in (False, True):
        times = []
        for ep in range(epochs):
            if hasattr(dl, "set_epoch"):
                dl.set_epoch(ep)
            t0 = time.perf_counter()
            n, _ = consume(dl, full)
            times.append(time.perf_counter() - t0)
        med = sorted(times)[len(times) // 2]
        rates["sum" if full else "produce"] = n / med
    print(
        f"  {name:44s} {rates['produce']:>9,.0f} img/s produce | "
        f"{rates['sum']:>9,.0f} np.sum-consumed"
    )
    if hasattr(dl, "close"):
        dl.close()
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette-dir", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    d = os.path.expanduser(args.imagenette_dir)
    tar_path = os.path.join(d, "imagenette_bench.tar")
    tbl_path = os.path.join(d, f"imagenette_{args.size}.tbl")
    build_tar(d, tar_path)

    if not os.path.exists(tbl_path):
        t0 = time.perf_counter()
        n = tl.preprocess_to_tbl(tar_path, tbl_path, image_size=args.size, num_workers=args.workers)
        print(f"preprocess_to_tbl: {n} images in {time.perf_counter() - t0:.1f}s (one-time)")

    tar_mb = os.path.getsize(tar_path) / 1e6
    tbl_mb = os.path.getsize(tbl_path) / 1e6
    print(f"file sizes: TAR {tar_mb:.0f} MB | TBL-RAW({args.size}px) {tbl_mb:.0f} MB")

    # honest LZ4-on-decoded-photos number (first 512 samples)
    lz4_path = tbl_path + ".lz4probe"
    if not os.path.exists(lz4_path):
        view, H, W = tl.tbl.open_raw_view(tbl_path)
        w = tl.TblWriterV2(lz4_path, enable_compression=True)
        k = min(512, view.shape[0])
        for i in range(k):
            w.add_sample(view[i].tobytes(), tl.SampleFormat.RAW_U8, width=W, height=H)
        w.finalize()
        raw_bytes = k * H * W * 3
        print(
            f"LZ4 on decoded photos: {raw_bytes / os.path.getsize(lz4_path):.2f}x "
            f"({k} samples) — why RAW defaults to compression=False"
        )

    print(f"\n{args.size}px, bs={args.batch_size}, identical consumption:")
    base = rss_mb()

    r_fly = bench(
        "on-the-fly TAR (decode every epoch)",
        lambda: tl.DataLoader(
            tar_path,
            batch_size=args.batch_size,
            output_format="pytorch",
            image_size=args.size,
            transform=tl.ImageNetNormalize(),
            num_workers=args.workers,
            shuffle=True,
            seed=0,
        ),
    )
    m_fly = rss_mb()

    r_tbl = bench(
        "TBL-RAW mmap (zero decode)",
        lambda: tl.DataLoader(
            tbl_path,
            batch_size=args.batch_size,
            transform=tl.ImageNetNormalize(),
            shuffle=True,
            seed=0,
        ),
    )
    m_tbl = rss_mb()

    r_cache = bench(
        "cache_decoded=True (float32 in RAM)",
        lambda: tl.DataLoader(
            tar_path,
            batch_size=args.batch_size,
            output_format="pytorch",
            image_size=args.size,
            transform=tl.ImageNetNormalize(),
            num_workers=args.workers,
            shuffle=True,
            seed=0,
            cache_decoded=True,
        ),
    )
    m_cache = rss_mb()

    print(
        f"\npeak-RSS growth while running each stage (MB): "
        f"on-the-fly +{m_fly - base:.0f} | TBL-RAW +{m_tbl - m_fly:.0f} | "
        f"float32 cache +{m_cache - m_tbl:.0f}"
    )
    for kind in ("produce", "sum"):
        print(
            f"speedups ({kind}): TBL-RAW {r_tbl[kind] / r_fly[kind]:.2f}x vs "
            f"on-the-fly, {r_tbl[kind] / r_cache[kind]:.2f}x vs float32 cache"
        )
    print(
        f"(cache owns ~{4 * tbl_mb:.0f} MB anonymous RAM; the mmap's resident "
        "pages are file-backed — shared, clean, evicted under pressure)"
    )


if __name__ == "__main__":
    main()
