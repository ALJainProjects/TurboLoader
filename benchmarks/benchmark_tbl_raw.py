"""TBL-RAW pre-processed pipeline vs on-the-fly vs RAM cache — speed AND memory.

Methodology matches docs/benchmarks/index.md, with per-library lessons from the
video benchmark applied: every configuration runs in its OWN subprocess (thread
pool / allocator state from one stage measurably contaminates the next — first
observed as a 6x swing on the sync serve path), warmed one epoch, median of 5
epochs, identical consumption reported at TWO levels so under-consumption
artifacts are ruled out by construction: "produce" (full batch genuinely written
by the loader; consumer touches one element) and "np.sum" (a full extra read
pass over every batch). Per-stage peak RSS comes free with the subprocesses.
Also reports file sizes, one-time costs, and the honest LZ4-on-raw ratio.

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


STAGES = {
    "fly": dict(kind="tar"),
    "tbl": dict(kind="tbl"),
    "tbl-sync": dict(kind="tbl", prefetch_batches=0),
    "cache": dict(kind="tar", cache_decoded=True),
}


def make_loader(stage, tar_path, tbl_path, args):
    cfg = STAGES[stage]
    if cfg["kind"] == "tbl":
        return tl.DataLoader(
            tbl_path,
            batch_size=args.batch_size,
            transform=tl.ImageNetNormalize(),
            shuffle=True,
            seed=0,
            prefetch_batches=cfg.get("prefetch_batches", 4),
        )
    return tl.DataLoader(
        tar_path,
        batch_size=args.batch_size,
        output_format="pytorch",
        image_size=args.size,
        transform=tl.ImageNetNormalize(),
        num_workers=args.workers,
        shuffle=True,
        seed=0,
        cache_decoded=cfg.get("cache_decoded", False),
    )


def consume(loader, full):
    n, sink = 0, 0.0
    for batch, _meta in loader:
        x = np.asarray(batch)
        # "produce": the loader already wrote every byte of x this epoch;
        # touching one element guards against lazy/aliased-buffer artifacts.
        sink += float(np.sum(x)) if full else float(x[0, 0, 0, 0])
        n += x.shape[0]
    return n, sink


def run_stage(stage, tar_path, tbl_path, args):
    """Child-process body: one loader, warm + 2x5 timed epochs, prints a
    machine-line 'produce sum rss_mb'."""
    dl = make_loader(stage, tar_path, tbl_path, args)
    consume(dl, full=True)  # warmup epoch (page cache, pools)
    rates = {}
    for full in (False, True):
        times = []
        for ep in range(ROUNDS):
            if hasattr(dl, "set_epoch"):
                dl.set_epoch(ep)
            t0 = time.perf_counter()
            n, _ = consume(dl, full)
            times.append(time.perf_counter() - t0)
        rates["sum" if full else "produce"] = n / sorted(times)[len(times) // 2]
    if hasattr(dl, "close"):
        dl.close()
    print(f"__STAGE__ {rates['produce']:.0f} {rates['sum']:.0f} {rss_mb():.0f}")


def bench(stage, name, tar_path, tbl_path, args):
    import subprocess

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--imagenette-dir",
        args.imagenette_dir,
        "--size",
        str(args.size),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--stage",
        stage,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    line = [ln for ln in out.splitlines() if ln.startswith("__STAGE__")][-1]
    produce, ssum, rss = (float(v) for v in line.split()[1:])
    print(
        f"  {name:46s} {produce:>9,.0f} img/s produce | "
        f"{ssum:>9,.0f} np.sum-consumed | peak RSS {rss:,.0f} MB"
    )
    return {"produce": produce, "sum": ssum, "rss": rss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette-dir", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stage", choices=list(STAGES), help="internal: child-process mode")
    args = ap.parse_args()

    d = os.path.expanduser(args.imagenette_dir)
    tar_path = os.path.join(d, "imagenette_bench.tar")
    tbl_path = os.path.join(d, f"imagenette_{args.size}.tbl")
    if args.stage:
        run_stage(args.stage, tar_path, tbl_path, args)
        return
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

    print(f"\n{args.size}px, bs={args.batch_size}, per-stage subprocesses, identical consumption:")
    r_fly = bench("fly", "on-the-fly TAR (decode every epoch)", tar_path, tbl_path, args)
    r_tbl = bench("tbl", "TBL-RAW mmap (zero decode, prefetch default)", tar_path, tbl_path, args)
    bench("tbl-sync", "TBL-RAW sync (prefetch_batches=0, raw serve)", tar_path, tbl_path, args)
    r_cache = bench("cache", "cache_decoded=True (float32 in RAM)", tar_path, tbl_path, args)

    for kind in ("produce", "sum"):
        print(
            f"speedups ({kind}): TBL-RAW {r_tbl[kind] / r_fly[kind]:.2f}x vs "
            f"on-the-fly, {r_tbl[kind] / r_cache[kind]:.2f}x vs float32 cache"
        )
    print(
        f"memory: cache peak RSS {r_cache['rss']:,.0f} MB (owns ~{4 * tbl_mb:.0f} MB anonymous) "
        f"vs TBL-RAW {r_tbl['rss']:,.0f} MB (file-backed pages — shared, clean, evictable)"
    )


if __name__ == "__main__":
    main()
