"""Measure the v2.39 speedups on real Imagenette data, one configuration per subprocess.

    python benchmarks/benchmark_fp16_cropdecode.py --imagenette-dir ~/data/imagenette2-320

Stages (each runs in its own process so earlier stages cannot contaminate later ones):

* tar-aug         TAR fast path, train_aug=True (RandomResizedCrop + flip fused), 160 px.
                  Run under two builds to measure crop-window decode: pass --pythonpath to a
                  checkout of the previous release for the "before" number.
* tbl-f32 / tbl-f16 / tbl-f32-nhwc / tbl-f16-nhwc
                  TBL-RAW identity serve (no crop), prefetch 2, 160 px file.
* tbl-aug-f32 / tbl-aug-f16-nhwc
                  TBL-RAW serve-time RandomResizedCrop from the 192 px file to 160 px.
* metal-f32 / metal-f16 / metal-f32-nhwc / metal-f16-nhwc   (Apple Silicon only)
                  MetalResidentLoader from the 160 px file.

Throughput is reported at two consumption levels: "produce" (iterate, touch nothing) and
"consumed" (a full np.sum over every batch, the honest lower bound). Medians of 3 epochs.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np


def _epochs(make_loader, epochs, consume):
    rates = []
    for ep in range(epochs):
        loader = make_loader()
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(ep)
        n = 0
        acc = 0.0
        t0 = time.perf_counter()
        for item in loader:
            x = item[0] if isinstance(item, tuple) else item
            n += x.shape[0]
            if consume:
                # Touch every element in its native width. float16 sums through
                # numpy's slow half path, so reduce the raw 16-bit view instead;
                # the point is a full read of the batch, not the arithmetic.
                xa = np.asarray(x)
                acc += (
                    float(xa.view(np.int16).sum(dtype=np.int64))
                    if xa.dtype == np.float16
                    else float(xa.sum())
                )
        rates.append(n / (time.perf_counter() - t0))
        if hasattr(loader, "close"):
            loader.close()
    return statistics.median(rates)


def run_stage(stage, d, size, batch_size, epochs):
    import turboloader as tl

    tar = os.path.join(d, "imagenette_bench.tar")
    tbl = os.path.join(d, f"imagenette_{size}.tbl")
    tbl_big = os.path.join(d, f"imagenette_{int(size * 1.2)}.tbl")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if stage == "tar-aug":
        mk = lambda: tl.DataLoader(  # noqa: E731
            tar,
            batch_size=batch_size,
            output_format="numpy_chw",
            image_size=size,
            transform=tl.ImageNetNormalize(),
            train_aug=True,
            shuffle=True,
            prefetch_batches=2,
        )
    elif stage.startswith("tbl-aug"):
        f16 = "f16" in stage
        mk = lambda: tl.TblRawImageLoader(  # noqa: E731
            tbl_big,
            batch_size=batch_size,
            mean=mean,
            std=std,
            train_aug=True,
            hflip_prob=0.5,
            image_size=size,
            dtype="float16" if f16 else "float32",
            channels_last="nhwc" in stage,
            prefetch_batches=2,
        )
    elif stage.startswith("tbl-"):
        mk = lambda: tl.TblRawImageLoader(  # noqa: E731
            tbl,
            batch_size=batch_size,
            mean=mean,
            std=std,
            dtype="float16" if "f16" in stage else "float32",
            channels_last="nhwc" in stage,
            prefetch_batches=2,
        )
    elif stage.startswith("metal-"):
        from turboloader.metal_loader import MetalResidentLoader

        mk = lambda: MetalResidentLoader(  # noqa: E731
            tbl,
            batch_size=batch_size,
            shuffle=True,
            dtype="float16" if "f16" in stage else "float32",
            channels_last="nhwc" in stage,
        )
    else:
        raise SystemExit(f"unknown stage {stage}")
    produce = _epochs(mk, epochs, consume=False)
    consumed = _epochs(mk, epochs, consume=True)
    print(
        json.dumps(
            {"stage": stage, "produce": produce, "consumed": consumed, "version": tl.__version__}
        )
    )


STAGES = [
    "tar-aug",
    "tbl-f32",
    "tbl-f16",
    "tbl-f32-nhwc",
    "tbl-f16-nhwc",
    "tbl-aug-f32",
    "tbl-aug-f16-nhwc",
    "metal-f32",
    "metal-f16",
    "metal-f32-nhwc",
    "metal-f16-nhwc",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette-dir", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument(
        "--pythonpath", default=None, help="checkout whose turboloader to import (before/after)"
    )
    ap.add_argument("--stage", choices=STAGES, help="internal: child-process mode")
    args = ap.parse_args()
    d = os.path.expanduser(args.imagenette_dir)
    if args.stage:
        run_stage(args.stage, d, args.size, args.batch_size, args.epochs)
        return
    env = dict(os.environ)
    if args.pythonpath:
        env["PYTHONPATH"] = os.path.abspath(args.pythonpath)
    for stage in args.stages.split(","):
        if stage.startswith("metal-") and sys.platform != "darwin":
            continue
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--imagenette-dir",
            d,
            "--size",
            str(args.size),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--stage",
            stage,
        ]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
        if not line:
            print(
                f"{stage:18s} FAILED: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else 'no output'}"
            )
            continue
        r = json.loads(line[-1])
        print(
            f"{stage:18s} produce {r['produce']:>9,.0f} img/s   consumed {r['consumed']:>9,.0f} img/s   ({r['version']})"
        )


if __name__ == "__main__":
    main()
