"""Interactive benchmark backend for benchmark_app.html.

Serves the dashboard and exposes POST /api/benchmark, which accepts a user's own
uploaded images and runs the REAL TurboLoader vs PyTorch (and tf.data if available)
throughput benchmark on them — same honest methodology as run_benchmark.py (real
consumption via np.sum, warmup + median of timed passes).

    pip install flask
    python benchmark_server.py        # then open http://localhost:8000

The static dashboard still works without this server (it falls back to the embedded
reference results); this just makes the "Benchmark my images" upload do real work.
"""

import io
import os
import shutil
import statistics
import tarfile
import tempfile
import time

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_FILES = 2000
TARGET_LOADS = 4000  # repeat the user's set until ~this many image-loads, for stable timing

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(HERE, "benchmark_app.html")


@app.route("/benchmark_results.json")
def results():
    return send_from_directory(HERE, "benchmark_results.json")


def _median_rate(epoch_fn, n_per_epoch, target_loads, warmup=1, timed=3):
    """Run epoch_fn (returns images processed) enough times to cover target_loads per
    measurement; report median img/s across `timed` measurements."""
    reps = max(1, target_loads // max(1, n_per_epoch))
    for _ in range(warmup):
        epoch_fn()
    rates = []
    for _ in range(timed):
        t0 = time.perf_counter()
        total = 0
        for _ in range(reps):
            total += epoch_fn()
        rates.append(total / (time.perf_counter() - t0))
    return round(statistics.median(rates))


def _bench_turbo(tar_path, n, image_size):
    import turboloader as t

    dl = t.DataLoader(
        tar_path,
        batch_size=64,
        num_workers=4,
        output_format="pytorch",
        image_size=image_size,
        transform=t.ImageNetNormalize(),
    )

    def ep():
        seen = 0
        for img, _ in dl:
            a = np.asarray(img)
            seen += a.shape[0]
            float(a.sum())  # force consumption
        return seen

    return _median_rate(ep, n, TARGET_LOADS)


def _bench_pytorch(img_dir, n, image_size):
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder

    tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = ImageFolder(img_dir, transform=tf)
    dl = DataLoader(
        ds, batch_size=64, num_workers=4, shuffle=False, persistent_workers=True, prefetch_factor=4
    )

    def ep():
        seen = 0
        for img, _ in dl:
            seen += img.shape[0]
            float(img.sum())
        return seen

    return _median_rate(ep, n, TARGET_LOADS)


@app.route("/api/benchmark", methods=["POST"])
def benchmark():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded."}), 400
    try:
        image_size = max(16, min(512, int(request.form.get("image_size", 160))))
    except ValueError:
        image_size = 160

    tmp = tempfile.mkdtemp(prefix="tbl_bench_")
    cls_dir = os.path.join(tmp, "imgs", "user")
    os.makedirs(cls_dir, exist_ok=True)
    tar_path = os.path.join(tmp, "data.tar")
    n = 0
    try:
        with tarfile.open(tar_path, "w") as tar:
            for f in files[:MAX_FILES]:
                raw = f.read()
                try:
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    continue  # skip non-images
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=90)
                jpg = buf.getvalue()
                name = f"{n:05d}.jpg"
                ti = tarfile.TarInfo(name)
                ti.size = len(jpg)
                tar.addfile(ti, io.BytesIO(jpg))
                with open(os.path.join(cls_dir, name), "wb") as out:
                    out.write(jpg)
                n += 1
        if n == 0:
            return jsonify({"error": "None of the uploaded files were valid images."}), 400

        frameworks = [{"name": "TurboLoader", "throughput": _bench_turbo(tar_path, n, image_size)}]
        try:
            frameworks.append(
                {
                    "name": "PyTorch DataLoader",
                    "throughput": _bench_pytorch(os.path.join(tmp, "imgs"), n, image_size),
                }
            )
        except Exception as e:  # torch/torchvision not installed — TurboLoader-only
            app.logger.warning("PyTorch benchmark skipped: %s", e)

        base = next((f["throughput"] for f in frameworks if "PyTorch" in f["name"]), None)
        return jsonify(
            {
                "frameworks": frameworks,
                "user": {
                    "n_images": n,
                    "image_size": image_size,
                    "speedup_vs_pytorch": (
                        round(frameworks[0]["throughput"] / base, 1) if base else None
                    ),
                },
                "meta": {
                    "consumption": "real (np.sum forces materialization)",
                    "method": "warmup + median of 3 timed passes",
                    "your_images": True,
                },
            }
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"TurboLoader benchmark dashboard: http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
