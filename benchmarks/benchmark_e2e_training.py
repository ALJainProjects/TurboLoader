"""End-to-end training benchmark: wall-clock per epoch, identical model + step.

Trains a REAL ResNet-18 (torchvision, num_classes=10) on Imagenette with the standard
train recipe — RandomResizedCrop(160) + horizontal flip + ImageNet normalize — comparing
only the input pipeline:

  1. PyTorch DataLoader: torchvision ImageFolder + PIL transforms, num_workers=N,
     pin_memory=True, persistent_workers=True (the standard recipe).
  2. TurboLoader: DataLoader(tar, train_aug=True, pin_memory=True, prefetch_batches=4).

Both feed the same forward+backward+optimizer step on CUDA. Loss is printed so you can
see training is real (decreasing), not a no-op loop. img/s here is END-TO-END training
throughput, not loader-only throughput.

Usage (on a CUDA box):
    python benchmarks/benchmark_e2e_training.py --imagenette-dir /data/imagenette2-160 \
        --epochs 3 --batch-size 128 --workers 8

The TAR for TurboLoader is built automatically next to the dataset (class labels are
carried via an aligned .npy so both pipelines train on identical supervision).
"""

import argparse
import glob
import io
import os
import tarfile
import time

import numpy as np


def build_labeled_tar(imagenette_dir, tar_path, labels_path):
    train_dir = os.path.join(imagenette_dir, "train")
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    assert len(classes) == 10, f"expected 10 Imagenette classes, got {len(classes)}: {classes}"
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for p in sorted(glob.glob(os.path.join(imagenette_dir, "train", c, "*.JPEG"))):
            paths.append(p)
            labels.append(cls_to_idx[c])
    with tarfile.open(tar_path, "w") as tf:
        for i, p in enumerate(paths):
            tf.add(p, arcname=f"{i:06d}.jpg")
    np.save(labels_path, np.asarray(labels, dtype=np.int64))
    return len(paths)


def dev_sync(device):
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_model_and_step(device, lr=0.05):
    import torch
    import torchvision

    model = torchvision.models.resnet18(num_classes=10).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step(x, y):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        return float(loss.detach())

    return model, step


def bench_pytorch(imagenette_dir, epochs, batch_size, workers, size, device):
    import torch
    import torchvision
    from torchvision import transforms as T

    tfm = T.Compose(
        [
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = torchvision.datasets.ImageFolder(os.path.join(imagenette_dir, "train"), transform=tfm)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device == "cuda"),
        persistent_workers=workers > 0,
        drop_last=True,
    )
    _model, step = make_model_and_step(device)
    times = []
    for ep in range(epochs):
        dev_sync(device)
        t0 = time.perf_counter()
        n, last = 0, 0.0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            last = step(x, y)
            n += x.shape[0]
        dev_sync(device)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [pytorch] epoch {ep}: {dt:.2f}s  ({n / dt:.0f} img/s)  loss {last:.3f}")
    return times


def bench_turboloader(tar_path, labels_path, epochs, batch_size, size, device):
    import torch

    import turboloader as tl

    labels = np.load(labels_path)
    loader = tl.DataLoader(
        tar_path,
        batch_size=batch_size,
        output_format="pytorch",
        image_size=size,
        transform=tl.ImageNetNormalize(),
        shuffle=True,
        seed=0,
        train_aug=True,
        pin_memory=True,
        prefetch_batches=4,
        drop_last=True,
    )
    _model, step = make_model_and_step(device)
    times = []
    for ep in range(epochs):
        loader.set_epoch(ep)
        dev_sync(device)
        t0 = time.perf_counter()
        n, last = 0, 0.0
        for x, meta in loader:
            xb = (x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))).to(
                device, non_blocking=True
            )
            yb = torch.from_numpy(labels[np.asarray(meta["indices"])]).to(device, non_blocking=True)
            last = step(xb, yb)
            n += xb.shape[0]
        dev_sync(device)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [turboloader] epoch {ep}: {dt:.2f}s  ({n / dt:.0f} img/s)  loss {last:.3f}")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps"])
    ap.add_argument(
        "--floor",
        action="store_true",
        help="also measure the pure-GPU floor (same steps, resident batch)",
    )
    args = ap.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else None)
        )
    assert device in ("cuda", "mps"), "end-to-end benchmark needs CUDA or MPS"
    # GPU training needs no torch CPU intraop pool; letting it spin on all cores fights the
    # input pipeline's decode threads (measured: +40% epoch time for EITHER loader when the
    # CPU is oversubscribed). Applied identically to both pipelines.
    torch.set_num_threads(1)
    tar_path = os.path.join(args.imagenette_dir, "imagenette_e2e.tar")
    labels_path = os.path.join(args.imagenette_dir, "imagenette_e2e_labels.npy")
    if not (os.path.exists(tar_path) and os.path.exists(labels_path)):
        n = build_labeled_tar(args.imagenette_dir, tar_path, labels_path)
        print(f"built labeled tar: {n} images")

    print(
        f"ResNet-18 / Imagenette-160 / bs={args.batch_size} / {args.epochs} epochs / "
        f"RandomResizedCrop({args.size}) + flip + normalize"
    )
    if args.floor:
        _m, step = make_model_and_step(device)
        x = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
        for _ in range(10):
            step(x, y)
        dev_sync(device)
        n_steps = int(np.load(labels_path).shape[0]) // args.batch_size
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step(x, y)
        dev_sync(device)
        print(
            f"pure-{device} floor ({n_steps} steps, resident batch): "
            f"{time.perf_counter() - t0:.2f}s"
        )

    print("== TurboLoader (train_aug + pin_memory + prefetch) ==")
    t_tl = bench_turboloader(tar_path, labels_path, args.epochs, args.batch_size, args.size, device)
    print("== PyTorch DataLoader (ImageFolder + PIL, workers=%d) ==" % args.workers)
    t_pt = bench_pytorch(
        args.imagenette_dir, args.epochs, args.batch_size, args.workers, args.size, device
    )
    med = lambda v: sorted(v)[len(v) // 2]
    # epoch 0 is warmup (cudnn autotune, page cache, worker spawn) — excluded from stats.
    s_tl, s_pt = t_tl[1:] or t_tl, t_pt[1:] or t_pt
    print(
        f"\nmedian steady-state epoch (excl. warmup): "
        f"turboloader {med(s_tl):.2f}s | pytorch {med(s_pt):.2f}s | "
        f"speedup {med(s_pt) / med(s_tl):.2f}x"
    )


if __name__ == "__main__":
    main()
