"""Real fine-tune with CudaResidentLoader — the GPU-resident (FFCV-beating) loader.

Fine-tunes ResNet-18 on real Imagenette fed entirely from GPU memory: the dataset is
decoded+resized to uint8 once, uploaded once, and every epoch is normalized on-GPU with a
single-launch kernel (zero H2D per epoch). Labels align via return_indices=True.

Usage (CUDA box):
    python examples/finetune_resnet_residentloader.py --imagenette-dir /data/imagenette2-160
"""

import argparse
import glob
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchvision

    import turboloader as tl

    torch.set_num_threads(1)
    train_dir = os.path.join(args.imagenette_dir, "train")
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    assert len(classes) == 10, classes
    paths, labels = [], []
    for ci, c in enumerate(classes):
        for p in sorted(glob.glob(os.path.join(train_dir, c, "*.JPEG"))):
            paths.append(p)
            labels.append(ci)
    labels_t = torch.tensor(labels, dtype=torch.long, device="cuda")
    print(f"real dataset: {len(paths)} Imagenette JPEGs, {len(classes)} classes")

    t0 = time.perf_counter()
    loader = tl.CudaResidentLoader(
        paths,
        image_size=args.size,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        return_indices=True,
    )
    print(
        f"one-time preprocess+upload: {time.perf_counter() - t0:.1f}s "
        f"(~{len(paths) * args.size * args.size * 3 / 1e6:.0f} MB resident uint8)"
    )

    model = torchvision.models.resnet18(num_classes=10).cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        loader.set_epoch(ep)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n, last = 0, 0.0
        for batch, idx in loader:
            x = torch.as_tensor(batch, device="cuda")  # zero-copy adoption
            y = labels_t[idx if torch.is_tensor(idx) else torch.as_tensor(idx, device="cuda")]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            last = float(loss.detach())
            n += x.shape[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  epoch {ep}: {dt:.2f}s  ({n / dt:.0f} img/s)  loss {last:.3f}")


if __name__ == "__main__":
    main()
