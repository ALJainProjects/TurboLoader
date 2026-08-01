"""End-to-end VIDEO training benchmark: wall-clock per epoch, identical model + step.

Trains a REAL video classifier (torchvision r3d_18, 2 classes) on real footage —
Big Buck Bunny vs Jellyfish (test-videos.co.uk, CC), split locally into 1-second
segment files per class (20 videos, 600 frames total) — comparing only the input
pipeline at the standard video recipe (8-frame clips, RandomResizedCrop(112) with
ONE window per clip + hflip + ImageNet normalize):

  1. PyTorch DataLoader + PyAV: per-sample av.open/seek/decode -> torchvision
     resized_crop per frame (what people actually write; pytorchvideo-shaped).
  2. TurboLoader VideoDatasetLoader: N decoder threads -> fused CUDA clip kernel
     (YUV->RGB + crop + flip + resize + normalize in ONE launch per clip).

Both feed the same forward+backward+optimizer step. Loss decreasing = the task
(which film is this clip from?) is genuinely learned. CUDA only — the Metal clip
path is roadmap (docs/benchmarks/index.md).

Usage (on a CUDA box):
    python benchmarks/benchmark_e2e_video_training.py --epochs 4
"""

import argparse
import os
import time
import urllib.request

import numpy as np

VIDS = {  # first URL that works wins (some hosts 403 non-browser agents)
    "bunny": [
        "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_30MB.mp4",
        "https://download.blender.org/peach/trailer/trailer_480p.mov",
    ],
    "jellyfish": [
        "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_30MB.mp4",
        "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
    ],
}
SEG_FRAMES = 30  # 1s segments @30fps


def _download(urls, dst):
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return url
        except Exception as e:
            print(f"  {url}: {e}")
    raise RuntimeError(f"could not download any of {urls}")


def build_dataset(root):
    """Download the two source films and split each into 1s mpeg4 segment files
    (PyAV; LGPL wheels decode h264 but encode mpeg4)."""
    import av

    if os.path.exists(os.path.join(root, "bunny")):
        return
    os.makedirs(root, exist_ok=True)
    for cls, urls in VIDS.items():
        src = os.path.join(root, f"_{cls}.mp4")
        if not os.path.exists(src):
            print(f"downloading {cls} ...")
            _download(urls, src)
        cdir = os.path.join(root, cls)
        os.makedirs(cdir, exist_ok=True)
        with av.open(src) as ic:
            istream = ic.streams.video[0]
            W, H = istream.codec_context.width, istream.codec_context.height
            seg, out, ostream = -1, None, None

            def close(o):
                if o is not None:
                    o.mux(ostream.encode())  # flush
                    o.close()

            n = 0
            for frame in ic.decode(istream):
                if n % SEG_FRAMES == 0:
                    close(out)
                    seg += 1
                    out = av.open(os.path.join(cdir, f"{cls}_{seg:03d}.mp4"), "w")
                    ostream = out.add_stream("mpeg4", rate=30)
                    ostream.width, ostream.height = W, H
                    ostream.pix_fmt = "yuv420p"
                    ostream.bit_rate = 4_000_000
                # fresh frame from pixels — passing decoded frames straight to
                # the encoder trips avcodec_send_frame EINVAL on some streams
                rgb = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                out.mux(ostream.encode(rgb))
                n += 1
            close(out)
        print(f"  {cls}: {seg + 1} segments of {SEG_FRAMES} frames")


def make_model_and_step(device, lr=0.01):
    import torch
    import torchvision

    model = torchvision.models.video.r3d_18(num_classes=2).to(device)
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


def bench_pytorch(root, epochs, steps, batch_size, clip_len, size, workers, device):
    import av
    import torch
    from torchvision.transforms import v2 as T
    from torchvision.transforms.v2 import functional as TF

    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = []
    for ci, cls in enumerate(classes):
        cdir = os.path.join(root, cls)
        items += [(os.path.join(cdir, f), ci) for f in sorted(os.listdir(cdir))]

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    class ClipDataset(torch.utils.data.Dataset):
        """Random clip per index — the standard PyAV recipe: open, decode to the
        sampled window, ONE RandomResizedCrop window + flip for the whole clip."""

        def __init__(self, n):
            self.n = n
            self.epoch = 0

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            rng = np.random.default_rng((self.epoch, idx))
            path, label = items[rng.integers(len(items))]
            with av.open(path) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                frames = []
                start = int(rng.integers(0, max(1, SEG_FRAMES - clip_len + 1)))
                for i, frame in enumerate(c.decode(s)):
                    if i < start:
                        continue
                    if len(frames) == clip_len:
                        break
                    frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            while len(frames) < clip_len:
                frames.append(frames[-1])
            clip = torch.stack(frames).permute(0, 3, 1, 2)  # (T,C,H,W) uint8
            i, j, h, w = T.RandomResizedCrop.get_params(
                clip[0], scale=[0.08, 1.0], ratio=[3 / 4, 4 / 3]
            )
            clip = TF.resized_crop(clip, i, j, h, w, [size, size], antialias=True)
            if rng.random() < 0.5:
                clip = TF.hflip(clip)
            clip = norm(clip.float() / 255.0)
            return clip, label

    ds = ClipDataset(steps * batch_size)
    _model, step = make_model_and_step(device)
    times = []
    for ep in range(epochs):
        ds.epoch = ep
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n, last = 0, 0.0
        for x, y in dl:
            x = x.to(device, non_blocking=True).permute(0, 2, 1, 3, 4)  # (B,C,T,H,W)
            y = y.to(device, non_blocking=True)
            last = step(x, y)
            n += x.shape[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [pytorch+pyav] epoch {ep}: {dt:.2f}s  ({n / dt:.0f} clips/s)  loss {last:.3f}")
    return times


def bench_turboloader(root, epochs, steps, batch_size, clip_len, size, workers, device):
    import torch

    import turboloader as tl

    loader = tl.VideoDatasetLoader(
        root,
        clip_len=clip_len,
        batch_size=batch_size,
        image_size=size,
        workers=workers,
        train_aug=True,
        seed=0,
        steps_per_epoch=steps,
    )
    _model, step = make_model_and_step(device)
    times = []
    for ep in range(epochs):
        loader.set_epoch(ep)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n, last = 0, 0.0
        for clips, labels, _meta in loader:
            x = clips.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W) for r3d_18
            last = step(x, labels)
            n += clips.shape[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  [turboloader] epoch {ep}: {dt:.2f}s  ({n / dt:.0f} clips/s)  loss {last:.3f}")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/data/video2cls"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--floor", action="store_true")
    args = ap.parse_args()

    import torch

    assert torch.cuda.is_available(), "video e2e benchmark needs CUDA"
    device = "cuda"
    torch.set_num_threads(1)
    build_dataset(args.root)
    print(
        f"r3d_18 / BBB-vs-Jellyfish / bs={args.batch_size} x {args.clip_len} frames @ "
        f"{args.size}px / {args.steps} steps x {args.epochs} epochs"
    )
    if args.floor:
        _m, step = make_model_and_step(device)
        x = torch.randn(args.batch_size, 3, args.clip_len, args.size, args.size, device=device)
        y = torch.randint(0, 2, (args.batch_size,), device=device)
        for _ in range(10):
            step(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step(x, y)
        torch.cuda.synchronize()
        print(
            f"pure-cuda floor ({args.steps} steps, resident batch): {time.perf_counter() - t0:.2f}s"
        )

    print("== TurboLoader VideoDatasetLoader (threaded decode + fused clip kernel) ==")
    t_tl = bench_turboloader(
        args.root,
        args.epochs,
        args.steps,
        args.batch_size,
        args.clip_len,
        args.size,
        args.workers,
        device,
    )
    print(f"== PyTorch DataLoader + PyAV (workers={args.workers}) ==")
    t_pt = bench_pytorch(
        args.root,
        args.epochs,
        args.steps,
        args.batch_size,
        args.clip_len,
        args.size,
        args.workers,
        device,
    )
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    s_tl, s_pt = t_tl[1:] or t_tl, t_pt[1:] or t_pt
    print(
        f"\nmedian steady-state epoch (excl. warmup): "
        f"turboloader {med(s_tl):.2f}s | pytorch+pyav {med(s_pt):.2f}s | "
        f"speedup {med(s_pt) / med(s_tl):.2f}x"
    )


if __name__ == "__main__":
    main()
