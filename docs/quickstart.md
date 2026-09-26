<!-- Generated from the project wiki page https://github.com/ALJainProjects/TurboLoader/wiki/Quickstart — edit there. -->
> Canonical, always-current version: **[Quickstart](https://github.com/ALJainProjects/TurboLoader/wiki/Quickstart)** on the wiki.

# Quickstart

## 1. The training fast path (start here)

Your data is a TAR of JPEGs (any layout; WebDataset-style `000000.jpg` works). The loader you want for training returns **one contiguous normalized tensor per batch**, assembled in parallel C++:

```python
import turboloader as tl

loader = tl.DataLoader(
    "imagenette_train.tar",
    batch_size=128,
    output_format="pytorch",          # (N, 3, H, W) float32, CHW
    image_size=160,                   # fixed size => one contiguous batch
    transform=tl.ImageNetNormalize(),
    shuffle=True, seed=0,
    train_aug=True,                   # fused RandomResizedCrop + hflip inside the C++ pass
    pin_memory=True,                  # recycled pinned torch buffers (async H2D)
    prefetch_batches=4,               # decode-ahead while your GPU trains
)

for epoch in range(epochs):
    loader.set_epoch(epoch)           # reproducible, different shuffle + crops per epoch
    for x, meta in loader:            # x: torch tensor (pinned) or numpy; meta['indices']: sample ids
        x = x.to("cuda", non_blocking=True)
        y = torch.from_numpy(labels[meta["indices"]]).to("cuda", non_blocking=True)
        train_step(x, y)
```

Things to know on day one:

- **Labels come from you.** A TAR is a flat archive; samples carry **no `label` key**. Build an aligned label array once (e.g. from folder names when you write the TAR) and index it with `meta["indices"]`. `benchmarks/benchmark_e2e_training.py` shows the pattern (`build_labeled_tar` writes an aligned `.npy`).
- `image_size` and a `Resize` transform are two ways to say the same thing; passing both with different sizes raises `ValueError("Conflicting sizes...")` rather than silently training on the wrong size.
- With `pin_memory=True`, yielded tensors come from a **reused ring** — consume (`.to(device)`) or `.clone()` a batch before taking the next one; the batch you hold is safe, the previous one may already be recycled. See [Memory and Lifetime Contracts](https://github.com/ALJainProjects/TurboLoader/wiki/Memory-and-Lifetime-Contracts).
- `num_workers` does **not** scale the fast path the way PyTorch's does: it is one process-wide C++ thread pool, already saturated at one worker.

## 2. Resume mid-epoch, exactly

```python
sd = loader.state_dict()             # {'epoch': e, 'batches_served': k}
new_loader.load_state_dict(sd)       # continues byte-exactly, decode-free skip to batch k
```

## 3. Decode once, serve every epoch (TBL-RAW)

If your recipe is resize (+ hflip) + normalize, pay the JPEG decode **once**:

```python
tl.preprocess_to_tbl("imagenette_train.tar", "imagenette_160.tbl", image_size=160)   # ~6 s for 9,469 images (M4)

loader = tl.DataLoader("imagenette_160.tbl", batch_size=128, transform=tl.ImageNetNormalize(), shuffle=True)
```

Batches are bit-identical to the TAR path, served from a memory map through one fused SIMD pass: 586k img/s raw serve on an M4 Max and the fastest end-to-end epochs we've measured on the 3090. Random crop can't be applied to pre-resized samples — keep `train_aug=True` on the TAR path when you need it. Full details: [TBL RAW Preprocessed Pipeline](https://github.com/ALJainProjects/TurboLoader/wiki/TBL-RAW-Preprocessed-Pipeline).

## 4. Other modalities, same ergonomics

```python
# LLM tokens (memmap of uint16 GPT-2 BPE ids) -> (B, seq_len) int64 x, y (y = x shifted by one)
for x, y in tl.TokenDataLoader("train.bin", seq_len=1024, batch_size=32, device="cuda"):
    loss = model(x, y)

# Any (N, ...) arrays / memmaps, aligned
for feats, labels in tl.ArrayDataLoader(features, labels, batch_size=256, shuffle=True):
    ...

# Anything with __len__/__getitem__ (torch Dataset protocol) — flexibility, not the C++ fast path
for xb, yb in tl.MapDataLoader(MyDataset(), batch_size=64, num_workers=8):
    ...

# Video (Apple: hardware decode, in the wheel; NVIDIA: CUDA build)
for batch in tl.MetalVideoLoader("clip.mp4", image_size=224, batch_size=32):
    ...
```

See [Tokens and Arrays](https://github.com/ALJainProjects/TurboLoader/wiki/Tokens-and-Arrays) and [Video](https://github.com/ALJainProjects/TurboLoader/wiki/Video).

## 5. Inspect samples (the slow, flexible path)

```python
loader = tl.DataLoader("data.tar", batch_size=8)      # output_format='dict' (default)
for batch in loader:
    for s in batch:
        s["image"]        # (H, W, 3) uint8 numpy, original size
        s["filename"]     # TAR member name
        s["index"]        # sample index
```

The dict path stacks nothing and is several times slower — use it for inspection or irregular data, never for the training loop.

## 6. Clean up

Loaders are context managers and have `close()`; abandoned loaders are also cleaned up at interpreter exit. Use `with tl.DataLoader(...) as loader:` in scripts that create many loaders.

Next: [Which Loader Do I Use](https://github.com/ALJainProjects/TurboLoader/wiki/Which-Loader-Do-I-Use) · [Image DataLoader API](https://github.com/ALJainProjects/TurboLoader/wiki/Image-DataLoader-API).
