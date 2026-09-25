# PyTorch integration

> Canonical, always-current material: the [Quickstart](https://github.com/ALJainProjects/TurboLoader/wiki/Quickstart), [DataLoader API](https://github.com/ALJainProjects/TurboLoader/wiki/Image-DataLoader-API), and [Determinism / DDP](https://github.com/ALJainProjects/TurboLoader/wiki/Determinism-Resume-and-Distributed) wiki pages.

## The training loop

```python
import torch, numpy as np, turboloader as tl

labels = np.load("labels.npy")                       # aligned to the TAR's member order
loader = tl.DataLoader("train.tar", batch_size=128, output_format="pytorch", image_size=160,
                       transform=tl.ImageNetNormalize(), shuffle=True, seed=0,
                       train_aug=True, pin_memory=True, prefetch_batches=4, drop_last=True)

for epoch in range(epochs):
    loader.set_epoch(epoch)
    for x, meta in loader:                                    # x: pinned torch.FloatTensor (N,3,H,W)
        x = x.to("cuda", non_blocking=True)                   # consume within the pinned ring's window
        y = torch.from_numpy(labels[np.asarray(meta["indices"])]).to("cuda", non_blocking=True)
        loss = criterion(model(x), y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
```

Rules that matter: samples carry **no label key** (align by `meta['indices']`); with `pin_memory=True` a yielded tensor's buffer is reused after `prefetch_batches + 1` more batches — move it to the device (or `.clone()`) before then; call `set_epoch` every epoch; `torch.set_num_threads(1)` for GPU-bound training so torch's intraop pool doesn't fight the decode threads.

## Many epochs, fixed recipe: TBL-RAW

```python
tl.preprocess_to_tbl("train.tar", "train_192.tbl", image_size=192)       # once
loader = tl.DataLoader("train_192.tbl", batch_size=128, image_size=160, transform=tl.ImageNetNormalize(),
                       train_aug=True, shuffle=True, pin_memory=True)     # serve-time RandomResizedCrop + hflip
```

Fastest end-to-end pipeline we've measured (3.64 s ResNet-18 epochs on a 3090 vs 3.92 s for the PyTorch recipe, floor 3.39 s). `TblRawImageLoader(..., dtype="float16")` halves H2D for AMP.

## Mid-epoch resume

```python
sd = loader.state_dict()             # {'epoch', 'batches_served'}
loader.load_state_dict(sd)           # byte-exact continuation
```

## DDP

```python
loader = tl.DataLoader("train.tar", ..., enable_distributed=True,
                       world_rank=dist.get_rank(), world_size=dist.get_world_size(), drop_last=True)
```

Disjoint equal shards per rank; the same `world_rank`/`world_size` arguments exist on `TblRawImageLoader`, `TokenDataLoader`, `VideoDatasetLoader`, `CudaResidentLoader`, `MetalResidentLoader`.

## GPU-side loaders (CUDA build)

```python
ld = tl.CudaImageLoader(paths, batch_size=64, image_size=160, decode="nvimgcodec", return_indices=True)
for batch, idx in ld:                                    # out-of-order completion — align labels via idx
    x = torch.as_tensor(batch, device="cuda")            # zero-copy adoption, valid until the next batch
```

`CudaResidentLoader.from_tbl(...)`, `CudaStreamLoader`, `CudaPrefetcher`, `VideoDatasetLoader` — see the [CUDA wiki page](https://github.com/ALJainProjects/TurboLoader/wiki/GPU-NVIDIA-CUDA).

## PyTorch Lightning

Wrap the loader in an `IterableDataset` and hand Lightning a `torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)` — `examples/pytorch_lightning_example.py`.

## `PyTorchCompatibleLoader`

Returns `(images, labels)` tuples with pluggable label extractors (`FolderLabelExtractor`, `FilenamePatternExtractor`, `JSONSidecarExtractor`, `CallableLabelExtractor`) and `TransformAdapter` for torchvision transforms — convenient for drop-in replacement, not the fastest path.
