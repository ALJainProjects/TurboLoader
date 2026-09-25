<!-- Generated from the project wiki page https://github.com/ALJainProjects/TurboLoader/wiki/Determinism-Resume-and-Distributed — edit there. -->
> Canonical, always-current version: **[Determinism Resume and Distributed](https://github.com/ALJainProjects/TurboLoader/wiki/Determinism-Resume-and-Distributed)** on the wiki.

# Determinism, resume, and distributed training

## Determinism: everything is a function of `(seed, epoch)`

Every loader with randomness derives it from its `seed` plus the epoch you set with `set_epoch(e)` — the `DistributedSampler` contract:

- `DataLoader`: shuffle order **and** the fused `train_aug` crops/flips.
- `TblRawImageLoader`: permutation and serve-time hflips.
- `TokenDataLoader`: window start positions.
- `VideoDatasetLoader`: `(video, start)` samples, crop windows, flips — drawn on one thread in task order and re-sequenced, so the result does not depend on which decoder thread finished first.
- `CudaResidentLoader` / `MetalResidentLoader`: the permutation.

Two loaders built with the same arguments, given the same `set_epoch`, yield the same batches in the same order (tests assert this, including under multi-threaded producers). Forgetting `set_epoch` gives you epoch 0's order every epoch — an audit-era bug ("set_epoch loss") that is now pinned by tests.

## Exact mid-epoch resume

```python
sd = loader.state_dict()        # {'version': 1, 'epoch': e, 'batches_served': k}
loader2.load_state_dict(sd)     # next iteration starts at batch k of epoch e
```

Supported by `DataLoader` (fast path and cached path — a decode-free skip), `TblRawImageLoader`, and `TokenDataLoader`. Because the order is a pure function of `(seed, epoch)`, resumption is byte-exact; `batches_served` counts batches actually yielded to you (with prefetch, batches sitting in the queue are not counted).

## Distributed data parallel

```python
loader = turboloader.DataLoader(
    "train.tar", batch_size=256, output_format="pytorch", image_size=160,
    transform=turboloader.ImageNetNormalize(), shuffle=True,
    enable_distributed=True, world_rank=rank, world_size=world_size,
    distributed_seed=42, drop_last=True,
)
for epoch in range(epochs):
    loader.set_epoch(epoch)
    ...
```

- Sharding is **disjoint and equal-size** across ranks and deterministic given `distributed_seed`; `drop_last=True` keeps batch counts identical on every rank (otherwise ranks can finish at different times and hang collective ops).
- `world_rank`/`world_size` are explicit — TurboLoader does not read them from `torch.distributed` for you; pass `dist.get_rank()` / `dist.get_world_size()` (works the same with Horovod or DeepSpeed launchers).
- `cache_decoded=True` under sharding caches only this rank's shard (a warning says so).
- Data must be reachable from every node: local copies or an NFS mount. Object storage is not read directly by the training loaders — stage it locally first.

**Every loader shards by parameter** since v2.38: `TblRawImageLoader`, `TokenDataLoader`, `VideoDatasetLoader`, `CudaResidentLoader` (+ `from_tbl`) and `MetalResidentLoader` all take `world_rank`/`world_size` and slice one global `(seed, epoch)` order into disjoint, equal-size per-rank shards (`num_samples // world_size` samples per rank; token/video ranks keep their `steps_per_epoch` batches, drawn from a `world_size`× larger global draw). `DataLoader('x.tbl', enable_distributed=True, world_rank=..., world_size=...)` forwards them. Tests assert disjointness, equal lengths and determinism.

Examples: `examples/distributed_ddp.py`, `examples/imagenet_resnet50.py`, `examples/pytorch_lightning_example.py`. Best practices (rank-0 checkpoints, scaled LR, all-reduce metrics, `dist.barrier()` per epoch) are standard PyTorch and unchanged by TurboLoader.

## Labels, again

Nothing above changes the rule from [Quickstart](https://github.com/ALJainProjects/TurboLoader/wiki/Quickstart): samples carry no label; align your label array through `meta['indices']` (fast path), `sample['index']` (dict path), or the indices returned by `return_indices=True` (GPU loaders — mandatory for `CudaImageLoader(decode='nvimgcodec')`, whose batches complete out of order).
