# Beyond Images: Tokens, Arrays & Any Python Dataset


TurboLoader also ships loaders for non-image modalities with the same ergonomics
(re-iterable, `shuffle`, `set_epoch`, batched arrays):

```python
# LLM pretraining: memory-mapped token stream -> (B, seq_len) next-token batches
loader = turboloader.TokenDataLoader('train.bin', seq_len=1024, batch_size=8,
                                     dtype='uint16', shuffle=True)
for x, y in loader:          # x, y: (8, 1024) int64; y is x shifted by one
    loss = model(x, y)

# Generic arrays/memmaps (embeddings, tabular features, labels, pre-tokenized data)
loader = turboloader.ArrayDataLoader(features, labels, batch_size=256, shuffle=True)
for xb, yb in loader:
    ...
```

`TokenDataLoader` uses a vectorized fancy-index gather over a `np.memmap` (so multi-GB
corpora stream without loading into RAM): measured on an RTX 3090 at GPT-2 shape
(batch 32 × seq 1024), it delivers **168M tok/s to device vs 88M for the exact nanoGPT
`get_batch` idiom (1.9×)** — `benchmarks/benchmark_token_loader.py`. The image pipeline
(decode/transform/TBL) remains C++; these modality loaders are NumPy-based and
modality-agnostic.

### CUDA fast path: pinned ring and `device=`

```python
# yields pinned torch int64 tensors from a reused ring (zero steady-state allocs);
# a yielded tensor's buffer is overwritten `ring` (default 4) batches later
loader = turboloader.TokenDataLoader('train.bin', seq_len=1024, batch_size=32,
                                     pin_memory=True)

# or let the loader manage transfers: ONE seq_len+1 gather feeds both x and y
# (half the memory traffic of separate x/y gathers), H2D runs on a side CUDA
# stream overlapped with your model's compute, buffer reuse is guarded by CUDA
# events — batches arrive as CUDA tensors with no lifetime rules to track
loader = turboloader.TokenDataLoader('train.bin', seq_len=1024, batch_size=32,
                                     device='cuda')
for x, y in loader:        # already on GPU
    loss = model(x, y)
```

Honest note: in a full GPT training loop the model usually hides the pipeline —
on a 3090 with a small GPT all paths land within ~2% (`examples/train_gpt_tokenloader.py`).
The fast path pays off when the input path is the bottleneck: eval sweeps, big
batch × seq, or CPU-bound steps.

All three modalities are also reachable from the **single `DataLoader` entry point**:

```python
turboloader.DataLoader('train.bin', modality='tokens', seq_len=1024, batch_size=8)
turboloader.DataLoader(arrays=[feats, labels], data_path=None, modality='array', batch_size=256)
turboloader.DataLoader('data.tar', image_size=160, output_format='pytorch')   # modality='image' (default)
```

### Wrap *any* Python dataset (`MapDataLoader`)

When your data doesn't fit the native paths, `MapDataLoader` batches **any** map-style
dataset — anything with `__len__` and `__getitem__(i)`, i.e. exactly the
`torch.utils.data.Dataset` protocol — so your loading/decoding/business logic can be
arbitrary Python:

```python
class MyDataset:
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = decode_however_you_like(self.records[i])   # any Python logic
        return x, self.labels[i]                       # (features, label)

# directly, or via the unified entry point with dataset=...
for xb, yb in turboloader.MapDataLoader(MyDataset(), batch_size=64, shuffle=True, num_workers=8):
    train_step(xb, yb)
```

It parallelizes `__getitem__` on a bounded thread pool with read-ahead and collates
(tuples/dicts/arrays, or a custom `collate_fn`). **Honest tradeoff:** because the
per-sample work runs in Python, this path is roughly PyTorch-`DataLoader` speed (and
GIL-bound for pure-Python CPU work — threads help most when `__getitem__` releases the
GIL, e.g. NumPy/PIL/file/network I/O). It's about *flexibility*, not the C++ fast path —
use the image/token/array loaders above when you want maximum throughput.
