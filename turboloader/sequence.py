"""Non-image data loaders: LLM token streams and generic array/tensor datasets.

TurboLoader's image path (TAR/TBL -> decode -> SIMD transform -> batched tensor) does
not fit token or tabular data, so these loaders provide the same ergonomics (one object,
re-iterable, shuffle, set_epoch, batched arrays) for other modalities:

    * TokenDataLoader  -- memory-mapped token streams for causal-LM pretraining
    * ArrayDataLoader  -- batches one or more aligned (N, ...) arrays/memmaps
                          (embeddings, tabular rows, pre-tokenized sequences, labels)
"""

from __future__ import annotations

import numpy as np

__all__ = ["TokenDataLoader", "ArrayDataLoader", "MapDataLoader", "default_collate"]


class TokenDataLoader:
    """High-throughput loader for LLM token streams (next-token batches).

    Reads a flat array of token IDs (memory-mapped, so multi-GB corpora stream without
    loading into RAM) and yields ``(inputs, targets)`` int64 batches of shape
    ``(batch_size, seq_len)``, where ``targets`` is ``inputs`` shifted by one position
    (standard causal language-model objective).

    Args:
        source: path to a raw token file (``np.memmap``) or an in-memory array.
        seq_len: context length (tokens per sequence).
        batch_size: sequences per batch.
        dtype: token dtype for the on-disk file (e.g. ``'uint16'`` for GPT-2 BPE).
        shuffle: random start positions each epoch (set_epoch makes it reproducible).
        steps_per_epoch: batches per epoch (default: cover the corpus once).
        return_targets: also yield shifted-by-one targets (causal LM). If False, yields
            just inputs (e.g. for encoder/MLM pipelines that build their own targets).

    Example:
        >>> dl = TokenDataLoader("train.bin", seq_len=1024, batch_size=8)
        >>> for x, y in dl:            # x, y: (8, 1024) int64
        ...     logits = model(x); loss = loss_fn(logits, y)
    """

    def __init__(
        self,
        source,
        seq_len,
        batch_size=32,
        *,
        dtype="uint16",
        shuffle=True,
        seed=42,
        steps_per_epoch=None,
        return_targets=True,
    ):
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
            self._tokens = np.memmap(source, dtype=np.dtype(dtype), mode="r")
        else:
            self._tokens = np.asarray(source)
        if self._tokens.ndim != 1:
            self._tokens = self._tokens.reshape(-1)
        self._n = int(self._tokens.shape[0])
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.return_targets = bool(return_targets)
        self._epoch = 0
        self._served = 0
        self._resume_batches = 0
        # A window needs seq_len tokens (+1 more for the shifted target).
        self._max_start = self._n - self.seq_len - (1 if return_targets else 0)
        if self._max_start <= 0:
            raise ValueError(f"token stream of length {self._n} is too short for seq_len={seq_len}")
        if steps_per_epoch is None:
            steps_per_epoch = max(1, self._max_start // (self.batch_size * self.seq_len))
        self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        """Reproducible per-epoch shuffling (matches PyTorch DistributedSampler)."""
        self._epoch = int(epoch)

    def _start_positions(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        total = self.steps_per_epoch * self.batch_size
        if self.shuffle:
            return rng.integers(0, self._max_start, size=total, dtype=np.int64)
        # contiguous, non-overlapping windows (wraps if corpus < requested span)
        return (np.arange(total, dtype=np.int64) * self.seq_len) % self._max_start

    def _gather(self, starts):
        # Vectorized gather: (B, seq_len) index matrix -> one fancy-index read.
        rows = starts[:, None] + np.arange(self.seq_len, dtype=np.int64)[None, :]
        x = np.asarray(self._tokens[rows], dtype=np.int64)
        if not self.return_targets:
            return x
        y = np.asarray(self._tokens[rows + 1], dtype=np.int64)
        return x, y

    def __iter__(self):
        starts = self._start_positions()
        bs = self.batch_size
        resume = self._resume_batches
        self._resume_batches = 0
        self._served = resume
        for b in range(resume, self.steps_per_epoch):
            self._served += 1
            yield self._gather(starts[b * bs : (b + 1) * bs])

    def state_dict(self):
        """{'epoch', 'batches_served'} — start positions are deterministic per
        (seed, epoch), so resumption is exact."""
        return {
            "version": 1,
            "epoch": int(self._epoch),
            "batches_served": int(getattr(self, "_served", 0)),
        }

    def load_state_dict(self, sd):
        self._epoch = int(sd["epoch"])
        self._resume_batches = int(sd["batches_served"])


class ArrayDataLoader:
    """Batches one or more aligned ``(N, ...)`` arrays or memmaps (modality-agnostic).

    Use for embeddings, tabular features, pre-tokenized sequences, labels, or any data
    that already lives in numpy/memmap form. Returns a single array if given one array,
    else a tuple — like ``torch.utils.data.TensorDataset`` + ``DataLoader``.

    Example:
        >>> dl = ArrayDataLoader(features, labels, batch_size=256, shuffle=True)
        >>> for xb, yb in dl: ...
    """

    def __init__(self, *arrays, batch_size=32, shuffle=False, seed=42, drop_last=False):
        if not arrays:
            raise ValueError("ArrayDataLoader needs at least one array")
        self.arrays = [a if isinstance(a, np.memmap) else np.asarray(a) for a in arrays]
        self._n = len(self.arrays[0])
        for a in self.arrays:
            if len(a) != self._n:
                raise ValueError("all arrays must share the first (sample) dimension")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._single = len(self.arrays) == 1

    def __len__(self):
        n_full = self._n // self.batch_size
        return n_full if self.drop_last else -(-self._n // self.batch_size)

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            order = np.random.default_rng(self.seed + self._epoch).permutation(self._n)
        else:
            order = np.arange(self._n)
        bs = self.batch_size
        end = (self._n // bs) * bs if self.drop_last else self._n
        for start in range(0, end, bs):
            sel = order[start : start + bs]
            if self._single:
                yield np.ascontiguousarray(self.arrays[0][sel])
            else:
                yield tuple(np.ascontiguousarray(a[sel]) for a in self.arrays)


def default_collate(samples):
    """Collate a list of samples into a batch (numpy-based, like torch's default_collate).

    - tuples/lists (e.g. ``(image, label)``) -> a tuple with each position collated
    - dicts -> a dict with each value collated
    - numpy arrays / array-likes -> stacked along a new leading axis
    - scalars (int/float/bool) -> a 1-D array
    """
    first = samples[0]
    if isinstance(first, tuple):
        return tuple(default_collate([s[i] for s in samples]) for i in range(len(first)))
    if isinstance(first, list):
        return [default_collate([s[i] for s in samples]) for i in range(len(first))]
    if isinstance(first, dict):
        return {k: default_collate([s[k] for s in samples]) for k in first}
    if isinstance(first, np.ndarray):
        return np.stack(samples)
    return np.asarray(samples)


class MapDataLoader:
    """Batch ANY map-style dataset — the flexible, "wrap arbitrary Python" path.

    Accepts anything indexable with ``__len__`` and ``__getitem__(i)`` (exactly the
    ``torch.utils.data.Dataset`` protocol), so your loading/decoding/business logic can
    be arbitrary Python. Items are fetched on a bounded thread pool (with read-ahead) and
    collated into batches.

    Honest tradeoff: because the per-sample work runs in Python, this path is roughly
    PyTorch-DataLoader speed (and GIL-bound for pure-Python CPU work — threads help most
    when ``__getitem__`` releases the GIL, e.g. NumPy / PIL / file or network I/O). For
    maximum throughput on images/tokens/arrays, use the native fast paths
    (``DataLoader(... image_size=...)``, ``TokenDataLoader``, ``ArrayDataLoader``); use
    this when the data simply doesn't fit those.

    Args:
        dataset: object with ``__getitem__(i)`` (and ``__len__``), or any sized sequence.
        batch_size: samples per batch.
        shuffle: shuffle indices each epoch (reproducible via ``set_epoch``).
        num_workers: thread-pool size for concurrent ``__getitem__`` calls.
        collate_fn: how to merge a list of samples into a batch (default: ``default_collate``).
        drop_last: drop a ragged final batch.
        prefetch_batches: how many batches' worth of items to keep in flight.

    Example:
        >>> class Squares:
        ...     def __len__(self): return 1000
        ...     def __getitem__(self, i): return np.full((4,), i), i % 10   # (x, label)
        >>> for x, y in MapDataLoader(Squares(), batch_size=32, shuffle=True):
        ...     train_step(x, y)
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        *,
        shuffle=False,
        num_workers=4,
        collate_fn=None,
        seed=42,
        drop_last=False,
        prefetch_batches=2,
    ):
        if not hasattr(dataset, "__getitem__"):
            raise TypeError(
                "MapDataLoader needs a map-style dataset with __getitem__(i) "
                "(and __len__) — the torch.utils.data.Dataset protocol."
            )
        try:
            self._n = len(dataset)
        except TypeError:
            raise TypeError("MapDataLoader dataset must define __len__")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = max(1, int(num_workers))
        self.collate_fn = collate_fn or default_collate
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.prefetch_batches = max(1, int(prefetch_batches))
        self._epoch = 0

    def __len__(self):
        n_full = self._n // self.batch_size
        return n_full if self.drop_last else -(-self._n // self.batch_size)

    def set_epoch(self, epoch):
        """Reproducible per-epoch shuffling (matches PyTorch DistributedSampler)."""
        self._epoch = int(epoch)

    def __iter__(self):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        if self.shuffle:
            order = np.random.default_rng(self.seed + self._epoch).permutation(self._n)
        else:
            order = np.arange(self._n)
        bs = self.batch_size
        end = (self._n // bs) * bs if self.drop_last else self._n
        indices = (int(i) for i in order[:end])
        window = max(bs * 2, self.num_workers * 2, bs * self.prefetch_batches)
        ds = self.dataset

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = deque()
            for _ in range(window):
                try:
                    futures.append(ex.submit(ds.__getitem__, next(indices)))
                except StopIteration:
                    break
            batch = []
            while futures:
                batch.append(futures.popleft().result())  # ordered
                try:
                    futures.append(ex.submit(ds.__getitem__, next(indices)))
                except StopIteration:
                    pass
                if len(batch) == bs:
                    yield self.collate_fn(batch)
                    batch = []
            if batch:
                yield self.collate_fn(batch)
