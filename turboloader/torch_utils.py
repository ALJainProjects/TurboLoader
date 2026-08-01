"""Torch-side training utilities.

CudaPrefetcher: overlap host->device copies with compute. While the model trains
on batch k, batch k+1's H2D copy runs on a side CUDA stream — the standard
double-buffering technique (NVIDIA apex / DALI iterators), packaged for
TurboLoader's iterators. Pairs with ``DataLoader(pin_memory=True, ...)``:
pinned sources make ``non_blocking=True`` genuinely asynchronous (pageable
memory silently degrades to a synchronous copy and the prefetcher buys nothing).

Honest measurement (RTX 3090, ResNet-18/Imagenette e2e, 160px AND 224px):
NEUTRAL — within noise of the plain pinned non_blocking loop, because on that
box decode delivery, not the 1.6-3 ms/batch H2D, binds the epoch. Overlap pays
when transfers are large relative to the step (big batches / high-res / small
models) or the copy demonstrably sits on the compute stream in a profile;
measure on your setup before adopting.
"""

import numpy as np

__all__ = ["CudaPrefetcher"]


class CudaPrefetcher:
    """Wrap a loader so batches arrive on-device with copies overlapped.

    Yields ``(device_tensor, meta)`` for loaders yielding ``(batch, meta)``,
    or bare device tensors for loaders yielding bare batches. The yielded
    tensor is safe to use on the current stream (the prefetcher inserts the
    stream dependency and records the consumer stream on the tensor).

    LIFETIME: staging runs ONE batch ahead, which stays within the pinned
    ring's reuse window (``prefetch_batches + 2`` buffers).
    """

    def __init__(self, loader, device="cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CudaPrefetcher requires CUDA")
        self.loader = loader
        self.device = torch.device(device)
        self._torch = torch

    def _to_tensor(self, x):
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)  # zero-copy view of the (pinned) ring buffer
        return torch.as_tensor(x)  # e.g. __cuda_array_interface__ objects

    def __iter__(self):
        torch = self._torch
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(device=self.device)

        staged = None  # (device_tensor, meta) staged on the side stream

        def _stage(item):
            batch, meta = item if isinstance(item, tuple) else (item, None)
            src = self._to_tensor(batch)
            with torch.cuda.stream(stream):
                dev = src.to(self.device, non_blocking=True)
            return dev, meta, isinstance(item, tuple)

        # ORDER MATTERS: wait/yield BEFORE staging the next copy. The wait_stream
        # covers everything enqueued on the side stream so far — staging first
        # would make the consumer's kernels wait for batch k+1's copy as well,
        # re-serializing the very transfer this class exists to overlap. Staged
        # after the yield, the copy is enqueued when the generator resumes (the
        # consumer's step is already queued) and runs on the copy engine
        # concurrently with that step's kernels.
        for item in self.loader:
            if staged is not None:
                dev, meta, was_tuple = staged
                current.wait_stream(stream)   # consumer must not touch dev early
                dev.record_stream(current)    # nor may the allocator recycle it
                yield (dev, meta) if was_tuple else dev
            staged = _stage(item)
        if staged is not None:
            dev, meta, was_tuple = staged
            current.wait_stream(stream)
            dev.record_stream(current)
            yield (dev, meta) if was_tuple else dev
