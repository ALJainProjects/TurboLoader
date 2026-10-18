"""JAX and PyTorch/XLA handoff: TurboLoader batches onto TPUs, Trainium, GPUs.

TPUs and AWS Trainium/Inferentia are programmed through XLA (JAX, or PyTorch/XLA)
and have no custom-kernel path a data loader could use; the elementwise normalize
our GPU kernels fuse is something XLA fuses into the model for free. What those
platforms need from a loader is a fast host-side producer and a non-blocking
handoff. This module is the handoff:

* :class:`JaxDeviceLoader` wraps any TurboLoader (or any iterable of numpy
  batches) and yields ``jax.Array``s. Each batch is moved with ``jax.device_put``
  (asynchronous: the call returns while the transfer is in flight) from a
  background thread, with ``prefetch`` device batches kept ahead of the training
  step. Pass ``sharding=`` to place batches across local devices, or on a
  multi-host TPU pod to build the global array from this host's shard
  (``jax.make_array_from_process_local_data``); pair it with the loaders'
  ``world_rank``/``world_size`` so each host reads a disjoint slice.
* :func:`xla_device_loader` wraps a loader in PyTorch/XLA's ``MpDeviceLoader``,
  which does the same job for ``torch_xla`` (TPU) and ``torch-neuronx``
  (Trainium).

Lifetime: use loaders that yield fresh arrays (the default). A ``pin_memory``
ring hands out buffers that are recycled after the next batch, and an
asynchronous ``device_put`` may still be reading one; torch tensors are therefore
copied before the transfer here, which costs the ring its advantage.
"""

import queue
import threading

import numpy as np

__all__ = ["JaxDeviceLoader", "xla_device_loader"]

_DONE = object()


class JaxDeviceLoader:
    """Yield a loader's batches as ``jax.Array``s, moved asynchronously ahead of use.

    Args:
        loader: any iterable yielding ``batch`` or ``(batch, meta)``; TurboLoader
            image, TBL, token and array loaders all fit.
        sharding: a ``jax.sharding.Sharding`` to place each batch with. On a
            multi-process JAX program the batch is treated as this process's
            shard of the global array.
        device: a single ``jax.Device`` (default: ``jax.devices()[0]``). Mutually
            exclusive with ``sharding``.
        prefetch: device batches kept in flight ahead of the consumer (default 2).
        dtype: optional numpy dtype to cast to on the host before the transfer
            (``'bfloat16'`` is a common TPU choice; cast on host halves the bytes
            moved).
        with_meta: yield ``(array, meta)`` when the loader yields pairs (default);
            ``False`` yields the array alone.
    """

    def __init__(
        self, loader, *, sharding=None, device=None, prefetch=2, dtype=None, with_meta=True
    ):
        import jax  # lazy: the package must not require jax

        if sharding is not None and device is not None:
            raise ValueError("pass sharding= or device=, not both")
        self._jax = jax
        self.loader = loader
        self.sharding = sharding
        self.device = device if (device is not None or sharding is not None) else jax.devices()[0]
        self.prefetch = max(1, int(prefetch))
        self._dtype = None if dtype is None else np.dtype(dtype)
        self.with_meta = bool(with_meta)
        self._multi_process = sharding is not None and jax.process_count() > 1

    # ---------------------------------------------------------------- transfer
    def _to_device(self, x):
        jax = self._jax
        if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
            # torch tensor (possibly a recycled pinned-ring view): copy so the async
            # transfer can never race the ring's next fill.
            x = x.detach().cpu().numpy().copy()
        x = np.asarray(x)
        if self._dtype is not None and x.dtype != self._dtype:
            x = x.astype(self._dtype)
        if self._multi_process:
            return jax.make_array_from_process_local_data(self.sharding, x)
        return jax.device_put(x, self.sharding if self.sharding is not None else self.device)

    def _convert(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            batch, meta = item
        else:
            batch, meta = item, None
        arr = self._to_device(batch)
        if self.with_meta and meta is not None:
            return arr, meta
        return arr

    # ------------------------------------------------------------------- api
    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch):
        if hasattr(self.loader, "set_epoch"):
            self.loader.set_epoch(epoch)

    def __iter__(self):
        it = iter(self.loader)
        q = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        convert = self._convert  # the thread binds the method, not a strong loop over self

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            try:
                for item in it:
                    if not put(convert(item)):
                        return
                put(_DONE)
            except Exception as e:  # surfaced on the consumer thread
                put(("__jax_err__", e))

        th = threading.Thread(target=producer, daemon=True, name="jax-device-put")
        th.start()
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    break
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    if item[0] == "__jax_err__":
                        raise RuntimeError("JaxDeviceLoader producer failed") from item[1]
                yield item
        finally:
            stop.set()
            th.join(timeout=5)


def xla_device_loader(loader, device=None, **kwargs):
    """Wrap a TurboLoader for PyTorch/XLA (TPU via ``torch_xla``, Trainium via
    ``torch-neuronx``): returns ``MpDeviceLoader(loader, device)``, which moves each
    batch to the XLA device in a background thread. ``device`` defaults to
    ``xm.xla_device()``. Extra keyword arguments go to ``MpDeviceLoader``."""
    import torch_xla.core.xla_model as xm  # lazy: only on XLA installs
    from torch_xla.distributed.parallel_loader import MpDeviceLoader

    return MpDeviceLoader(loader, device if device is not None else xm.xla_device(), **kwargs)
