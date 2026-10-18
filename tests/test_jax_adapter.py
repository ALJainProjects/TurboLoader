"""JaxDeviceLoader: asynchronous device_put handoff for JAX (TPU, GPU, CPU) and the
PyTorch/XLA wrapper. Runs on the CPU JAX backend; the contract is the same on TPU."""

import io
import sys
import tarfile
import types

import numpy as np
import pytest

import turboloader as tl
from turboloader.jax_adapter import JaxDeviceLoader, xla_device_loader

jax = pytest.importorskip("jax")
Image = pytest.importorskip("PIL.Image")


class FakeLoader:
    def __init__(self, n=5, with_meta=True, fail_at=None):
        self.n, self.with_meta, self.fail_at, self.epoch = n, with_meta, fail_at, None

    def __len__(self):
        return self.n

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        for i in range(self.n):
            if self.fail_at == i:
                raise RuntimeError("boom")
            x = np.full((4, 3, 8, 8), i, np.float32)
            yield (x, {"indices": np.arange(4) + 4 * i}) if self.with_meta else x


def test_batches_arrive_as_jax_arrays_with_meta_in_order():
    ld = JaxDeviceLoader(FakeLoader(), prefetch=2)
    assert len(ld) == 5
    out = list(ld)
    assert len(out) == 5
    for i, (x, meta) in enumerate(out):
        assert isinstance(x, jax.Array) and x.shape == (4, 3, 8, 8) and x.dtype == np.float32
        assert float(x[0, 0, 0, 0]) == i and list(meta["indices"]) == list(np.arange(4) + 4 * i)


def test_without_meta_and_host_side_dtype_cast():
    ld = JaxDeviceLoader(FakeLoader(with_meta=False), dtype="bfloat16", prefetch=1)
    xs = list(ld)
    assert all(isinstance(x, jax.Array) and str(x.dtype) == "bfloat16" for x in xs)


def test_set_epoch_and_explicit_device_and_errors():
    inner = FakeLoader()
    ld = JaxDeviceLoader(inner, device=jax.devices()[0])
    ld.set_epoch(3)
    assert inner.epoch == 3
    with pytest.raises(ValueError):
        JaxDeviceLoader(inner, device=jax.devices()[0], sharding=object())
    with pytest.raises(RuntimeError, match="producer failed"):
        list(JaxDeviceLoader(FakeLoader(fail_at=2)))


def test_sharded_placement_on_local_devices():
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    x, _ = next(iter(JaxDeviceLoader(FakeLoader(), sharding=sharding)))
    assert x.sharding == sharding


def test_dataloader_to_jax_on_a_real_tar(tmp_path):
    tar = tmp_path / "x.tar"
    rng = np.random.default_rng(0)
    with tarfile.open(tar, "w") as tf:
        for i in range(6):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (40, 40, 3), np.uint8)).save(buf, format="JPEG")
            info = tarfile.TarInfo(f"{i}.jpg")
            info.size = len(buf.getvalue())
            tf.addfile(info, io.BytesIO(buf.getvalue()))
    dl = tl.DataLoader(
        str(tar), batch_size=3, image_size=16, output_format="numpy_chw", num_workers=1
    )
    try:
        batches = list(dl.to_jax(prefetch=1))
    finally:
        dl.close()
    assert len(batches) == 2
    x, meta = batches[0]
    assert isinstance(x, jax.Array) and x.shape == (3, 3, 16, 16) and "indices" in meta


def test_torch_tensors_are_copied_before_transfer():
    torch = pytest.importorskip("torch")

    class TorchLoader:
        def __iter__(self):
            yield torch.full((2, 3), 7.0), {"i": 0}

    x, _ = next(iter(JaxDeviceLoader(TorchLoader())))
    assert isinstance(x, jax.Array) and float(x[0, 0]) == 7.0


def test_xla_wrapper_uses_mp_device_loader(monkeypatch):
    seen = {}

    class MpDeviceLoader:
        def __init__(self, loader, device, **kw):
            seen.update(loader=loader, device=device, kw=kw)

    xm = types.SimpleNamespace(xla_device=lambda: "xla:0")
    mod = types.ModuleType("torch_xla")
    core = types.ModuleType("torch_xla.core")
    core_xm = types.ModuleType("torch_xla.core.xla_model")
    core_xm.xla_device = xm.xla_device
    dist = types.ModuleType("torch_xla.distributed")
    pl = types.ModuleType("torch_xla.distributed.parallel_loader")
    pl.MpDeviceLoader = MpDeviceLoader
    for name, m in [
        ("torch_xla", mod),
        ("torch_xla.core", core),
        ("torch_xla.core.xla_model", core_xm),
        ("torch_xla.distributed", dist),
        ("torch_xla.distributed.parallel_loader", pl),
    ]:
        monkeypatch.setitem(sys.modules, name, m)
    inner = FakeLoader()
    xla_device_loader(inner, batches_per_execution=2)
    assert (
        seen["loader"] is inner
        and seen["device"] == "xla:0"
        and seen["kw"] == {"batches_per_execution": 2}
    )
