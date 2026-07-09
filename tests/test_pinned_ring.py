"""pin_memory=True: recycled (optionally pinned) torch buffer ring on the fast path.

The default path allocates a fresh ~20-40 MB numpy array per batch under the GIL;
``pin_memory=True`` fills a fixed ring of recycled torch buffers via ``next_batch_into``
(GIL released for the whole decode+fill) and yields torch tensors. When the torch build
supports pinning, ``.to(device, non_blocking=True)`` becomes a genuinely async H2D copy.

Contract pinned here: values/order identical to the numpy path; the ring actually recycles
(bounded distinct data_ptrs); a yielded tensor is safe until ring-1 further batches.
"""

import tarfile

import numpy as np
import pytest

import turboloader

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tar40(tmp_path_factory):
    root = tmp_path_factory.mktemp("pinring")
    tar_path = str(root / "data.tar")
    rng = np.random.default_rng(11)
    with tarfile.open(tar_path, "w") as tf:
        for i in range(40):
            img = rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)
            p = str(root / f"{i:03d}.jpg")
            Image.fromarray(img.astype(np.uint8)).save(p, quality=92)
            tf.add(p, arcname=f"{i:03d}.jpg")
    return tar_path


def _mk(tar, **kw):
    return turboloader.DataLoader(
        tar,
        batch_size=8,
        output_format="pytorch",
        image_size=56,
        transform=turboloader.ImageNetNormalize(),
        shuffle=False,
        **kw,
    )


def test_pinned_ring_matches_numpy_path(tar40):
    ref = [(np.asarray(x).copy(), list(m["indices"])) for x, m in _mk(tar40, prefetch_batches=0)]
    got = list(_mk(tar40, prefetch_batches=2, pin_memory=True))
    assert len(got) == len(ref) == 5
    for (xr, ir), (xt, mt) in zip(ref, got):
        assert torch.is_tensor(xt)
        assert list(mt["indices"]) == ir
        # NOTE: xt's buffer is recycled ring memory — compare via the copies we hold order-wise
    # re-iterate and compare eagerly (copy-free check while each tensor is live)
    it = iter(_mk(tar40, prefetch_batches=2, pin_memory=True))
    for xr, ir in ref:
        xt, mt = next(it)
        assert np.allclose(xt.numpy(), xr, atol=1e-6)


def test_ring_actually_recycles(tar40):
    loader = _mk(tar40, prefetch_batches=2, pin_memory=True)
    ptrs = {x.data_ptr() for x, _m in loader}
    assert len(ptrs) <= 4, f"expected <= ring(4) distinct buffers, saw {len(ptrs)} (no recycling?)"


def test_pinned_serial_path_too(tar40):
    """pin_memory with prefetch_batches=0 must also work (serial ring)."""
    ref = [np.asarray(x).copy() for x, _ in _mk(tar40, prefetch_batches=0)]
    it = iter(_mk(tar40, prefetch_batches=0, pin_memory=True))
    for xr in ref:
        xt, _ = next(it)
        assert torch.is_tensor(xt) and np.allclose(xt.numpy(), xr, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for true pinning")
def test_buffers_actually_pinned_on_cuda_builds(tar40):
    x, _ = next(iter(_mk(tar40, prefetch_batches=2, pin_memory=True)))
    assert x.is_pinned()
    y = x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert np.allclose(y.cpu().numpy(), x.numpy(), atol=0)
