"""Regression: per-epoch reshuffling on the worker/queue pipeline (v2.8.0 feature).

Bug (<= 2.31.0): ``UnifiedPipeline::reset()`` destroyed all TarWorkers and ``start()``
recreated them WITHOUT re-applying ``current_epoch_`` — so ``set_epoch(N)`` was silently
discarded on every new epoch (each ``for`` loop calls reset()), and every epoch shuffled
with epoch 0. Reproducible-but-DIFFERENT per-epoch orders, the documented contract, never
happened on this path. (``TarWorker::epoch_`` was also a plain size_t racing between the
control and worker threads; now atomic.)
"""

import tarfile

import numpy as np
import pytest

import turboloader

Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tar32(tmp_path_factory):
    root = tmp_path_factory.mktemp("epoch_shuffle")
    tar_path = str(root / "data.tar")
    with tarfile.open(tar_path, "w") as tf:
        for i in range(32):
            img = np.full((32, 32, 3), i * 8, dtype=np.uint8)
            p = str(root / f"{i:03d}.jpg")
            Image.fromarray(img).save(p, quality=95)
            tf.add(p, arcname=f"{i:03d}.jpg")
    return tar_path


def _epoch_order(loader):
    order = []
    for _images, meta in loader:
        order.extend(int(i) for i in meta["indices"])
    return order


def test_set_epoch_survives_reset_and_reshuffles(tar32):
    fl = turboloader.FastDataLoader(
        tar32, batch_size=8, num_workers=1, output_format="pytorch",
        target_height=32, target_width=32, shuffle=True,
    )
    fl.set_epoch(0)
    order0 = _epoch_order(fl)
    fl.set_epoch(1)  # the second `for` loop triggers reset(); the epoch must survive it
    order1 = _epoch_order(fl)
    fl.set_epoch(1)
    order1_again = _epoch_order(fl)

    assert sorted(order0) == sorted(order1) == list(range(32)), "each epoch must cover all samples"
    assert order0 != order1, (
        "epoch 0 and epoch 1 produced IDENTICAL shuffle orders: set_epoch was discarded by "
        "reset() (workers recreated with epoch 0)"
    )
    assert order1 == order1_again, "same epoch + seed must reproduce the same order"
