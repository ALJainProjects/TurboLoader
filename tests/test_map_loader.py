"""Tests for the flexible map-style loader (wrap any Python __getitem__)."""

import numpy as np
import pytest

tl = pytest.importorskip("turboloader")


class TupleDS:
    """Arbitrary Python dataset returning (features, label) tuples."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.full((3,), i, dtype=np.float32), i % 4


def test_maploader_basic_shapes_and_coverage():
    dl = tl.MapDataLoader(TupleDS(100), batch_size=16, shuffle=False)
    xb, yb = next(iter(dl))
    assert xb.shape == (16, 3) and xb.dtype == np.float32 and yb.shape == (16,)
    assert np.array_equal(xb[0], [0, 0, 0]) and yb.tolist()[:4] == [0, 1, 2, 3]
    assert sum(len(x) for x, _ in dl) == 100
    assert len(dl) == 7  # ceil(100/16)


def test_maploader_threadpool_preserves_order():
    # With num_workers>1 the items still come back in index order (no shuffle).
    dl = tl.MapDataLoader(TupleDS(50), batch_size=10, shuffle=False, num_workers=8)
    ys = [int(y) for _x, ys_ in dl for y in ys_]
    assert ys == [i % 4 for i in range(50)]


def test_maploader_shuffle_is_deterministic_per_epoch():
    dl = tl.MapDataLoader(TupleDS(60), batch_size=10, shuffle=True, seed=7)
    dl.set_epoch(0)
    a = [int(y) for _x, ys in dl for y in ys]
    dl.set_epoch(0)
    b = [int(y) for _x, ys in dl for y in ys]
    dl.set_epoch(1)
    c = [int(y) for _x, ys in dl for y in ys]
    assert a == b and a != c and sorted(a) == sorted(c)


def test_maploader_drop_last():
    full = list(tl.MapDataLoader(TupleDS(25), batch_size=10, drop_last=False))
    dropped = list(tl.MapDataLoader(TupleDS(25), batch_size=10, drop_last=True))
    assert [len(x) for x, _ in full] == [10, 10, 5]
    assert [len(x) for x, _ in dropped] == [10, 10]


def test_default_collate_dict_and_array():
    class DictDS:
        def __len__(self):
            return 12

        def __getitem__(self, i):
            return {"img": np.zeros((2, 2)), "id": i}

    batches = list(tl.MapDataLoader(DictDS(), batch_size=4))
    assert len(batches) == 3
    assert set(batches[0].keys()) == {"img", "id"}
    assert batches[0]["img"].shape == (4, 2, 2)
    assert batches[0]["id"].tolist() == [0, 1, 2, 3]


def test_unified_dataloader_dataset_kwarg_and_modality_map():
    ds = TupleDS(40)
    # dataset= kwarg, no data_path
    dl = tl.DataLoader(dataset=ds, batch_size=8, num_workers=4)
    assert sum(len(x) for x, _ in dl) == 40 and len(dl) == 5
    # modality='map' positional
    dl2 = tl.DataLoader(ds, modality="map", batch_size=8)
    assert sum(len(x) for x, _ in dl2) == 40


def test_maploader_rejects_non_mapstyle():
    with pytest.raises(TypeError):
        tl.MapDataLoader(object())  # no __getitem__


def test_custom_collate_fn():
    def my_collate(samples):
        return np.array([s for s, _ in samples]).sum()

    dl = tl.MapDataLoader(TupleDS(10), batch_size=5, collate_fn=my_collate)
    out = list(dl)
    assert len(out) == 2 and all(np.isscalar(o) or o.ndim == 0 for o in out)
