"""TBL-RAW pre-processed pipeline: preprocess_to_tbl + TblRawImageLoader.

The core claim under test: serving from the RAW mmap is BIT-IDENTICAL to the
TAR fast path with ImageNetNormalize — same uint8 pixels (exact rint recovery
from the [0,1] floats), same fused SIMD normalize (deinterleave_hwc_to_chw_f32),
so np.array_equal, not allclose.
"""

import io
import tarfile

import numpy as np
import pytest
from PIL import Image

import turboloader as tl

SIZE = 64
N_IMGS = 37  # deliberately not a multiple of the batch size


@pytest.fixture(scope="module")
def tar_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("tblraw")
    p = root / "imgs.tar"
    rng = np.random.default_rng(0)
    with tarfile.open(p, "w") as tf:
        for i in range(N_IMGS):
            arr = rng.integers(0, 256, size=(80, 96, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=95)
            data = buf.getvalue()
            info = tarfile.TarInfo(f"{i:04d}.jpg")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return str(p)


@pytest.fixture(scope="module")
def tbl_path(tar_path, tmp_path_factory):
    p = tmp_path_factory.mktemp("tblraw_out") / "imgs.tbl"
    n = tl.preprocess_to_tbl(tar_path, p, image_size=SIZE, batch_size=8)
    assert n == N_IMGS
    return str(p)


def _tar_batches(tar_path, batch_size=8, transform=None, shuffle=False):
    loader = tl.DataLoader(
        tar_path,
        batch_size=batch_size,
        output_format="pytorch",
        image_size=SIZE,
        transform=transform,
        shuffle=shuffle,
        num_workers=2,
    )
    try:
        out = {}
        for batch, meta in loader:
            for row, i in zip(np.asarray(batch), np.asarray(meta["indices"])):
                out[int(i)] = row.copy()
        return out
    finally:
        loader.close()


class TestRoundtrip:
    def test_bit_identical_to_tar_pipeline(self, tar_path, tbl_path):
        ref = _tar_batches(tar_path, transform=tl.ImageNetNormalize())
        dl = tl.TblRawImageLoader(tbl_path, batch_size=8, shuffle=False)
        got = {}
        for batch, meta in dl:
            for row, i in zip(batch, meta["indices"]):
                got[int(i)] = row.copy()
        assert set(got) == set(ref)
        for i in ref:
            assert np.array_equal(got[i], ref[i]), f"sample {i} differs"

    def test_unit_range_mode(self, tar_path, tbl_path):
        ref = _tar_batches(tar_path, transform=None)  # [0,1] floats
        dl = tl.TblRawImageLoader(tbl_path, batch_size=8, shuffle=False, mean=None, std=None)
        for batch, meta in dl:
            for row, i in zip(batch, meta["indices"]):
                assert np.array_equal(row, ref[int(i)])

    def test_dataloader_routing(self, tbl_path):
        dl = tl.DataLoader(
            tbl_path, batch_size=8, transform=tl.ImageNetNormalize(), image_size=SIZE
        )
        batch, meta = next(iter(dl))
        assert batch.shape == (8, 3, SIZE, SIZE) and batch.dtype == np.float32
        assert len(dl) == -(-N_IMGS // 8)

    def test_routing_rejects_other_transforms(self, tbl_path):
        with pytest.raises(ValueError, match="baked in"):
            tl.DataLoader(tbl_path, batch_size=8, transform=tl.Resize(32, 32))

    def test_routing_rejects_wrong_size(self, tbl_path):
        with pytest.raises(ValueError, match="does not resize"):
            tl.DataLoader(tbl_path, batch_size=8, image_size=SIZE * 2)


class TestContract:
    def test_epoch_determinism_and_shuffle(self, tbl_path):
        a = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=3)
        b = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=3)
        a.set_epoch(2)
        b.set_epoch(2)
        ia = [m["indices"].tolist() for _, m in a]
        ib = [m["indices"].tolist() for _, m in b]
        assert ia == ib
        a.set_epoch(3)
        assert ia != [m["indices"].tolist() for _, m in a]
        flat = [i for bt in ia for i in bt]
        assert sorted(flat) == list(range(N_IMGS))  # a true permutation

    def test_drop_last_and_len(self, tbl_path):
        keep = tl.TblRawImageLoader(tbl_path, batch_size=8, drop_last=False)
        drop = tl.TblRawImageLoader(tbl_path, batch_size=8, drop_last=True)
        assert len(keep) == -(-N_IMGS // 8)
        assert len(drop) == N_IMGS // 8
        sizes = [b.shape[0] for b, _ in keep]
        assert sizes[-1] == N_IMGS % 8 and all(s == 8 for s in sizes[:-1])
        assert all(b.shape[0] == 8 for b, _ in drop)

    def test_state_dict_resume(self, tbl_path):
        a = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=5)
        a.set_epoch(1)
        it = iter(a)
        next(it)
        next(it)
        sd = a.state_dict()

        b = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=5)
        b.load_state_dict(sd)
        resumed = [m["indices"].tolist() for _, m in b]

        c = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=5)
        c.set_epoch(1)
        assert resumed == [m["indices"].tolist() for _, m in c][2:]

    def test_fresh_arrays_by_default(self, tbl_path):
        dl = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=0)
        it = iter(dl)
        first, _ = next(it)
        keep = first.copy()
        next(it)
        next(it)
        assert np.array_equal(first, keep)  # no ring reuse without pin_memory


class TestValidation:
    def test_rejects_non_raw_tbl(self, tmp_path):
        p = tmp_path / "jpeg.tbl"
        w = tl.TblWriterV2(str(p), enable_compression=False)
        w.add_sample(b"\xff\xd8\xff\xe0fakejpeg", tl.SampleFormat.JPEG, width=8, height=8)
        w.finalize()
        with pytest.raises(ValueError, match="RAW_U8"):
            tl.TblRawImageLoader(str(p))

    def test_rejects_compressed_raw(self, tmp_path):
        p = tmp_path / "comp.tbl"
        w = tl.TblWriterV2(str(p), enable_compression=True)
        payload = np.zeros((8, 8, 3), dtype=np.uint8).tobytes()
        w.add_sample(payload, tl.SampleFormat.RAW_U8, width=8, height=8)
        w.finalize()
        with pytest.raises(ValueError, match="compressed"):
            tl.TblRawImageLoader(str(p))

    def test_rejects_mixed_dims(self, tmp_path):
        p = tmp_path / "mixed.tbl"
        w = tl.TblWriterV2(str(p), enable_compression=False)
        for hw in ((8, 8), (16, 8)):
            w.add_sample(
                np.zeros((hw[0], hw[1], 3), dtype=np.uint8).tobytes(),
                tl.SampleFormat.RAW_U8,
                width=hw[1],
                height=hw[0],
            )
        w.finalize()
        with pytest.raises(ValueError, match="mixed"):
            tl.TblRawImageLoader(str(p))


class TestNormalizeOp:
    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 256, size=(5, 17, 13, 3), dtype=np.uint8)
        out = np.empty((5, 3, 17, 13), dtype=np.float32)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        tl.normalize_u8_batch(x, out, mean=mean, std=std)
        ref = ((x.astype(np.float32) / 255.0 - mean) / std).transpose(0, 3, 1, 2)
        assert np.abs(out - ref).max() < 1e-5

    def test_scale_only(self):
        x = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
        out = np.empty((2, 3, 4, 4), dtype=np.float32)
        tl.normalize_u8_batch(x, out)
        assert np.abs(out - (x.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).max() < 1e-7

    def test_shape_validation(self):
        with pytest.raises(Exception):
            tl.normalize_u8_batch(
                np.zeros((2, 4, 4, 3), dtype=np.uint8),
                np.zeros((2, 3, 5, 4), dtype=np.float32),
            )

    def test_mean_without_std_rejected(self):
        with pytest.raises(Exception, match="both"):
            tl.normalize_u8_batch(
                np.zeros((1, 4, 4, 3), dtype=np.uint8),
                np.zeros((1, 3, 4, 4), dtype=np.float32),
                mean=[0.5, 0.5, 0.5],
            )


class TestResidentIngestion:
    """GPU-resident loaders fed by the RAW mmap (no decode-all pass)."""

    @pytest.mark.skipif(not getattr(tl, "metal_available", lambda: False)(), reason="needs Metal")
    def test_metal_resident_from_tbl(self, tbl_path):
        dl = tl.MetalResidentLoader(tbl_path, batch_size=8, return_indices=True)
        ref = tl.TblRawImageLoader(tbl_path, batch_size=8, shuffle=False, drop_last=True)
        for (mb, midx), (rb, rm) in zip(dl, ref):
            assert np.array_equal(np.asarray(midx), rm["indices"])
            assert np.abs(np.asarray(mb) - rb).max() < 1e-5

    @pytest.mark.skipif(
        not (getattr(tl, "cuda_available", lambda: False)() and hasattr(tl, "CudaResidentLoader")),
        reason="needs CUDA",
    )
    def test_cuda_resident_from_tbl(self, tbl_path):
        import torch

        dl = tl.CudaResidentLoader.from_tbl(tbl_path, batch_size=8, return_indices=True)
        ref = tl.TblRawImageLoader(tbl_path, batch_size=8, shuffle=False, drop_last=True)
        for (cb, cidx), (rb, rm) in zip(dl, ref):
            got = torch.as_tensor(cb, device="cuda").cpu().numpy()
            assert np.array_equal(np.asarray(cidx), rm["indices"])
            assert np.abs(got - rb).max() < 1e-5


class TestHflip:
    def test_flip_correct_and_deterministic(self, tbl_path):
        plain = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=9, shuffle=False)
        f1 = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=9, shuffle=False, hflip_prob=0.5)
        f2 = tl.TblRawImageLoader(tbl_path, batch_size=8, seed=9, shuffle=False, hflip_prob=0.5)
        f1.set_epoch(1)
        f2.set_epoch(1)
        flipped_any = 0
        for (pb, _), (a, _), (b, _) in zip(plain, f1, f2):
            assert np.array_equal(a, b)  # deterministic per (seed, epoch)
            for r in range(a.shape[0]):
                same = np.array_equal(a[r], pb[r])
                mirrored = np.array_equal(a[r], pb[r, :, :, ::-1])
                assert same or mirrored
                flipped_any += mirrored and not same
        assert flipped_any > 0

    def test_train_aug_on_tbl_rejected(self, tbl_path):
        with pytest.raises(ValueError, match="TAR pipeline"):
            tl.DataLoader(tbl_path, batch_size=8, train_aug=True)


class TestGatherOp:
    def test_gather_matches_take_plus_batch(self):
        rng = np.random.default_rng(2)
        ds = rng.integers(0, 256, size=(23, 12, 10, 3), dtype=np.uint8)
        idx = rng.permutation(23)[:9].astype(np.int64)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        a = np.empty((9, 3, 12, 10), dtype=np.float32)
        tl.normalize_u8_gather(ds, idx, a, mean=mean, std=std)
        b = np.empty((9, 3, 12, 10), dtype=np.float32)
        tl.normalize_u8_batch(np.ascontiguousarray(ds[idx]), b, mean=mean, std=std)
        assert np.array_equal(a, b)

    def test_gather_rejects_out_of_range(self):
        ds = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        out = np.zeros((1, 3, 4, 4), dtype=np.float32)
        with pytest.raises(Exception, match="range"):
            tl.normalize_u8_gather(ds, np.array([4], dtype=np.int64), out)
        with pytest.raises(Exception, match="range"):
            tl.normalize_u8_gather(ds, np.array([-1], dtype=np.int64), out)
