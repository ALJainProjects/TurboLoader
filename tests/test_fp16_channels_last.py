"""float16 and channels_last (NHWC) output on the pre-processed paths.

The serve kernels take their layout from the output shape ((B,3,H,W) CHW or
(B,H,W,3) NHWC) and their dtype from the array (float32 or float16). Contract:
every combination equals the float32 CHW result transposed and/or cast with
numpy's own round-to-nearest-even, so the fast paths cost nothing in accuracy.
"""

import io
import tarfile

import numpy as np
import pytest

import turboloader as tl
from turboloader.tbl import TblRawImageLoader, open_raw_view

Image = pytest.importorskip("PIL.Image")

SIZE = 48
N_IMGS = 21
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def tbl_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("fp16cl")
    tar = root / "imgs.tar"
    rng = np.random.default_rng(0)
    with tarfile.open(tar, "w") as tf:
        for i in range(N_IMGS):
            arr = rng.integers(0, 256, size=(60, 72, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=95)
            data = buf.getvalue()
            info = tarfile.TarInfo(f"{i:04d}.jpg")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    out = root / "imgs.tbl"
    assert tl.preprocess_to_tbl(str(tar), out, image_size=SIZE, batch_size=8) == N_IMGS
    return str(out)


@pytest.fixture(scope="module")
def view(tbl_path):
    v, h, w = open_raw_view(tbl_path)
    assert (h, w) == (SIZE, SIZE)
    return v


def _ref_chw(view, idx):
    x = view[idx].astype(np.float32) / 255.0
    x = (x - np.float32(MEAN)) / np.float32(STD)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class TestGatherOp:
    def test_nhwc_f32_is_the_chw_result_transposed(self, view):
        idx = np.array([3, 0, 7], dtype=np.int64)
        chw = np.empty((3, 3, SIZE, SIZE), np.float32)
        nhwc = np.empty((3, SIZE, SIZE, 3), np.float32)
        tl.normalize_u8_gather(view, idx, chw, mean=MEAN, std=STD)
        tl.normalize_u8_gather(view, idx, nhwc, mean=MEAN, std=STD)
        np.testing.assert_allclose(nhwc.transpose(0, 3, 1, 2), chw, atol=1e-6)

    @pytest.mark.parametrize("nhwc", [False, True])
    def test_f16_is_the_f32_result_rounded_to_nearest_even(self, view, nhwc):
        idx = np.array([5, 1], dtype=np.int64)
        shape = (2, SIZE, SIZE, 3) if nhwc else (2, 3, SIZE, SIZE)
        f32 = np.empty(shape, np.float32)
        f16 = np.empty(shape, np.float16)
        tl.normalize_u8_gather(view, idx, f32, mean=MEAN, std=STD)
        tl.normalize_u8_gather(view, idx, f16, mean=MEAN, std=STD)
        np.testing.assert_array_equal(f16, f32.astype(np.float16))

    def test_f32_chw_stays_bit_identical_to_the_simd_path(self, view):
        idx = np.array([2, 9], dtype=np.int64)
        out = np.empty((2, 3, SIZE, SIZE), np.float32)
        tl.normalize_u8_gather(view, idx, out, mean=MEAN, std=STD)
        np.testing.assert_allclose(out, _ref_chw(view, idx), atol=2e-6)

    def test_rejects_bad_outputs(self, view):
        idx = np.array([0], dtype=np.int64)
        with pytest.raises(ValueError):
            tl.normalize_u8_gather(view, idx, np.empty((1, 3, SIZE, SIZE), np.float64))
        with pytest.raises(ValueError):
            tl.normalize_u8_gather(view, idx, np.empty((1, 4, SIZE, SIZE), np.float32))
        with pytest.raises(ValueError):
            tl.normalize_u8_gather(view, idx, np.empty((1, 3, SIZE, SIZE), np.float32)[:, ::-1])


class TestCropOp:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_nhwc_matches_chw(self, view, dtype):
        idx = np.array([4, 6, 8], dtype=np.int64)
        crops = np.array([[3, 2, 30, 40], [0, 0, SIZE, SIZE], [10, 9, 20, 20]], np.float32)
        flips = np.array([1, 0, 1], np.uint8)
        chw = np.empty((3, 3, 32, 32), dtype)
        nhwc = np.empty((3, 32, 32, 3), dtype)
        tl.crop_resize_normalize_u8_gather(view, idx, crops, flips, chw, mean=MEAN, std=STD)
        tl.crop_resize_normalize_u8_gather(view, idx, crops, flips, nhwc, mean=MEAN, std=STD)
        np.testing.assert_array_equal(nhwc.transpose(0, 3, 1, 2), chw)


class TestLoader:
    def _rows(self, loader):
        out = {}
        for x, m in loader:
            for row, i in zip(np.asarray(x), m["indices"]):
                out[int(i)] = row.copy()
        return out

    def test_channels_last_identity_path(self, tbl_path, view):
        a = TblRawImageLoader(
            tbl_path, batch_size=8, shuffle=False, channels_last=True, prefetch_batches=0
        )
        assert not a._resample
        rows = self._rows(a)
        x = rows[5]
        assert x.shape == (SIZE, SIZE, 3) and x.dtype == np.float32
        np.testing.assert_allclose(x.transpose(2, 0, 1), _ref_chw(view, [5])[0], atol=2e-6)

    def test_float16_identity_path_uses_the_exact_gather(self, tbl_path, view):
        a = TblRawImageLoader(
            tbl_path, batch_size=8, shuffle=False, dtype="float16", prefetch_batches=0
        )
        assert not a._resample  # no longer forced through the crop kernel
        x = self._rows(a)[11]
        assert x.dtype == np.float16 and x.shape == (3, SIZE, SIZE)
        np.testing.assert_array_equal(x, _ref_chw(view, [11])[0].astype(np.float16))

    def test_flip_in_float16_nhwc_equals_the_float32_flip_path(self, tbl_path):
        kw = dict(batch_size=8, shuffle=False, hflip_prob=0.5, seed=9, prefetch_batches=0)
        ref = self._rows(TblRawImageLoader(tbl_path, **kw))  # float32 CHW: exact mirror path
        got = TblRawImageLoader(tbl_path, dtype="float16", channels_last=True, **kw)
        assert got._flip_via_kernel and got._resample
        rows = self._rows(got)
        for i in range(N_IMGS):
            np.testing.assert_array_equal(rows[i].transpose(2, 0, 1), ref[i].astype(np.float16))

    def test_dataloader_routes_dtype_and_layout_for_tbl(self, tbl_path):
        with tl.DataLoader(tbl_path, batch_size=8, dtype="float16", channels_last=True) as dl:
            x, _ = next(iter(dl))
        assert x.dtype == np.float16 and x.shape == (8, SIZE, SIZE, 3)

    def test_dataloader_rejects_them_on_the_tar_path(self, tmp_path):
        tar = tmp_path / "x.tar"
        with tarfile.open(tar, "w") as tf:
            buf = io.BytesIO()
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format="JPEG")
            info = tarfile.TarInfo("a.jpg")
            info.size = len(buf.getvalue())
            tf.addfile(info, io.BytesIO(buf.getvalue()))
        with pytest.raises(ValueError, match="channels_last"):
            tl.DataLoader(str(tar), image_size=8, output_format="numpy_chw", channels_last=True)
        with pytest.raises(ValueError, match="dtype"):
            tl.DataLoader(str(tar), image_size=8, output_format="numpy_chw", dtype="float16")


@pytest.mark.skipif(
    not getattr(tl, "metal_available", lambda: False)(), reason="Metal (Apple Silicon) only"
)
class TestMetalResident:
    @pytest.mark.parametrize(
        "dtype,channels_last", [("float32", True), ("float16", False), ("float16", True)]
    )
    def test_formats_match_the_float32_chw_reference(self, view, dtype, channels_last):
        from turboloader.metal_loader import MetalResidentLoader

        arr = np.ascontiguousarray(view[:16])
        with MetalResidentLoader(
            arr,
            batch_size=8,
            shuffle=False,
            dtype=dtype,
            channels_last=channels_last,
            mean=MEAN,
            std=STD,
        ) as ld:
            batch = next(iter(ld)).copy()
        ref = _ref_chw(arr, np.arange(8))
        assert batch.dtype == np.dtype(dtype)
        if channels_last:
            assert batch.shape == (8, SIZE, SIZE, 3)
            batch = batch.transpose(0, 3, 1, 2)
        else:
            assert batch.shape == (8, 3, SIZE, SIZE)
        tol = 2e-3 if dtype == "float16" else 1e-5
        np.testing.assert_allclose(batch.astype(np.float32), ref, atol=tol)


class TestCudaPlumbing:
    """GPU-free: the Python side must pass fmt and describe the output correctly."""

    @pytest.fixture
    def stubbed(self, monkeypatch):
        pytest.importorskip("torch")
        calls = []

        def fake_normalize(ptr, n, h, w, mean=None, std=None, **kw):
            calls.append(kw)
            return 0x1000

        monkeypatch.setattr(tl, "cuda_available", lambda: True, raising=False)
        monkeypatch.setattr(tl, "cuda_normalize_resident", fake_normalize, raising=False)
        monkeypatch.setattr(tl, "cuda_normalize_resident_gather", None, raising=False)
        return calls

    def test_setup_derives_fmt_and_typestr(self, stubbed):
        from turboloader.cuda_loader import CudaResidentLoader

        ld = CudaResidentLoader.__new__(CudaResidentLoader)
        ld._setup(16, 4, MEAN, STD, True, False, 0, False, dtype="float16", channels_last=True)
        assert ld._fmt == 3 and ld._typestr == "<f2"
        ld._setup(16, 4, MEAN, STD, True, False, 0, False)
        assert ld._fmt == 0 and ld._typestr == "<f4"
        with pytest.raises(ValueError):
            ld._setup(16, 4, MEAN, STD, True, False, 0, False, dtype="bfloat16")

    def test_iter_passes_fmt_and_shapes_the_batch(self, stubbed):
        from turboloader.cuda_loader import CudaResidentLoader

        class FakeGpu:
            def data_ptr(self):
                return 0x2000

        ld = CudaResidentLoader.__new__(CudaResidentLoader)
        ld._setup(16, 4, MEAN, STD, True, False, 0, False, dtype="float16", channels_last=True)
        ld._n, ld._gpu = 8, FakeGpu()
        batches = list(ld)
        assert len(batches) == 2 and stubbed and all(c == {"fmt": 3} for c in stubbed)
        cai = batches[0].__cuda_array_interface__
        assert cai["shape"] == (4, 16, 16, 3) and cai["typestr"] == "<f2"
        ld._setup(16, 4, MEAN, STD, True, False, 0, False)
        ld._n, ld._gpu = 8, FakeGpu()
        stubbed.clear()
        first = next(iter(ld))
        assert stubbed == [{}]  # fmt=0 is never passed: old CUDA builds keep working
        assert first.__cuda_array_interface__["shape"] == (4, 3, 16, 16)
