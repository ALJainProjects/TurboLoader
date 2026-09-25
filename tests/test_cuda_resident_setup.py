"""CudaResidentLoader._setup — the constructor logic shared by __init__ and
from_tbl, exercised WITHOUT a GPU by stubbing the capability probe.

The CUDA suite (self-hosted 3090) covers the real from_tbl upload; this pins
the Python-side contract everywhere: capability gate, field normalization,
and that both entry points route through the same setup."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import turboloader as tl  # noqa: E402
from turboloader.cuda_loader import CudaResidentLoader  # noqa: E402


@pytest.fixture
def cuda_stubbed(monkeypatch):
    monkeypatch.setattr(tl, "cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(tl, "cuda_normalize_resident", lambda *a, **k: None, raising=False)


class TestSetup:
    def test_fields_normalized(self, cuda_stubbed):
        self_ = CudaResidentLoader.__new__(CudaResidentLoader)
        self_._setup(
            image_size="96",
            batch_size=16.0,
            mean=(0.1, 0.2, 0.3),
            std=np.array([0.4, 0.5, 0.6]),
            drop_last=0,
            shuffle=1,
            seed="7",
            return_indices=None,
        )
        assert (self_._H, self_._W) == (96, 96)
        assert self_.batch_size == 16 and isinstance(self_.batch_size, int)
        assert self_.mean == [0.1, 0.2, 0.3] and self_.std == [0.4, 0.5, 0.6]
        assert self_.drop_last is False and self_.shuffle is True
        assert self_.seed == 7 and self_.return_indices is False
        assert self_._epoch == 0 and self_._t is tl and self_._torch is torch

    def test_capability_gate(self, monkeypatch):
        monkeypatch.setattr(tl, "cuda_available", lambda: False, raising=False)
        self_ = CudaResidentLoader.__new__(CudaResidentLoader)
        with pytest.raises(RuntimeError, match="CUDA build"):
            self_._setup(160, 64, (0, 0, 0), (1, 1, 1), True, False, 0, False)

    def test_len_uses_setup_fields(self, cuda_stubbed):
        self_ = CudaResidentLoader.__new__(CudaResidentLoader)
        self_._setup(160, 64, (0, 0, 0), (1, 1, 1), True, False, 0, False)
        self_._n = 130
        assert len(self_) == 2
        self_.drop_last = False
        assert len(self_) == 3

    def test_from_tbl_reaches_setup_before_any_cuda_call(self, cuda_stubbed, tmp_path):
        # A non-square file must be rejected BEFORE setup/upload; a square one
        # must call _setup (asserted by patching it) before touching CUDA.
        w = tl.TblWriterV2(str(tmp_path / "ns.tbl"), enable_compression=False)
        w.add_sample(np.zeros((8, 12, 3), np.uint8).tobytes(), tl.SampleFormat.RAW_U8, 12, 8)
        w.finalize()
        with pytest.raises(ValueError, match="square"):
            CudaResidentLoader.from_tbl(str(tmp_path / "ns.tbl"))

        w = tl.TblWriterV2(str(tmp_path / "sq.tbl"), enable_compression=False)
        w.add_sample(np.zeros((8, 8, 3), np.uint8).tobytes(), tl.SampleFormat.RAW_U8, 8, 8)
        w.finalize()
        calls = []

        def fake_setup(self, *args):
            calls.append(args)
            raise RuntimeError("stop-before-cuda")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(CudaResidentLoader, "_setup", fake_setup)
            with pytest.raises(RuntimeError, match="stop-before-cuda"):
                CudaResidentLoader.from_tbl(
                    str(tmp_path / "sq.tbl"), batch_size=4, shuffle=True, seed=3
                )
        (args,) = calls
        assert args[0] == 8 and args[1] == 4  # H from the file, batch_size passed through
        assert args[5] is True and args[6] == 3  # shuffle, seed
