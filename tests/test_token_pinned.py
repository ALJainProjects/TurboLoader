"""TokenDataLoader pinned / device fast path.

The zero-alloc gather (`_gather_into`) is pure numpy and tested everywhere; the
pinned-ring and device= paths need torch + CUDA and are skipped elsewhere (they
run on the CUDA box — see benchmarks/GPT example for the e2e numbers).
"""

import numpy as np
import pytest

from turboloader.sequence import TokenDataLoader

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False

cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs torch + CUDA")


def _corpus(n=100_000, seed=7):
    return np.random.default_rng(seed).integers(0, 50257, size=n).astype(np.uint16)


class TestGatherInto:
    def test_matches_reference_gather(self):
        tokens = _corpus()
        dl = TokenDataLoader(tokens, seq_len=64, batch_size=8, seed=3)
        starts = dl._start_positions()[:8]
        ref_x, ref_y = dl._gather(starts)

        win = dl.seq_len + 1
        offs = np.arange(win, dtype=np.int64)
        idx = np.empty((8, win), dtype=np.int64)
        stage = np.empty((8, win), dtype=tokens.dtype)
        x = np.empty((8, dl.seq_len), dtype=np.int64)
        y = np.empty((8, dl.seq_len), dtype=np.int64)
        dl._gather_into(starts, offs, idx, stage, x, y)

        np.testing.assert_array_equal(x, ref_x)
        np.testing.assert_array_equal(y, ref_y)
        assert x.dtype == np.int64

    def test_no_targets_variant(self):
        tokens = _corpus()
        dl = TokenDataLoader(tokens, seq_len=32, batch_size=4, seed=1, return_targets=False)
        starts = dl._start_positions()[:4]
        ref = dl._gather(starts)

        offs = np.arange(32, dtype=np.int64)
        idx = np.empty((4, 32), dtype=np.int64)
        stage = np.empty((4, 32), dtype=tokens.dtype)
        x = np.empty((4, 32), dtype=np.int64)
        dl._gather_into(starts, offs, idx, stage, x, None)
        np.testing.assert_array_equal(x, ref)


class TestValidation:
    def test_ring_too_small(self):
        with pytest.raises(ValueError, match="ring"):
            TokenDataLoader(_corpus(), seq_len=8, batch_size=2, pin_memory=True, ring=1)

    @pytest.mark.skipif(torch is None or HAS_CUDA, reason="needs torch WITHOUT CUDA")
    def test_pin_without_cuda_raises(self):
        with pytest.raises(RuntimeError, match="CUDA"):
            TokenDataLoader(_corpus(), seq_len=8, batch_size=2, pin_memory=True)

    @cuda_only
    def test_non_cuda_device_rejected(self):
        with pytest.raises(ValueError, match="CUDA"):
            TokenDataLoader(_corpus(), seq_len=8, batch_size=2, device="cpu")


@cuda_only
class TestPinnedPath:
    def test_values_match_numpy_path(self):
        tokens = _corpus()
        kw = dict(seq_len=64, batch_size=8, seed=5, steps_per_epoch=6)
        ref = TokenDataLoader(tokens, **kw)
        pin = TokenDataLoader(tokens, **kw, pin_memory=True)
        for (rx, ry), (px, py) in zip(ref, pin):
            assert px.is_pinned() and py.is_pinned()
            assert px.dtype == torch.int64
            np.testing.assert_array_equal(px.numpy(), rx)
            np.testing.assert_array_equal(py.numpy(), ry)

    def test_ring_buffers_are_reused(self):
        # Documented LIFETIME contract: slot r is overwritten `ring` batches later.
        tokens = _corpus()
        dl = TokenDataLoader(
            tokens, seq_len=64, batch_size=8, seed=5, steps_per_epoch=6, pin_memory=True, ring=2
        )
        it = iter(dl)
        first = next(it)[0]
        keep = first.clone()
        next(it)
        next(it)  # ring=2 -> slot 0 overwritten here
        assert not torch.equal(first, keep), "buffer should have been recycled"

    def test_device_path_matches_and_is_cuda(self):
        tokens = _corpus()
        kw = dict(seq_len=64, batch_size=8, seed=5, steps_per_epoch=6)
        ref = TokenDataLoader(tokens, **kw)
        dev = TokenDataLoader(tokens, **kw, device="cuda")
        for (rx, ry), (dx, dy) in zip(ref, dev):
            assert dx.is_cuda and dy.is_cuda
            np.testing.assert_array_equal(dx.cpu().numpy(), rx)
            np.testing.assert_array_equal(dy.cpu().numpy(), ry)

    def test_state_dict_resume(self):
        tokens = _corpus()
        kw = dict(seq_len=64, batch_size=8, seed=5, steps_per_epoch=6, device="cuda")
        a = TokenDataLoader(tokens, **kw)
        a.set_epoch(1)
        it = iter(a)
        for _ in range(3):
            next(it)
        sd = a.state_dict()

        b = TokenDataLoader(tokens, **kw)
        b.load_state_dict(sd)
        resumed = [x.cpu().numpy() for x, _ in b]

        full = TokenDataLoader(tokens, **kw)
        full.set_epoch(1)
        expect = [x.cpu().numpy() for x, _ in full][3:]
        assert len(resumed) == len(expect)
        for r, e in zip(resumed, expect):
            np.testing.assert_array_equal(r, e)
