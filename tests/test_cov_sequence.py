"""Coverage tests for turboloader/sequence.py edge cases.

These exercise branches of TokenDataLoader and ArrayDataLoader that the existing
suite (tests/test_audit_fixes.py) does not cover:

  TokenDataLoader
    * return_targets=False  (single-array yield, no shifted target, +0 in _max_start)
    * shuffle=False         (contiguous, non-overlapping windows from _start_positions)
    * memmap-from-file path  (np.memmap dtype/mode='r' branch via a real uint16 .bin)
    * steps_per_epoch default (max(1, _max_start // (batch_size*seq_len)) math)
    * "too short" ValueError (_max_start <= 0)

  ArrayDataLoader
    * drop_last True/False   (iteration count + __len__ math)
    * multiple aligned arrays (tuple yield, row alignment preserved under shuffle)
    * mismatched-length ValueError
    * single-array yield (np.ndarray, not a tuple)
    * set_epoch determinism (same epoch reproduces order, different epoch differs)

The token streams use np.arange so a token's *value* equals its position in the
stream; that lets every gathered window be checked against an exact expectation.
"""

import numpy as np
import pytest

tl = pytest.importorskip("turboloader")
from turboloader.sequence import TokenDataLoader, ArrayDataLoader  # noqa: E402


# --------------------------------------------------------------- TokenDataLoader
class TestTokenDataLoader:
    def test_return_targets_false_yields_single_array(self):
        """return_targets=False yields just inputs (no tuple, no shifted target)."""
        toks = np.arange(10000, dtype=np.uint16)
        dl = TokenDataLoader(
            toks,
            seq_len=4,
            batch_size=3,
            shuffle=False,
            return_targets=False,
            steps_per_epoch=2,
        )
        batch = next(iter(dl))
        # A bare ndarray, not (inputs, targets).
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (3, 4)
        assert batch.dtype == np.int64
        # Without targets the loader needs no extra +1 token of headroom.
        assert dl._max_start == len(toks) - 4

    def test_no_targets_allows_one_shorter_stream_than_targets(self):
        """The (1 if return_targets) headroom term: n == seq_len+1 is legal only
        when no shifted target is requested."""
        toks = np.arange(65, dtype=np.uint16)  # seq_len + 1
        ok = TokenDataLoader(toks, seq_len=64, batch_size=1, shuffle=False, return_targets=False)
        assert ok._max_start == 1
        x = next(iter(ok))
        assert x.shape == (1, 64)
        assert np.array_equal(x[0], np.arange(64))
        # The same stream is too short once a shifted target is required.
        with pytest.raises(ValueError):
            TokenDataLoader(toks, seq_len=64, batch_size=1, return_targets=True)

    def test_non_shuffle_contiguous_windows(self):
        """shuffle=False -> start positions are i*seq_len: contiguous, non-overlapping."""
        toks = np.arange(10000, dtype=np.uint16)
        seq_len, bs, steps = 4, 3, 2
        dl = TokenDataLoader(
            toks,
            seq_len=seq_len,
            batch_size=bs,
            shuffle=False,
            return_targets=False,
            steps_per_epoch=steps,
        )
        batches = list(dl)
        assert len(batches) == steps
        # Flatten the per-epoch windows and check they tile the stream with no gaps.
        flat = np.concatenate([b.reshape(-1) for b in batches])
        # total = steps*bs = 6 windows of length 4 starting at 0,4,8,...,20.
        assert np.array_equal(flat, np.arange(steps * bs * seq_len))
        # Spot-check the exact window layout of the first batch.
        assert np.array_equal(batches[0][0], np.arange(0, 4))
        assert np.array_equal(batches[0][1], np.arange(4, 8))
        assert np.array_equal(batches[0][2], np.arange(8, 12))

    def test_non_shuffle_with_targets_shift(self):
        """Contiguous windows still carry a correct shifted-by-one target."""
        toks = np.arange(2000, dtype=np.uint16)
        dl = TokenDataLoader(
            toks,
            seq_len=8,
            batch_size=4,
            shuffle=False,
            return_targets=True,
            steps_per_epoch=3,
        )
        x, y = next(iter(dl))
        assert x.shape == (4, 8) and y.shape == (4, 8)
        # Deterministic contiguous starts: 0, 8, 16, 24.
        assert np.array_equal(x[0], np.arange(0, 8))
        assert np.array_equal(x[1], np.arange(8, 16))
        # target == input shifted by one position in the stream.
        assert np.array_equal(y, x + 1)

    def test_memmap_from_file_path(self, tmp_path):
        """Passing a path triggers the np.memmap(dtype, mode='r') branch and reads
        the on-disk uint16 tokens correctly."""
        path = tmp_path / "tokens.bin"
        data = np.arange(2000, dtype=np.uint16)
        data.tofile(path)
        dl = TokenDataLoader(
            str(path),
            seq_len=8,
            batch_size=4,
            dtype="uint16",
            shuffle=False,
            return_targets=True,
            steps_per_epoch=2,
        )
        # Loaded as a read-only memmap, not copied into RAM.
        assert isinstance(dl._tokens, np.memmap)
        assert dl._tokens.mode == "r"
        assert dl._n == 2000
        x, y = next(iter(dl))
        # token value == position, so the file content round-trips exactly.
        assert np.array_equal(x[0], np.arange(0, 8))
        assert np.array_equal(y[0], np.arange(1, 9))
        assert x.dtype == np.int64

    def test_memmap_pathlike_accepted(self, tmp_path):
        """A PathLike (not just str) also takes the memmap branch via __fspath__."""
        path = tmp_path / "tok2.bin"
        np.arange(500, dtype=np.uint16).tofile(path)
        dl = TokenDataLoader(path, seq_len=16, batch_size=2, shuffle=False)
        assert isinstance(dl._tokens, np.memmap)
        x, _ = next(iter(dl))
        assert np.array_equal(x[0], np.arange(0, 16))

    def test_steps_per_epoch_default(self):
        """Default steps_per_epoch == max(1, _max_start // (batch_size*seq_len))."""
        toks = np.arange(10000, dtype=np.uint16)
        dl = TokenDataLoader(toks, seq_len=64, batch_size=8, return_targets=True)
        max_start = 10000 - 64 - 1
        expected = max(1, max_start // (8 * 64))
        assert expected == 19  # guards the arithmetic itself
        assert dl.steps_per_epoch == expected
        assert len(dl) == expected
        assert len(list(dl)) == expected

    def test_steps_per_epoch_default_floor_is_one(self):
        """When the corpus barely exceeds one window, the default floors to 1."""
        toks = np.arange(66, dtype=np.uint16)  # _max_start == 1 with targets
        dl = TokenDataLoader(toks, seq_len=64, batch_size=8, return_targets=True)
        assert dl._max_start == 1
        # 1 // (8*64) == 0 -> max(1, 0) == 1
        assert dl.steps_per_epoch == 1
        assert len(list(dl)) == 1

    def test_too_short_value_error(self):
        """A stream shorter than one window raises with an informative message."""
        with pytest.raises(ValueError, match="too short"):
            TokenDataLoader(np.arange(10, dtype=np.uint16), seq_len=64, batch_size=2)

    def test_too_short_boundary_max_start_zero(self):
        """_max_start == 0 (n == seq_len+1 with targets) is rejected; +1 is accepted."""
        with pytest.raises(ValueError):
            TokenDataLoader(np.arange(65, dtype=np.uint16), seq_len=64, batch_size=1)
        ok = TokenDataLoader(np.arange(66, dtype=np.uint16), seq_len=64, batch_size=1)
        assert ok._max_start == 1


# --------------------------------------------------------------- ArrayDataLoader
class TestArrayDataLoader:
    def test_single_array_yields_ndarray(self):
        """One array in -> bare ndarray batches (no tuple wrapping)."""
        arr = np.arange(100).reshape(100, 1).astype("float32")
        dl = ArrayDataLoader(arr, batch_size=25, shuffle=False)
        batch = next(iter(dl))
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (25, 1)
        # shuffle=False preserves natural order.
        assert np.array_equal(batch.reshape(-1), np.arange(25))

    def test_drop_last_true_drops_partial_batch(self):
        """drop_last=True omits the trailing partial batch in both __len__ and iteration."""
        a = np.arange(100)
        dl = ArrayDataLoader(a, batch_size=32, shuffle=False, drop_last=True)
        assert len(dl) == 3  # 100 // 32
        batches = list(dl)
        assert len(batches) == 3
        assert all(len(b) == 32 for b in batches)
        seen = sum(len(b) for b in batches)
        assert seen == 96  # last 4 samples dropped

    def test_drop_last_false_keeps_partial_batch(self):
        """drop_last=False keeps the partial last batch; __len__ uses a ceil."""
        a = np.arange(100)
        dl = ArrayDataLoader(a, batch_size=32, shuffle=False, drop_last=False)
        assert len(dl) == 4  # ceil(100/32)
        batches = list(dl)
        assert len(batches) == 4
        assert [len(b) for b in batches] == [32, 32, 32, 4]
        assert sum(len(b) for b in batches) == 100

    def test_len_math_exact_multiple(self):
        """When n is an exact multiple, drop_last makes no difference."""
        a = np.arange(96)
        assert len(ArrayDataLoader(a, batch_size=32, drop_last=False)) == 3
        assert len(ArrayDataLoader(a, batch_size=32, drop_last=True)) == 3

    def test_multiple_arrays_tuple_and_alignment(self):
        """Multiple arrays -> tuple per batch, with row alignment preserved even
        under shuffle (each array indexed by the same permutation)."""
        n = 257
        a = np.arange(n)
        b = np.arange(n) * 10 + 7  # bijection of a, so b == a*10+7 must always hold
        c = (np.arange(n) * np.arange(n)).astype("float64")
        dl = ArrayDataLoader(a, b, c, batch_size=64, shuffle=True, seed=123)
        total = 0
        for xa, xb, xc in dl:
            assert isinstance((xa, xb, xc), tuple)
            assert np.array_equal(xb, xa * 10 + 7)
            assert np.allclose(xc, xa.astype("float64") ** 2)
            total += len(xa)
        assert total == n  # full coverage, no drop_last
        assert len(dl) == -(-n // 64)  # ceil

    def test_mismatched_length_value_error(self):
        with pytest.raises(ValueError, match="first .*dimension"):
            ArrayDataLoader(np.arange(100), np.arange(99), batch_size=16)

    def test_no_arrays_value_error(self):
        with pytest.raises(ValueError, match="at least one array"):
            ArrayDataLoader(batch_size=16)

    def test_set_epoch_determinism(self):
        """Same epoch reproduces the shuffle order; a different epoch reshuffles,
        but every epoch still covers the whole dataset."""
        a = np.arange(500)
        dl = ArrayDataLoader(a, batch_size=64, shuffle=True, seed=7)

        def order():
            return np.concatenate([np.asarray(b).reshape(-1) for b in dl])

        dl.set_epoch(0)
        o0 = order()
        dl.set_epoch(0)
        o0b = order()
        dl.set_epoch(1)
        o1 = order()
        assert np.array_equal(o0, o0b), "same epoch must reproduce the order"
        assert not np.array_equal(o0, o1), "different epoch must reshuffle"
        # both epochs are full permutations of the dataset
        assert np.array_equal(np.sort(o0), np.arange(500))
        assert np.array_equal(np.sort(o1), np.arange(500))

    def test_no_shuffle_is_natural_order(self):
        """shuffle=False yields samples in original order across batches."""
        a = np.arange(200)
        dl = ArrayDataLoader(a, batch_size=64, shuffle=False)
        out = np.concatenate([np.asarray(b).reshape(-1) for b in dl])
        assert np.array_equal(out, np.arange(200))

    def test_batches_are_contiguous_copies(self):
        """Yielded batches are contiguous and independent of the source array."""
        src = np.arange(40, dtype=np.int64).reshape(40, 1)
        dl = ArrayDataLoader(src, batch_size=10, shuffle=False)
        b = next(iter(dl))
        assert b.flags["C_CONTIGUOUS"]
        b[...] = -1  # mutating the batch must not corrupt the source
        assert np.array_equal(src[:10].reshape(-1), np.arange(10))
