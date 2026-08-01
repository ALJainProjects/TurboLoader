"""Token pipeline benchmark (CUDA): batches DELIVERED TO DEVICE per second.

Compares, at a realistic GPT-2 pretraining shape (batch 32 x seq 1024, uint16
memmap corpus), the full path from memmap to CUDA tensors:

  1. nanoGPT ``get_batch`` — the exact idiom (per-row slice + np.stack, then
     ``.pin_memory().to(device, non_blocking=True)``), the strong baseline
     everyone actually uses.
  2. TokenDataLoader numpy path + ``torch.from_numpy(...).to(device)``.
  3. TokenDataLoader ``device="cuda"`` — one seq_len+1 gather into a pinned
     ring, H2D on a side stream, CUDA-event guarded reuse.

Loader-only by design: no model, sync once per measured repeat. This isolates
what the e2e GPT example (examples/train_gpt_tokenloader.py) shows is hidden
behind compute at small model sizes.

Usage:  python benchmarks/benchmark_token_loader.py [--tokens 200000000]
"""

import argparse
import os
import time

import numpy as np


def make_corpus(path, n):
    if not (os.path.exists(path) and os.path.getsize(path) == n * 2):
        rng = np.random.default_rng(0)
        chunk = 10_000_000
        with open(path, "wb") as f:
            done = 0
            while done < n:
                m = min(chunk, n - done)
                f.write(rng.integers(0, 50257, size=m).astype(np.uint16).tobytes())
                done += m
    return np.memmap(path, dtype=np.uint16, mode="r")


def bench(fn, steps, device, repeats=5):
    import torch

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(steps)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "loader-to-device benchmark needs CUDA"
    device = "cuda"
    tokens = make_corpus(os.path.expanduser("~/data/tokens_bench.bin"), args.tokens)
    bs, sl, steps = args.batch_size, args.seq_len, args.steps
    tok_per_run = steps * bs * sl

    import turboloader as tl

    def run_get_batch(n):
        rng = np.random.default_rng(0)
        for _ in range(n):
            ix = rng.integers(0, len(tokens) - sl - 1, size=bs)
            x = np.stack([np.asarray(tokens[i : i + sl]).astype(np.int64) for i in ix])
            y = np.stack([np.asarray(tokens[i + 1 : i + 1 + sl]).astype(np.int64) for i in ix])
            torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
            torch.from_numpy(y).pin_memory().to(device, non_blocking=True)

    dl_np = tl.TokenDataLoader(tokens, seq_len=sl, batch_size=bs, seed=0, steps_per_epoch=steps)

    def run_numpy(n):
        it = iter(dl_np)
        for _ in range(n):
            x, y = next(it)
            torch.from_numpy(x).to(device, non_blocking=True)
            torch.from_numpy(y).to(device, non_blocking=True)

    dl_dev = tl.TokenDataLoader(
        tokens, seq_len=sl, batch_size=bs, seed=0, steps_per_epoch=steps, device=device
    )

    def run_device(n):
        it = iter(dl_dev)
        for _ in range(n):
            next(it)

    print(f"corpus {len(tokens):,} uint16 tokens | bs={bs} seq={sl} | {steps} steps/run, median of 5")
    for name, fn in [
        ("nanoGPT get_batch (+pin+to)", run_get_batch),
        ("TokenDataLoader numpy +to", run_numpy),
        ("TokenDataLoader device=cuda", run_device),
    ]:
        fn(20)  # warmup
        dt = bench(fn, steps, device)
        print(f"  {name:30s} {dt:6.3f}s  {tok_per_run / dt / 1e6:8.1f}M tok/s to device")


if __name__ == "__main__":
    main()
