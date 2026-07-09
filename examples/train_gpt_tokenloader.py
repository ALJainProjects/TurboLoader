"""Train a small GPT on real text with TokenDataLoader (nanoGPT-style).

Downloads Tiny Shakespeare (~1.1 MB of real text), byte-tokenizes it to a uint16 memmap,
and trains a 4-layer GPT for a few hundred steps — comparing the input pipeline only:

  1. the classic nanoGPT ``get_batch`` numpy-memmap idiom
  2. ``turboloader.TokenDataLoader`` (same (x, y) next-token batches)

Both feed the identical model/step; loss decreasing proves the data is real and correctly
aligned (x shifted by one = y). Usage:  python examples/train_gpt_tokenloader.py
"""

import os
import time
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_data(path="/tmp/tinyshakespeare.bin"):
    if not os.path.exists(path):
        txt = urllib.request.urlopen(URL).read()
        np.frombuffer(txt, dtype=np.uint8).astype(np.uint16).tofile(path)
    return np.memmap(path, dtype=np.uint16, mode="r")


def make_model(device):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x, mask):
            a, _ = self.attn(
                self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False
            )
            x = x + a
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self, vocab=256, d=128, heads=4, layers=4, block=128):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(block, d)
            self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab)
            mask = torch.triu(torch.full((block, block), float("-inf")), diagonal=1)
            self.register_buffer("mask", mask)

        def forward(self, idx):
            b, t = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
            for blk in self.blocks:
                x = blk(x, self.mask[:t, :t])
            return self.head(self.ln(x))

    return GPT().to(device)


def train(loader_kind, tokens, steps, batch_size, seq_len, device):
    import torch

    model = make_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    if loader_kind == "turboloader":
        import turboloader as tl

        dl = tl.TokenDataLoader(tokens, seq_len=seq_len, batch_size=batch_size, seed=0)

        def batches():
            ep = 0
            while True:
                dl.set_epoch(ep)
                for x, y in dl:
                    yield torch.from_numpy(np.ascontiguousarray(x)).long(), torch.from_numpy(
                        np.ascontiguousarray(y)
                    ).long()
                ep += 1

    else:  # nanoGPT get_batch idiom

        def batches():
            rng = np.random.default_rng(0)
            while True:
                ix = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
                x = np.stack([np.asarray(tokens[i : i + seq_len]) for i in ix])
                y = np.stack([np.asarray(tokens[i + 1 : i + 1 + seq_len]) for i in ix])
                yield torch.from_numpy(x).long(), torch.from_numpy(y).long()

    gen = batches()
    # warmup
    for _ in range(10):
        x, y = next(gen)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).transpose(1, 2), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    first = last = None
    for s in range(steps):
        x, y = next(gen)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).transpose(1, 2), y)
        loss.backward()
        opt.step()
        if s == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.perf_counter() - t0
    print(
        f"  [{loader_kind}] {steps} steps in {dt:.2f}s ({steps / dt:.1f} steps/s)  "
        f"loss {first:.3f} -> {last:.3f}"
    )
    return dt, first, last


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1) if device == "cuda" else None
    tokens = get_data()
    print(f"real corpus: tiny shakespeare, {len(tokens):,} byte-tokens | device={device}")
    steps, bs, sl = 300, 64, 128
    dt_tl, f_tl, l_tl = train("turboloader", tokens, steps, bs, sl, device)
    dt_np, f_np, l_np = train("get_batch", tokens, steps, bs, sl, device)
    assert l_tl < f_tl - 0.5 and l_np < f_np - 0.5, "loss must clearly decrease (real training)"
    print(f"pipeline speed: turboloader {steps/dt_tl:.1f} vs get_batch {steps/dt_np:.1f} steps/s")


if __name__ == "__main__":
    main()
