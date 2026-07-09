# End-to-end training benchmark (RTX 3090, WSL2)

`benchmarks/benchmark_e2e_training.py` — real ResNet-18 (torchvision, 10 classes) on real
Imagenette-160 (9,469 JPEGs), identical training step, identical recipe
(RandomResizedCrop(160) + hflip + ImageNet normalize), identical settings
(`torch.set_num_threads(1)` both sides — see note), bs=128, warmup epoch excluded,
median of 4 steady-state epochs. Loss decreases on both sides (real training, not a no-op).

| Input pipeline | median epoch | end-to-end img/s |
|---|---:|---:|
| **TurboLoader** (`train_aug=True, pin_memory=True, prefetch_batches=4`) | **4.37 s** | ~2,170 |
| PyTorch `DataLoader` (ImageFolder + PIL, 8 workers, pinned, persistent) | 5.11 s | ~1,835 |

**1.17× faster end-to-end.** The GPU compute floor (same 73 steps on resident data) is
3.81 s: TurboLoader trains ~0.5 s above the floor — the input pipeline is nearly fully
hidden (its loader-only epoch is 0.83 s) — while PyTorch leaves ~1.3 s/epoch of input work
exposed. On a faster GPU, larger images, or a smaller model the exposed gap widens toward
the raw loader-throughput difference.

**Oversubscription note (applies to any loader):** with torch's default intraop CPU pool
spinning on all cores, epoch time inflated ~40% for either pipeline (decode threads and
torch threads fight for cores). `torch.set_num_threads(1)` is standard practice for
GPU-bound training and is applied identically to both sides.

**Found during this benchmark (honesty log):** a `.DS_Store` stray in the dataset dir
shifted ImageFolder-style class indices to 1..10 → CUDA device assert; the harness now
filters non-directories and asserts 10 classes.
