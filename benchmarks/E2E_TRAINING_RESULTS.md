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

## GPU-resident variant (CudaResidentLoader, build-from-source CUDA path)

Same ResNet-18/Imagenette fine-tune fed from GPU-resident uint8
(`examples/finetune_resnet_residentloader.py`, normalize-only pipeline — no random crop,
so a lighter recipe than the augmented runs above):

| Input pipeline | median epoch | notes |
|---|---:|---|
| **CudaResidentLoader** (zero H2D/epoch) | **3.42 s** | + 9.4 s one-time decode+upload (727 MB) |
| TurboLoader CPU pipeline (augmented) | 4.37 s | |
| PyTorch DataLoader (augmented) | 5.11 s | |

Loss 1.72 → 0.77 over 5 epochs (real training). The loader contributes ~zero per-epoch
overhead; the one-time upload amortizes vs PyTorch in ~11 epochs.
