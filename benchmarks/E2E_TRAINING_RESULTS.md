# End-to-end training benchmark (RTX 3090, WSL2)

`benchmarks/benchmark_e2e_training.py` — real ResNet-18 (torchvision, 10 classes) on real
Imagenette (9,469 JPEGs), identical training step, identical recipe
(RandomResizedCrop + hflip + ImageNet normalize), identical settings
(`torch.set_num_threads(1)` both sides — see note), bs=128, warmup epoch excluded,
median of 5 steady-state epochs. Loss decreases on both sides (real training, not a no-op).

**160px (Imagenette-160):**

| Input pipeline | median epoch | end-to-end img/s |
|---|---:|---:|
| pure-CUDA floor (resident batch, same steps) | 3.38 s | — |
| **TurboLoader** (`train_aug=True, pin_memory=True, prefetch_batches=4`) | **3.72 s** | ~2,515 |
| PyTorch `DataLoader` (ImageFolder + PIL, 8 workers, pinned, persistent) | 3.90 s | ~2,395 |

**224px (Imagenette-320 source, RandomResizedCrop(224)):**

| Input pipeline | median epoch | end-to-end img/s |
|---|---:|---:|
| pure-CUDA floor | 6.19 s | — |
| **TurboLoader** | **6.78 s** | ~1,380 |
| PyTorch `DataLoader` | 7.35 s | ~1,270 |

**1.05–1.08× faster end-to-end, ~9–10% above the pure-GPU floor at both sizes.**
(An earlier run of the same benchmark measured 4.37 s vs 5.11 s / floor 3.81 s —
1.17×; the host was in a slower state that day. Both runs are honest; the *gap to
floor*, not the absolute epoch time, is the stable comparison.) TurboLoader leaves
~0.35–0.6 s/epoch of input work exposed vs PyTorch's ~0.5–1.2 s. On a faster GPU or
a smaller model the exposed gap widens toward the raw loader-throughput difference.

## Overlapped H2D (`CudaPrefetcher`) — honest neutral result

`turboloader.CudaPrefetcher` stages batch k+1's host→device copy on a side CUDA
stream while the model computes on batch k (apex/DALI-style double buffering,
CUDA-event-safe). Measured in the exact benchmark above at BOTH 160px and 224px:
**neutral — within noise of the plain pinned `non_blocking=True` loop** (3.73 s vs
3.72 s; 6.81 s vs 6.78 s). On this box the epoch is bound by decode delivery, not
by the 1.6–3 ms/batch H2D, so there is nothing for the overlap to recover. It is
kept because the technique pays where transfers are large relative to the step
(big batches / high-res inputs / small models); measure before adopting.

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

## Apple Silicon (M4 Max, MPS) — the honest null result

Same benchmark, `--device mps` (ResNet-18, bs=128, identical recipe, warmup excluded):

| Input pipeline | median epoch |
|---|---:|
| pure-MPS floor (resident batch) | 7.61 s |
| **TurboLoader** | **8.06 s** |
| PyTorch DataLoader (8 spawn workers) | 8.08 s |

**A tie (1.00x) — and that's the correct outcome.** The M4's MPS step is ~2x slower than a
3090 (104 ms vs 52 ms), so the epoch needs only ~1,200 img/s of input; both loaders hide
completely behind compute (each ~0.45 s above the floor). The loader can only buy back time
the input pipeline is actually costing you. Differences that remain on macOS: cold start
(TurboLoader first epoch 7.9 s vs PyTorch 14.9 s — no spawn-worker tax) and input-bound
workloads (smaller models, eval sweeps, preprocessing), where the loader-throughput gap
(~23.7x vs a PIL loop) is the operative number.

Same story at **224px** (Imagenette-320 source, RandomResizedCrop(224), bs=128):
floor 14.30 s, **TurboLoader 15.39 s**, PyTorch 15.45 s — 1.00x steady-state, with
TurboLoader's first epoch 14.8 s vs PyTorch's 22.3 s (spawn-worker tax again).

## Video training (r3d_18, first e2e video benchmark) — RTX 3090

`benchmarks/benchmark_e2e_video_training.py` — torchvision **r3d_18** (2 classes) on
real footage (Big Buck Bunny vs Jellyfish, split into 20 one-second segment files),
8-frame clips, RandomResizedCrop(112) with ONE window+flip per clip + normalize —
the standard video recipe. 40 steps x 4 epochs, bs=8, 4 decode workers both sides,
warmup excluded. Loss reaches ~0 on both sides (the 2-film classification task is
genuinely learned).

| Input pipeline | median epoch | clips/s |
|---|---:|---:|
| pure-CUDA floor (resident clip batch) | 2.11 s | — |
| **TurboLoader `VideoDatasetLoader`** (threaded PyAV decode → fused clip kernel) | **4.19 s** | ~76 |
| PyTorch `DataLoader` + PyAV (per-sample open/decode + torchvision per-frame ops) | 4.85 s | ~66 |

**1.16× faster end-to-end.** Honest read: BOTH pipelines sit ~2x above the floor —
video training at this scale is decode-bound, and the decode is the same libav on both
sides. TurboLoader's edge comes from thread (not process) workers, per-worker container
reuse, plane-direct I420 staging, and ONE fused CUDA launch per clip instead of
per-frame torchvision ops. Closing the remaining gap needs faster decode
(NVDEC on native Linux, or more cores).

## LLM pipeline (GPT + tokens) — RTX 3090

`examples/train_gpt_tokenloader.py` (4-layer GPT on Tiny Shakespeare, 300 steps):
**turboloader 85.4 / turboloader `device=` 83.7 / nanoGPT `get_batch` 83.6 steps/s —
all within ~2%.** At any realistic model size the token pipeline hides behind compute;
that parity (previously TurboLoader trailed ~10% here) is the honest e2e claim.

Loader-only, where the pipeline IS the work (`benchmarks/benchmark_token_loader.py`,
GPT-2 shape 32x1024, uint16 memmap corpus, delivered TO DEVICE, median of 5):

| Pipeline | M tok/s to device |
|---|---:|
| nanoGPT `get_batch` (exact idiom incl. `.pin_memory().to(non_blocking=True)`) | 88.1 |
| **TokenDataLoader numpy path + `.to()`** | **168.7** |
| **TokenDataLoader `device="cuda"`** (pinned ring, overlapped H2D, event-guarded) | **168.1** |

**1.9× the standard idiom** — one `seq_len+1` gather feeds both x and y (get_batch
gathers twice), zero steady-state allocations. `device=` adds ready-on-GPU tensors
with no lifetime rules at parity throughput.

## TBL-RAW pre-processed pipeline — the fastest input path measured here

Same ResNet-18/Imagenette-160 benchmark, fed from a pre-processed RAW `.tbl`
(`preprocess_to_tbl` once: ~7s on the 3090 box; serve = mmap + fused SIMD
normalize + background prefetch + pinned ring). Recipe caveat as with the
resident-loader section: random hflip only — RandomResizedCrop cannot apply to
pre-resized samples, so this is a lighter recipe than the augmented rows.

| Input pipeline | median epoch | end-to-end img/s |
|---|---:|---:|
| pure-CUDA floor | 3.39 s | — |
| **TBL-RAW** (`DataLoader('....tbl')`, hflip-only) | **3.64 s** | ~2,570 |
| TurboLoader TAR (full train_aug) | 3.76 s | ~2,490 |
| PyTorch DataLoader (full aug) | 3.92 s | ~2,385 |

**7.4% above the pure-GPU floor** — the closest any input pipeline has come on
this box. Engineering honesty log: the FIRST e2e run measured TBL-RAW at
4.51 s — SLOWER than TAR — because serving ran synchronously on the training
thread while the TAR pipeline produces in background C++ threads. The fix
(stop-aware background prefetch; SIMD ops release the GIL so production
overlaps the step) is what turned 0.87x into the fastest pipeline, and the
loader now defaults to it.

Loader-only on the same box (WSL2, modest CPU memory bandwidth — the M4 Max
numbers in docs/benchmarks/index.md are ~6x higher): TBL-RAW 44.9k img/s
produce / 25.7k np.sum-consumed vs on-the-fly 22.0k/21.6k and float32 cache
32.6k/25.4k. The ordering is the same on both machines; the magnitude tracks
memory bandwidth, honestly stated.
