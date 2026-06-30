"""CUDA loader head-to-head on the 3090: TurboLoader (CPU-out + GPU-out) vs PyTorch vs DALI.
Real consumption (force materialization), warmup + median of timed epochs."""
import glob, sys, time
import numpy as np

ROOT = "/home/arnav/data/imagenette2-160/train"
BS, SZ, NW = 64, 160, 8
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
paths = sorted(glob.glob(ROOT + "/*/*.JPEG"))[:N]
print(f"{len(paths)} images | batch {BS} | {SZ}x{SZ} | {NW} workers\n")

def timeit(make_iter, epochs=3, warmup=1):
    rates = []
    for e in range(warmup + epochs):
        n = 0; t0 = time.perf_counter()
        for cnt in make_iter():
            n += cnt
        dt = time.perf_counter() - t0
        if e >= warmup:
            rates.append(n / dt)
    return float(np.median(rates))

results = {}

try:
    import turboloader as t
    def tbl():
        for b in t.CudaImageLoader(paths, batch_size=BS, image_size=SZ, num_workers=NW,
                                   mean=MEAN, std=STD, decode="gpu"):
            float(b.sum()); yield b.shape[0]
    results["TurboLoader-CUDA (->cpu numpy)"] = timeit(tbl); print("TurboLoader-CPU done")
except Exception as e:
    print("TurboLoader-CPU failed:", repr(e)[:200])

try:
    import turboloader as t, torch
    def tblg():
        for arr in t.CudaImageLoader(paths, batch_size=BS, image_size=SZ, num_workers=NW,
                                     mean=MEAN, std=STD, decode="gpu", gpu_output=True):
            x = torch.as_tensor(arr, device="cuda"); float(x.sum()); yield x.shape[0]
    results["TurboLoader-CUDA (gpu-resident)"] = timeit(tblg); print("TurboLoader-GPU done")
except Exception as e:
    print("TurboLoader-GPU failed:", repr(e)[:200])

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((SZ, SZ)), T.ToTensor(), T.Normalize(MEAN, STD)])
    class DS(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i): return tfm(Image.open(paths[i]).convert("RGB")), 0
    def pt():
        for x, _ in DataLoader(DS(), batch_size=BS, num_workers=NW, shuffle=False):
            float(x.sum()); yield x.shape[0]
    results["PyTorch DataLoader (PIL cpu)"] = timeit(pt); print("PyTorch done")
except Exception as e:
    print("PyTorch failed:", repr(e)[:200])

try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    @pipeline_def(batch_size=BS, num_threads=NW, device_id=0)
    def pipe():
        jpegs, labels = fn.readers.file(file_root=ROOT, random_shuffle=False, name="Reader")
        img = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        img = fn.resize(img, resize_x=SZ, resize_y=SZ)
        img = fn.crop_mirror_normalize(img, dtype=types.FLOAT, output_layout="CHW",
                                       mean=[m*255 for m in MEAN], std=[s*255 for s in STD])
        return img, labels
    def dali():
        p = pipe(); p.build()
        it = DALIGenericIterator([p], ["x", "y"], size=len(paths), auto_reset=True)
        for d in it:
            x = d[0]["x"]; float(x.sum()); yield x.shape[0]
    results["DALI (gpu decode->gpu)"] = timeit(dali); print("DALI done")
except Exception as e:
    print("DALI failed:", repr(e)[:300])

# FFCV (only if installed; needs a .beton conversion done separately)
try:
    import ffcv  # noqa
    print("(FFCV installed — add .beton benchmark separately)")
except Exception:
    pass

print("\n=== RESULTS (img/s, higher = better) ===")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"  {k:34s} {v:9.0f} img/s")
