import time
import turboloader as tl

print("TurboLoader — 9,469 real ImageNet JPEGs (Imagenette-160)\n")

loader = tl.DataLoader(
    "imagenette_train.tar"  # TAR archive of JPEGs,
    batch_size=256,
    image_size=160,
    output_format="pytorch",        # (N, 3, H, W) float32, normalized
    transform=tl.ImageNetNormalize(),
    shuffle=True,
    train_aug=True,                 # fused RandomResizedCrop + flip in C++
)

n = sum(b.shape[0] for b, _ in loader)   # warmup epoch: OS page cache
print(f"  warmup:   {n:,} images (filling the OS page cache)")

for epoch in range(5):
    loader.set_epoch(epoch)
    n, t0 = 0, time.perf_counter()
    for images, meta in loader:
        n += images.shape[0]
    dt = time.perf_counter() - t0
    print(f"  epoch {epoch + 1}:  {n:,} images in {dt:.2f}s   ->  {n / dt:>7,.0f} img/s")

print("\none pip install. zero FFmpeg/DALI/beton. Apple Silicon + Linux + CUDA.")
