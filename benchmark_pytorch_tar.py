#!/usr/bin/env python3
"""
Benchmark PyTorch DataLoader with TAR dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
import tarfile
import io
from PIL import Image
import time
import sys
import numpy as np

class TarDataset(Dataset):
    def __init__(self, tar_path):
        self.tar_path = tar_path
        # Open TAR to get member list
        with tarfile.open(tar_path, 'r') as tar:
            self.members = [m.name for m in tar.getmembers() if m.name.endswith('.jpg')]
        print(f"Found {len(self.members)} JPEG images in TAR")
        self._tar = None

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        # Open TAR lazily per worker (avoids pickle issues)
        if self._tar is None:
            self._tar = tarfile.open(self.tar_path, 'r')

        member_name = self.members[idx]
        member = self._tar.getmember(member_name)
        f = self._tar.extractfile(member)
        img = Image.open(io.BytesIO(f.read()))
        # Convert to tensor
        img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_array

if __name__ == '__main__':
    tar_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/benchmark_1k.tar'
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    print("=" * 80)
    print("PYTORCH DATALOADER BENCHMARK (TAR Dataset)")
    print("=" * 80)
    print(f"TAR file: {tar_path}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    # Create dataset and dataloader
    dataset = TarDataset(tar_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )

    # Warmup
    print("\nWarming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break

    # Benchmark
    print("\nRunning benchmark...")
    start_time = time.perf_counter()
    total_images = 0
    num_batches = 0

    for batch in dataloader:
        total_images += len(batch)
        num_batches += 1
        if num_batches % 20 == 0:
            elapsed = time.perf_counter() - start_time
            throughput = total_images / elapsed
            print(f"  Processed {num_batches} batches, {total_images} images @ {throughput:.1f} img/s")

    total_time = time.perf_counter() - start_time
    throughput = total_images / total_time

    print("\n" + "=" * 80)
    print("PYTORCH RESULTS")
    print("=" * 80)
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} img/s")
    print(f"Batch time: {(total_time / num_batches) * 1000:.2f}ms")
    print("=" * 80)
