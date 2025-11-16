#!/usr/bin/env python3
"""
DataLoader Comparison Example

Side-by-side comparison of TurboLoader vs PyTorch DataLoader.
Demonstrates the speedup and ease of migration.
"""

import sys
import time
import argparse

sys.path.insert(0, 'build/python')
import turboloader

import torch
from torch.utils.data import Dataset, DataLoader
import tarfile
from PIL import Image
import io
import torchvision.transforms as transforms


class TarDataset(Dataset):
    """PyTorch Dataset for loading from TAR files"""
    def __init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform

        # Build index
        print("Building TAR index...")
        self.samples = []
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith(('.jpg', '.jpeg', '.JPEG', '.png')):
                    self.samples.append(member.name)
        print(f"Found {len(self.samples)} images in TAR")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Extract single file from TAR
        with tarfile.open(self.tar_path, 'r') as tar:
            member = tar.getmember(self.samples[idx])
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read())).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Dummy label
        label = 0
        return img, label


def benchmark_pytorch(tar_path, num_workers, batch_size, num_batches):
    """Benchmark PyTorch DataLoader"""
    print("\n" + "=" * 80)
    print("PYTORCH DATALOADER BENCHMARK")
    print("=" * 80)

    # Standard ImageNet transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = TarDataset(tar_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    print(f"\nConfiguration:")
    print(f"  Workers: {num_workers}")
    print(f"  Batch size: {batch_size}")

    # Warmup
    print("\nWarming up (5 batches)...")
    for i, (images, labels) in enumerate(dataloader):
        if i >= 5:
            break

    # Benchmark
    print(f"\nBenchmarking {num_batches} batches...")
    total_samples = 0
    batch_times = []

    start_time = time.perf_counter()

    for i, (images, labels) in enumerate(dataloader):
        batch_start = time.perf_counter()

        total_samples += images.size(0)

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_batches} batches...")

        if i + 1 >= num_batches:
            break

    total_time = time.perf_counter() - start_time

    return {
        'framework': 'PyTorch DataLoader',
        'total_samples': total_samples,
        'total_time': total_time,
        'throughput': total_samples / total_time,
        'avg_batch_time': sum(batch_times) / len(batch_times),
        'batch_times': batch_times
    }


def benchmark_turboloader(tar_path, num_workers, batch_size, num_batches):
    """Benchmark TurboLoader"""
    print("\n" + "=" * 80)
    print("TURBOLOADER BENCHMARK")
    print("=" * 80)

    # Configure TurboLoader
    transform_config = turboloader.TransformConfig()
    transform_config.target_width = 224
    transform_config.target_height = 224
    transform_config.resize_mode = "bilinear"
    transform_config.normalize = True
    transform_config.mean = [0.485, 0.456, 0.406]
    transform_config.std = [0.229, 0.224, 0.225]
    transform_config.to_chw = True

    config = turboloader.Config()
    config.num_workers = num_workers
    config.queue_size = 512
    config.decode_jpeg = True
    config.enable_simd_transforms = True
    config.transform_config = transform_config

    pipeline = turboloader.Pipeline([tar_path], config)

    print(f"\nConfiguration:")
    print(f"  Workers: {num_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Queue size: 512")
    print(f"  SIMD: Enabled")

    # Start pipeline
    pipeline.start()

    # Warmup
    print("\nWarming up (5 batches)...")
    for _ in range(5):
        batch = pipeline.next_batch(batch_size)

    # Benchmark
    print(f"\nBenchmarking {num_batches} batches...")
    total_samples = 0
    batch_times = []

    start_time = time.perf_counter()

    for i in range(num_batches):
        batch_start = time.perf_counter()

        batch = pipeline.next_batch(batch_size)
        total_samples += len(batch)

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_batches} batches...")

    total_time = time.perf_counter() - start_time

    pipeline.stop()

    return {
        'framework': 'TurboLoader',
        'total_samples': total_samples,
        'total_time': total_time,
        'throughput': total_samples / total_time,
        'avg_batch_time': sum(batch_times) / len(batch_times),
        'batch_times': batch_times
    }


def print_comparison(pytorch_results, turboloader_results):
    """Print comparison summary"""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nThroughput:")
    print(f"  PyTorch:     {pytorch_results['throughput']:>12,.2f} images/sec")
    print(f"  TurboLoader: {turboloader_results['throughput']:>12,.2f} images/sec")
    speedup = turboloader_results['throughput'] / pytorch_results['throughput']
    print(f"  Speedup:     {speedup:>12.2f}x faster")

    print(f"\nAverage Batch Time:")
    pytorch_batch_ms = pytorch_results['avg_batch_time'] * 1000
    turbo_batch_ms = turboloader_results['avg_batch_time'] * 1000
    print(f"  PyTorch:     {pytorch_batch_ms:>12.2f}ms")
    print(f"  TurboLoader: {turbo_batch_ms:>12.2f}ms")
    improvement = pytorch_batch_ms / turbo_batch_ms
    print(f"  Improvement: {improvement:>12.2f}x faster")

    print(f"\nTotal Processing Time:")
    print(f"  PyTorch:     {pytorch_results['total_time']:>12.2f}s")
    print(f"  TurboLoader: {turboloader_results['total_time']:>12.2f}s")
    time_saved = pytorch_results['total_time'] - turboloader_results['total_time']
    print(f"  Time saved:  {time_saved:>12.2f}s")

    print("\n" + "=" * 80)
    print(f"TurboLoader is {speedup:.1f}x FASTER!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compare TurboLoader vs PyTorch DataLoader')
    parser.add_argument('tar_path', type=str, help='Path to TAR file')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers (default: 8)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to benchmark (default: 100)')
    parser.add_argument('--skip-pytorch', action='store_true',
                        help='Skip PyTorch benchmark (TurboLoader only)')

    args = parser.parse_args()

    print("=" * 80)
    print("DATALOADER COMPARISON")
    print("=" * 80)
    print(f"\nDataset: {args.tar_path}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num batches: {args.num_batches}")

    # Run benchmarks
    if not args.skip_pytorch:
        pytorch_results = benchmark_pytorch(
            args.tar_path, args.workers, args.batch_size, args.num_batches
        )
    else:
        pytorch_results = None
        print("\nSkipping PyTorch benchmark (--skip-pytorch specified)")

    turboloader_results = benchmark_turboloader(
        args.tar_path, args.workers, args.batch_size, args.num_batches
    )

    # Print comparison
    if pytorch_results:
        print_comparison(pytorch_results, turboloader_results)
    else:
        print("\n" + "=" * 80)
        print("TURBOLOADER RESULTS")
        print("=" * 80)
        print(f"\nThroughput: {turboloader_results['throughput']:,.2f} images/sec")
        print(f"Avg batch time: {turboloader_results['avg_batch_time'] * 1000:.2f}ms")
        print("=" * 80)


if __name__ == "__main__":
    main()
