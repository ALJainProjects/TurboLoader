#!/usr/bin/env python3
"""
PyTorch DataLoader Throughput Benchmark

Measures raw throughput (images/second) for PyTorch's standard DataLoader.
This serves as the baseline for comparison.
"""

import argparse
import gc
import json
import os
import sys
import tarfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    import torchvision.transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    library: str
    metric: str
    value: float
    unit: str
    config: Dict[str, Any]
    timestamp: str
    extra: Optional[Dict[str, Any]] = None


class TarDataset(Dataset):
    """PyTorch Dataset that reads from a TAR archive"""

    def __init__(self, tar_path: str, transform=None):
        self.tar_path = tar_path
        self.transform = transform
        self.samples = []

        # Index the TAR file
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG')):
                    self.samples.append(member.name)

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]

        # Read from TAR
        with tarfile.open(self.tar_path, 'r') as tar:
            member = tar.getmember(filename)
            f = tar.extractfile(member)
            data = f.read()

        # Decode image
        if HAS_TORCHVISION:
            img = Image.open(BytesIO(data)).convert('RGB')
            if self.transform:
                img = self.transform(img)
        else:
            img = np.frombuffer(data, dtype=np.uint8)

        # Simple label from filename
        label = idx % 1000

        return img, label


class CachedTarDataset(Dataset):
    """
    PyTorch Dataset that caches TAR contents in memory.
    More comparable to TurboLoader's approach.
    """

    def __init__(self, tar_path: str, transform=None):
        self.transform = transform
        self.samples = []
        self.data_cache = {}

        print("Loading TAR into memory...")
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG')):
                    f = tar.extractfile(member)
                    self.data_cache[member.name] = f.read()
                    self.samples.append(member.name)

        self.samples.sort()
        print(f"Loaded {len(self.samples)} images into memory")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        data = self.data_cache[filename]

        if HAS_TORCHVISION:
            img = Image.open(BytesIO(data)).convert('RGB')
            if self.transform:
                img = self.transform(img)
        else:
            img = np.frombuffer(data, dtype=np.uint8)

        label = idx % 1000
        return img, label


def warmup_loader(loader, num_batches: int = 5):
    """Warmup the loader"""
    count = 0
    for batch in loader:
        count += 1
        if count >= num_batches:
            break


def benchmark_throughput(
    loader,
    num_batches: int = 100,
    batch_size: int = 64
) -> Dict[str, float]:
    """Measure throughput in images/second"""
    gc.collect()
    if HAS_TORCH:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total_images = 0
    batch_times = []

    start = time.perf_counter()

    for i, (images, labels) in enumerate(loader):
        batch_start = time.perf_counter()

        if isinstance(images, torch.Tensor):
            batch_count = images.shape[0]
        else:
            batch_count = len(images)

        total_images += batch_count
        batch_times.append(time.perf_counter() - batch_start)

        if i >= num_batches - 1:
            break

    elapsed = time.perf_counter() - start

    return {
        'throughput': total_images / elapsed,
        'total_time': elapsed,
        'total_images': total_images,
        'avg_batch_time': np.mean(batch_times) * 1000,
        'p50_batch_time': np.percentile(batch_times, 50) * 1000,
        'p95_batch_time': np.percentile(batch_times, 95) * 1000,
        'p99_batch_time': np.percentile(batch_times, 99) * 1000,
    }


def run_pytorch_benchmark(
    tar_path: str,
    batch_sizes: List[int] = [32, 64, 128],
    num_workers_list: List[int] = [0, 1, 2, 4, 8],
    num_batches: int = 100,
    with_transforms: bool = True,
    cached: bool = False
) -> List[BenchmarkResult]:
    """Run PyTorch DataLoader benchmarks"""

    if not HAS_TORCH:
        print("Error: PyTorch not installed")
        return []

    results = []
    timestamp = datetime.now().isoformat()

    print(f"\n{'='*60}")
    print(f"PyTorch DataLoader Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {tar_path}")
    print(f"Batches per run: {num_batches}")
    print(f"With transforms: {with_transforms}")
    print(f"Cached mode: {cached}")

    # Create transforms
    if with_transforms and HAS_TORCHVISION:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        transform = T.ToTensor() if HAS_TORCHVISION else None

    # Create dataset once for cached mode
    if cached:
        dataset = CachedTarDataset(tar_path, transform=transform)
    else:
        dataset = TarDataset(tar_path, transform=transform)

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            print(f"\n--- Batch Size: {batch_size}, Workers: {num_workers} ---")

            config = {
                'batch_size': batch_size,
                'num_workers': num_workers,
                'with_transforms': with_transforms,
                'cached': cached,
                'dataset': os.path.basename(tar_path),
            }

            try:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    pin_memory=True if torch.cuda.is_available() else False,
                    persistent_workers=num_workers > 0,
                )

                # Warmup
                print("  Warming up...")
                warmup_loader(loader, num_batches=5)

                # Benchmark
                print("  Running benchmark...")
                stats = benchmark_throughput(loader, num_batches, batch_size)

                print(f"  Throughput: {stats['throughput']:.1f} images/sec")
                print(f"  Avg batch time: {stats['avg_batch_time']:.2f} ms")
                print(f"  P95 batch time: {stats['p95_batch_time']:.2f} ms")

                results.append(BenchmarkResult(
                    library='pytorch' + ('_cached' if cached else ''),
                    metric='throughput',
                    value=stats['throughput'],
                    unit='images/sec',
                    config=config,
                    timestamp=timestamp,
                    extra=stats
                ))

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    return results


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    data = [asdict(r) for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[BenchmarkResult]):
    """Print a summary table"""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Library':>15} {'Batch':>8} {'Workers':>8} {'Throughput':>15}")
    print(f"{'':>15} {'Size':>8} {'':>8} {'(img/sec)':>15}")
    print("-" * 70)

    for r in results:
        if r.metric == 'throughput':
            lib = r.library
            batch = r.config.get('batch_size', 'N/A')
            workers = r.config.get('num_workers', 'N/A')
            throughput = r.value
            print(f"{lib:>15} {batch:>8} {workers:>8} {throughput:>15.1f}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch DataLoader Benchmark')
    parser.add_argument('--tar-path', type=str, required=True,
                        help='Path to TAR dataset')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 64, 128],
                        help='Batch sizes to test')
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 1, 2, 4, 8],
                        help='Number of workers to test')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches per benchmark')
    parser.add_argument('--no-transforms', action='store_true',
                        help='Run without transforms')
    parser.add_argument('--cached', action='store_true',
                        help='Use cached dataset (loads TAR into memory)')
    parser.add_argument('--output', type=str,
                        default='benchmarks/results/throughput/pytorch.json',
                        help='Output path for results')

    args = parser.parse_args()

    if not HAS_TORCH:
        print("Error: PyTorch is required for this benchmark")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    # Run standard benchmark
    results = run_pytorch_benchmark(
        args.tar_path,
        batch_sizes=args.batch_sizes,
        num_workers_list=args.workers,
        num_batches=args.num_batches,
        with_transforms=not args.no_transforms,
        cached=args.cached
    )

    # Save and print results
    save_results(results, args.output)
    print_summary(results)


if __name__ == '__main__':
    main()
