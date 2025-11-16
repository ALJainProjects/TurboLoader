#!/usr/bin/env python3
"""
Full ImageNet Production-Scale Benchmark

Benchmarks TurboLoader vs PyTorch DataLoader on full ImageNet dataset (1.3M images).

Requirements:
1. Convert ImageNet using imagenet_converter.py first
2. Ensure sufficient RAM (16GB+ recommended)
3. SSD storage recommended for best results

Usage:
    # Single TAR file
    python3 benchmarks/full_imagenet_benchmark.py --tar-paths imagenet_train.tar

    # Sharded TARs
    python3 benchmarks/full_imagenet_benchmark.py --shard-dir imagenet_train_shards/

    # Custom settings
    python3 benchmarks/full_imagenet_benchmark.py \
        --tar-paths imagenet_train.tar \
        --num-workers 16 \
        --batch-size 256 \
        --num-batches 500
"""

import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# TurboLoader
sys.path.insert(0, 'build/python')
import turboloader

# PyTorch for comparison
try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    import tarfile
    from PIL import Image
    import io
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - will only benchmark TurboLoader")


class PyTorchImageNetDataset(IterableDataset):
    """PyTorch baseline for comparison"""

    def __init__(self, tar_paths):
        self.tar_paths = tar_paths if isinstance(tar_paths, list) else [tar_paths]

    def __iter__(self):
        for tar_path in self.tar_paths:
            with tarfile.open(tar_path, 'r') as tar:
                members = [m for m in tar.getmembers() if m.name.endswith('.jpg')]

                for member in members:
                    # Extract image
                    img_file = tar.extractfile(member)
                    img_data = img_file.read()

                    # Decode JPEG
                    img = Image.open(io.BytesIO(img_data))
                    img_array = np.array(img, dtype=np.float32) / 255.0

                    # Basic transforms
                    if img_array.shape != (224, 224, 3):
                        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                        img_pil = img_pil.resize((224, 224))
                        img_array = np.array(img_pil, dtype=np.float32) / 255.0

                    # Normalize (ImageNet stats)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_array = (img_array - mean) / std

                    # Convert to CHW
                    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

                    yield img_tensor, 0  # Dummy label


def benchmark_turboloader(tar_paths, num_workers, batch_size, num_batches):
    """Benchmark TurboLoader on ImageNet"""

    print("\n" + "=" * 80)
    print("TURBOLOADER BENCHMARK")
    print("=" * 80)

    # SIMD transform configuration
    transform_config = turboloader.TransformConfig()
    transform_config.target_width = 224
    transform_config.target_height = 224
    transform_config.resize_mode = "bilinear"
    transform_config.normalize = True
    transform_config.mean = [0.485, 0.456, 0.406]
    transform_config.std = [0.229, 0.224, 0.225]
    transform_config.to_chw = True

    # Create pipeline
    pipeline = turboloader.Pipeline(
        tar_paths=tar_paths,
        num_workers=num_workers,
        queue_size=512,
        decode_jpeg=True,
        enable_simd_transforms=True,
        transform_config=transform_config
    )

    print(f"Configuration:")
    print(f"  TAR files: {len(tar_paths)}")
    print(f"  Workers: {num_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Queue size: 512")
    print(f"  SIMD transforms: Enabled")
    print(f"  Target size: 224x224")

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        batch = pipeline.next_batch(batch_size)

    # Benchmark
    print(f"\nRunning benchmark ({num_batches} batches)...")
    batch_times = []
    total_samples = 0

    start_time = time.time()

    for i in tqdm(range(num_batches), desc="TurboLoader"):
        batch_start = time.time()
        batch = pipeline.next_batch(batch_size)
        batch_time = time.time() - batch_start

        batch_times.append(batch_time)
        total_samples += batch_size

    total_time = time.time() - start_time

    # Statistics
    throughput = total_samples / total_time
    avg_batch_time = np.mean(batch_times) * 1000  # ms
    p50_batch_time = np.percentile(batch_times, 50) * 1000
    p95_batch_time = np.percentile(batch_times, 95) * 1000
    p99_batch_time = np.percentile(batch_times, 99) * 1000

    print("\n" + "-" * 80)
    print("RESULTS:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"  Avg batch time: {avg_batch_time:.2f}ms")
    print(f"  P50 batch time: {p50_batch_time:.2f}ms")
    print(f"  P95 batch time: {p95_batch_time:.2f}ms")
    print(f"  P99 batch time: {p99_batch_time:.2f}ms")
    print("-" * 80)

    return {
        'framework': 'TurboLoader',
        'total_samples': total_samples,
        'total_time': total_time,
        'throughput': throughput,
        'avg_batch_time_ms': avg_batch_time,
        'p50_batch_time_ms': p50_batch_time,
        'p95_batch_time_ms': p95_batch_time,
        'p99_batch_time_ms': p99_batch_time,
        'batch_times': batch_times
    }


def benchmark_pytorch(tar_paths, num_workers, batch_size, num_batches):
    """Benchmark PyTorch DataLoader on ImageNet"""

    if not PYTORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available - skipping PyTorch benchmark")
        return None

    print("\n" + "=" * 80)
    print("PYTORCH DATALOADER BENCHMARK")
    print("=" * 80)

    # Create dataset and dataloader
    dataset = PyTorchImageNetDataset(tar_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"Configuration:")
    print(f"  TAR files: {len(tar_paths)}")
    print(f"  Workers: {num_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target size: 224x224")

    # Warmup
    print("\nWarming up...")
    dataloader_iter = iter(dataloader)
    for _ in range(5):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

    # Benchmark
    print(f"\nRunning benchmark ({num_batches} batches)...")
    batch_times = []
    total_samples = 0

    dataloader_iter = iter(dataloader)
    start_time = time.time()

    for i in tqdm(range(num_batches), desc="PyTorch"):
        try:
            batch_start = time.time()
            batch = next(dataloader_iter)
            batch_time = time.time() - batch_start

            batch_times.append(batch_time)
            total_samples += batch[0].shape[0]

        except StopIteration:
            print("\n⚠️  Dataset exhausted, restarting...")
            dataloader_iter = iter(dataloader)

    total_time = time.time() - start_time

    # Statistics
    throughput = total_samples / total_time
    avg_batch_time = np.mean(batch_times) * 1000
    p50_batch_time = np.percentile(batch_times, 50) * 1000
    p95_batch_time = np.percentile(batch_times, 95) * 1000
    p99_batch_time = np.percentile(batch_times, 99) * 1000

    print("\n" + "-" * 80)
    print("RESULTS:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"  Avg batch time: {avg_batch_time:.2f}ms")
    print(f"  P50 batch time: {p50_batch_time:.2f}ms")
    print(f"  P95 batch time: {p95_batch_time:.2f}ms")
    print(f"  P99 batch_time: {p99_batch_time:.2f}ms")
    print("-" * 80)

    return {
        'framework': 'PyTorch',
        'total_samples': total_samples,
        'total_time': total_time,
        'throughput': throughput,
        'avg_batch_time_ms': avg_batch_time,
        'p50_batch_time_ms': p50_batch_time,
        'p95_batch_time_ms': p95_batch_time,
        'p99_batch_time_ms': p99_batch_time,
        'batch_times': batch_times
    }


def print_comparison(turbo_results, pytorch_results):
    """Print comparison summary"""

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if pytorch_results is None:
        print("\nPyTorch benchmark not run (PyTorch not available)")
        print(f"\nTurboLoader: {turbo_results['throughput']:.2f} images/sec")
        print("=" * 80)
        return

    speedup = turbo_results['throughput'] / pytorch_results['throughput']
    batch_time_improvement = pytorch_results['avg_batch_time_ms'] / turbo_results['avg_batch_time_ms']

    print(f"\nThroughput:")
    print(f"  TurboLoader: {turbo_results['throughput']:>10,.2f} images/sec")
    print(f"  PyTorch:     {pytorch_results['throughput']:>10,.2f} images/sec")
    print(f"  Speedup:     {speedup:>10.2f}x ⚡")

    print(f"\nAverage Batch Time:")
    print(f"  TurboLoader: {turbo_results['avg_batch_time_ms']:>10.2f}ms")
    print(f"  PyTorch:     {pytorch_results['avg_batch_time_ms']:>10.2f}ms")
    print(f"  Improvement: {batch_time_improvement:>10.2f}x faster")

    print(f"\nP99 Latency:")
    print(f"  TurboLoader: {turbo_results['p99_batch_time_ms']:>10.2f}ms")
    print(f"  PyTorch:     {pytorch_results['p99_batch_time_ms']:>10.2f}ms")

    print("\n" + "=" * 80)
    print(f"🚀 TurboLoader is {speedup:.1f}× FASTER on full ImageNet!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Full ImageNet Benchmark')
    parser.add_argument('--tar-paths', type=str, nargs='+',
                       help='TAR file paths (can specify multiple)')
    parser.add_argument('--shard-dir', type=str,
                       help='Directory containing sharded TARs')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of worker threads (default: 8)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--num-batches', type=int, default=500,
                       help='Number of batches to benchmark (default: 500)')
    parser.add_argument('--output', type=str, default='benchmark_results/imagenet_benchmark.json',
                       help='Output JSON file for results')
    parser.add_argument('--skip-pytorch', action='store_true',
                       help='Skip PyTorch benchmark')

    args = parser.parse_args()

    # Determine TAR paths
    if args.shard_dir:
        shard_dir = Path(args.shard_dir)
        shard_list = shard_dir / 'shard_list.txt'

        if shard_list.exists():
            with open(shard_list) as f:
                tar_paths = [line.strip() for line in f if line.strip()]
        else:
            tar_paths = sorted(str(p) for p in shard_dir.glob('*.tar'))

        print(f"Found {len(tar_paths)} shards in {shard_dir}")

    elif args.tar_paths:
        tar_paths = args.tar_paths
        print(f"Using {len(tar_paths)} TAR file(s)")

    else:
        print("Error: Must specify --tar-paths or --shard-dir")
        return

    # Verify files exist
    for tar_path in tar_paths:
        if not Path(tar_path).exists():
            print(f"Error: TAR file not found: {tar_path}")
            return

    print("=" * 80)
    print("FULL IMAGENET PRODUCTION-SCALE BENCHMARK")
    print("=" * 80)
    print(f"Dataset: {len(tar_paths)} TAR file(s)")
    print(f"Workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches: {args.num_batches}")
    print(f"Total samples: ~{args.num_batches * args.batch_size:,}")
    print("=" * 80)

    # Run benchmarks
    turbo_results = benchmark_turboloader(
        tar_paths,
        args.num_workers,
        args.batch_size,
        args.num_batches
    )

    pytorch_results = None
    if not args.skip_pytorch:
        pytorch_results = benchmark_pytorch(
            tar_paths,
            args.num_workers,
            args.batch_size,
            args.num_batches
        )

    # Print comparison
    print_comparison(turbo_results, pytorch_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'ImageNet (Full)',
        'tar_files': len(tar_paths),
        'num_workers': args.num_workers,
        'batch_size': args.batch_size,
        'num_batches': args.num_batches,
        'turboloader': turbo_results,
        'pytorch': pytorch_results,
    }

    if pytorch_results:
        results['speedup'] = turbo_results['throughput'] / pytorch_results['throughput']

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
