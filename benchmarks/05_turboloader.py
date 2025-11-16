#!/usr/bin/env python3
"""
TurboLoader Benchmark

Measures the performance of TurboLoader's optimized C++ data loading pipeline.

TurboLoader Features:
- C++ multi-threaded pipeline (TAR reading, JPEG decoding, batching)
- Lock-free SPSC queues for zero-copy data transfer
- SIMD-optimized image transforms
- Efficient TAR streaming (no file extraction)
- Memory-mapped I/O for maximum throughput

This benchmark demonstrates TurboLoader's design goal: maximize data throughput
to ensure the GPU is never starved waiting for data.
"""

import os
import sys
import time
import json
import argparse
import psutil
from pathlib import Path
from typing import Dict, Any

try:
    import numpy as np
    import torch
except ImportError:
    print("Error: numpy and torch required")
    print("Install with: pip install numpy torch")
    sys.exit(1)

# Import TurboLoader
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from _turboloader import DataLoader as TurboDataLoader
except ImportError:
    print("Error: TurboLoader not found")
    print("Build TurboLoader first: cd build && cmake .. && make")
    sys.exit(1)


def run_benchmark(tar_path: str,
                  batch_size: int = 32,
                  num_workers: int = 8,
                  num_epochs: int = 3) -> Dict[str, Any]:
    """
    Run TurboLoader benchmark.

    Args:
        tar_path: Path to TAR file containing images
        batch_size: Batch size
        num_workers: Number of worker threads
        num_epochs: Number of epochs

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("TURBOLOADER BENCHMARK")
    print("="*80)
    print(f"Dataset: {tar_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print(f"Features: C++ pipeline, lock-free queues, SIMD transforms")
    print("="*80)

    # Track metrics
    epoch_times = []
    batch_times = []
    memory_usage = []
    process = psutil.Process()

    # Get initial memory baseline
    initial_mem = process.memory_info().rss / (1024 * 1024)

    # Run benchmark
    total_start = time.time()

    for epoch in range(num_epochs):
        # Create TurboLoader (fresh for each epoch to match PyTorch DataLoader behavior)
        loader = TurboDataLoader(
            data_path=tar_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

        epoch_start = time.time()
        batch_count = 0
        sample_count = 0

        # Iterate through batches
        while not loader.is_finished():
            batch_start = time.time()

            # Get next batch
            batch = loader.next_batch()

            if not batch:
                break

            # Simulate model forward pass (access the data)
            for sample in batch:
                if 'image' in sample:
                    # Access image data
                    img = sample['image']
                    if isinstance(img, np.ndarray):
                        _ = img.mean()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            batch_count += 1
            sample_count += len(batch)

            # Track memory every 10 batches
            if batch_count % 10 == 0:
                mem_info = process.memory_info()
                memory_usage.append(mem_info.rss / (1024 * 1024))  # MB

        # Stop loader
        loader.stop()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        throughput = sample_count / epoch_time if epoch_time > 0 else 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Batches: {batch_count}")
        print(f"  Samples: {sample_count}")
        print(f"  Throughput: {throughput:.1f} images/sec")

    total_time = time.time() - total_start

    # Calculate total samples
    total_samples = sum([e * (sample_count // num_epochs) for e in range(num_epochs)]) if num_epochs > 0 else sample_count * num_epochs

    # Calculate statistics
    results = {
        'framework': 'TurboLoader',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'backend': 'C++ multi-threaded pipeline',
        'total_time': total_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'std_epoch_time': np.std(epoch_times) if epoch_times else 0,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'std_batch_time': np.std(batch_times) if batch_times else 0,
        'throughput': (sample_count * num_epochs) / total_time if total_time > 0 else 0,
        'peak_memory_mb': max(memory_usage) if memory_usage else initial_mem,
        'avg_memory_mb': np.mean(memory_usage) if memory_usage else initial_mem,
    }

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average epoch time: {results['avg_epoch_time']:.2f}s ± {results['std_epoch_time']:.2f}s")
    print(f"Average batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
    print(f"Throughput: {results['throughput']:.1f} images/sec")
    print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    print(f"Average memory: {results['avg_memory_mb']:.1f} MB")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='TurboLoader benchmark')
    parser.add_argument('tar_path', type=str,
                       help='Path to TAR file containing images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-w', type=int, default=8,
                       help='Number of worker threads (default: 8)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Validate TAR file exists
    if not os.path.exists(args.tar_path):
        print(f"Error: TAR file not found: {args.tar_path}")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        tar_path=args.tar_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
