#!/usr/bin/env python3
"""
Benchmark: TurboLoader DataLoader vs FastDataLoader

Compares:
1. DataLoader (original) - returns list of dicts
2. FastDataLoader (v2.5.0) - returns batched numpy arrays
3. FastDataLoader + next_batch_torch() - returns PyTorch tensors directly

Usage:
    python benchmarks/benchmark_fastdataloader.py <tar_file> [--batch-size 32] [--workers 4] [--epochs 3]

Example:
    python benchmarks/benchmark_fastdataloader.py /tmp/benchmark.tar --batch-size 64 --workers 8 --epochs 5
"""

import argparse
import time
import gc
import sys
import os

# Add turboloader to path if running from repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def benchmark_dataloader(tar_path, batch_size, num_workers, num_epochs):
    """Benchmark original DataLoader (list of dicts)."""
    import turboloader

    total_images = 0
    total_time = 0

    for epoch in range(num_epochs):
        loader = turboloader.DataLoader(
            tar_path,
            batch_size=batch_size,
            num_workers=num_workers
        )

        epoch_images = 0
        start = time.perf_counter()

        for batch in loader:
            epoch_images += len(batch)

        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_images += epoch_images

        print(f"  Epoch {epoch + 1}: {epoch_images:,} images in {elapsed:.2f}s = {epoch_images/elapsed:,.0f} img/s")
        gc.collect()

    return total_images, total_time


def benchmark_fastdataloader_numpy(tar_path, batch_size, num_workers, num_epochs, target_size=None):
    """Benchmark FastDataLoader with numpy output."""
    import turboloader

    total_images = 0
    total_time = 0

    for epoch in range(num_epochs):
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'output_format': 'numpy'
        }
        if target_size:
            kwargs['target_height'] = target_size[0]
            kwargs['target_width'] = target_size[1]

        loader = turboloader.FastDataLoader(tar_path, **kwargs)

        epoch_images = 0
        start = time.perf_counter()

        for images, metadata in loader:
            epoch_images += images.shape[0]

        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_images += epoch_images

        print(f"  Epoch {epoch + 1}: {epoch_images:,} images in {elapsed:.2f}s = {epoch_images/elapsed:,.0f} img/s")
        gc.collect()

    return total_images, total_time


def benchmark_fastdataloader_torch(tar_path, batch_size, num_workers, num_epochs, target_size=None):
    """Benchmark FastDataLoader with direct PyTorch tensor output."""
    try:
        import torch
    except ImportError:
        print("  [SKIPPED] PyTorch not installed")
        return 0, 0

    import turboloader

    total_images = 0
    total_time = 0

    for epoch in range(num_epochs):
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }
        if target_size:
            kwargs['target_height'] = target_size[0]
            kwargs['target_width'] = target_size[1]

        loader = turboloader.FastDataLoader(tar_path, **kwargs)

        epoch_images = 0
        start = time.perf_counter()

        while not loader.is_finished():
            images, metadata = loader.next_batch_torch()
            epoch_images += images.shape[0]

        loader.stop()
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_images += epoch_images

        print(f"  Epoch {epoch + 1}: {epoch_images:,} images in {elapsed:.2f}s = {epoch_images/elapsed:,.0f} img/s")
        gc.collect()

    return total_images, total_time


def benchmark_fastdataloader_tf(tar_path, batch_size, num_workers, num_epochs, target_size=None):
    """Benchmark FastDataLoader with direct TensorFlow tensor output."""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
    except ImportError:
        print("  [SKIPPED] TensorFlow not installed")
        return 0, 0

    import turboloader

    total_images = 0
    total_time = 0

    for epoch in range(num_epochs):
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }
        if target_size:
            kwargs['target_height'] = target_size[0]
            kwargs['target_width'] = target_size[1]

        loader = turboloader.FastDataLoader(tar_path, **kwargs)

        epoch_images = 0
        start = time.perf_counter()

        while not loader.is_finished():
            images, metadata = loader.next_batch_tf()
            epoch_images += images.shape[0]

        loader.stop()
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_images += epoch_images

        print(f"  Epoch {epoch + 1}: {epoch_images:,} images in {elapsed:.2f}s = {epoch_images/elapsed:,.0f} img/s")
        gc.collect()

    return total_images, total_time


def main():
    parser = argparse.ArgumentParser(description='Benchmark TurboLoader DataLoader vs FastDataLoader')
    parser.add_argument('tar_file', help='Path to tar file containing images')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--target-size', type=int, nargs=2, metavar=('H', 'W'),
                        help='Target size for resize (e.g., --target-size 224 224)')
    parser.add_argument('--skip-original', action='store_true', help='Skip original DataLoader benchmark')
    parser.add_argument('--skip-torch', action='store_true', help='Skip PyTorch tensor benchmark')
    parser.add_argument('--skip-tf', action='store_true', help='Skip TensorFlow tensor benchmark')

    args = parser.parse_args()

    if not os.path.exists(args.tar_file):
        print(f"Error: tar file not found: {args.tar_file}")
        sys.exit(1)

    target_size = tuple(args.target_size) if args.target_size else None

    print("=" * 80)
    print("TurboLoader Benchmark: DataLoader vs FastDataLoader")
    print("=" * 80)
    print(f"Tar file: {args.tar_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Epochs: {args.epochs}")
    if target_size:
        print(f"Target size: {target_size[0]}x{target_size[1]}")
    print("=" * 80)

    results = {}

    # 1. Original DataLoader
    if not args.skip_original:
        print("\n[1] DataLoader (original - list of dicts)")
        print("-" * 40)
        images, elapsed = benchmark_dataloader(
            args.tar_file, args.batch_size, args.workers, args.epochs
        )
        if elapsed > 0:
            results['DataLoader'] = images / elapsed
            print(f"  TOTAL: {images:,} images in {elapsed:.2f}s = {images/elapsed:,.0f} img/s")

    # 2. FastDataLoader with numpy
    print("\n[2] FastDataLoader (numpy arrays)")
    print("-" * 40)
    images, elapsed = benchmark_fastdataloader_numpy(
        args.tar_file, args.batch_size, args.workers, args.epochs, target_size
    )
    if elapsed > 0:
        results['FastDataLoader (numpy)'] = images / elapsed
        print(f"  TOTAL: {images:,} images in {elapsed:.2f}s = {images/elapsed:,.0f} img/s")

    # 3. FastDataLoader with PyTorch tensors
    if not args.skip_torch:
        print("\n[3] FastDataLoader (PyTorch tensors)")
        print("-" * 40)
        images, elapsed = benchmark_fastdataloader_torch(
            args.tar_file, args.batch_size, args.workers, args.epochs, target_size
        )
        if elapsed > 0:
            results['FastDataLoader (torch)'] = images / elapsed
            print(f"  TOTAL: {images:,} images in {elapsed:.2f}s = {images/elapsed:,.0f} img/s")

    # 4. FastDataLoader with TensorFlow tensors
    if not args.skip_tf:
        print("\n[4] FastDataLoader (TensorFlow tensors)")
        print("-" * 40)
        images, elapsed = benchmark_fastdataloader_tf(
            args.tar_file, args.batch_size, args.workers, args.epochs, target_size
        )
        if elapsed > 0:
            results['FastDataLoader (tf)'] = images / elapsed
            print(f"  TOTAL: {images:,} images in {elapsed:.2f}s = {images/elapsed:,.0f} img/s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Configuration':<35} | {'Throughput':>15} | {'Speedup':>10}")
    print("-" * 65)

    baseline = results.get('DataLoader', results.get('FastDataLoader (numpy)', 1))

    for name, throughput in sorted(results.items(), key=lambda x: -x[1]):
        speedup = throughput / baseline if baseline > 0 else 0
        print(f"{name:<35} | {throughput:>12,.0f}/s | {speedup:>9.2f}x")

    print("-" * 65)

    if 'DataLoader' in results and 'FastDataLoader (numpy)' in results:
        speedup = results['FastDataLoader (numpy)'] / results['DataLoader']
        print(f"\nFastDataLoader is {speedup:.2f}x faster than DataLoader")


if __name__ == '__main__':
    main()
