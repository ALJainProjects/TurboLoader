#!/usr/bin/env python3
"""
GPU Decode Benchmark: TurboLoader vs DALI vs FFCV

Compares GPU-accelerated JPEG decoding performance across frameworks.

Requirements:
- NVIDIA GPU with CUDA
- TurboLoader compiled with -DTURBOLOADER_WITH_CUDA=ON
- NVIDIA DALI: pip install nvidia-dali-cuda110
- FFCV: pip install ffcv

Usage:
    python gpu_decode_benchmark.py /path/to/dataset.tar --batch-size 64 --num-workers 4
"""

import argparse
import time
import sys
import os
from pathlib import Path

# Try importing GPU-enabled TurboLoader
try:
    sys.path.insert(0, 'build/python')
    import turboloader
    TURBOLOADER_AVAILABLE = turboloader.gpu_available()
except Exception as e:
    print(f"Warning: TurboLoader GPU not available: {e}")
    TURBOLOADER_AVAILABLE = False

# Try importing DALI
try:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
except ImportError:
    print("Warning: NVIDIA DALI not installed")
    DALI_AVAILABLE = False

# Try importing FFCV
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import SimpleRGBImageDecoder
    FFCV_AVAILABLE = True
except ImportError:
    print("Warning: FFCV not installed")
    FFCV_AVAILABLE = False

import torch
import numpy as np


def benchmark_turboloader_gpu(tar_path, batch_size, num_workers, num_epochs=3):
    """Benchmark TurboLoader with GPU decode"""
    if not TURBOLOADER_AVAILABLE:
        print("  [SKIP] TurboLoader GPU not available")
        return None

    print(f"\n=== TurboLoader GPU Decode ===")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")

    # Create pipeline with GPU decode enabled
    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path],
        num_workers=num_workers,
        decode_jpeg=True,
        gpu_decode=True,  # Enable GPU decoding
        device_id=0
    )

    total_samples = 0
    total_time = 0.0

    for epoch in range(num_epochs):
        pipeline.start()
        epoch_start = time.time()
        epoch_samples = 0

        while True:
            batch = pipeline.next_batch(batch_size)
            if len(batch) == 0:
                break

            # Get GPU tensors directly (zero-copy)
            for sample in batch:
                gpu_tensor = sample.get_gpu_tensor()  # Returns CUDA tensor
                epoch_samples += 1

        epoch_time = time.time() - epoch_start
        total_samples += epoch_samples
        total_time += epoch_time

        throughput = epoch_samples / epoch_time
        print(f"  Epoch {epoch+1}: {throughput:.2f} img/s ({epoch_samples} images in {epoch_time:.2f}s)")

        pipeline.stop()

    avg_throughput = total_samples / total_time
    print(f"  Average: {avg_throughput:.2f} img/s")

    return {
        'framework': 'TurboLoader (GPU)',
        'throughput': avg_throughput,
        'batch_size': batch_size,
        'num_workers': num_workers
    }


def benchmark_dali(tar_path, batch_size, num_workers, num_epochs=3):
    """Benchmark NVIDIA DALI with GPU decode"""
    if not DALI_AVAILABLE:
        print("  [SKIP] NVIDIA DALI not available")
        return None

    print(f"\n=== NVIDIA DALI GPU Decode ===")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")

    @pipeline_def
    def create_dali_pipeline():
        # Read from TAR file
        jpegs, labels = fn.readers.file(
            file_root=os.path.dirname(tar_path),
            files=[os.path.basename(tar_path)],
            random_shuffle=False
        )
        # Decode on GPU
        images = fn.decoders.image(jpegs, device='mixed')  # GPU decode
        return images, labels

    # Create pipeline
    pipe = create_dali_pipeline(
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0
    )
    pipe.build()

    total_samples = 0
    total_time = 0.0

    for epoch in range(num_epochs):
        pipe.reset()
        epoch_start = time.time()
        epoch_samples = 0

        while True:
            try:
                batch = pipe.run()
                images = batch[0].as_cpu()  # Transfer to CPU for comparison
                epoch_samples += len(images)
            except StopIteration:
                break

        epoch_time = time.time() - epoch_start
        total_samples += epoch_samples
        total_time += epoch_time

        throughput = epoch_samples / epoch_time
        print(f"  Epoch {epoch+1}: {throughput:.2f} img/s ({epoch_samples} images in {epoch_time:.2f}s)")

    avg_throughput = total_samples / total_time
    print(f"  Average: {avg_throughput:.2f} img/s")

    return {
        'framework': 'DALI (GPU)',
        'throughput': avg_throughput,
        'batch_size': batch_size,
        'num_workers': num_workers
    }


def benchmark_ffcv(beton_path, batch_size, num_workers, num_epochs=3):
    """Benchmark FFCV (requires pre-converted .beton file)"""
    if not FFCV_AVAILABLE:
        print("  [SKIP] FFCV not available")
        return None

    if not Path(beton_path).exists():
        print(f"  [SKIP] FFCV requires .beton file (not found: {beton_path})")
        print(f"  To create: ffcv write --dataset imagefolder /data /tmp/dataset.beton")
        return None

    print(f"\n=== FFCV GPU Decode ===")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")

    # Create FFCV loader
    loader = Loader(
        beton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        pipelines={'image': [SimpleRGBImageDecoder()]}
    )

    total_samples = 0
    total_time = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_samples = 0

        for batch in loader:
            images = batch['image']
            epoch_samples += len(images)

        epoch_time = time.time() - epoch_start
        total_samples += epoch_samples
        total_time += epoch_time

        throughput = epoch_samples / epoch_time
        print(f"  Epoch {epoch+1}: {throughput:.2f} img/s ({epoch_samples} images in {epoch_time:.2f}s)")

    avg_throughput = total_samples / total_time
    print(f"  Average: {avg_throughput:.2f} img/s")

    return {
        'framework': 'FFCV',
        'throughput': avg_throughput,
        'batch_size': batch_size,
        'num_workers': num_workers
    }


def print_summary(results):
    """Print comparison summary"""
    print("\n" + "="*70)
    print("GPU DECODE BENCHMARK SUMMARY")
    print("="*70)

    if not results:
        print("No results to display")
        return

    # Sort by throughput
    results = sorted(results, key=lambda x: x['throughput'], reverse=True)

    print(f"\n{'Framework':<20} {'Throughput':<15} {'Batch':<10} {'Workers':<10}")
    print("-"*70)

    baseline = results[0]['throughput']
    for r in results:
        speedup = r['throughput'] / baseline
        print(f"{r['framework']:<20} {r['throughput']:>10.2f} img/s  "
              f"{r['batch_size']:<10} {r['num_workers']:<10}  ({speedup:.2f}x)")

    print("\nKey Insights:")
    print(f"  - Fastest: {results[0]['framework']} at {results[0]['throughput']:.0f} img/s")
    if len(results) > 1:
        slowest = results[-1]
        fastest = results[0]
        speedup = fastest['throughput'] / slowest['throughput']
        print(f"  - Speedup: {speedup:.2f}x faster than {slowest['framework']}")


def main():
    parser = argparse.ArgumentParser(description='GPU Decode Benchmark')
    parser.add_argument('tar_path', help='Path to TAR dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--ffcv-path', default=None, help='Path to FFCV .beton file (optional)')
    parser.add_argument('--skip-turboloader', action='store_true', help='Skip TurboLoader')
    parser.add_argument('--skip-dali', action='store_true', help='Skip DALI')
    parser.add_argument('--skip-ffcv', action='store_true', help='Skip FFCV')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU benchmarks require NVIDIA GPU with CUDA.")
        sys.exit(1)

    print("="*70)
    print("GPU JPEG DECODE BENCHMARK")
    print("="*70)
    print(f"Dataset: {args.tar_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Epochs: {args.epochs}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("="*70)

    results = []

    # Benchmark TurboLoader GPU
    if not args.skip_turboloader:
        result = benchmark_turboloader_gpu(
            args.tar_path,
            args.batch_size,
            args.num_workers,
            args.epochs
        )
        if result:
            results.append(result)

    # Benchmark DALI
    if not args.skip_dali:
        result = benchmark_dali(
            args.tar_path,
            args.batch_size,
            args.num_workers,
            args.epochs
        )
        if result:
            results.append(result)

    # Benchmark FFCV
    if not args.skip_ffcv:
        beton_path = args.ffcv_path or args.tar_path.replace('.tar', '.beton')
        result = benchmark_ffcv(
            beton_path,
            args.batch_size,
            args.num_workers,
            args.epochs
        )
        if result:
            results.append(result)

    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
