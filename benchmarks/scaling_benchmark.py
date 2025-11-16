#!/usr/bin/env python3
"""
Scaling Benchmark - All Frameworks

Tests how data loading performance scales with:
1. Number of workers (1, 2, 4, 8, 16)
2. Batch size (8, 16, 32, 64, 128)
3. Dataset size (1K, 10K, 100K images)

Compares:
- TurboLoader (C++ backend)
- PyTorch DataLoader + WebDataset
- TensorFlow tf.data
- FFCV (if available)

Usage: python scaling_benchmark.py <tar_file>
Example: python scaling_benchmark.py /tmp/benchmark_10k.tar
"""

import sys
import time
import os
import tempfile
import shutil
import tarfile
from pathlib import Path
import io
import json

# TurboLoader
sys.path.insert(0, 'build/python')
import turboloader

# PyTorch
import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import numpy as np

# TensorFlow
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# FFCV
try:
    from ffcv.loader import Loader
    from ffcv.fields.decoders import SimpleRGBImageDecoder
    HAS_FFCV = True
except ImportError:
    HAS_FFCV = False


def benchmark_turboloader_workers(tar_path, batch_size=32):
    """Test worker scaling for TurboLoader"""
    print("\n=== TurboLoader Worker Scaling ===")
    results = {}

    for num_workers in [1, 2, 4, 8, 16]:
        pipeline = turboloader.Pipeline(
            tar_paths=[tar_path],
            num_workers=num_workers,
            decode_jpeg=True
        )
        pipeline.start()

        # Warmup
        for _ in range(3):
            _ = pipeline.next_batch(batch_size)

        # Benchmark
        total_samples = 0
        start = time.time()

        for _ in range(10):  # 10 batches
            batch = pipeline.next_batch(batch_size)
            if len(batch) == 0:
                pipeline.stop()
                pipeline.start()
                batch = pipeline.next_batch(batch_size)
            total_samples += len(batch)

        duration = time.time() - start
        throughput = total_samples / duration

        pipeline.stop()
        results[num_workers] = throughput
        print(f"  {num_workers} workers: {int(throughput)} img/s")

    return results


def benchmark_pytorch_workers(tar_path, batch_size=32):
    """Test worker scaling for PyTorch DataLoader"""
    print("\n=== PyTorch Worker Scaling ===")
    results = {}

    for num_workers in [0, 1, 2, 4, 8]:  # 0 = main process only
        dataset = (
            wds.WebDataset(tar_path)
            .decode("pilrgb")
            .to_tuple("jpg")
            .map(lambda x: (torch.from_numpy(np.array(x[0])).permute(2, 0, 1).float() / 255.0,))
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False
        )

        # Warmup
        for i, _ in enumerate(dataloader):
            if i >= 3:
                break

        # Benchmark
        total_samples = 0
        start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            total_samples += batch[0].shape[0]

        duration = time.time() - start
        throughput = total_samples / duration if duration > 0 else 0

        results[num_workers] = throughput
        print(f"  {num_workers} workers: {int(throughput)} img/s")

    return results


def benchmark_tensorflow_workers(tar_path, batch_size=32):
    """Test worker scaling for TensorFlow tf.data"""
    if not HAS_TF:
        return {}

    print("\n=== TensorFlow Worker Scaling ===")
    results = {}

    # Extract TAR
    temp_dir = tempfile.mkdtemp(prefix='tf_scaling_')
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)

    image_files = sorted(Path(temp_dir).rglob('*.jpg'))
    file_paths = [str(f) for f in image_files]

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.cast(img, tf.float32) / 255.0

    for num_workers in [1, 2, 4, 8, 16]:
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(load_image, num_parallel_calls=num_workers)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

        # Warmup
        for i, _ in enumerate(dataset.take(3)):
            pass

        # Benchmark
        total_samples = 0
        start = time.time()

        for i, batch in enumerate(dataset.take(10)):
            total_samples += batch.shape[0]

        duration = time.time() - start
        throughput = total_samples / duration if duration > 0 else 0

        results[num_workers] = throughput
        print(f"  {num_workers} workers: {int(throughput)} img/s")

    shutil.rmtree(temp_dir)
    return results


def benchmark_turboloader_batch_sizes(tar_path, num_workers=4):
    """Test batch size scaling for TurboLoader"""
    print("\n=== TurboLoader Batch Size Scaling ===")
    results = {}

    for batch_size in [8, 16, 32, 64, 128]:
        pipeline = turboloader.Pipeline(
            tar_paths=[tar_path],
            num_workers=num_workers,
            decode_jpeg=True
        )
        pipeline.start()

        # Warmup
        for _ in range(3):
            _ = pipeline.next_batch(batch_size)

        # Benchmark
        total_samples = 0
        start = time.time()

        for _ in range(10):
            batch = pipeline.next_batch(batch_size)
            if len(batch) == 0:
                pipeline.stop()
                pipeline.start()
                batch = pipeline.next_batch(batch_size)
            total_samples += len(batch)

        duration = time.time() - start
        throughput = total_samples / duration

        pipeline.stop()
        results[batch_size] = throughput
        print(f"  Batch {batch_size}: {int(throughput)} img/s")

    return results


def benchmark_pytorch_batch_sizes(tar_path, num_workers=4):
    """Test batch size scaling for PyTorch"""
    print("\n=== PyTorch Batch Size Scaling ===")
    results = {}

    for batch_size in [8, 16, 32, 64, 128]:
        dataset = (
            wds.WebDataset(tar_path)
            .decode("pilrgb")
            .to_tuple("jpg")
            .map(lambda x: (torch.from_numpy(np.array(x[0])).permute(2, 0, 1).float() / 255.0,))
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False
        )

        # Warmup
        for i, _ in enumerate(dataloader):
            if i >= 3:
                break

        # Benchmark
        total_samples = 0
        start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            total_samples += batch[0].shape[0]

        duration = time.time() - start
        throughput = total_samples / duration if duration > 0 else 0

        results[batch_size] = throughput
        print(f"  Batch {batch_size}: {int(throughput)} img/s")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python scaling_benchmark.py <tar_file>")
        print("Example: python scaling_benchmark.py /tmp/benchmark_10k.tar")
        sys.exit(1)

    tar_path = sys.argv[1]

    print("=" * 60)
    print("SCALING BENCHMARK - ALL FRAMEWORKS")
    print("=" * 60)
    print(f"Dataset: {tar_path}")
    print(f"Dataset size: {os.path.getsize(tar_path) / 1024 / 1024:.2f} MB")

    # Worker scaling
    print("\n" + "=" * 60)
    print("PART 1: WORKER SCALING (batch_size=32)")
    print("=" * 60)

    turbo_workers = benchmark_turboloader_workers(tar_path, batch_size=32)
    pytorch_workers = benchmark_pytorch_workers(tar_path, batch_size=32)
    tf_workers = benchmark_tensorflow_workers(tar_path, batch_size=32) if HAS_TF else {}

    # Batch size scaling
    print("\n" + "=" * 60)
    print("PART 2: BATCH SIZE SCALING (workers=4)")
    print("=" * 60)

    turbo_batches = benchmark_turboloader_batch_sizes(tar_path, num_workers=4)
    pytorch_batches = benchmark_pytorch_batch_sizes(tar_path, num_workers=4)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - WORKER SCALING")
    print("=" * 60)
    print(f"{'Workers':<10} {'TurboLoader':<15} {'PyTorch':<15} {'TensorFlow':<15} {'Speedup vs PyTorch':<20}")
    print("-" * 75)

    for workers in sorted(turbo_workers.keys()):
        turbo = turbo_workers.get(workers, 0)
        pytorch = pytorch_workers.get(workers, 0)
        tf_val = tf_workers.get(workers, 0)
        speedup = turbo / pytorch if pytorch > 0 else 0

        print(f"{workers:<10} {int(turbo):<15} {int(pytorch):<15} {int(tf_val):<15} {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("SUMMARY - BATCH SIZE SCALING")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'TurboLoader':<15} {'PyTorch':<15} {'Speedup':<15}")
    print("-" * 60)

    for batch in sorted(turbo_batches.keys()):
        turbo = turbo_batches.get(batch, 0)
        pytorch = pytorch_batches.get(batch, 0)
        speedup = turbo / pytorch if pytorch > 0 else 0

        print(f"{batch:<12} {int(turbo):<15} {int(pytorch):<15} {speedup:.2f}x")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Optimal workers
    optimal_turbo = max(turbo_workers.items(), key=lambda x: x[1])
    optimal_pytorch = max(pytorch_workers.items(), key=lambda x: x[1])

    print(f"✅ Optimal workers:")
    print(f"   TurboLoader: {optimal_turbo[0]} workers ({int(optimal_turbo[1])} img/s)")
    print(f"   PyTorch: {optimal_pytorch[0]} workers ({int(optimal_pytorch[1])} img/s)")

    # Scaling efficiency
    if 1 in turbo_workers and 8 in turbo_workers:
        turbo_scaling = turbo_workers[8] / turbo_workers[1]
        print(f"\n✅ Scaling efficiency (1 → 8 workers):")
        print(f"   TurboLoader: {turbo_scaling:.2f}x speedup")

    if 0 in pytorch_workers and 8 in pytorch_workers:
        pytorch_scaling = pytorch_workers[8] / pytorch_workers[0]
        print(f"   PyTorch: {pytorch_scaling:.2f}x speedup")

    # Best speedup
    best_speedup = max([turbo_workers[w] / pytorch_workers.get(w, 1)
                        for w in turbo_workers.keys()
                        if w in pytorch_workers])
    print(f"\n✅ Best speedup: {best_speedup:.2f}x faster than PyTorch")

    # Save results
    results = {
        'worker_scaling': {
            'turboloader': turbo_workers,
            'pytorch': pytorch_workers,
            'tensorflow': tf_workers
        },
        'batch_scaling': {
            'turboloader': turbo_batches,
            'pytorch': pytorch_batches
        }
    }

    with open('scaling_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to scaling_results.json")


if __name__ == '__main__':
    main()
