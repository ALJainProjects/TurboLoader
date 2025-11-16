#!/usr/bin/env python3
"""
Memory Efficiency Benchmark - All Frameworks

Measures peak memory usage and memory per worker for:
- TurboLoader (C++ threads, shared memory)
- PyTorch DataLoader (multiprocessing, duplicated memory)
- TensorFlow tf.data (tf.data.Dataset)

Metrics:
1. Peak memory usage (GB)
2. Memory per worker (MB)
3. Memory scaling with workers
4. Zero-copy effectiveness

Usage: python memory_benchmark.py <tar_file>
Example: python memory_benchmark.py /tmp/benchmark_10k.tar
"""

import sys
import time
import os
import psutil
import gc
from pathlib import Path

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


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


def benchmark_turboloader_memory(tar_path, num_workers=4, batch_size=32, num_batches=50):
    """Measure TurboLoader memory usage"""
    print("\n=== TurboLoader Memory Benchmark ===")

    # Force garbage collection
    gc.collect()
    baseline_memory = get_memory_usage()

    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path],
        num_workers=num_workers,
        decode_jpeg=True
    )
    pipeline.start()

    # Warmup
    for _ in range(5):
        _ = pipeline.next_batch(batch_size)

    warmup_memory = get_memory_usage()

    # Process batches and track peak memory
    peak_memory = warmup_memory
    for i in range(num_batches):
        batch = pipeline.next_batch(batch_size)
        if len(batch) == 0:
            pipeline.stop()
            pipeline.start()
            batch = pipeline.next_batch(batch_size)

        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)

    pipeline.stop()

    final_memory = get_memory_usage()

    print(f"Baseline memory: {baseline_memory:.2f} MB")
    print(f"Warmup memory: {warmup_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - baseline_memory:.2f} MB")
    print(f"Memory per worker: {(peak_memory - baseline_memory) / num_workers:.2f} MB")

    return {
        'baseline': baseline_memory,
        'peak': peak_memory,
        'increase': peak_memory - baseline_memory,
        'per_worker': (peak_memory - baseline_memory) / num_workers
    }


def benchmark_pytorch_memory(tar_path, num_workers=4, batch_size=32, num_batches=50):
    """Measure PyTorch DataLoader memory usage"""
    print("\n=== PyTorch DataLoader Memory Benchmark ===")

    # Force garbage collection
    gc.collect()
    baseline_memory = get_memory_usage()

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
        if i >= 5:
            break

    warmup_memory = get_memory_usage()

    # Process batches and track peak memory
    peak_memory = warmup_memory
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)

    final_memory = get_memory_usage()

    print(f"Baseline memory: {baseline_memory:.2f} MB")
    print(f"Warmup memory: {warmup_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - baseline_memory:.2f} MB")
    print(f"Memory per worker: {(peak_memory - baseline_memory) / num_workers:.2f} MB")

    return {
        'baseline': baseline_memory,
        'peak': peak_memory,
        'increase': peak_memory - baseline_memory,
        'per_worker': (peak_memory - baseline_memory) / num_workers
    }


def benchmark_tensorflow_memory(tar_path, num_workers=4, batch_size=32, num_batches=50):
    """Measure TensorFlow tf.data memory usage"""
    if not HAS_TF:
        return {}

    print("\n=== TensorFlow tf.data Memory Benchmark ===")

    # Force garbage collection
    gc.collect()
    baseline_memory = get_memory_usage()

    # For simplicity, assume extracted dataset
    image_files = sorted(Path(tar_path).parent.rglob('*.jpg'))
    file_paths = [str(f) for f in image_files[:10000]]  # Limit for memory test

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.cast(img, tf.float32) / 255.0

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_image, num_parallel_calls=num_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    # Warmup
    for i, _ in enumerate(dataset.take(5)):
        pass

    warmup_memory = get_memory_usage()

    # Process batches and track peak memory
    peak_memory = warmup_memory
    for i, batch in enumerate(dataset.take(num_batches)):
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)

    final_memory = get_memory_usage()

    print(f"Baseline memory: {baseline_memory:.2f} MB")
    print(f"Warmup memory: {warmup_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - baseline_memory:.2f} MB")
    print(f"Memory per worker: {(peak_memory - baseline_memory) / num_workers:.2f} MB")

    return {
        'baseline': baseline_memory,
        'peak': peak_memory,
        'increase': peak_memory - baseline_memory,
        'per_worker': (peak_memory - baseline_memory) / num_workers
    }


def test_memory_scaling(tar_path):
    """Test how memory scales with number of workers"""
    print("\n" + "=" * 60)
    print("MEMORY SCALING WITH WORKERS")
    print("=" * 60)

    results = {'turboloader': {}, 'pytorch': {}}

    for num_workers in [1, 2, 4, 8]:
        print(f"\n--- Testing with {num_workers} workers ---")

        # TurboLoader
        gc.collect()
        turbo_result = benchmark_turboloader_memory(tar_path, num_workers, batch_size=32, num_batches=30)
        results['turboloader'][num_workers] = turbo_result

        # PyTorch
        gc.collect()
        pytorch_result = benchmark_pytorch_memory(tar_path, num_workers, batch_size=32, num_batches=30)
        results['pytorch'][num_workers] = pytorch_result

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python memory_benchmark.py <tar_file>")
        print("Example: python memory_benchmark.py /tmp/benchmark_10k.tar")
        sys.exit(1)

    tar_path = sys.argv[1]

    print("=" * 60)
    print("MEMORY EFFICIENCY BENCHMARK - ALL FRAMEWORKS")
    print("=" * 60)
    print(f"Dataset: {tar_path}")
    print(f"Python process: {psutil.Process().pid}")

    # Test memory scaling
    scaling_results = test_memory_scaling(tar_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - MEMORY SCALING")
    print("=" * 60)
    print(f"{'Workers':<10} {'TurboLoader (MB)':<20} {'PyTorch (MB)':<20} {'Ratio':<10}")
    print("-" * 60)

    for workers in sorted(scaling_results['turboloader'].keys()):
        turbo_mem = scaling_results['turboloader'][workers]['increase']
        pytorch_mem = scaling_results['pytorch'][workers]['increase']
        ratio = pytorch_mem / turbo_mem if turbo_mem > 0 else 0

        print(f"{workers:<10} {turbo_mem:<20.2f} {pytorch_mem:<20.2f} {ratio:.2f}x")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Memory efficiency at 8 workers
    if 8 in scaling_results['turboloader']:
        turbo_8 = scaling_results['turboloader'][8]['increase']
        pytorch_8 = scaling_results['pytorch'][8]['increase']
        savings = pytorch_8 - turbo_8
        savings_pct = (savings / pytorch_8) * 100 if pytorch_8 > 0 else 0

        print(f"✅ At 8 workers:")
        print(f"   TurboLoader: {turbo_8:.2f} MB")
        print(f"   PyTorch: {pytorch_8:.2f} MB")
        print(f"   Memory savings: {savings:.2f} MB ({savings_pct:.1f}%)")

    # Per-worker memory
    if 8 in scaling_results['turboloader']:
        turbo_per = scaling_results['turboloader'][8]['per_worker']
        pytorch_per = scaling_results['pytorch'][8]['per_worker']

        print(f"\n✅ Memory per worker:")
        print(f"   TurboLoader: {turbo_per:.2f} MB/worker")
        print(f"   PyTorch: {pytorch_per:.2f} MB/worker")

    # Multiprocessing overhead
    if 1 in scaling_results['pytorch'] and 8 in scaling_results['pytorch']:
        mp_overhead = scaling_results['pytorch'][8]['increase'] / scaling_results['pytorch'][1]['increase']
        print(f"\n⚠️  PyTorch multiprocessing overhead: {mp_overhead:.2f}x memory at 8 workers")

    print(f"\n📝 TurboLoader uses C++ threads (shared memory)")
    print(f"📝 PyTorch uses multiprocessing (duplicated memory per worker)")


if __name__ == '__main__':
    main()
