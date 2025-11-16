#!/usr/bin/env python3
"""
Comprehensive Data Loading Benchmark

Compares data loading performance across:
1. TurboLoader (C++ with Python bindings)
2. PyTorch DataLoader with WebDataset
3. TensorFlow tf.data pipeline

Measures pure data loading throughput (no training).

Usage: python comprehensive_benchmark.py <tar_file> <num_workers> <batch_size>
Example: python comprehensive_benchmark.py /tmp/dataset.tar 4 32
"""

import sys
import time
import tarfile
import tempfile
import shutil
from pathlib import Path

# TurboLoader
sys.path.insert(0, 'build/python')
import turboloader

# PyTorch
import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import io

# TensorFlow
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not installed, skipping TF benchmark")


def benchmark_turboloader(tar_path, num_workers, batch_size):
    """Benchmark TurboLoader with C++ backend"""
    print("\n=== TurboLoader Benchmark ===")

    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path],
        num_workers=num_workers,
        decode_jpeg=True
    )
    pipeline.start()

    # Warmup
    for _ in range(5):
        _ = pipeline.next_batch(batch_size)

    # Benchmark
    total_samples = 0
    start = time.time()

    while True:
        batch = pipeline.next_batch(batch_size)
        if len(batch) == 0:
            break
        total_samples += len(batch)

    duration = time.time() - start
    pipeline.stop()

    throughput = total_samples / duration
    print(f"Samples: {total_samples}")
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {int(throughput)} img/s")

    return throughput


def benchmark_pytorch(tar_path, num_workers, batch_size):
    """Benchmark PyTorch DataLoader with WebDataset"""
    print("\n=== PyTorch DataLoader Benchmark ===")

    def decode_image(sample):
        """Decode JPEG from bytes"""
        img_bytes = sample['.jpg']
        img = Image.open(io.BytesIO(img_bytes))
        return torch.tensor(list(img.getdata())).reshape(img.size[1], img.size[0], 3)

    dataset = (
        wds.WebDataset(tar_path)
        .decode("pilrgb")
        .to_tuple("jpg")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break

    # Benchmark
    total_samples = 0
    start = time.time()

    for batch in dataloader:
        total_samples += batch[0].shape[0]

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Samples: {total_samples}")
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {int(throughput)} img/s")

    return throughput


def benchmark_tensorflow(tar_path, num_workers, batch_size):
    """Benchmark TensorFlow tf.data with extracted dataset"""
    if not HAS_TF:
        return 0

    print("\n=== TensorFlow tf.data Benchmark ===")
    print("Note: TensorFlow requires extraction to disk")

    # Extract TAR to temp directory
    temp_dir = tempfile.mkdtemp(prefix='tf_benchmark_')
    print(f"Extracting to {temp_dir}...")

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)

    # Find all JPEG files
    image_files = sorted(Path(temp_dir).rglob('*.jpg'))
    file_paths = [str(f) for f in image_files]

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_image, num_parallel_calls=num_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Warmup
    for i, batch in enumerate(dataset.take(5)):
        pass

    # Benchmark
    total_samples = 0
    start = time.time()

    for batch in dataset:
        total_samples += batch.shape[0]

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Samples: {total_samples}")
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {int(throughput)} img/s")

    # Cleanup
    shutil.rmtree(temp_dir)

    return throughput


def main():
    if len(sys.argv) < 4:
        print("Usage: python comprehensive_benchmark.py <tar_file> <num_workers> <batch_size>")
        print("Example: python comprehensive_benchmark.py /tmp/dataset.tar 4 32")
        sys.exit(1)

    tar_path = sys.argv[1]
    num_workers = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    print("=== Comprehensive Data Loading Benchmark ===")
    print(f"Dataset: {tar_path}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")

    results = {}

    # Run benchmarks
    results['TurboLoader'] = benchmark_turboloader(tar_path, num_workers, batch_size)
    results['PyTorch'] = benchmark_pytorch(tar_path, num_workers, batch_size)

    if HAS_TF:
        results['TensorFlow'] = benchmark_tensorflow(tar_path, num_workers, batch_size)

    # Summary
    print("\n=== Summary ===")
    baseline = results['PyTorch']

    for name, throughput in results.items():
        speedup = throughput / baseline if baseline > 0 else 0
        print(f"{name:15} {int(throughput):6} img/s  ({speedup:.2f}x)")

    print("\n=== Key Findings ===")
    if results['TurboLoader'] > baseline:
        speedup = results['TurboLoader'] / baseline
        print(f"✅ TurboLoader is {speedup:.2f}x faster than PyTorch")

    if HAS_TF and results['TensorFlow'] > baseline:
        speedup = results['TensorFlow'] / baseline
        print(f"📊 TensorFlow is {speedup:.2f}x faster than PyTorch (with disk extraction)")


if __name__ == '__main__':
    main()
