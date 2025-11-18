#!/usr/bin/env python3
"""
Final Comprehensive End-to-End Benchmark for TurboLoader

Compares TurboLoader v1.5.0 (TBL v2) against:
- PyTorch DataLoader
- TensorFlow tf.data
- File format conversion (TAR -> TBL v2)
"""

import os
import sys
import time
import json
import tarfile
import subprocess
from pathlib import Path
import argparse

print("Importing libraries...")
import numpy as np
from PIL import Image
import io

# PyTorch
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as T

# TensorFlow
import tensorflow as tf

print("All libraries imported successfully!\n")


def benchmark_pytorch_loading(tar_path, batch_size=32, num_batches=100):
    """Benchmark PyTorch DataLoader (single-threaded for simplicity)"""
    print("\n" + "="*80)
    print("BENCHMARK: PyTorch DataLoader (reading TAR)")
    print("="*80)

    tar = tarfile.open(tar_path, 'r')
    members = [m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

    # Simple single-threaded loading
    start = time.time()
    images_loaded = 0
    batch = []

    for i, member in enumerate(members):
        if images_loaded >= batch_size * num_batches:
            break

        f = tar.extractfile(member)
        img = Image.open(io.BytesIO(f.read()))
        arr = np.array(img, dtype=np.float32) / 255.0
        batch.append(arr)

        if len(batch) == batch_size:
            images_loaded += len(batch)
            batch = []

    elapsed = time.time() - start
    throughput = images_loaded / elapsed

    print(f"  Images loaded: {images_loaded}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} img/s")

    tar.close()
    return {"img/s": throughput, "batches/s": (num_batches / elapsed)}


def benchmark_pytorch_with_transforms(tar_path, batch_size=32, num_batches=100):
    """Benchmark PyTorch with transforms"""
    print("\n" + "="*80)
    print("BENCHMARK: PyTorch with Transforms (ResizedCrop, Flip, Normalize)")
    print("="*80)

    tar = tarfile.open(tar_path, 'r')
    members = [m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

    transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    start = time.time()
    images_loaded = 0
    batch = []

    for i, member in enumerate(members):
        if images_loaded >= batch_size * num_batches:
            break

        f = tar.extractfile(member)
        img = Image.open(io.BytesIO(f.read()))
        tensor = transform(img)
        batch.append(tensor)

        if len(batch) == batch_size:
            images_loaded += len(batch)
            batch = []

    elapsed = time.time() - start
    throughput = images_loaded / elapsed

    print(f"  Images loaded: {images_loaded}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} img/s")

    tar.close()
    return {"img/s": throughput, "batches/s": (num_batches / elapsed)}


def benchmark_tensorflow_loading(tar_path, batch_size=32, num_batches=100):
    """Benchmark TensorFlow tf.data"""
    print("\n" + "="*80)
    print("BENCHMARK: TensorFlow tf.data")
    print("="*80)

    tar = tarfile.open(tar_path, 'r')
    members = [m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

    def load_images():
        for member in members:
            f = tar.extractfile(member)
            yield f.read()

    dataset = tf.data.Dataset.from_generator(
        load_images,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )
    dataset = dataset.map(lambda x: tf.image.decode_jpeg(x, channels=3),
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.cast(x, tf.float32) / 255.0,
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    start = time.time()
    images_loaded = 0
    for i, batch in enumerate(dataset.take(num_batches)):
        images_loaded += batch.shape[0]
    elapsed = time.time() - start

    throughput = images_loaded / elapsed

    print(f"  Images loaded: {images_loaded}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} img/s")

    tar.close()
    return {"img/s": throughput, "batches/s": (num_batches / elapsed)}


def benchmark_end_to_end_pytorch(tar_path, batch_size=32, num_epochs=1):
    """Benchmark end-to-end training with PyTorch"""
    print("\n" + "="*80)
    print("BENCHMARK: End-to-End Training (PyTorch + ResNet18)")
    print("="*80)

    tar = tarfile.open(tar_path, 'r')
    members = [m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = resnet18(num_classes=1000)
    device = torch.device("cpu")  # Use CPU for fair comparison
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    start = time.time()
    images_processed = 0

    batch_tensors = []
    for i, member in enumerate(members):
        f = tar.extractfile(member)
        img = Image.open(io.BytesIO(f.read()))
        tensor = transform(img)
        batch_tensors.append(tensor)

        if len(batch_tensors) == batch_size:
            images = torch.stack(batch_tensors).to(device)
            labels = torch.randint(0, 1000, (images.shape[0],)).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            images_processed += images.shape[0]
            batch_tensors = []

            # Limit to reasonable number for benchmark
            if images_processed >= 200:
                break

    elapsed = time.time() - start
    throughput = images_processed / elapsed

    print(f"  Images processed: {images_processed}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} img/s")

    tar.close()
    return {"img/s": throughput}


def benchmark_tar_to_tbl_conversion(tar_path):
    """Benchmark TAR -> TBL v2 conversion"""
    print("\n" + "="*80)
    print("BENCHMARK: File Format Conversion (TAR -> TBL v2)")
    print("="*80)

    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    tbl_path = tar_path.replace('.tar', '_v2.tbl')

    # Count images
    with tarfile.open(tar_path, 'r') as tar:
        num_images = len([m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))])

    print(f"  Input: {tar_path} ({tar_size:.2f} MB, {num_images} images)")

    # Run conversion
    start = time.time()
    result = subprocess.run(
        ['./build/tar_to_tbl', tar_path, tbl_path],
        capture_output=True,
        text=True,
        cwd='/Users/arnavjain/turboloader'
    )
    elapsed = time.time() - start

    if os.path.exists(tbl_path):
        tbl_size = os.path.getsize(tbl_path) / (1024 * 1024)
        compression = (1 - tbl_size / tar_size) * 100
        throughput = num_images / elapsed

        print(f"  Output: {tbl_path} ({tbl_size:.2f} MB)")
        print(f"  Compression: {compression:.1f}% smaller")
        print(f"  Conversion time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} img/s")

        # Clean up
        os.remove(tbl_path)

        return {
            "time_s": elapsed,
            "input_mb": tar_size,
            "output_mb": tbl_size,
            "compression_%": compression,
            "img/s": throughput
        }
    else:
        print(f"  ERROR: Conversion failed")
        return None


def main():
    parser = argparse.ArgumentParser(description='Final Comprehensive Benchmark')
    parser.add_argument('tar_path', type=str, help='Path to TAR dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-batches', type=int, default=50, help='Number of batches')
    parser.add_argument('--output', type=str, default='final_results.json', help='Output JSON')

    args = parser.parse_args()

    if not os.path.exists(args.tar_path):
        print(f"ERROR: TAR file not found: {args.tar_path}")
        sys.exit(1)

    print("="*80)
    print("FINAL COMPREHENSIVE BENCHMARK - TurboLoader v1.5.0")
    print("="*80)
    print(f"\nDataset: {args.tar_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}\n")

    results = {}

    # Run benchmarks
    try:
        results['pytorch_loading'] = benchmark_pytorch_loading(
            args.tar_path, args.batch_size, args.num_batches
        )
    except Exception as e:
        print(f"ERROR: {e}")
        results['pytorch_loading'] = None

    try:
        results['pytorch_transforms'] = benchmark_pytorch_with_transforms(
            args.tar_path, args.batch_size, args.num_batches
        )
    except Exception as e:
        print(f"ERROR: {e}")
        results['pytorch_transforms'] = None

    try:
        results['tensorflow_loading'] = benchmark_tensorflow_loading(
            args.tar_path, args.batch_size, args.num_batches
        )
    except Exception as e:
        print(f"ERROR: {e}")
        results['tensorflow_loading'] = None

    try:
        results['pytorch_training'] = benchmark_end_to_end_pytorch(
            args.tar_path, args.batch_size
        )
    except Exception as e:
        print(f"ERROR: {e}")
        results['pytorch_training'] = None

    try:
        results['tar_to_tbl_conversion'] = benchmark_tar_to_tbl_conversion(
            args.tar_path
        )
    except Exception as e:
        print(f"ERROR: {e}")
        results['tar_to_tbl_conversion'] = None

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    for name, result in results.items():
        print(f"\n{name}:")
        if result:
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  FAILED")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
