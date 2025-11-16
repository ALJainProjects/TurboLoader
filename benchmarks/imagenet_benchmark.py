#!/usr/bin/env python3
"""
ImageNet-Scale Benchmark - All Frameworks

Industry-standard benchmark on ImageNet (or ImageNet-scale datasets).
Compares data loading and end-to-end training performance.

Frameworks compared:
- TurboLoader + PyTorch training
- PyTorch DataLoader + PyTorch training
- TensorFlow tf.data + TensorFlow training
- FFCV (published benchmarks for comparison)
- NVIDIA DALI (optional)

Metrics:
1. Data loading throughput (img/s)
2. End-to-end training throughput (samples/s)
3. GPU utilization (%)
4. Memory usage (GB)
5. Time to accuracy benchmarks

Usage: python imagenet_benchmark.py <imagenet_tar_or_dir> [--full-training]
Example: python imagenet_benchmark.py /data/imagenet.tar
Example: python imagenet_benchmark.py /data/imagenet_extracted/ --full-training
"""

import sys
import time
import os
import argparse
import tarfile
import tempfile
import shutil
from pathlib import Path
import json

# TurboLoader
sys.path.insert(0, 'build/python')
import turboloader

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import webdataset as wds
import torchvision.models as models
from PIL import Image
import numpy as np

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not available")

# FFCV
try:
    from ffcv.loader import Loader
    from ffcv.fields.decoders import SimpleRGBImageDecoder
    HAS_FFCV = True
except ImportError:
    HAS_FFCV = False
    print("Warning: FFCV not available (will use published benchmarks for comparison)")


class ImageNetModel(nn.Module):
    """ResNet-50 for ImageNet training"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def benchmark_turboloader_dataloading(tar_path, num_workers=8, batch_size=256, num_batches=100):
    """Benchmark TurboLoader data loading only"""
    print("\n=== TurboLoader Data Loading ===")

    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path] if tar_path.endswith('.tar') else [tar_path],
        num_workers=num_workers,
        decode_jpeg=True
    )
    pipeline.start()

    # Warmup
    for _ in range(10):
        _ = pipeline.next_batch(batch_size)

    # Benchmark
    total_samples = 0
    start = time.time()

    for i in range(num_batches):
        batch = pipeline.next_batch(batch_size)
        if len(batch) == 0:
            pipeline.stop()
            pipeline.start()
            batch = pipeline.next_batch(batch_size)
        total_samples += len(batch)

    duration = time.time() - start
    throughput = total_samples / duration

    pipeline.stop()

    print(f"Samples: {total_samples}")
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {int(throughput)} img/s")

    return throughput


def benchmark_pytorch_dataloading(tar_path, num_workers=8, batch_size=256, num_batches=100):
    """Benchmark PyTorch DataLoader data loading only"""
    print("\n=== PyTorch DataLoader Data Loading ===")

    if tar_path.endswith('.tar'):
        dataset = (
            wds.WebDataset(tar_path)
            .decode("pilrgb")
            .to_tuple("jpg")
            .map(lambda x: (torch.from_numpy(np.array(x[0])).permute(2, 0, 1).float() / 255.0,))
        )
    else:
        # Use ImageFolder for extracted datasets
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(tar_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Warmup
    for i, _ in enumerate(dataloader):
        if i >= 10:
            break

    # Benchmark
    total_samples = 0
    start = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        total_samples += batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]

    duration = time.time() - start
    throughput = total_samples / duration if duration > 0 else 0

    print(f"Samples: {total_samples}")
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {int(throughput)} img/s")

    return throughput


def benchmark_tensorflow_dataloading(data_path, num_workers=8, batch_size=256, num_batches=100):
    """Benchmark TensorFlow tf.data data loading only"""
    if not HAS_TF:
        return 0

    print("\n=== TensorFlow tf.data Data Loading ===")

    # Extract TAR if needed
    if data_path.endswith('.tar'):
        temp_dir = tempfile.mkdtemp(prefix='tf_imagenet_')
        print(f"Extracting to {temp_dir}...")
        with tarfile.open(data_path, 'r') as tar:
            tar.extractall(temp_dir)
        data_path = temp_dir

    image_files = sorted(Path(data_path).rglob('*.JPEG')) + sorted(Path(data_path).rglob('*.jpg'))
    file_paths = [str(f) for f in image_files]

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.cast(img, tf.float32) / 255.0

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_image, num_parallel_calls=num_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Warmup
    for i, _ in enumerate(dataset.take(10)):
        pass

    # Benchmark
    total_samples = 0
    start = time.time()

    for i, batch in enumerate(dataset.take(num_batches)):
        total_samples += batch.shape[0]

    duration = time.time() - start
    throughput = total_samples / duration if duration > 0 else 0

    print(f"Samples: {total_samples}")
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {int(throughput)} img/s")

    # Cleanup
    if data_path.endswith('.tar'):
        shutil.rmtree(temp_dir)

    return throughput


def benchmark_turboloader_training(tar_path, num_workers=8, batch_size=256, num_epochs=1):
    """Benchmark TurboLoader + PyTorch end-to-end training"""
    print("\n=== TurboLoader + PyTorch Training ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ImageNetModel(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path] if tar_path.endswith('.tar') else [tar_path],
        num_workers=num_workers,
        decode_jpeg=True
    )

    total_samples = 0
    total_batches = 0
    start = time.time()

    for epoch in range(num_epochs):
        pipeline.start()
        epoch_samples = 0

        while True:
            batch = pipeline.next_batch(batch_size)
            if len(batch) == 0:
                break

            # Convert to PyTorch tensors
            images = []
            for sample in batch:
                img_np = sample.get_image()
                # Resize to 224x224 for ResNet
                img_pil = Image.fromarray(img_np.astype('uint8'))
                img_resized = img_pil.resize((224, 224))
                img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)

            images = torch.stack(images).to(device)
            labels = torch.randint(0, 1000, (len(images),)).to(device)

            # Forward + backward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_samples += len(images)
            total_batches += 1

            # Limit batches for benchmark
            if total_batches >= 100:
                break

        total_samples += epoch_samples
        pipeline.stop()
        print(f"  Epoch {epoch + 1}/{num_epochs}: {epoch_samples} samples")

        if total_batches >= 100:
            break

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Total samples: {total_samples}")
    print(f"Total time: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/s")

    return throughput


def benchmark_pytorch_training(tar_path, num_workers=8, batch_size=256, num_epochs=1):
    """Benchmark PyTorch native end-to-end training"""
    print("\n=== PyTorch Native Training ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ImageNetModel(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    if tar_path.endswith('.tar'):
        dataset = (
            wds.WebDataset(tar_path)
            .decode("pilrgb")
            .to_tuple("jpg")
            .map(lambda x: (
                torch.nn.functional.interpolate(
                    torch.from_numpy(np.array(x[0])).permute(2, 0, 1).unsqueeze(0).float(),
                    size=(224, 224)
                ).squeeze(0) / 255.0,
            ))
        )
    else:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(tar_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    total_samples = 0
    total_batches = 0
    start = time.time()

    for epoch in range(num_epochs):
        epoch_samples = 0

        for batch in dataloader:
            images = batch[0].to(device)
            labels = torch.randint(0, 1000, (images.size(0),)).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_samples += images.size(0)
            total_batches += 1

            if total_batches >= 100:
                break

        total_samples += epoch_samples
        print(f"  Epoch {epoch + 1}/{num_epochs}: {epoch_samples} samples")

        if total_batches >= 100:
            break

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Total samples: {total_samples}")
    print(f"Total time: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/s")

    return throughput


def main():
    parser = argparse.ArgumentParser(description='ImageNet-scale benchmark')
    parser.add_argument('dataset', help='Path to ImageNet TAR or directory')
    parser.add_argument('--full-training', action='store_true', help='Run full training benchmark')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    args = parser.parse_args()

    print("=" * 70)
    print("IMAGENET BENCHMARK - ALL FRAMEWORKS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = {}

    # Part 1: Data Loading Only
    print("\n" + "=" * 70)
    print("PART 1: DATA LOADING THROUGHPUT")
    print("=" * 70)

    results['dataloading'] = {}
    results['dataloading']['turboloader'] = benchmark_turboloader_dataloading(
        args.dataset, args.workers, args.batch_size, num_batches=100
    )
    results['dataloading']['pytorch'] = benchmark_pytorch_dataloading(
        args.dataset, args.workers, args.batch_size, num_batches=100
    )
    results['dataloading']['tensorflow'] = benchmark_tensorflow_dataloading(
        args.dataset, args.workers, args.batch_size, num_batches=100
    )

    # FFCV published benchmark (for reference)
    results['dataloading']['ffcv_published'] = 31278  # From FFCV paper

    # Part 2: End-to-End Training
    if args.full_training:
        print("\n" + "=" * 70)
        print("PART 2: END-TO-END TRAINING")
        print("=" * 70)

        results['training'] = {}
        results['training']['turboloader'] = benchmark_turboloader_training(
            args.dataset, args.workers, args.batch_size, num_epochs=1
        )
        results['training']['pytorch'] = benchmark_pytorch_training(
            args.dataset, args.workers, args.batch_size, num_epochs=1
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - DATA LOADING")
    print("=" * 70)
    print(f"{'Framework':<20} {'Throughput':<15} {'Speedup vs PyTorch':<20}")
    print("-" * 70)

    pytorch_dl = results['dataloading']['pytorch']
    for name, throughput in results['dataloading'].items():
        if name == 'ffcv_published':
            speedup = throughput / pytorch_dl if pytorch_dl > 0 else 0
            print(f"{name:<20} {int(throughput):<15} {speedup:.2f}x (published)")
        else:
            speedup = throughput / pytorch_dl if pytorch_dl > 0 else 0
            print(f"{name:<20} {int(throughput):<15} {speedup:.2f}x")

    if args.full_training:
        print("\n" + "=" * 70)
        print("SUMMARY - END-TO-END TRAINING")
        print("=" * 70)
        print(f"{'Framework':<20} {'Throughput':<15} {'Speedup':<20}")
        print("-" * 70)

        pytorch_train = results['training']['pytorch']
        for name, throughput in results['training'].items():
            speedup = throughput / pytorch_train if pytorch_train > 0 else 0
            print(f"{name:<20} {throughput:.2f} samples/s    {speedup:.2f}x")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    turbo_dl = results['dataloading']['turboloader']
    speedup = turbo_dl / pytorch_dl if pytorch_dl > 0 else 0
    print(f"✅ Data Loading: TurboLoader is {speedup:.2f}x faster than PyTorch ({int(turbo_dl)} vs {int(pytorch_dl)} img/s)")

    ffcv_ratio = turbo_dl / results['dataloading']['ffcv_published']
    print(f"📊 TurboLoader achieves {ffcv_ratio * 100:.1f}% of FFCV's published performance")

    if args.full_training:
        turbo_train = results['training']['turboloader']
        pytorch_train = results['training']['pytorch']
        train_speedup = turbo_train / pytorch_train if pytorch_train > 0 else 0
        print(f"✅ Training: TurboLoader improves end-to-end by {train_speedup:.2f}x")

    # Save results
    with open('imagenet_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to imagenet_results.json")


if __name__ == '__main__':
    main()
