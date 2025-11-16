#!/usr/bin/env python3
"""
Full ML Pipeline Benchmark

Measures end-to-end training performance (data loading + model training).
Compares:
1. TurboLoader + PyTorch training
2. PyTorch DataLoader + PyTorch training
3. TensorFlow tf.data + TensorFlow training (optional)

This reveals the true impact of data loading on overall training speed.

Usage: python ml_pipeline_benchmark.py <tar_file> <num_workers> <batch_size> <num_epochs>
Example: python ml_pipeline_benchmark.py /tmp/dataset.tar 4 32 2
"""

import sys
import time
import tarfile
import tempfile
import shutil
from pathlib import Path
import io

# TurboLoader
sys.path.insert(0, 'build/python')
import turboloader

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import numpy as np

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not installed, skipping TF benchmark")


class SimpleModel(nn.Module):
    """Simple CNN for benchmarking"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def benchmark_turboloader_pytorch(tar_path, num_workers, batch_size, num_epochs):
    """Benchmark TurboLoader data loading + PyTorch training"""
    print("\n=== TurboLoader + PyTorch Training ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path],
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
                img_np = sample.get_image()  # NumPy array (H, W, C)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                images.append(img_tensor)

            images = torch.stack(images).to(device)
            labels = torch.randint(0, 10, (len(images),)).to(device)  # Dummy labels

            # Forward + backward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_samples += len(images)
            total_batches += 1

        total_samples += epoch_samples
        pipeline.stop()
        print(f"  Epoch {epoch + 1}/{num_epochs}: {epoch_samples} samples")

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Total samples: {total_samples}")
    print(f"Total time: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Time per epoch: {duration / num_epochs:.2f}s")

    return throughput


def benchmark_pytorch_native(tar_path, num_workers, batch_size, num_epochs):
    """Benchmark PyTorch DataLoader + PyTorch training"""
    print("\n=== PyTorch DataLoader + PyTorch Training ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

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
        pin_memory=(device.type == 'cuda')
    )

    total_samples = 0
    total_batches = 0
    start = time.time()

    for epoch in range(num_epochs):
        epoch_samples = 0

        for batch in dataloader:
            images = batch[0].to(device)
            labels = torch.randint(0, 10, (images.size(0),)).to(device)

            # Forward + backward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_samples += images.size(0)
            total_batches += 1

        total_samples += epoch_samples
        print(f"  Epoch {epoch + 1}/{num_epochs}: {epoch_samples} samples")

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Total samples: {total_samples}")
    print(f"Total time: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Time per epoch: {duration / num_epochs:.2f}s")

    return throughput


def benchmark_tensorflow(tar_path, num_workers, batch_size, num_epochs):
    """Benchmark TensorFlow tf.data + TensorFlow training"""
    if not HAS_TF:
        return 0

    print("\n=== TensorFlow tf.data + TensorFlow Training ===")

    # Extract TAR to temp directory
    temp_dir = tempfile.mkdtemp(prefix='tf_pipeline_')
    print(f"Extracting to {temp_dir}...")

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)

    image_files = sorted(Path(temp_dir).rglob('*.jpg'))
    file_paths = [str(f) for f in image_files]

    def load_and_preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.random.uniform((), 0, 10, dtype=tf.int32)  # Dummy label
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=num_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Simple model
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(256, 256, 3)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    total_samples = 0
    start = time.time()

    for epoch in range(num_epochs):
        epoch_samples = 0
        for images, labels in dataset:
            model.train_on_batch(images, labels)
            epoch_samples += images.shape[0]

        total_samples += epoch_samples
        print(f"  Epoch {epoch + 1}/{num_epochs}: {epoch_samples} samples")

    duration = time.time() - start
    throughput = total_samples / duration

    print(f"Total samples: {total_samples}")
    print(f"Total time: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Time per epoch: {duration / num_epochs:.2f}s")

    # Cleanup
    shutil.rmtree(temp_dir)

    return throughput


def main():
    if len(sys.argv) < 5:
        print("Usage: python ml_pipeline_benchmark.py <tar_file> <num_workers> <batch_size> <num_epochs>")
        print("Example: python ml_pipeline_benchmark.py /tmp/dataset.tar 4 32 2")
        sys.exit(1)

    tar_path = sys.argv[1]
    num_workers = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    num_epochs = int(sys.argv[4])

    print("=== Full ML Pipeline Benchmark ===")
    print(f"Dataset: {tar_path}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = {}

    # Run benchmarks
    results['TurboLoader + PyTorch'] = benchmark_turboloader_pytorch(tar_path, num_workers, batch_size, num_epochs)
    results['PyTorch Native'] = benchmark_pytorch_native(tar_path, num_workers, batch_size, num_epochs)

    if HAS_TF:
        results['TensorFlow'] = benchmark_tensorflow(tar_path, num_workers, batch_size, num_epochs)

    # Summary
    print("\n=== Summary ===")
    baseline = results['PyTorch Native']

    for name, throughput in results.items():
        speedup = throughput / baseline if baseline > 0 else 0
        print(f"{name:25} {throughput:6.2f} samples/s  ({speedup:.2f}x)")

    print("\n=== Key Findings ===")
    if results['TurboLoader + PyTorch'] > baseline:
        speedup = results['TurboLoader + PyTorch'] / baseline
        improvement = (speedup - 1) * 100
        print(f"✅ TurboLoader improves training by {speedup:.2f}x ({improvement:.1f}% faster)")
    else:
        print(f"⚠️  Data loading not the bottleneck (compute-bound on small dataset)")


if __name__ == '__main__':
    main()
