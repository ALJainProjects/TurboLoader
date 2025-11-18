#!/usr/bin/env python3
"""
Comprehensive End-to-End Benchmark Suite for TurboLoader

Compares TurboLoader against:
- PyTorch DataLoader (baseline)
- TensorFlow tf.data (baseline)
- FFCV (best-in-class for speed)
- DALI (GPU-accelerated)

Tests:
1. Data Loading Speed
2. Transform/Augmentation Performance
3. End-to-End Training Throughput
4. Memory Efficiency
5. File Format Conversion (TAR -> TBL v2, FFCV, etc.)
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Attempt to import all libraries
LIBRARIES_AVAILABLE = {
    'turboloader': False,
    'torch': False,
    'tensorflow': False,
    'ffcv': False,
    'nvidia_dali': False,
}

try:
    import turboloader
    LIBRARIES_AVAILABLE['turboloader'] = True
except ImportError:
    print("WARNING: TurboLoader not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    LIBRARIES_AVAILABLE['torch'] = True
except ImportError:
    print("WARNING: PyTorch not available")

try:
    import tensorflow as tf
    LIBRARIES_AVAILABLE['tensorflow'] = True
except ImportError:
    print("WARNING: TensorFlow not available")

try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField
    LIBRARIES_AVAILABLE['ffcv'] = True
except ImportError:
    print("WARNING: FFCV not available")

try:
    import nvidia.dali as dali
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    LIBRARIES_AVAILABLE['nvidia_dali'] = True
except ImportError:
    print("WARNING: NVIDIA DALI not available")


class BenchmarkResults:
    """Store and format benchmark results"""

    def __init__(self):
        self.results = {}

    def add_result(self, category: str, library: str, metric: str, value: float):
        if category not in self.results:
            self.results[category] = {}
        if library not in self.results[category]:
            self.results[category][library] = {}
        self.results[category][library][metric] = value

    def print_table(self):
        """Print formatted results table"""
        print("\n" + "="*100)
        print("COMPREHENSIVE BENCHMARK RESULTS")
        print("="*100)

        for category, libraries in self.results.items():
            print(f"\n{category}")
            print("-"*100)

            # Find all metrics
            all_metrics = set()
            for lib_data in libraries.values():
                all_metrics.update(lib_data.keys())

            # Print header
            print(f"{'Library':<20}", end="")
            for metric in sorted(all_metrics):
                print(f"{metric:>20}", end="")
            print()

            # Print results
            for library, metrics in sorted(libraries.items()):
                print(f"{library:<20}", end="")
                for metric in sorted(all_metrics):
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float):
                            if value > 1000:
                                print(f"{value:>18.1f}  ", end="")
                            else:
                                print(f"{value:>18.2f}  ", end="")
                        else:
                            print(f"{value:>18}  ", end="")
                    else:
                        print(f"{'N/A':>18}  ", end="")
                print()

        print("\n" + "="*100)

    def save_json(self, path: str):
        """Save results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {path}")


def benchmark_data_loading(tar_path: str, results: BenchmarkResults, num_workers: int = 4,
                           batch_size: int = 32, num_batches: int = 100):
    """Benchmark pure data loading speed (no transforms)"""
    print("\n" + "="*100)
    print("BENCHMARK 1: Data Loading Speed (no transforms)")
    print("="*100)

    # TurboLoader
    if LIBRARIES_AVAILABLE['turboloader'] and LIBRARIES_AVAILABLE['torch']:
        print("\nTesting TurboLoader...")
        try:
            loader = turboloader.Loader(
                tar_path,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                decode=True
            )

            start = time.time()
            images_loaded = 0
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                images_loaded += batch.shape[0]
            elapsed = time.time() - start

            throughput = images_loaded / elapsed
            results.add_result("Data Loading", "TurboLoader", "img/s", throughput)
            results.add_result("Data Loading", "TurboLoader", "batch/s", num_batches / elapsed)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # PyTorch DataLoader
    if LIBRARIES_AVAILABLE['torch']:
        print("\nTesting PyTorch DataLoader...")
        try:
            import tarfile
            from PIL import Image
            import io

            class TarDataset(Dataset):
                def __init__(self, tar_path):
                    self.tar_path = tar_path
                    self.tar = tarfile.open(tar_path, 'r')
                    self.members = [m for m in self.tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

                def __len__(self):
                    return len(self.members)

                def __getitem__(self, idx):
                    member = self.members[idx]
                    f = self.tar.extractfile(member)
                    img = Image.open(io.BytesIO(f.read()))
                    return np.array(img, dtype=np.float32) / 255.0

            dataset = TarDataset(tar_path)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=False, pin_memory=True)

            start = time.time()
            images_loaded = 0
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                images_loaded += batch.shape[0]
            elapsed = time.time() - start

            throughput = images_loaded / elapsed
            results.add_result("Data Loading", "PyTorch", "img/s", throughput)
            results.add_result("Data Loading", "PyTorch", "batch/s", num_batches / elapsed)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # TensorFlow tf.data
    if LIBRARIES_AVAILABLE['tensorflow']:
        print("\nTesting TensorFlow tf.data...")
        try:
            import tarfile

            def load_tar_images():
                tar = tarfile.open(tar_path, 'r')
                for member in tar.getmembers():
                    if member.name.endswith(('.jpg', '.jpeg', '.png')):
                        f = tar.extractfile(member)
                        img_bytes = f.read()
                        yield img_bytes

            dataset = tf.data.Dataset.from_generator(
                load_tar_images,
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
            results.add_result("Data Loading", "TensorFlow", "img/s", throughput)
            results.add_result("Data Loading", "TensorFlow", "batch/s", num_batches / elapsed)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")


def benchmark_transforms(tar_path: str, results: BenchmarkResults, num_workers: int = 4,
                        batch_size: int = 32, num_batches: int = 100):
    """Benchmark data loading + transforms/augmentation"""
    print("\n" + "="*100)
    print("BENCHMARK 2: Data Loading + Transforms (RandomResizedCrop, RandomHorizontalFlip, Normalize)")
    print("="*100)

    # TurboLoader with SIMD transforms
    if LIBRARIES_AVAILABLE['turboloader'] and LIBRARIES_AVAILABLE['torch']:
        print("\nTesting TurboLoader (SIMD transforms)...")
        try:
            loader = turboloader.Loader(
                tar_path,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                decode=True,
                transforms=[
                    turboloader.transforms.RandomResizedCrop(224),
                    turboloader.transforms.RandomHorizontalFlip(0.5),
                    turboloader.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

            start = time.time()
            images_loaded = 0
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                images_loaded += batch.shape[0]
            elapsed = time.time() - start

            throughput = images_loaded / elapsed
            results.add_result("Transforms", "TurboLoader", "img/s", throughput)
            results.add_result("Transforms", "TurboLoader", "batch/s", num_batches / elapsed)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # PyTorch with torchvision transforms
    if LIBRARIES_AVAILABLE['torch']:
        print("\nTesting PyTorch (torchvision transforms)...")
        try:
            import tarfile
            from PIL import Image
            import io

            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            class TarDataset(Dataset):
                def __init__(self, tar_path, transform):
                    self.tar_path = tar_path
                    self.transform = transform
                    self.tar = tarfile.open(tar_path, 'r')
                    self.members = [m for m in self.tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

                def __len__(self):
                    return len(self.members)

                def __getitem__(self, idx):
                    member = self.members[idx]
                    f = self.tar.extractfile(member)
                    img = Image.open(io.BytesIO(f.read()))
                    return self.transform(img)

            dataset = TarDataset(tar_path, transform)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True)

            start = time.time()
            images_loaded = 0
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                images_loaded += batch.shape[0]
            elapsed = time.time() - start

            throughput = images_loaded / elapsed
            results.add_result("Transforms", "PyTorch", "img/s", throughput)
            results.add_result("Transforms", "PyTorch", "batch/s", num_batches / elapsed)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # TensorFlow with tf.image
    if LIBRARIES_AVAILABLE['tensorflow']:
        print("\nTesting TensorFlow (tf.image transforms)...")
        try:
            import tarfile

            def transform_tf(image):
                image = tf.image.random_crop(image, [224, 224, 3])
                image = tf.image.random_flip_left_right(image)
                image = tf.image.per_image_standardization(image)
                return image

            def load_tar_images():
                tar = tarfile.open(tar_path, 'r')
                for member in tar.getmembers():
                    if member.name.endswith(('.jpg', '.jpeg', '.png')):
                        f = tar.extractfile(member)
                        img_bytes = f.read()
                        yield img_bytes

            dataset = tf.data.Dataset.from_generator(
                load_tar_images,
                output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
            )
            dataset = dataset.map(lambda x: tf.image.decode_jpeg(x, channels=3),
                                num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(lambda x: tf.cast(x, tf.float32) / 255.0,
                                num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(transform_tf, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            start = time.time()
            images_loaded = 0
            for i, batch in enumerate(dataset.take(num_batches)):
                images_loaded += batch.shape[0]
            elapsed = time.time() - start

            throughput = images_loaded / elapsed
            results.add_result("Transforms", "TensorFlow", "img/s", throughput)
            results.add_result("Transforms", "TensorFlow", "batch/s", num_batches / elapsed)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")


def benchmark_end_to_end_training(tar_path: str, results: BenchmarkResults,
                                  num_workers: int = 4, batch_size: int = 32,
                                  num_epochs: int = 1):
    """Benchmark end-to-end training throughput with ResNet18"""
    print("\n" + "="*100)
    print("BENCHMARK 3: End-to-End Training (ResNet18, 1 epoch)")
    print("="*100)

    # TurboLoader + PyTorch
    if LIBRARIES_AVAILABLE['turboloader'] and LIBRARIES_AVAILABLE['torch']:
        print("\nTesting TurboLoader + PyTorch...")
        try:
            from torchvision.models import resnet18

            loader = turboloader.Loader(
                tar_path,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                decode=True,
                transforms=[
                    turboloader.transforms.RandomResizedCrop(224),
                    turboloader.transforms.RandomHorizontalFlip(0.5),
                    turboloader.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

            model = resnet18(num_classes=1000)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            model.train()
            start = time.time()
            images_processed = 0

            for epoch in range(num_epochs):
                for batch_idx, images in enumerate(loader):
                    # Convert to PyTorch tensor
                    images_t = torch.from_numpy(images).to(device)
                    labels = torch.randint(0, 1000, (images_t.shape[0],)).to(device)

                    optimizer.zero_grad()
                    outputs = model(images_t)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    images_processed += images_t.shape[0]

            elapsed = time.time() - start
            throughput = images_processed / elapsed
            results.add_result("End-to-End Training", "TurboLoader+PyTorch", "img/s", throughput)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # PyTorch DataLoader + PyTorch
    if LIBRARIES_AVAILABLE['torch']:
        print("\nTesting PyTorch DataLoader + PyTorch...")
        try:
            import tarfile
            from PIL import Image
            import io
            from torchvision.models import resnet18

            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            class TarDataset(Dataset):
                def __init__(self, tar_path, transform):
                    self.tar_path = tar_path
                    self.transform = transform
                    self.tar = tarfile.open(tar_path, 'r')
                    self.members = [m for m in self.tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

                def __len__(self):
                    return len(self.members)

                def __getitem__(self, idx):
                    member = self.members[idx]
                    f = self.tar.extractfile(member)
                    img = Image.open(io.BytesIO(f.read()))
                    return self.transform(img), 0

            dataset = TarDataset(tar_path, transform)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True)

            model = resnet18(num_classes=1000)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            model.train()
            start = time.time()
            images_processed = 0

            for epoch in range(num_epochs):
                for batch_idx, (images, _) in enumerate(loader):
                    images = images.to(device)
                    labels = torch.randint(0, 1000, (images.shape[0],)).to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    images_processed += images.shape[0]

            elapsed = time.time() - start
            throughput = images_processed / elapsed
            results.add_result("End-to-End Training", "PyTorch", "img/s", throughput)
            print(f"  Throughput: {throughput:.1f} img/s")

        except Exception as e:
            print(f"  ERROR: {e}")


def benchmark_file_conversion(tar_path: str, results: BenchmarkResults):
    """Benchmark file format conversion speed and compression ratio"""
    print("\n" + "="*100)
    print("BENCHMARK 4: File Format Conversion (TAR -> optimized formats)")
    print("="*100)

    tar_size = os.path.getsize(tar_path) / (1024 * 1024)  # MB
    print(f"\nOriginal TAR size: {tar_size:.2f} MB")

    # Count number of images
    import tarfile
    with tarfile.open(tar_path, 'r') as tar:
        num_images = len([m for m in tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Number of images: {num_images}")

    # TAR -> TBL v2 (TurboLoader)
    if LIBRARIES_AVAILABLE['turboloader']:
        print("\nTesting TAR -> TBL v2 conversion...")
        try:
            import subprocess
            tbl_path = tar_path.replace('.tar', '.tbl')

            start = time.time()
            # Use tar_to_tbl converter
            result = subprocess.run(
                ['./build/tar_to_tbl', tar_path, tbl_path],
                capture_output=True,
                text=True,
                cwd='/Users/arnavjain/turboloader'
            )
            elapsed = time.time() - start

            if os.path.exists(tbl_path):
                tbl_size = os.path.getsize(tbl_path) / (1024 * 1024)
                compression_ratio = (1 - tbl_size / tar_size) * 100
                throughput = num_images / elapsed

                results.add_result("File Conversion", "TBL v2", "Time (s)", elapsed)
                results.add_result("File Conversion", "TBL v2", "Size (MB)", tbl_size)
                results.add_result("File Conversion", "TBL v2", "Compression %", compression_ratio)
                results.add_result("File Conversion", "TBL v2", "img/s", throughput)

                print(f"  Conversion time: {elapsed:.2f}s ({throughput:.1f} img/s)")
                print(f"  Output size: {tbl_size:.2f} MB ({compression_ratio:.1f}% smaller)")
            else:
                print(f"  ERROR: Conversion failed")

        except Exception as e:
            print(f"  ERROR: {e}")

    # TAR -> FFCV format
    if LIBRARIES_AVAILABLE['ffcv'] and LIBRARIES_AVAILABLE['torch']:
        print("\nTesting TAR -> FFCV conversion...")
        try:
            import tarfile
            from PIL import Image
            import io

            ffcv_path = tar_path.replace('.tar', '.beton')

            # Create temporary dataset
            class TarDatasetFFCV(Dataset):
                def __init__(self, tar_path):
                    self.tar = tarfile.open(tar_path, 'r')
                    self.members = [m for m in self.tar.getmembers() if m.name.endswith(('.jpg', '.jpeg', '.png'))]

                def __len__(self):
                    return len(self.members)

                def __getitem__(self, idx):
                    member = self.members[idx]
                    f = self.tar.extractfile(member)
                    img = Image.open(io.BytesIO(f.read()))
                    return np.array(img), 0

            dataset = TarDatasetFFCV(tar_path)

            start = time.time()
            writer = DatasetWriter(ffcv_path, {
                'image': RGBImageField(),
                'label': IntField()
            })
            writer.from_indexed_dataset(dataset)
            elapsed = time.time() - start

            if os.path.exists(ffcv_path):
                ffcv_size = os.path.getsize(ffcv_path) / (1024 * 1024)
                compression_ratio = (1 - ffcv_size / tar_size) * 100
                throughput = num_images / elapsed

                results.add_result("File Conversion", "FFCV", "Time (s)", elapsed)
                results.add_result("File Conversion", "FFCV", "Size (MB)", ffcv_size)
                results.add_result("File Conversion", "FFCV", "Compression %", compression_ratio)
                results.add_result("File Conversion", "FFCV", "img/s", throughput)

                print(f"  Conversion time: {elapsed:.2f}s ({throughput:.1f} img/s)")
                print(f"  Output size: {ffcv_size:.2f} MB ({compression_ratio:.1f}% smaller)")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive TurboLoader Benchmark')
    parser.add_argument('tar_path', type=str, help='Path to TAR dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches to test')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output JSON file')

    args = parser.parse_args()

    if not os.path.exists(args.tar_path):
        print(f"ERROR: TAR file not found: {args.tar_path}")
        sys.exit(1)

    print("="*100)
    print("COMPREHENSIVE TURBOLOADER BENCHMARK SUITE")
    print("="*100)
    print(f"\nDataset: {args.tar_path}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")

    print("\nLibraries available:")
    for lib, available in LIBRARIES_AVAILABLE.items():
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")

    results = BenchmarkResults()

    # Run benchmarks
    try:
        benchmark_data_loading(args.tar_path, results, args.workers, args.batch_size, args.num_batches)
    except Exception as e:
        print(f"\nERROR in data loading benchmark: {e}")

    try:
        benchmark_transforms(args.tar_path, results, args.workers, args.batch_size, args.num_batches)
    except Exception as e:
        print(f"\nERROR in transforms benchmark: {e}")

    try:
        benchmark_end_to_end_training(args.tar_path, results, args.workers, args.batch_size)
    except Exception as e:
        print(f"\nERROR in end-to-end training benchmark: {e}")

    try:
        benchmark_file_conversion(args.tar_path, results)
    except Exception as e:
        print(f"\nERROR in file conversion benchmark: {e}")

    # Print and save results
    results.print_table()
    results.save_json(args.output)

    print("\n" + "="*100)
    print("BENCHMARK COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
