#!/usr/bin/env python3
"""
FFCV Benchmark

Measures the performance of FFCV (Fast Forward Computer Vision Library),
a high-performance data loading library developed by researchers at MIT.

FFCV Features:
- Custom .beton file format for optimized storage
- OS-level caching and memory mapping
- JIT-compiled data pipelines
- GPU-accelerated transforms
- Efficient multi-threaded decoding

Note: FFCV requires converting datasets to .beton format first, which is
a one-time preprocessing cost.

Installation:
    pip install ffcv

Reference: https://ffcv.io/
"""

import os
import sys
import time
import json
import argparse
import psutil
import shutil
from pathlib import Path
from typing import Dict, Any

try:
    import numpy as np
    import torch
except ImportError:
    print("Error: numpy and torch required")
    print("Install with: pip install numpy torch")
    sys.exit(1)

try:
    from ffcv.writer import DatasetWriter
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields import RGBImageField, IntField
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
    from ffcv.transforms import Resize, CenterCrop
except ImportError:
    print("Error: FFCV not installed")
    print("Install with: pip install ffcv")
    print("Note: FFCV requires PyTorch and may have platform-specific requirements")
    sys.exit(1)


def extract_tar_to_images(tar_path: str, output_dir: str) -> str:
    """
    Extract TAR to individual image files for FFCV conversion.

    Args:
        tar_path: Path to TAR file
        output_dir: Directory to extract to

    Returns:
        Path to directory with extracted images
    """
    import tarfile

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting TAR to: {output_dir}")
    extract_start = time.time()

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(output_path)

    extract_time = time.time() - extract_start
    image_files = list(output_path.glob('*.jpg'))
    print(f"Extracted {len(image_files)} images in {extract_time:.2f}s")

    return str(output_path)


class ImageDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset for FFCV conversion"""

    def __init__(self, image_dir: str):
        self.image_files = sorted(Path(image_dir).glob('*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = 0  # Dummy label
        return image, label


def convert_to_beton(image_dir: str, beton_path: str, max_resolution: int = 256):
    """
    Convert image dataset to FFCV .beton format.

    Args:
        image_dir: Directory containing JPEG images
        beton_path: Output .beton file path
        max_resolution: Maximum image resolution for storage
    """
    print(f"\nConverting to FFCV .beton format...")
    print(f"  Source: {image_dir}")
    print(f"  Output: {beton_path}")
    print(f"  Max resolution: {max_resolution}x{max_resolution}")

    dataset = ImageDataset(image_dir)

    convert_start = time.time()

    writer = DatasetWriter(
        beton_path,
        {
            'image': RGBImageField(max_resolution=max_resolution),
            'label': IntField()
        }
    )

    writer.from_indexed_dataset(dataset)

    convert_time = time.time() - convert_start
    beton_size_mb = os.path.getsize(beton_path) / (1024 * 1024)

    print(f"Converted {len(dataset)} images in {convert_time:.2f}s")
    print(f"BETON file size: {beton_size_mb:.2f} MB")

    return convert_time


def run_benchmark(tar_path: str,
                  batch_size: int = 32,
                  num_workers: int = 8,
                  num_epochs: int = 3,
                  work_dir: str = "/tmp/ffcv_benchmark") -> Dict[str, Any]:
    """
    Run FFCV benchmark.

    Args:
        tar_path: Path to TAR file
        batch_size: Batch size
        num_workers: Number of worker threads
        num_epochs: Number of epochs
        work_dir: Working directory for conversion

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("FFCV BENCHMARK")
    print("="*80)
    print(f"TAR file: {tar_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print(f"Features: .beton format, OS caching, JIT compilation")
    print("="*80)

    # Setup working directory
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    image_dir = work_path / "images"
    beton_path = work_path / "dataset.beton"

    # Extract TAR
    extract_start = time.time()
    extract_tar_to_images(tar_path, str(image_dir))
    extraction_time = time.time() - extract_start

    # Convert to .beton format
    conversion_time = convert_to_beton(str(image_dir), str(beton_path))

    # Create FFCV Loader
    print("\nCreating FFCV Loader...")

    # Define FFCV pipeline
    pipelines = {
        'image': [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            ToTorchImage(),
            NormalizeImage(
                mean=np.array([0.485, 0.456, 0.406]) * 255,
                std=np.array([0.229, 0.224, 0.225]) * 255,
                type=np.float32
            )
        ],
        'label': [ToTensor()]
    }

    loader = Loader(
        str(beton_path),
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        pipelines=pipelines,
        drop_last=False
    )

    # Get dataset size
    dataset_size = len(loader)
    total_samples = dataset_size * batch_size

    print(f"\nDataset loaded:")
    print(f"  Total batches: {dataset_size}")
    print(f"  Approximate samples: {total_samples}")

    # Track metrics
    epoch_times = []
    batch_times = []
    memory_usage = []
    process = psutil.Process()

    # Run benchmark
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        batch_count = 0
        sample_count = 0

        for batch_images, batch_labels in loader:
            batch_start = time.time()

            # Simulate model forward pass
            _ = batch_images.mean()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            batch_count += 1
            sample_count += len(batch_images)

            # Track memory every 10 batches
            if batch_count % 10 == 0:
                mem_info = process.memory_info()
                memory_usage.append(mem_info.rss / (1024 * 1024))  # MB

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        throughput = sample_count / epoch_time if epoch_time > 0 else 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Batches: {batch_count}")
        print(f"  Samples: {sample_count}")
        print(f"  Throughput: {throughput:.1f} images/sec")

    total_time = time.time() - total_start

    # Clean up
    print(f"\nCleaning up: {work_dir}")
    shutil.rmtree(work_dir, ignore_errors=True)

    # Calculate statistics
    results = {
        'framework': 'FFCV',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'backend': '.beton format, OS caching, JIT compilation',
        'extraction_time': extraction_time,
        'conversion_time': conversion_time,
        'preprocessing_time': extraction_time + conversion_time,
        'total_time': total_time,
        'total_time_with_preprocessing': total_time + extraction_time + conversion_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'std_epoch_time': np.std(epoch_times) if epoch_times else 0,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'std_batch_time': np.std(batch_times) if batch_times else 0,
        'throughput': sample_count * num_epochs / total_time if total_time > 0 else 0,
        'throughput_with_preprocessing': sample_count * num_epochs / (total_time + extraction_time + conversion_time) if (total_time + extraction_time + conversion_time) > 0 else 0,
        'peak_memory_mb': max(memory_usage) if memory_usage else 0,
        'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
    }

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Extraction time: {extraction_time:.2f}s")
    print(f"Conversion time: {conversion_time:.2f}s")
    print(f"Training time: {total_time:.2f}s")
    print(f"Total time (with preprocessing): {total_time + extraction_time + conversion_time:.2f}s")
    print(f"Average epoch time: {results['avg_epoch_time']:.2f}s ± {results['std_epoch_time']:.2f}s")
    print(f"Average batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
    print(f"Throughput (training only): {results['throughput']:.1f} images/sec")
    print(f"Throughput (with preprocessing): {results['throughput_with_preprocessing']:.1f} images/sec")
    print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    print(f"Average memory: {results['avg_memory_mb']:.1f} MB")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='FFCV benchmark')
    parser.add_argument('tar_path', type=str,
                       help='Path to TAR file containing images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-w', type=int, default=8,
                       help='Number of worker threads (default: 8)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--work-dir', type=str, default='/tmp/ffcv_benchmark',
                       help='Working directory (default: /tmp/ffcv_benchmark)')
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
        num_epochs=args.epochs,
        work_dir=args.work_dir
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
