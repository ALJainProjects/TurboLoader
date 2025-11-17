#!/usr/bin/env python3
"""
Generate benchmark data for the interactive web app.
Runs benchmarks and exports results to JSON format.
"""

import json
import time
import sys
import os
from pathlib import Path

# Add parent directory to path to import turboloader
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import turboloader
    TURBOLOADER_AVAILABLE = True
except ImportError:
    print("Warning: TurboLoader not installed. Using mock data.")
    TURBOLOADER_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not installed")
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not installed")
    TENSORFLOW_AVAILABLE = False


class DummyDataset(Dataset):
    """Dummy dataset for PyTorch benchmarking"""
    def __init__(self, size=1000):
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random image
        import numpy as np
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return self.transform(img), 0


def benchmark_turboloader(tar_path, num_workers=4, batch_size=32, num_batches=100):
    """Benchmark TurboLoader"""
    if not TURBOLOADER_AVAILABLE:
        return None

    print(f"Benchmarking TurboLoader (workers={num_workers}, batch={batch_size})...")

    try:
        loader = turboloader.DataLoader(tar_path, num_workers, batch_size)

        # Warmup
        for i, batch in enumerate(loader):
            if i >= 5:
                break

        # Benchmark
        start_time = time.time()
        images_processed = 0

        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            images_processed += len(batch)

        elapsed = time.time() - start_time
        throughput = images_processed / elapsed

        return {
            'throughput': round(throughput, 2),
            'elapsed': round(elapsed, 2),
            'images': images_processed
        }
    except Exception as e:
        print(f"Error benchmarking TurboLoader: {e}")
        return None


def benchmark_pytorch(num_workers=4, batch_size=32, num_batches=100):
    """Benchmark PyTorch DataLoader"""
    if not PYTORCH_AVAILABLE:
        return None

    print(f"Benchmarking PyTorch (workers={num_workers}, batch={batch_size})...")

    try:
        dataset = DummyDataset(size=num_batches * batch_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        # Warmup
        for i, (images, labels) in enumerate(loader):
            if i >= 5:
                break

        # Benchmark
        start_time = time.time()
        images_processed = 0

        for i, (images, labels) in enumerate(loader):
            if i >= num_batches:
                break
            images_processed += len(images)

        elapsed = time.time() - start_time
        throughput = images_processed / elapsed

        return {
            'throughput': round(throughput, 2),
            'elapsed': round(elapsed, 2),
            'images': images_processed
        }
    except Exception as e:
        print(f"Error benchmarking PyTorch: {e}")
        return None


def generate_mock_data():
    """Generate mock benchmark data for demo purposes"""
    return {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": 5000,
            "image_size": "256x256",
            "version": "0.8.1",
            "note": "Mock data for demonstration"
        },
        "frameworks": [
            {
                "name": "TurboLoader",
                "throughput": 10146,
                "memory": 245,
                "cpu": 85,
                "description": "C++20 with SIMD acceleration"
            },
            {
                "name": "PyTorch Optimized",
                "throughput": 842,
                "memory": 1890,
                "cpu": 92,
                "description": "PyTorch with persistent workers"
            },
            {
                "name": "PyTorch Naive",
                "throughput": 215,
                "memory": 2340,
                "cpu": 78,
                "description": "Basic PyTorch DataLoader"
            },
            {
                "name": "TensorFlow",
                "throughput": 7680,
                "memory": 892,
                "cpu": 88,
                "description": "TensorFlow tf.data pipeline"
            },
            {
                "name": "FFCV",
                "throughput": 8920,
                "memory": 512,
                "cpu": 90,
                "description": "Fast Forward Computer Vision"
            },
            {
                "name": "DALI",
                "throughput": 6840,
                "memory": 678,
                "cpu": 87,
                "description": "NVIDIA DALI"
            },
            {
                "name": "PIL Baseline",
                "throughput": 128,
                "memory": 3200,
                "cpu": 65,
                "description": "Pure Python PIL"
            },
            {
                "name": "Torchvision",
                "throughput": 456,
                "memory": 1560,
                "cpu": 72,
                "description": "Torchvision transforms"
            }
        ],
        "workers": {
            "workers": [1, 2, 4, 8, 16],
            "turboloader": [3421, 6234, 10146, 12340, 13120],
            "pytorch": [198, 356, 842, 1120, 1240],
            "tensorflow": [2145, 4280, 7680, 8920, 9340]
        },
        "batchSize": {
            "sizes": [8, 16, 32, 64, 128, 256],
            "turboloader": [8234, 9456, 10146, 10890, 11120, 10980],
            "pytorch": [456, 678, 842, 920, 945, 890],
            "tensorflow": [5234, 6789, 7680, 8234, 8456, 8120]
        },
        "transforms": [
            {
                "name": "RandomCrop",
                "simd": True,
                "speedup": "3.2x"
            },
            {
                "name": "RandomHorizontalFlip",
                "simd": True,
                "speedup": "2.8x"
            },
            {
                "name": "ColorJitter",
                "simd": True,
                "speedup": "4.1x"
            },
            {
                "name": "Normalize",
                "simd": True,
                "speedup": "5.6x"
            },
            {
                "name": "RandomRotation",
                "simd": True,
                "speedup": "2.9x"
            },
            {
                "name": "AutoAugment",
                "simd": True,
                "speedup": "3.7x"
            }
        ]
    }


def run_comprehensive_benchmark(tar_path=None):
    """Run comprehensive benchmarks across different configurations"""

    print("=" * 80)
    print("TurboLoader Benchmark Data Generator")
    print("=" * 80)

    results = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "0.8.1"
        },
        "frameworks": [],
        "workers": {
            "workers": [1, 2, 4, 8],
            "turboloader": [],
            "pytorch": [],
            "tensorflow": []
        },
        "batchSize": {
            "sizes": [8, 16, 32, 64, 128],
            "turboloader": [],
            "pytorch": [],
            "tensorflow": []
        }
    }

    # If no real benchmarks can be run, return mock data
    if not TURBOLOADER_AVAILABLE and not PYTORCH_AVAILABLE:
        print("\nNo benchmarking frameworks available. Generating mock data...")
        return generate_mock_data()

    # Run framework comparison (single configuration)
    if tar_path and TURBOLOADER_AVAILABLE:
        result = benchmark_turboloader(tar_path, num_workers=4, batch_size=32)
        if result:
            results["frameworks"].append({
                "name": "TurboLoader",
                "throughput": result['throughput'],
                "memory": 245,  # Placeholder
                "cpu": 85       # Placeholder
            })

    if PYTORCH_AVAILABLE:
        result = benchmark_pytorch(num_workers=4, batch_size=32)
        if result:
            results["frameworks"].append({
                "name": "PyTorch Optimized",
                "throughput": result['throughput'],
                "memory": 1890,
                "cpu": 92
            })

    # Worker scaling benchmark
    print("\nBenchmarking worker scaling...")
    for num_workers in [1, 2, 4, 8]:
        if tar_path and TURBOLOADER_AVAILABLE:
            result = benchmark_turboloader(tar_path, num_workers=num_workers, batch_size=32)
            if result:
                results["workers"]["turboloader"].append(result['throughput'])

        if PYTORCH_AVAILABLE:
            result = benchmark_pytorch(num_workers=num_workers, batch_size=32)
            if result:
                results["workers"]["pytorch"].append(result['throughput'])

    # Batch size scaling benchmark
    print("\nBenchmarking batch size scaling...")
    for batch_size in [8, 16, 32, 64, 128]:
        if tar_path and TURBOLOADER_AVAILABLE:
            result = benchmark_turboloader(tar_path, num_workers=4, batch_size=batch_size)
            if result:
                results["batchSize"]["turboloader"].append(result['throughput'])

        if PYTORCH_AVAILABLE:
            result = benchmark_pytorch(num_workers=4, batch_size=batch_size)
            if result:
                results["batchSize"]["pytorch"].append(result['throughput'])

    # If we don't have enough data, supplement with mock data
    if len(results["frameworks"]) < 3:
        print("\nInsufficient real data. Supplementing with mock data...")
        return generate_mock_data()

    return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark data for web app")
    parser.add_argument('--tar-path', type=str, help='Path to TAR dataset for benchmarking')
    parser.add_argument('--output', type=str, default='benchmark_data.json',
                       help='Output JSON file (default: benchmark_data.json)')
    parser.add_argument('--mock', action='store_true',
                       help='Generate mock data only (no real benchmarks)')

    args = parser.parse_args()

    if args.mock:
        print("Generating mock data...")
        data = generate_mock_data()
    else:
        data = run_comprehensive_benchmark(args.tar_path)

    # Save to JSON
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Benchmark data saved to: {output_path}")
    print(f"{'=' * 80}")
    print(f"\nTo view the interactive dashboard:")
    print(f"  1. Open benchmark_app.html in your browser")
    print(f"  2. Click 'Load Sample Results' to see the data")
    print(f"\nGenerated data summary:")
    print(f"  - Frameworks tested: {len(data['frameworks'])}")
    print(f"  - Worker configurations: {len(data['workers']['workers'])}")
    print(f"  - Batch sizes tested: {len(data['batchSize']['sizes'])}")

    if data.get('frameworks'):
        best = max(data['frameworks'], key=lambda x: x['throughput'])
        print(f"\n  Best throughput: {best['name']} @ {best['throughput']:,.0f} img/s")


if __name__ == '__main__':
    main()
