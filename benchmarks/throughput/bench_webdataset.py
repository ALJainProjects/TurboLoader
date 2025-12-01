#!/usr/bin/env python3
"""
WebDataset Throughput Benchmark

Measures raw throughput (images/second) for WebDataset streaming loader.
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import webdataset as wds

    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False

try:
    from PIL import Image
    import torchvision.transforms as T

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""

    library: str
    metric: str
    value: float
    unit: str
    config: Dict[str, Any]
    timestamp: str
    extra: Optional[Dict[str, Any]] = None


def warmup_loader(loader, num_batches: int = 5):
    """Warmup the loader"""
    count = 0
    for batch in loader:
        count += 1
        if count >= num_batches:
            break


def benchmark_throughput(loader, num_batches: int = 100, batch_size: int = 64) -> Dict[str, float]:
    """Measure throughput in images/second"""
    gc.collect()

    total_images = 0
    batch_times = []

    start = time.perf_counter()

    for i, batch in enumerate(loader):
        batch_start = time.perf_counter()

        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("jpg", batch.get("png", None)))
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        if images is not None:
            if isinstance(images, torch.Tensor):
                batch_count = images.shape[0]
            elif hasattr(images, "__len__"):
                batch_count = len(images)
            else:
                batch_count = batch_size
        else:
            batch_count = batch_size

        total_images += batch_count
        batch_times.append(time.perf_counter() - batch_start)

        if i >= num_batches - 1:
            break

    elapsed = time.perf_counter() - start

    return {
        "throughput": total_images / elapsed,
        "total_time": elapsed,
        "total_images": total_images,
        "avg_batch_time": np.mean(batch_times) * 1000,
        "p50_batch_time": np.percentile(batch_times, 50) * 1000,
        "p95_batch_time": np.percentile(batch_times, 95) * 1000,
        "p99_batch_time": np.percentile(batch_times, 99) * 1000,
    }


def identity(x):
    """Identity function for transforms"""
    return x


def run_webdataset_benchmark(
    tar_path: str,
    batch_sizes: List[int] = [32, 64, 128],
    num_workers_list: List[int] = [0, 1, 2, 4, 8],
    num_batches: int = 100,
    with_transforms: bool = True,
) -> List[BenchmarkResult]:
    """Run WebDataset benchmarks"""

    if not HAS_WEBDATASET:
        print("Error: WebDataset not installed")
        print("Install with: pip install webdataset")
        return []

    if not HAS_TORCH:
        print("Error: PyTorch is required")
        return []

    results = []
    timestamp = datetime.now().isoformat()

    print(f"\n{'='*60}")
    print(f"WebDataset Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {tar_path}")
    print(f"Batches per run: {num_batches}")
    print(f"With transforms: {with_transforms}")

    # Create transforms
    if with_transforms and HAS_TORCHVISION:

        def preprocess(sample):
            image = sample["jpg"] if "jpg" in sample else sample["png"]
            if isinstance(image, bytes):
                from io import BytesIO

                image = Image.open(BytesIO(image)).convert("RGB")

            transform = T.Compose(
                [
                    T.Resize((256, 256)),
                    T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            return transform(image), 0  # Return dummy label

    else:

        def preprocess(sample):
            image = sample.get("jpg", sample.get("png", None))
            return image, 0

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            print(f"\n--- Batch Size: {batch_size}, Workers: {num_workers} ---")

            config = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "with_transforms": with_transforms,
                "dataset": os.path.basename(tar_path),
            }

            try:
                # Create WebDataset pipeline
                dataset = (
                    wds.WebDataset(tar_path)
                    .decode("pil" if HAS_TORCHVISION else "rgb")
                    .map(preprocess)
                    .batched(batch_size)
                )

                loader = DataLoader(
                    dataset,
                    batch_size=None,  # Already batched
                    num_workers=num_workers,
                    pin_memory=True if torch.cuda.is_available() else False,
                )

                # Warmup
                print("  Warming up...")
                warmup_loader(loader, num_batches=3)

                # Recreate for benchmark
                dataset = (
                    wds.WebDataset(tar_path)
                    .decode("pil" if HAS_TORCHVISION else "rgb")
                    .map(preprocess)
                    .batched(batch_size)
                )
                loader = DataLoader(
                    dataset,
                    batch_size=None,
                    num_workers=num_workers,
                    pin_memory=True if torch.cuda.is_available() else False,
                )

                # Benchmark
                print("  Running benchmark...")
                stats = benchmark_throughput(loader, num_batches, batch_size)

                print(f"  Throughput: {stats['throughput']:.1f} images/sec")
                print(f"  Avg batch time: {stats['avg_batch_time']:.2f} ms")
                print(f"  P95 batch time: {stats['p95_batch_time']:.2f} ms")

                results.append(
                    BenchmarkResult(
                        library="webdataset",
                        metric="throughput",
                        value=stats["throughput"],
                        unit="images/sec",
                        config=config,
                        timestamp=timestamp,
                        extra=stats,
                    )
                )

            except Exception as e:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()

    return results


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[BenchmarkResult]):
    """Print a summary table"""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch':>8} {'Workers':>8} {'Throughput':>15} {'Avg Batch':>12}")
    print(f"{'Size':>8} {'':>8} {'(img/sec)':>15} {'Time (ms)':>12}")
    print("-" * 70)

    for r in results:
        if r.metric == "throughput":
            batch = r.config.get("batch_size", "N/A")
            workers = r.config.get("num_workers", "N/A")
            throughput = r.value
            avg_time = r.extra.get("avg_batch_time", 0) if r.extra else 0
            print(f"{batch:>8} {workers:>8} {throughput:>15.1f} {avg_time:>12.2f}")


def main():
    parser = argparse.ArgumentParser(description="WebDataset Benchmark")
    parser.add_argument("--tar-path", type=str, required=True, help="Path to TAR dataset")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[32, 64, 128], help="Batch sizes to test"
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[0, 1, 2, 4], help="Number of workers to test"
    )
    parser.add_argument(
        "--num-batches", type=int, default=100, help="Number of batches per benchmark"
    )
    parser.add_argument("--no-transforms", action="store_true", help="Run without transforms")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/throughput/webdataset.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    if not HAS_WEBDATASET:
        print("Error: WebDataset is required for this benchmark")
        print("Install with: pip install webdataset")
        sys.exit(1)

    results = run_webdataset_benchmark(
        args.tar_path,
        batch_sizes=args.batch_sizes,
        num_workers_list=args.workers,
        num_batches=args.num_batches,
        with_transforms=not args.no_transforms,
    )

    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
