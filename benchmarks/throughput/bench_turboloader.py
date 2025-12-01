#!/usr/bin/env python3
"""
TurboLoader Throughput Benchmark

Measures raw throughput (images/second) for TurboLoader data loading.
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

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


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
    """Warmup the loader to ensure fair comparison"""
    count = 0
    for batch in loader:
        count += 1
        if count >= num_batches:
            break


def benchmark_throughput(loader, num_batches: int = 100, batch_size: int = 64) -> Dict[str, float]:
    """
    Measure throughput in images/second

    Returns:
        Dict with throughput, latency stats, and timing info
    """
    gc.collect()

    total_images = 0
    batch_times = []

    start = time.perf_counter()

    for i, batch in enumerate(loader):
        batch_start = time.perf_counter()

        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        if isinstance(images, np.ndarray):
            batch_count = images.shape[0]
        elif hasattr(images, "__len__"):
            batch_count = len(images)
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
        "avg_batch_time": np.mean(batch_times) * 1000,  # ms
        "p50_batch_time": np.percentile(batch_times, 50) * 1000,
        "p95_batch_time": np.percentile(batch_times, 95) * 1000,
        "p99_batch_time": np.percentile(batch_times, 99) * 1000,
    }


def run_turboloader_benchmark(
    tar_path: str,
    batch_sizes: List[int] = [32, 64, 128],
    num_workers_list: List[int] = [1, 2, 4, 8],
    num_batches: int = 100,
    with_transforms: bool = True,
) -> List[BenchmarkResult]:
    """Run TurboLoader benchmarks with various configurations"""

    try:
        import turboloader
    except ImportError:
        print("Error: TurboLoader not installed")
        return []

    results = []
    timestamp = datetime.now().isoformat()

    print(f"\n{'='*60}")
    print(f"TurboLoader Benchmark v{turboloader.__version__}")
    print(f"{'='*60}")
    print(f"Dataset: {tar_path}")
    print(f"Batches per run: {num_batches}")
    print(f"With transforms: {with_transforms}")

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
                # Create transform pipeline if requested
                if with_transforms:
                    transforms = turboloader.Compose(
                        [
                            turboloader.Resize(256, 256),
                            turboloader.RandomCrop(224, 224),
                            turboloader.RandomHorizontalFlip(0.5),
                            turboloader.ImageNetNormalize(),
                        ]
                    )
                else:
                    transforms = None

                # Create loader (note: transforms applied separately after loading)
                loader = turboloader.DataLoader(
                    tar_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                )

                # Warmup
                print("  Warming up...")
                warmup_count = 0
                for batch in loader:
                    if transforms:
                        # Apply transforms to batch
                        if isinstance(batch, (tuple, list)):
                            images = batch[0]
                        else:
                            images = batch
                        if hasattr(images, "__iter__"):
                            for img in images:
                                if isinstance(img, np.ndarray):
                                    transforms.apply(img)
                    warmup_count += 1
                    if warmup_count >= 5:
                        break

                # Reset loader for actual benchmark
                loader = turboloader.DataLoader(
                    tar_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                )

                # Benchmark
                print("  Running benchmark...")
                stats = benchmark_throughput(loader, num_batches, batch_size)

                print(f"  Throughput: {stats['throughput']:.1f} images/sec")
                print(f"  Avg batch time: {stats['avg_batch_time']:.2f} ms")
                print(f"  P95 batch time: {stats['p95_batch_time']:.2f} ms")

                results.append(
                    BenchmarkResult(
                        library="turboloader",
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
                results.append(
                    BenchmarkResult(
                        library="turboloader",
                        metric="error",
                        value=0,
                        unit="",
                        config=config,
                        timestamp=timestamp,
                        extra={"error": str(e)},
                    )
                )

    return results


def run_turboloader_pipe_benchmark(
    tar_path: str, batch_size: int = 64, num_workers: int = 4, num_batches: int = 100
) -> List[BenchmarkResult]:
    """Benchmark using pipe operator syntax"""

    try:
        import turboloader
    except ImportError:
        print("Error: TurboLoader not installed")
        return []

    results = []
    timestamp = datetime.now().isoformat()

    print(f"\n{'='*60}")
    print("TurboLoader Pipe Operator Benchmark")
    print(f"{'='*60}")

    config = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "syntax": "pipe_operator",
    }

    try:
        # Create pipeline using pipe operator
        transforms = (
            turboloader.Resize(256, 256)
            | turboloader.RandomCrop(224, 224)
            | turboloader.RandomHorizontalFlip(0.5)
            | turboloader.ImageNetNormalize()
        )

        loader = turboloader.DataLoader(
            tar_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        # Warmup and benchmark
        warmup_loader(loader, 5)

        loader = turboloader.DataLoader(
            tar_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        stats = benchmark_throughput(loader, num_batches, batch_size)

        print(f"Throughput with pipe operator: {stats['throughput']:.1f} images/sec")

        results.append(
            BenchmarkResult(
                library="turboloader_pipe",
                metric="throughput",
                value=stats["throughput"],
                unit="images/sec",
                config=config,
                timestamp=timestamp,
                extra=stats,
            )
        )

    except Exception as e:
        print(f"Error: {e}")

    return results


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[BenchmarkResult]):
    """Print a summary table of results"""
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
    parser = argparse.ArgumentParser(description="TurboLoader Throughput Benchmark")
    parser.add_argument("--tar-path", type=str, required=True, help="Path to TAR dataset")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[32, 64, 128], help="Batch sizes to test"
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 2, 4, 8], help="Number of workers to test"
    )
    parser.add_argument(
        "--num-batches", type=int, default=100, help="Number of batches per benchmark"
    )
    parser.add_argument("--no-transforms", action="store_true", help="Run without transforms")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/throughput/turboloader.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    # Run main benchmark
    results = run_turboloader_benchmark(
        args.tar_path,
        batch_sizes=args.batch_sizes,
        num_workers_list=args.workers,
        num_batches=args.num_batches,
        with_transforms=not args.no_transforms,
    )

    # Run pipe operator benchmark
    pipe_results = run_turboloader_pipe_benchmark(
        args.tar_path, batch_size=64, num_workers=4, num_batches=args.num_batches
    )
    results.extend(pipe_results)

    # Save and print results
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
