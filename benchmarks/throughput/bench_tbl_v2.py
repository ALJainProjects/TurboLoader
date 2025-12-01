#!/usr/bin/env python3
"""
Benchmark TBL v2 vs TAR format performance.

This script compares:
1. TAR format (standard)
2. TBL v2 format (optimized binary with LZ4 compression)

Usage:
    python benchmarks/throughput/bench_tbl_v2.py --tar data.tar --tbl data.tbl

    Or create TBL from TAR first:
    python benchmarks/throughput/bench_tbl_v2.py --tar data.tar --create-tbl
"""

import argparse
import os
import sys
import time
import statistics
import tempfile

# Add turboloader to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import turboloader
except ImportError:
    print("Error: turboloader not installed. Run: pip install -e .")
    sys.exit(1)


def create_tbl_from_tar(tar_path: str, tbl_path: str) -> None:
    """Convert TAR to TBL v2 format."""
    print(f"Converting {tar_path} to {tbl_path}...")

    # Use TblWriterV2 if available
    if hasattr(turboloader, "TblWriterV2"):
        writer = turboloader.TblWriterV2(tbl_path, compression=True)
        reader = turboloader.TarReader(tar_path)

        count = 0
        for sample in reader:
            writer.add_sample(
                data=sample.data,
                format=sample.format,
                metadata={"label": getattr(sample, "label", 0)},
            )
            count += 1
            if count % 1000 == 0:
                print(f"  Converted {count} samples...")

        writer.finalize()
        print(f"  Created {tbl_path} with {count} samples")
    else:
        print("Error: TblWriterV2 not available in this build")
        sys.exit(1)


def benchmark_format(
    path: str, format_name: str, batch_size: int = 64, num_workers: int = 4, iterations: int = 5
) -> dict:
    """Benchmark a specific format."""
    print(f"\nBenchmarking {format_name} format: {path}")

    times = []
    sample_counts = []

    for i in range(iterations):
        loader = turboloader.DataLoader(path, batch_size=batch_size, num_workers=num_workers)

        start = time.perf_counter()
        count = 0
        for batch in loader:
            count += len(batch)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed if elapsed > 0 else 0
        times.append(throughput)
        sample_counts.append(count)
        print(f"  Run {i+1}: {throughput:.1f} images/sec ({count} samples in {elapsed:.2f}s)")

    avg_throughput = sum(times) / len(times)
    std_throughput = statistics.stdev(times) if len(times) > 1 else 0

    return {
        "format": format_name,
        "path": path,
        "images_per_sec": avg_throughput,
        "std": std_throughput,
        "total_samples": sample_counts[0] if sample_counts else 0,
        "iterations": iterations,
    }


def benchmark_cache_effectiveness(
    path: str, batch_size: int = 64, num_workers: int = 4, epochs: int = 3
) -> dict:
    """Benchmark cache effectiveness across epochs."""
    print(f"\nBenchmarking cache effectiveness: {path}")
    print(f"  Running {epochs} epochs with L1 cache enabled...")

    epoch_times = []

    # Create loader with caching enabled
    loader = turboloader.DataLoader(
        path,
        batch_size=batch_size,
        num_workers=num_workers,
        enable_cache=True,
        cache_l1_mb=1024,  # 1GB L1 cache
    )

    for epoch in range(epochs):
        start = time.perf_counter()
        count = 0
        for batch in loader:
            count += len(batch)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed if elapsed > 0 else 0
        epoch_times.append(throughput)
        print(f"  Epoch {epoch+1}: {throughput:.1f} images/sec")

    # Calculate speedup from cache
    if len(epoch_times) >= 2:
        speedup = epoch_times[-1] / epoch_times[0] if epoch_times[0] > 0 else 0
    else:
        speedup = 1.0

    return {
        "first_epoch": epoch_times[0] if epoch_times else 0,
        "last_epoch": epoch_times[-1] if epoch_times else 0,
        "speedup": speedup,
        "epoch_times": epoch_times,
    }


def print_results(tar_result: dict, tbl_result: dict = None, cache_result: dict = None):
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("                      BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Format':<15} {'Throughput (img/s)':<20} {'Std Dev':<15}")
    print("-" * 50)

    print(f"{'TAR':<15} {tar_result['images_per_sec']:<20.1f} {tar_result['std']:<15.1f}")

    if tbl_result:
        print(f"{'TBL v2':<15} {tbl_result['images_per_sec']:<20.1f} {tbl_result['std']:<15.1f}")

        speedup = (
            tbl_result["images_per_sec"] / tar_result["images_per_sec"]
            if tar_result["images_per_sec"] > 0
            else 0
        )
        print(f"\nTBL v2 Speedup: {speedup:.2f}x")

    if cache_result:
        print("\n" + "-" * 50)
        print("Cache Effectiveness (across epochs):")
        print(f"  First epoch:  {cache_result['first_epoch']:.1f} images/sec")
        print(f"  Last epoch:   {cache_result['last_epoch']:.1f} images/sec")
        print(f"  Cache speedup: {cache_result['speedup']:.2f}x")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TBL v2 vs TAR format performance")
    parser.add_argument("--tar", required=True, help="Path to TAR file")
    parser.add_argument("--tbl", help="Path to TBL v2 file (optional)")
    parser.add_argument("--create-tbl", action="store_true", help="Create TBL v2 file from TAR")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers (default: 4)")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations (default: 5)"
    )
    parser.add_argument("--test-cache", action="store_true", help="Also test cache effectiveness")

    args = parser.parse_args()

    # Validate TAR exists
    if not os.path.exists(args.tar):
        print(f"Error: TAR file not found: {args.tar}")
        sys.exit(1)

    # Create TBL if requested
    tbl_path = args.tbl
    if args.create_tbl and not tbl_path:
        tbl_path = args.tar.replace(".tar", ".tbl")
        if not os.path.exists(tbl_path):
            create_tbl_from_tar(args.tar, tbl_path)

    # Benchmark TAR
    tar_result = benchmark_format(
        args.tar,
        "TAR",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        iterations=args.iterations,
    )

    # Benchmark TBL v2 if available
    tbl_result = None
    if tbl_path and os.path.exists(tbl_path):
        tbl_result = benchmark_format(
            tbl_path,
            "TBL v2",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            iterations=args.iterations,
        )

    # Test cache effectiveness if requested
    cache_result = None
    if args.test_cache:
        cache_result = benchmark_cache_effectiveness(
            args.tar, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # Print results
    print_results(tar_result, tbl_result, cache_result)


if __name__ == "__main__":
    main()
