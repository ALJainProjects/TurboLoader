#!/usr/bin/env python3
"""
TurboLoader Benchmark Suite v2.0

Comprehensive benchmark suite comparing TurboLoader against:
- PyTorch DataLoader
- WebDataset
- Albumentations (transforms)

Benchmark categories:
- Throughput (images/second)
- Transform performance
- Memory usage
- Scalability

Results are saved to benchmarks/results/ for analysis.
"""

import argparse
import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class BenchmarkRunner:
    """Run and collect benchmark results"""

    def __init__(
        self, dataset_path: str, output_dir: str = "benchmarks/results", quick: bool = False
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.quick = quick
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_path,
            "quick_mode": quick,
            "benchmarks": {},
        }

        # Create output directories
        (self.output_dir / "throughput").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "transforms").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "memory").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)

    def run_command(self, name: str, command: list, timeout: int = 600) -> bool:
        """Run a benchmark command and capture output"""
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(__file__).parent.parent),
            )
            elapsed = time.time() - start_time

            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(f"STDERR: {result.stderr}")

            self.results["benchmarks"][name] = {
                "duration": elapsed,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"  Timeout after {timeout}s")
            self.results["benchmarks"][name] = {
                "duration": timeout,
                "exit_code": -1,
                "error": "Timeout",
            }
            return False
        except Exception as e:
            print(f"  Error: {e}")
            self.results["benchmarks"][name] = {"error": str(e)}
            return False

    def run_throughput_benchmarks(self):
        """Run throughput benchmarks"""
        print("\n" + "=" * 70)
        print("THROUGHPUT BENCHMARKS")
        print("=" * 70)

        if self.quick:
            batch_sizes = "64"
            workers = "4"
            num_batches = "20"
        else:
            batch_sizes = "32 64 128"
            workers = "1 2 4 8"
            num_batches = "50"

        # TurboLoader throughput
        self.run_command(
            "TurboLoader Throughput",
            [
                sys.executable,
                "benchmarks/throughput/bench_turboloader.py",
                "--tar-path",
                self.dataset_path,
                "--batch-sizes",
                *batch_sizes.split(),
                "--workers",
                *workers.split(),
                "--num-batches",
                num_batches,
                "--output",
                str(self.output_dir / "throughput/turboloader.json"),
            ],
        )

        # PyTorch throughput
        self.run_command(
            "PyTorch Throughput",
            [
                sys.executable,
                "benchmarks/throughput/bench_pytorch.py",
                "--tar-path",
                self.dataset_path,
                "--batch-sizes",
                *batch_sizes.split(),
                "--workers",
                *workers.split(),
                "--num-batches",
                num_batches,
                "--cached",
                "--output",
                str(self.output_dir / "throughput/pytorch.json"),
            ],
        )

    def run_transform_benchmarks(self):
        """Run transform benchmarks"""
        print("\n" + "=" * 70)
        print("TRANSFORM BENCHMARKS")
        print("=" * 70)

        timed_runs = "50" if self.quick else "100"

        self.run_command(
            "Transform Performance",
            [
                sys.executable,
                "benchmarks/transforms/bench_transforms.py",
                "--num-images",
                "50",
                "--timed-runs",
                timed_runs,
                "--output",
                str(self.output_dir / "transforms/transforms.json"),
            ],
        )

    def run_memory_benchmarks(self):
        """Run memory benchmarks"""
        print("\n" + "=" * 70)
        print("MEMORY BENCHMARKS")
        print("=" * 70)

        if self.quick:
            batch_sizes = "64"
            num_batches = "20"
        else:
            batch_sizes = "32 64 128"
            num_batches = "30"

        self.run_command(
            "Memory Usage",
            [
                sys.executable,
                "benchmarks/memory/bench_memory.py",
                "--tar-path",
                self.dataset_path,
                "--batch-sizes",
                *batch_sizes.split(),
                "--num-batches",
                num_batches,
                "--output",
                str(self.output_dir / "memory/memory.json"),
            ],
        )

    def run_scalability_tests(self):
        """Test scalability with different worker counts"""
        print("\n" + "=" * 70)
        print("SCALABILITY TESTS")
        print("=" * 70)

        try:
            import turboloader
        except ImportError:
            print("TurboLoader not available, skipping scalability tests")
            return

        worker_counts = [1, 2, 4, 8] if not self.quick else [1, 4]
        scalability_results = {}

        for workers in worker_counts:
            print(f"\n  Testing {workers} workers...")
            try:
                loader = turboloader.DataLoader(
                    self.dataset_path, batch_size=64, num_workers=workers, shuffle=True
                )

                start = time.time()
                count = 0
                for batch in loader:
                    if isinstance(batch, (tuple, list)):
                        count += len(batch[0])
                    else:
                        count += len(batch)
                    if count >= 1000:
                        break
                elapsed = time.time() - start

                throughput = count / elapsed
                print(f"    Throughput: {throughput:.1f} img/s")
                scalability_results[workers] = throughput

            except Exception as e:
                print(f"    Error: {e}")

        self.results["benchmarks"]["Scalability"] = scalability_results

    def generate_report(self):
        """Generate summary report"""
        print("\n" + "=" * 70)
        print("GENERATING REPORT")
        print("=" * 70)

        lines = []
        lines.append("=" * 78)
        lines.append("                    TurboLoader Benchmark Report")
        lines.append("=" * 78)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Dataset: {self.dataset_path}")
        lines.append(f"Quick mode: {self.quick}")

        # Load results from JSON files
        throughput_results = {}
        transform_results = []
        memory_results = []

        # Load throughput results
        for lib in ["turboloader", "pytorch"]:
            path = self.output_dir / f"throughput/{lib}.json"
            if path.exists():
                try:
                    with open(path) as f:
                        throughput_results[lib] = json.load(f)
                except:
                    pass

        # Load transform results
        transform_path = self.output_dir / "transforms/transforms.json"
        if transform_path.exists():
            try:
                with open(transform_path) as f:
                    transform_results = json.load(f)
            except:
                pass

        # Load memory results
        memory_path = self.output_dir / "memory/memory.json"
        if memory_path.exists():
            try:
                with open(memory_path) as f:
                    memory_results = json.load(f)
            except:
                pass

        # Throughput comparison
        if throughput_results:
            lines.append("\n" + "-" * 78)
            lines.append("                       THROUGHPUT COMPARISON")
            lines.append("-" * 78)
            lines.append(f"{'Library':>20} {'Batch':>8} {'Workers':>8} {'Images/sec':>15}")
            lines.append("-" * 78)

            for lib, data in throughput_results.items():
                if isinstance(data, list):
                    for entry in data:
                        if entry.get("metric") == "throughput":
                            batch = entry.get("config", {}).get("batch_size", "N/A")
                            workers = entry.get("config", {}).get("num_workers", "N/A")
                            value = entry.get("value", 0)
                            lines.append(f"{lib:>20} {batch:>8} {workers:>8} {value:>15.1f}")

        # Transform comparison
        if transform_results:
            lines.append("\n" + "-" * 78)
            lines.append("                      TRANSFORM PERFORMANCE")
            lines.append("-" * 78)
            lines.append(f"{'Transform':>35} {'Library':>15} {'Time (ms)':>12}")
            lines.append("-" * 78)

            for entry in transform_results:
                transform = entry.get("transform", "N/A")
                library = entry.get("library", "N/A")
                time_ms = entry.get("time_per_image_ms", 0)
                lines.append(f"{transform:>35} {library:>15} {time_ms:>12.3f}")

        # Memory comparison
        if memory_results:
            lines.append("\n" + "-" * 78)
            lines.append("                         MEMORY USAGE")
            lines.append("-" * 78)
            lines.append(f"{'Library':>20} {'Batch':>8} {'Peak (MB)':>12} {'Delta (MB)':>12}")
            lines.append("-" * 78)

            for entry in memory_results:
                library = entry.get("library", "N/A")
                batch = entry.get("config", {}).get("batch_size", "N/A")
                peak = entry.get("peak_mb", 0)
                delta = entry.get("delta_mb", 0)
                lines.append(f"{library:>20} {batch:>8} {peak:>12.1f} {delta:>12.1f}")

        # Scalability
        if "Scalability" in self.results["benchmarks"]:
            lines.append("\n" + "-" * 78)
            lines.append("                         SCALABILITY")
            lines.append("-" * 78)
            lines.append(f"{'Workers':>12} {'Throughput (img/s)':>20}")
            lines.append("-" * 78)

            for workers, throughput in self.results["benchmarks"]["Scalability"].items():
                lines.append(f"{workers:>12} {throughput:>20.1f}")

        lines.append("\n" + "=" * 78)

        report = "\n".join(lines)

        # Save report
        report_path = (
            self.output_dir
            / f"reports/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_path, "w") as f:
            f.write(report)

        print(report)
        print(f"\nReport saved to: {report_path}")

    def save_results(self):
        """Save all results to JSON"""
        output_file = self.output_dir / "all_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nAll results saved to: {output_file}")

    def run_all(self):
        """Run complete benchmark suite"""
        print(
            f"""
{'='*70}
         TURBOLOADER COMPREHENSIVE BENCHMARK SUITE v2.0
{'='*70}

Dataset: {self.dataset_path}
Output directory: {self.output_dir}
Quick mode: {self.quick}
Timestamp: {self.results['timestamp']}

{'='*70}
"""
        )

        # Run all benchmark categories
        self.run_throughput_benchmarks()
        self.run_transform_benchmarks()
        self.run_memory_benchmarks()
        self.run_scalability_tests()

        # Save and report
        self.save_results()
        self.generate_report()


def generate_synthetic_dataset(output_path: str, num_images: int = 1000) -> bool:
    """Generate synthetic dataset for benchmarking"""
    print(f"\nGenerating synthetic dataset with {num_images} images...")

    cmd = [
        sys.executable,
        "benchmarks/datasets/generate_synthetic.py",
        "--output",
        output_path,
        "--num-images",
        str(num_images),
        "--type",
        "tar",
        "--format",
        "jpeg",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(Path(__file__).parent.parent)
        )
        if result.returncode == 0:
            print(f"  Created: {output_path}")
            return True
        else:
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TurboLoader Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with auto-generated data
  python run_all_benchmarks.py --quick

  # Full benchmark with custom dataset
  python run_all_benchmarks.py --dataset /path/to/data.tar --full

  # Run only throughput benchmarks
  python run_all_benchmarks.py --throughput-only
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset (TAR file). If not provided, synthetic data is generated.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of images for synthetic dataset (default: 1000)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer iterations, smaller configs"
    )
    parser.add_argument(
        "--full", action="store_true", help="Full mode: comprehensive testing (slower)"
    )
    parser.add_argument(
        "--throughput-only", action="store_true", help="Run only throughput benchmarks"
    )
    parser.add_argument(
        "--transforms-only", action="store_true", help="Run only transform benchmarks"
    )
    parser.add_argument("--memory-only", action="store_true", help="Run only memory benchmarks")
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Get or create dataset
    if args.dataset:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset not found: {args.dataset}")
            sys.exit(1)
        dataset_path = args.dataset
    else:
        dataset_path = "/tmp/turboloader_benchmark.tar"
        num_images = 500 if args.quick else args.num_images
        if not os.path.exists(dataset_path):
            if not generate_synthetic_dataset(dataset_path, num_images):
                print("Failed to generate synthetic dataset")
                sys.exit(1)

    runner = BenchmarkRunner(dataset_path, output_dir=args.output_dir, quick=args.quick)

    # Run selected benchmarks or all
    if args.throughput_only:
        runner.run_throughput_benchmarks()
    elif args.transforms_only:
        runner.run_transform_benchmarks()
    elif args.memory_only:
        runner.run_memory_benchmarks()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
