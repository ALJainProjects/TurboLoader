#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for TurboLoader v1.1.0+

Runs all benchmarks and collects real-world performance data:
- Transform benchmarks (all 19 transforms)
- Framework comparisons (PyTorch, TensorFlow)
- End-to-end pipeline benchmarks
- Memory profiling
- Scalability tests

Results are saved to benchmark_results/ for documentation updates.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

class BenchmarkRunner:
    def __init__(self, dataset_path: str, output_dir: str = "benchmark_results"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_path,
            "benchmarks": {}
        }

    def run_command(self, name: str, command: list, timeout: int = 600):
        """Run a benchmark command and capture output"""
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*80}\n")

        output_file = self.output_dir / f"{name.replace(' ', '_').lower()}.txt"

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/Users/arnavjain/turboloader"
            )
            elapsed = time.time() - start_time

            # Save output
            with open(output_file, 'w') as f:
                f.write(f"=== {name} ===\n")
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"Duration: {elapsed:.2f}s\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            # Extract key metrics from output
            metrics = self.extract_metrics(result.stdout + result.stderr)

            self.results["benchmarks"][name] = {
                "duration": elapsed,
                "exit_code": result.returncode,
                "metrics": metrics,
                "output_file": str(output_file)
            }

            print(f"✅ Completed in {elapsed:.2f}s")
            if metrics:
                print(f"   Metrics: {metrics}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout after {timeout}s")
            self.results["benchmarks"][name] = {
                "duration": timeout,
                "exit_code": -1,
                "error": "Timeout",
                "output_file": str(output_file)
            }
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["benchmarks"][name] = {
                "error": str(e),
                "output_file": str(output_file)
            }
            return False

    def extract_metrics(self, output: str) -> dict:
        """Extract performance metrics from benchmark output"""
        metrics = {}

        # Common patterns
        patterns = {
            "throughput": r"(\d+\.?\d*)\s*img/s",
            "fps": r"(\d+\.?\d*)\s*fps",
            "latency": r"(\d+\.?\d*)\s*ms",
            "memory": r"(\d+\.?\d*)\s*MB",
            "speedup": r"(\d+\.?\d*)x\s*faster",
        }

        import re
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))

        return metrics

    def run_transform_benchmarks(self):
        """Benchmark all 19 transforms"""
        print("\n" + "="*80)
        print("TRANSFORM BENCHMARKS")
        print("="*80)

        if not Path("benchmarks/benchmark_advanced_transforms.py").exists():
            print("⚠️  Transform benchmark script not found, skipping")
            return

        self.run_command(
            "Transform Benchmarks",
            ["python3.13", "benchmarks/benchmark_advanced_transforms.py", self.dataset_path]
        )

    def run_pytorch_comparison(self):
        """Compare against PyTorch DataLoader"""
        print("\n" + "="*80)
        print("PYTORCH COMPARISON")
        print("="*80)

        # PyTorch naive
        if Path("benchmarks/02_pytorch_naive.py").exists():
            self.run_command(
                "PyTorch Naive",
                ["python3.13", "benchmarks/02_pytorch_naive.py", self.dataset_path]
            )

        # PyTorch optimized
        if Path("benchmarks/03_pytorch_optimized.py").exists():
            self.run_command(
                "PyTorch Optimized",
                ["python3.13", "benchmarks/03_pytorch_optimized.py", self.dataset_path]
            )

        # TurboLoader
        if Path("benchmarks/05_turboloader.py").exists():
            self.run_command(
                "TurboLoader",
                ["python3.13", "benchmarks/05_turboloader.py", self.dataset_path]
            )

    def run_framework_comparison(self):
        """Compare against TensorFlow"""
        print("\n" + "="*80)
        print("FRAMEWORK COMPARISON")
        print("="*80)

        if Path("benchmarks/08_tensorflow.py").exists():
            self.run_command(
                "TensorFlow",
                ["python3.13", "benchmarks/08_tensorflow.py", self.dataset_path]
            )

    def run_scalability_tests(self):
        """Test scalability with different worker counts"""
        print("\n" + "="*80)
        print("SCALABILITY TESTS")
        print("="*80)

        # Test with 1, 2, 4, 8, 16 workers
        for workers in [1, 2, 4, 8, 16]:
            self.run_command(
                f"Scalability {workers} workers",
                ["python3.13", "-c", f"""
import turboloader
import time

loader = turboloader.DataLoader('{self.dataset_path}', batch_size=64, num_workers={workers})
start = time.time()
count = 0
for batch in loader:
    count += len(batch)
    if count >= 1000:
        break
elapsed = time.time() - start
print(f"Workers: {workers}, Throughput: {{count/elapsed:.1f}} img/s")
"""]
            )

    def run_memory_profiling(self):
        """Profile memory usage"""
        print("\n" + "="*80)
        print("MEMORY PROFILING")
        print("="*80)

        # This would need memory_profiler package
        # For now, just track basic stats
        pass

    def save_results(self):
        """Save all results to JSON"""
        output_file = self.output_dir / "all_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")

        # Print summary
        print("BENCHMARK SUMMARY:")
        print(f"Total benchmarks run: {len(self.results['benchmarks'])}")

        successful = sum(1 for b in self.results['benchmarks'].values()
                        if b.get('exit_code') == 0)
        print(f"Successful: {successful}")
        print(f"Failed: {len(self.results['benchmarks']) - successful}")

        # Print key metrics
        print("\nKEY METRICS:")
        for name, data in self.results['benchmarks'].items():
            if 'metrics' in data and data['metrics']:
                print(f"\n{name}:")
                for metric, value in data['metrics'].items():
                    print(f"  {metric}: {value}")

    def run_all(self):
        """Run complete benchmark suite"""
        print(f"""
{'='*80}
TURBOLOADER COMPREHENSIVE BENCHMARK SUITE
{'='*80}

Dataset: {self.dataset_path}
Output directory: {self.output_dir}
Timestamp: {self.results['timestamp']}

{'='*80}
""")

        # Run all benchmark categories
        self.run_transform_benchmarks()
        self.run_pytorch_comparison()
        self.run_framework_comparison()
        self.run_scalability_tests()
        self.run_memory_profiling()

        # Save results
        self.save_results()

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_all_benchmarks.py <dataset.tar>")
        print("\nExample:")
        print("  python run_all_benchmarks.py /tmp/benchmark_1000.tar")
        sys.exit(1)

    dataset_path = sys.argv[1]

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    runner = BenchmarkRunner(dataset_path)
    runner.run_all()

if __name__ == "__main__":
    main()
