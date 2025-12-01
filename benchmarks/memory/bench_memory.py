#!/usr/bin/env python3
"""
Memory Usage Benchmark

Measures memory consumption during data loading for:
- TurboLoader
- PyTorch DataLoader
- WebDataset (if available)
"""

import argparse
import gc
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory profiling will be limited.")

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import turboloader

    HAS_TURBOLOADER = True
except ImportError:
    HAS_TURBOLOADER = False


@dataclass
class MemoryResult:
    """Memory benchmark result"""

    library: str
    config: Dict[str, Any]
    baseline_mb: float
    peak_mb: float
    delta_mb: float
    avg_mb: float
    samples_per_mb: float
    timestamp: str


class MemoryMonitor:
    """Monitor memory usage in a separate thread"""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.measurements = []
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring"""
        if not HAS_PSUTIL:
            return

        self.measurements = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _monitor(self):
        """Monitoring loop"""
        process = psutil.Process()
        while self.running:
            try:
                mem = process.memory_info().rss / (1024 * 1024)  # MB
                self.measurements.append(mem)
            except:
                pass
            time.sleep(self.interval)

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        if not self.measurements:
            return {
                "min_mb": 0,
                "max_mb": 0,
                "avg_mb": 0,
                "std_mb": 0,
            }

        measurements = np.array(self.measurements)
        return {
            "min_mb": np.min(measurements),
            "max_mb": np.max(measurements),
            "avg_mb": np.mean(measurements),
            "std_mb": np.std(measurements),
        }


def get_current_memory_mb() -> float:
    """Get current memory usage in MB"""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0


def force_gc():
    """Force garbage collection"""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_turboloader_memory(
    tar_path: str, batch_size: int = 64, num_workers: int = 4, num_batches: int = 50
) -> Optional[MemoryResult]:
    """Benchmark TurboLoader memory usage"""
    if not HAS_TURBOLOADER:
        return None

    force_gc()
    time.sleep(0.5)

    baseline = get_current_memory_mb()
    monitor = MemoryMonitor(interval=0.05)

    try:
        transforms = turboloader.Compose(
            [
                turboloader.Resize(256, 256),
                turboloader.RandomCrop(224, 224),
                turboloader.RandomHorizontalFlip(0.5),
                turboloader.ImageNetNormalize(),
            ]
        )

        loader = turboloader.DataLoader(
            tar_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        monitor.start()
        total_samples = 0

        for i, batch in enumerate(loader):
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            if isinstance(images, np.ndarray):
                total_samples += images.shape[0]
            else:
                total_samples += batch_size

            if i >= num_batches - 1:
                break

        monitor.stop()
        stats = monitor.get_stats()

        peak = stats["max_mb"]
        delta = peak - baseline

        return MemoryResult(
            library="turboloader",
            config={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "num_batches": num_batches,
            },
            baseline_mb=baseline,
            peak_mb=peak,
            delta_mb=delta,
            avg_mb=stats["avg_mb"],
            samples_per_mb=total_samples / delta if delta > 0 else float("inf"),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        print(f"TurboLoader error: {e}")
        monitor.stop()
        return None


def benchmark_pytorch_memory(
    tar_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    num_batches: int = 50,
    cached: bool = True,
) -> Optional[MemoryResult]:
    """Benchmark PyTorch DataLoader memory usage"""
    if not HAS_TORCH:
        return None

    import tarfile
    from io import BytesIO
    from PIL import Image
    import torchvision.transforms as T

    class TarDataset(Dataset):
        def __init__(self, tar_path, transform=None, cache=True):
            self.tar_path = tar_path
            self.transform = transform
            self.samples = []
            self.cache = {} if cache else None

            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith((".jpg", ".jpeg", ".png", ".JPEG", ".JPG")):
                        self.samples.append(member.name)
                        if cache:
                            f = tar.extractfile(member)
                            self.cache[member.name] = f.read()

            self.samples.sort()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            filename = self.samples[idx]

            if self.cache:
                data = self.cache[filename]
            else:
                with tarfile.open(self.tar_path, "r") as tar:
                    f = tar.extractfile(tar.getmember(filename))
                    data = f.read()

            img = Image.open(BytesIO(data)).convert("RGB")
            if self.transform:
                img = self.transform(img)

            return img, idx % 1000

    force_gc()
    time.sleep(0.5)

    baseline = get_current_memory_mb()
    monitor = MemoryMonitor(interval=0.05)

    try:
        transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        dataset = TarDataset(tar_path, transform=transform, cache=cached)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=False,
        )

        monitor.start()
        total_samples = 0

        for i, (images, labels) in enumerate(loader):
            total_samples += images.shape[0]
            if i >= num_batches - 1:
                break

        monitor.stop()
        stats = monitor.get_stats()

        peak = stats["max_mb"]
        delta = peak - baseline

        return MemoryResult(
            library="pytorch" + ("_cached" if cached else ""),
            config={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "num_batches": num_batches,
                "cached": cached,
            },
            baseline_mb=baseline,
            peak_mb=peak,
            delta_mb=delta,
            avg_mb=stats["avg_mb"],
            samples_per_mb=total_samples / delta if delta > 0 else float("inf"),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        print(f"PyTorch error: {e}")
        import traceback

        traceback.print_exc()
        monitor.stop()
        return None


def run_memory_benchmarks(
    tar_path: str,
    batch_sizes: List[int] = [32, 64, 128],
    num_workers: int = 4,
    num_batches: int = 50,
) -> List[MemoryResult]:
    """Run all memory benchmarks"""
    results = []

    print(f"\n{'='*60}")
    print("Memory Usage Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {tar_path}")
    print(f"Workers: {num_workers}")
    print(f"Batches: {num_batches}")

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        # TurboLoader
        if HAS_TURBOLOADER:
            print("  TurboLoader...")
            result = benchmark_turboloader_memory(tar_path, batch_size, num_workers, num_batches)
            if result:
                print(f"    Peak: {result.peak_mb:.1f} MB, Delta: {result.delta_mb:.1f} MB")
                results.append(result)

        # PyTorch (cached)
        if HAS_TORCH:
            print("  PyTorch (cached)...")
            result = benchmark_pytorch_memory(
                tar_path, batch_size, num_workers, num_batches, cached=True
            )
            if result:
                print(f"    Peak: {result.peak_mb:.1f} MB, Delta: {result.delta_mb:.1f} MB")
                results.append(result)

        # PyTorch (uncached)
        if HAS_TORCH:
            print("  PyTorch (uncached)...")
            result = benchmark_pytorch_memory(
                tar_path, batch_size, num_workers, num_batches, cached=False
            )
            if result:
                print(f"    Peak: {result.peak_mb:.1f} MB, Delta: {result.delta_mb:.1f} MB")
                results.append(result)

        force_gc()
        time.sleep(1.0)

    return results


def save_results(results: List[MemoryResult], output_path: str):
    """Save results to JSON"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[MemoryResult]):
    """Print summary table"""
    print(f"\n{'='*80}")
    print("MEMORY USAGE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Library':>20} {'Batch':>8} {'Peak MB':>12} {'Delta MB':>12} {'Samples/MB':>12}")
    print("-" * 80)

    for r in results:
        batch = r.config.get("batch_size", "N/A")
        print(
            f"{r.library:>20} {batch:>8} {r.peak_mb:>12.1f} {r.delta_mb:>12.1f} {r.samples_per_mb:>12.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Memory Usage Benchmark")
    parser.add_argument("--tar-path", type=str, required=True, help="Path to TAR dataset")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[32, 64, 128], help="Batch sizes to test"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--num-batches", type=int, default=50, help="Number of batches per benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/memory/memory.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    if not HAS_PSUTIL:
        print("Error: psutil is required for memory benchmarking")
        print("Install with: pip install psutil")
        sys.exit(1)

    results = run_memory_benchmarks(
        args.tar_path,
        batch_sizes=args.batch_sizes,
        num_workers=args.workers,
        num_batches=args.num_batches,
    )

    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
