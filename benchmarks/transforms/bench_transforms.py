#!/usr/bin/env python3
"""
Transform Performance Benchmark

Compares individual transform performance between:
- TurboLoader (SIMD-accelerated)
- torchvision
- Albumentations (if available)
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
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
    from PIL import Image

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    import turboloader

    HAS_TURBOLOADER = True
except ImportError:
    HAS_TURBOLOADER = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""

    library: str
    transform: str
    time_per_image_ms: float
    throughput: float
    config: Dict[str, Any]
    timestamp: str


def generate_test_images(num_images: int, size: tuple = (256, 256)) -> List[np.ndarray]:
    """Generate test images as numpy arrays"""
    images = []
    for _ in range(num_images):
        img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        images.append(img)
    return images


def benchmark_transform(
    transform_fn, images: List[np.ndarray], warmup_runs: int = 10, timed_runs: int = 100
) -> Dict[str, float]:
    """Benchmark a single transform"""
    gc.collect()

    # Warmup
    for i in range(min(warmup_runs, len(images))):
        _ = transform_fn(images[i % len(images)])

    # Timed runs
    times = []
    for i in range(timed_runs):
        img = images[i % len(images)]
        start = time.perf_counter()
        _ = transform_fn(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "throughput": 1.0 / np.mean(times),
    }


def run_turboloader_transforms(images: List[np.ndarray], timed_runs: int) -> List[BenchmarkResult]:
    """Benchmark TurboLoader transforms"""
    if not HAS_TURBOLOADER:
        return []

    results = []
    timestamp = datetime.now().isoformat()

    transforms = [
        ("Resize(256, 256)", turboloader.Resize(256, 256)),
        ("Resize(224, 224)", turboloader.Resize(224, 224)),
        ("CenterCrop(224, 224)", turboloader.CenterCrop(224, 224)),
        ("RandomCrop(224, 224)", turboloader.RandomCrop(224, 224)),
        ("RandomHorizontalFlip(1.0)", turboloader.RandomHorizontalFlip(1.0)),
        ("RandomVerticalFlip(1.0)", turboloader.RandomVerticalFlip(1.0)),
        ("ColorJitter(0.4, 0.4, 0.4, 0.2)", turboloader.ColorJitter(0.4, 0.4, 0.4, 0.2)),
        ("GaussianBlur(5)", turboloader.GaussianBlur(5)),
        ("Grayscale", turboloader.Grayscale()),
        ("ImageNetNormalize", turboloader.ImageNetNormalize()),
    ]

    print("\nTurboLoader Transforms:")
    print("-" * 60)

    for name, transform in transforms:
        try:
            stats = benchmark_transform(
                lambda img, t=transform: t.apply(img), images, timed_runs=timed_runs
            )
            print(f"  {name:40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)")

            results.append(
                BenchmarkResult(
                    library="turboloader",
                    transform=name,
                    time_per_image_ms=stats["mean_ms"],
                    throughput=stats["throughput"],
                    config={"input_size": images[0].shape[:2]},
                    timestamp=timestamp,
                )
            )
        except Exception as e:
            print(f"  {name:40} Error: {e}")

    # Test pipeline
    try:
        pipeline = turboloader.Compose(
            [
                turboloader.Resize(256, 256),
                turboloader.RandomCrop(224, 224),
                turboloader.RandomHorizontalFlip(0.5),
                turboloader.ColorJitter(0.4, 0.4, 0.4, 0.2),
                turboloader.ImageNetNormalize(),
            ]
        )
        stats = benchmark_transform(lambda img: pipeline.apply(img), images, timed_runs=timed_runs)
        print(
            f"  {'Full Pipeline (5 transforms)':40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)"
        )

        results.append(
            BenchmarkResult(
                library="turboloader",
                transform="Full Pipeline (5 transforms)",
                time_per_image_ms=stats["mean_ms"],
                throughput=stats["throughput"],
                config={"input_size": images[0].shape[:2]},
                timestamp=timestamp,
            )
        )
    except Exception as e:
        print(f"  {'Full Pipeline':40} Error: {e}")

    # Test pipe operator pipeline
    try:
        pipeline = (
            turboloader.Resize(256, 256)
            | turboloader.RandomCrop(224, 224)
            | turboloader.RandomHorizontalFlip(0.5)
            | turboloader.ColorJitter(0.4, 0.4, 0.4, 0.2)
            | turboloader.ImageNetNormalize()
        )
        stats = benchmark_transform(lambda img: pipeline.apply(img), images, timed_runs=timed_runs)
        print(
            f"  {'Pipe Operator Pipeline':40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)"
        )

        results.append(
            BenchmarkResult(
                library="turboloader_pipe",
                transform="Pipe Operator Pipeline",
                time_per_image_ms=stats["mean_ms"],
                throughput=stats["throughput"],
                config={"input_size": images[0].shape[:2]},
                timestamp=timestamp,
            )
        )
    except Exception as e:
        print(f"  {'Pipe Operator Pipeline':40} Error: {e}")

    return results


def run_torchvision_transforms(images: List[np.ndarray], timed_runs: int) -> List[BenchmarkResult]:
    """Benchmark torchvision transforms"""
    if not HAS_TORCHVISION:
        return []

    results = []
    timestamp = datetime.now().isoformat()

    # Convert numpy images to PIL for torchvision
    pil_images = [Image.fromarray(img) for img in images]

    transforms = [
        ("Resize(256, 256)", T.Resize((256, 256))),
        ("Resize(224, 224)", T.Resize((224, 224))),
        ("CenterCrop(224)", T.CenterCrop(224)),
        ("RandomCrop(224)", T.RandomCrop(224)),
        ("RandomHorizontalFlip(1.0)", T.RandomHorizontalFlip(1.0)),
        ("RandomVerticalFlip(1.0)", T.RandomVerticalFlip(1.0)),
        ("ColorJitter(0.4, 0.4, 0.4, 0.2)", T.ColorJitter(0.4, 0.4, 0.4, 0.2)),
        ("GaussianBlur(5)", T.GaussianBlur(5)),
        ("Grayscale", T.Grayscale(num_output_channels=3)),
        (
            "Normalize",
            T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        ),
    ]

    print("\ntorchvision Transforms:")
    print("-" * 60)

    for name, transform in transforms:
        try:
            if "Normalize" in name:
                # Normalize needs tensor input
                stats = benchmark_transform(
                    lambda img, t=transform: t(img), pil_images, timed_runs=timed_runs
                )
            else:
                stats = benchmark_transform(
                    lambda img, t=transform: t(img), pil_images, timed_runs=timed_runs
                )
            print(f"  {name:40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)")

            results.append(
                BenchmarkResult(
                    library="torchvision",
                    transform=name,
                    time_per_image_ms=stats["mean_ms"],
                    throughput=stats["throughput"],
                    config={"input_size": images[0].shape[:2]},
                    timestamp=timestamp,
                )
            )
        except Exception as e:
            print(f"  {name:40} Error: {e}")

    # Test pipeline
    try:
        pipeline = T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.4, 0.4, 0.4, 0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        stats = benchmark_transform(lambda img: pipeline(img), pil_images, timed_runs=timed_runs)
        print(
            f"  {'Full Pipeline (5 transforms)':40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)"
        )

        results.append(
            BenchmarkResult(
                library="torchvision",
                transform="Full Pipeline (5 transforms)",
                time_per_image_ms=stats["mean_ms"],
                throughput=stats["throughput"],
                config={"input_size": images[0].shape[:2]},
                timestamp=timestamp,
            )
        )
    except Exception as e:
        print(f"  {'Full Pipeline':40} Error: {e}")

    return results


def run_albumentations_transforms(
    images: List[np.ndarray], timed_runs: int
) -> List[BenchmarkResult]:
    """Benchmark Albumentations transforms"""
    if not HAS_ALBUMENTATIONS:
        return []

    results = []
    timestamp = datetime.now().isoformat()

    transforms = [
        ("Resize(256, 256)", A.Resize(256, 256)),
        ("Resize(224, 224)", A.Resize(224, 224)),
        ("CenterCrop(224, 224)", A.CenterCrop(224, 224)),
        ("RandomCrop(224, 224)", A.RandomCrop(224, 224)),
        ("HorizontalFlip(p=1.0)", A.HorizontalFlip(p=1.0)),
        ("VerticalFlip(p=1.0)", A.VerticalFlip(p=1.0)),
        (
            "ColorJitter",
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1.0),
        ),
        ("GaussianBlur(blur_limit=5)", A.GaussianBlur(blur_limit=5, p=1.0)),
        ("ToGray", A.ToGray(p=1.0)),
        ("Normalize", A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    ]

    print("\nAlbumentations Transforms:")
    print("-" * 60)

    for name, transform in transforms:
        try:
            stats = benchmark_transform(
                lambda img, t=transform: t(image=img)["image"], images, timed_runs=timed_runs
            )
            print(f"  {name:40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)")

            results.append(
                BenchmarkResult(
                    library="albumentations",
                    transform=name,
                    time_per_image_ms=stats["mean_ms"],
                    throughput=stats["throughput"],
                    config={"input_size": images[0].shape[:2]},
                    timestamp=timestamp,
                )
            )
        except Exception as e:
            print(f"  {name:40} Error: {e}")

    # Test pipeline
    try:
        pipeline = A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        stats = benchmark_transform(
            lambda img: pipeline(image=img)["image"], images, timed_runs=timed_runs
        )
        print(
            f"  {'Full Pipeline (5 transforms)':40} {stats['mean_ms']:8.3f} ms ({stats['throughput']:8.1f} img/sec)"
        )

        results.append(
            BenchmarkResult(
                library="albumentations",
                transform="Full Pipeline (5 transforms)",
                time_per_image_ms=stats["mean_ms"],
                throughput=stats["throughput"],
                config={"input_size": images[0].shape[:2]},
                timestamp=timestamp,
            )
        )
    except Exception as e:
        print(f"  {'Full Pipeline':40} Error: {e}")

    return results


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Group by transform
    transform_map = {}
    for r in results:
        key = r.transform
        if key not in transform_map:
            transform_map[key] = {}
        transform_map[key][r.library] = r.time_per_image_ms

    print(f"{'Transform':40} {'TurboLoader':>12} {'torchvision':>12} {'Speedup':>10}")
    print("-" * 80)

    for transform, libs in sorted(transform_map.items()):
        tl = libs.get("turboloader", libs.get("turboloader_pipe", float("inf")))
        tv = libs.get("torchvision", float("inf"))

        tl_str = f"{tl:.3f} ms" if tl != float("inf") else "N/A"
        tv_str = f"{tv:.3f} ms" if tv != float("inf") else "N/A"

        if tl != float("inf") and tv != float("inf"):
            speedup = tv / tl
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "N/A"

        print(f"{transform:40} {tl_str:>12} {tv_str:>12} {speedup_str:>10}")


def main():
    parser = argparse.ArgumentParser(description="Transform Performance Benchmark")
    parser.add_argument(
        "--num-images", type=int, default=50, help="Number of test images to generate"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Input image size (width height)",
    )
    parser.add_argument(
        "--timed-runs", type=int, default=100, help="Number of timed runs per transform"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/transforms/transforms.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Transform Performance Benchmark")
    print(f"{'='*60}")
    print(f"Test images: {args.num_images}")
    print(f"Image size: {args.image_size[0]}x{args.image_size[1]}")
    print(f"Timed runs per transform: {args.timed_runs}")

    # Generate test images
    print("\nGenerating test images...")
    images = generate_test_images(args.num_images, tuple(args.image_size))

    # Run benchmarks
    results = []

    if HAS_TURBOLOADER:
        results.extend(run_turboloader_transforms(images, args.timed_runs))
    else:
        print("\nTurboLoader not available, skipping...")

    if HAS_TORCHVISION:
        results.extend(run_torchvision_transforms(images, args.timed_runs))
    else:
        print("\ntorchvision not available, skipping...")

    if HAS_ALBUMENTATIONS:
        results.extend(run_albumentations_transforms(images, args.timed_runs))
    else:
        print("\nAlbumentations not available, skipping...")

    # Save and print results
    save_results(results, args.output)
    print_comparison(results)


if __name__ == "__main__":
    main()
