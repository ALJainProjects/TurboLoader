#!/usr/bin/env python3
"""Benchmark TurboLoader vs PyTorch DataLoader"""

import sys
sys.path.insert(0, '/Users/arnavjain/turboloader/build/python')

import turboloader
import time
import tarfile
import io
from PIL import Image
import numpy as np

def benchmark_turboloader(tar_files, num_workers=4):
    """Benchmark TurboLoader"""
    print(f"\n{'='*60}")
    print(f"TurboLoader (workers={num_workers})")
    print(f"{'='*60}")

    pipeline = turboloader.Pipeline(
        tar_paths=tar_files,
        num_workers=num_workers,
        decode_jpeg=True
    )

    print(f"Total samples: {len(pipeline)}")

    pipeline.start()

    start = time.time()
    total_samples = 0
    total_pixels = 0

    while True:
        batch = pipeline.next_batch(32)
        if not batch:
            break

        total_samples += len(batch)
        for sample in batch:
            if sample.width > 0:
                total_pixels += sample.width * sample.height * sample.channels

    elapsed = time.time() - start

    pipeline.stop()

    print(f"  Samples: {total_samples}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {total_samples / elapsed:.1f} samples/sec")
    print(f"  Decode speed: {total_pixels / elapsed / 1e6:.1f} Mpixels/sec")

    return total_samples / elapsed

def benchmark_pytorch_style(tar_files):
    """Benchmark Python + PIL (PyTorch DataLoader style)"""
    print(f"\n{'='*60}")
    print(f"Python + PIL (PyTorch style, single-threaded)")
    print(f"{'='*60}")

    # Load all samples
    samples = []
    for tar_path in tar_files:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            # Group by sample key
            sample_files = {}
            for member in members:
                if member.isfile():
                    parts = member.name.split('.')
                    if len(parts) >= 2:
                        key = '.'.join(parts[:-1])
                        ext = parts[-1]
                        if key not in sample_files:
                            sample_files[key] = {}
                        sample_files[key][ext] = member

            samples.extend(sample_files.values())

    print(f"Total samples: {len(samples)}")

    start = time.time()
    total_samples = 0
    total_pixels = 0

    # Process in batches like DataLoader
    batch_size = 32
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]

        for sample in batch_samples:
            if 'jpg' in sample:
                # Read and decode JPEG
                tar_path = tar_files[0] if i < len(samples) // len(tar_files) else tar_files[1]
                with tarfile.open(tar_path, 'r') as tar:
                    member = sample['jpg']
                    f = tar.extractfile(member)
                    data = f.read()
                    img = Image.open(io.BytesIO(data))
                    arr = np.array(img)

                    total_samples += 1
                    total_pixels += arr.size

    elapsed = time.time() - start

    print(f"  Samples: {total_samples}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {total_samples / elapsed:.1f} samples/sec")
    print(f"  Decode speed: {total_pixels / elapsed / 1e6:.1f} Mpixels/sec")

    return total_samples / elapsed

def main():
    tar_files = [
        "/tmp/turboloader_real_images/shard_0000.tar",
        "/tmp/turboloader_real_images/shard_0001.tar"
    ]

    print("\n" + "="*60)
    print("TurboLoader vs PyTorch DataLoader Benchmark")
    print("="*60)
    print(f"\nDataset: {len(tar_files)} TAR files")
    print("Image size: 224x224 JPEG")

    # Baseline
    baseline_throughput = benchmark_pytorch_style(tar_files)

    # TurboLoader with different worker counts
    results = []
    for num_workers in [1, 2, 4, 8]:
        throughput = benchmark_turboloader(tar_files, num_workers)
        speedup = throughput / baseline_throughput
        results.append((num_workers, throughput, speedup))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline (Python + PIL): {baseline_throughput:.1f} samples/sec\n")
    print("TurboLoader Results:")
    print(f"{'Workers':<10} {'Throughput':<20} {'Speedup':<10}")
    print("-" * 40)
    for workers, throughput, speedup in results:
        print(f"{workers:<10} {throughput:>10.1f} samples/sec {speedup:>7.2f}x")

    best_workers, best_throughput, best_speedup = max(results, key=lambda x: x[2])
    print(f"\nBest: {best_speedup:.2f}x speedup with {best_workers} workers")

if __name__ == "__main__":
    main()
