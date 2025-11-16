#!/usr/bin/env python3
"""Simple benchmark of TurboLoader Python API"""

import turboloader
import time

def benchmark(tar_files, num_workers, decode_jpeg=True):
    """Run benchmark with given configuration"""
    pipeline = turboloader.Pipeline(
        tar_paths=tar_files,
        num_workers=num_workers,
        decode_jpeg=decode_jpeg
    )

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

    throughput = total_samples / elapsed if elapsed > 0 else 0
    decode_speed = total_pixels / elapsed / 1e6 if elapsed > 0 else 0

    return total_samples, elapsed, throughput, decode_speed

def main():
    tar_files = [
        "/tmp/turboloader_real_images/shard_0000.tar",
        "/tmp/turboloader_real_images/shard_0001.tar"
    ]

    print("="*70)
    print("TurboLoader Python API Benchmark")
    print("="*70)
    print(f"\nDataset: {len(tar_files)} TAR files, 1000 JPEG images (224x224)")
    print()

    # Test different worker counts
    results = []
    for num_workers in [1, 2, 4, 8]:
        print(f"Workers: {num_workers}...", end=" ", flush=True)
        samples, elapsed, throughput, decode_speed = benchmark(tar_files, num_workers)
        results.append((num_workers, samples, elapsed, throughput, decode_speed))
        print(f"{throughput:.1f} samples/sec")

    # Print results table
    print()
    print("="*70)
    print("Results")
    print("="*70)
    print()
    print(f"{'Workers':<10} {'Samples':<10} {'Time':<10} {'Throughput':<20} {'Decode Speed':<15}")
    print("-"*70)
    for workers, samples, elapsed, throughput, decode_speed in results:
        print(f"{workers:<10} {samples:<10} {elapsed:>6.3f}s    {throughput:>10.1f} samples/sec  {decode_speed:>8.1f} Mpx/s")

    print()
    print("Baseline (from C++ benchmark): 2,558 samples/sec")
    print()
    best = max(results, key=lambda x: x[3])
    speedup = best[3] / 2558.0
    print(f"Best: {best[3]:.1f} samples/sec with {best[0]} workers = {speedup:.2f}x vs Python PIL")
    print()

if __name__ == "__main__":
    main()
