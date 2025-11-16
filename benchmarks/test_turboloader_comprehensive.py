#!/usr/bin/env python3
"""
Comprehensive TurboLoader Benchmark
Tests actual end-to-end performance with JPEG decoding
"""

import sys
import time
sys.path.insert(0, '/Users/arnavjain/turboloader/build/lib.macosx-15.0-arm64-cpython-313')

import turboloader

def benchmark_turboloader(tar_path, num_workers=8, batch_size=256, decode=True, transforms=False):
    """Benchmark TurboLoader with various configurations"""

    print(f"\n{'='*70}")
    print(f"TurboLoader Benchmark")
    print(f"{'='*70}")
    print(f"Workers: {num_workers}, Batch Size: {batch_size}, Decode: {decode}, Transforms: {transforms}")

    # Create pipeline
    if transforms:
        config = turboloader.TransformConfig()
        config.enable_resize = True
        config.resize_width = 224
        config.resize_height = 224
        config.enable_normalize = True
        config.mean = [0.485, 0.456, 0.406]
        config.std = [0.229, 0.224, 0.225]

        pipeline = turboloader.Pipeline(
            tar_paths=[tar_path],
            num_workers=num_workers,
            decode_jpeg=decode,
            enable_simd_transforms=True,
            transform_config=config,
            queue_size=512
        )
    else:
        pipeline = turboloader.Pipeline(
            tar_paths=[tar_path],
            num_workers=num_workers,
            decode_jpeg=decode,
            queue_size=512
        )

    total_samples = pipeline.total_samples()
    print(f"Total samples: {total_samples}")

    pipeline.start()

    # Warmup
    print("Warming up...")
    for _ in range(5):
        batch = pipeline.next_batch(batch_size)

    # Benchmark
    print("Benchmarking...")
    samples_processed = 0
    start_time = time.time()

    while samples_processed < total_samples:
        batch = pipeline.next_batch(batch_size)
        if not batch:
            break
        samples_processed += len(batch)

    elapsed = time.time() - start_time

    pipeline.stop()

    throughput = samples_processed / elapsed
    time_per_sample = (elapsed / samples_processed) * 1000  # ms

    print(f"\nResults:")
    print(f"  Samples processed: {samples_processed}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Throughput: {throughput:.0f} images/second")
    print(f"  Time per image: {time_per_sample:.2f} ms")

    return {
        'throughput': throughput,
        'time_per_sample': time_per_sample,
        'total_time': elapsed
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_turboloader_comprehensive.py <tar_path>")
        sys.exit(1)

    tar_path = sys.argv[1]

    print(f"TurboLoader v{turboloader.__version__}")
    print(f"Testing with: {tar_path}")

    # Test 1: No decode, no transforms (raw TAR reading)
    print("\n" + "="*70)
    print("TEST 1: Raw TAR reading (no decode, no transforms)")
    print("="*70)
    results_raw = benchmark_turboloader(tar_path, num_workers=8, batch_size=256, decode=False, transforms=False)

    # Test 2: JPEG decoding only
    print("\n" + "="*70)
    print("TEST 2: JPEG Decoding (no transforms)")
    print("="*70)
    results_decode = benchmark_turboloader(tar_path, num_workers=8, batch_size=256, decode=True, transforms=False)

    # Test 3: JPEG + SIMD transforms
    print("\n" + "="*70)
    print("TEST 3: JPEG Decoding + SIMD Transforms (resize + normalize)")
    print("="*70)
    results_full = benchmark_turboloader(tar_path, num_workers=8, batch_size=256, decode=True, transforms=True)

    # Test 4: Worker scaling
    print("\n" + "="*70)
    print("TEST 4: Worker Scaling (decode + transforms)")
    print("="*70)
    for workers in [1, 2, 4, 8, 16]:
        print(f"\nWorkers: {workers}")
        result = benchmark_turboloader(tar_path, num_workers=workers, batch_size=256, decode=True, transforms=True)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Raw TAR reading:          {results_raw['throughput']:.0f} img/s")
    print(f"JPEG decoding:            {results_decode['throughput']:.0f} img/s")
    print(f"JPEG + SIMD transforms:   {results_full['throughput']:.0f} img/s")
    print("="*70)
