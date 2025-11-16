#!/usr/bin/env python3
"""
Benchmark TurboLoader Pipeline with TAR dataset
"""
import turboloader
import time
import sys

if __name__ == '__main__':
    tar_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/benchmark_5k.tar'
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    print("=" * 80)
    print("TURBOLOADER PIPELINE BENCHMARK (TAR Dataset)")
    print("=" * 80)
    print(f"TAR file: {tar_path}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    # Create pipeline
    pipeline = turboloader.Pipeline(
        tar_paths=[tar_path],
        num_workers=num_workers,
        queue_size=256,
        decode_jpeg=True
    )

    print(f"\nTotal samples: {pipeline.total_samples()}")

    # Start pipeline
    pipeline.start()

    # Warmup
    print("\nWarming up...")
    for i in range(5):
        batch = pipeline.next_batch(batch_size)
        if len(batch) == 0:
            print(f"  Warning: Empty batch during warmup (iteration {i})")

    # Reset for clean benchmark
    pipeline.reset()
    time.sleep(0.5)  # Give pipeline time to reset

    # Benchmark
    print("\nRunning benchmark...")
    start_time = time.perf_counter()
    total_images = 0
    num_batches = 0

    while True:
        batch = pipeline.next_batch(batch_size)
        if len(batch) == 0:
            break

        total_images += len(batch)
        num_batches += 1

        if num_batches % 20 == 0:
            elapsed = time.perf_counter() - start_time
            throughput = total_images / elapsed
            print(f"  Processed {num_batches} batches, {total_images} images @ {throughput:.1f} img/s")

    total_time = time.perf_counter() - start_time
    throughput = total_images / total_time

    pipeline.stop()

    print("\n" + "=" * 80)
    print("TURBOLOADER RESULTS")
    print("=" * 80)
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} img/s")
    print(f"Batch time: {(total_time / num_batches) * 1000:.2f}ms")
    print("=" * 80)
