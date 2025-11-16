#!/usr/bin/env python3
"""
Benchmark TurboLoader Pipeline with TAR dataset
"""
import turboloader
import time
import sys

if __name__ == '__main__':
    tar_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/benchmark_1k.tar'
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
    time.sleep(1.0)  # Give pipeline time to start and fill queue

    # Benchmark (no warmup to avoid reset issues)
    print("\nRunning benchmark...")
    start_time = time.perf_counter()
    total_images = 0
    num_batches = 0

    # Calculate how many batches we need to process all samples
    total_samples = pipeline.total_samples()
    expected_batches = (total_samples + batch_size - 1) // batch_size

    for _ in range(expected_batches):
        batch = pipeline.next_batch(batch_size)
        if len(batch) == 0:
            break  # Stop if pipeline is exhausted

        total_images += len(batch)
        num_batches += 1

        if num_batches % 20 == 0:
            elapsed = time.perf_counter() - start_time
            throughput = total_images / elapsed if elapsed > 0 else 0
            print(f"  Processed {num_batches} batches, {total_images} images @ {throughput:.1f} img/s")

    total_time = time.perf_counter() - start_time

    pipeline.stop()

    print("\n" + "=" * 80)
    print("TURBOLOADER RESULTS")
    print("=" * 80)
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f}s")

    # Handle edge cases for division by zero
    if total_images > 0 and total_time > 0:
        throughput = total_images / total_time
        print(f"Throughput: {throughput:.2f} img/s")
    else:
        print(f"Throughput: N/A (no images processed)")

    if num_batches > 0 and total_time > 0:
        batch_time = (total_time / num_batches) * 1000
        print(f"Batch time: {batch_time:.2f}ms")
    else:
        print(f"Batch time: N/A (no batches processed)")

    print("=" * 80)
