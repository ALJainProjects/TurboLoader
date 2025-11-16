"""
TurboLoader v2.0 Example Usage

Demonstrates how to use the high-performance v2.0 DataLoader for training.
"""

import turboloader_v2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def example_1_basic_usage():
    """Example 1: Basic iteration over batches"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Create DataLoader
    loader = turboloader_v2.DataLoader(
        tar_path="/tmp/benchmark_v2.tar",
        num_workers=4,
        batch_size=32
    )

    print(f"Total samples: {loader.total_samples()}")
    print()

    # Iterate over batches
    batch_count = 0
    while not loader.is_finished():
        batch = loader.next_batch()

        if len(batch) == 0:
            break

        batch_count += 1
        print(f"Batch {batch_count}: {len(batch)} samples")

        # Access first sample
        if len(batch) > 0:
            sample = batch[0]
            print(f"  Sample index: {sample['index']}")
            print(f"  Image shape: {sample['image'].shape if sample['is_decoded'] else 'N/A'}")

    loader.stop()
    print(f"\nProcessed {batch_count} batches")
    print()


def example_2_context_manager():
    """Example 2: Using context manager (recommended)"""
    print("=" * 80)
    print("Example 2: Context Manager")
    print("=" * 80)

    with turboloader_v2.DataLoader(
        tar_path="/tmp/benchmark_v2.tar",
        num_workers=4,
        batch_size=16
    ) as loader:
        for batch_idx, batch in enumerate(loader):
            print(f"Batch {batch_idx + 1}: {len(batch)} samples")

            if batch_idx >= 4:  # Process only first 5 batches
                break

    print("DataLoader automatically cleaned up\n")


def example_3_pytorch_training():
    """Example 3: Integration with PyTorch training loop"""
    print("=" * 80)
    print("Example 3: PyTorch Training Loop")
    print("=" * 80)

    # Simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(-1, 16 * 8 * 8)
            x = self.fc(x)
            return x

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Training loop
    with turboloader_v2.DataLoader(
        tar_path="/tmp/benchmark_v2.tar",
        num_workers=4,
        batch_size=32
    ) as loader:

        epoch_start = time.time()
        total_samples = 0

        for batch_idx, batch in enumerate(loader):
            # Convert batch to PyTorch tensors
            images = []
            for sample in batch:
                if sample['is_decoded']:
                    # Convert HWC to CHW and normalize
                    img = sample['image'].astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    images.append(img)

            if len(images) == 0:
                continue

            # Stack into batch tensor
            batch_tensor = torch.from_numpy(np.stack(images))

            # Dummy labels
            labels = torch.randint(0, 10, (len(images),))

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_tensor)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_samples += len(images)

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}: loss = {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        throughput = total_samples / epoch_time

        print(f"\nEpoch completed:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Samples: {total_samples}")
        print(f"  Throughput: {throughput:.2f} img/s")

    print()


def example_4_benchmark():
    """Example 4: Benchmark throughput"""
    print("=" * 80)
    print("Example 4: Throughput Benchmark")
    print("=" * 80)

    with turboloader_v2.DataLoader(
        tar_path="/tmp/benchmark_v2.tar",
        num_workers=4,
        batch_size=32
    ) as loader:

        start_time = time.time()
        total_samples = 0
        batch_count = 0

        for batch in loader:
            total_samples += len(batch)
            batch_count += 1

        elapsed_time = time.time() - start_time
        throughput = total_samples / elapsed_time

        print(f"Results:")
        print(f"  Total batches: {batch_count}")
        print(f"  Total samples: {total_samples}")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} img/s")

    print()


def example_5_worker_comparison():
    """Example 5: Compare different worker counts"""
    print("=" * 80)
    print("Example 5: Worker Count Comparison")
    print("=" * 80)

    worker_counts = [1, 2, 4, 8]

    for num_workers in worker_counts:
        with turboloader_v2.DataLoader(
            tar_path="/tmp/benchmark_v2.tar",
            num_workers=num_workers,
            batch_size=32
        ) as loader:

            start_time = time.time()
            total_samples = 0

            for batch in loader:
                total_samples += len(batch)

            elapsed_time = time.time() - start_time
            throughput = total_samples / elapsed_time

            print(f"Workers: {num_workers:2d} -> {throughput:6.2f} img/s ({elapsed_time:.2f}s)")

    print()


if __name__ == "__main__":
    print("\nTurboLoader v2.0 Examples")
    print(f"Version: {turboloader_v2.version()}")
    print(f"SIMD Support: {turboloader_v2.has_simd_support()}")
    print()

    # Run examples
    try:
        example_1_basic_usage()
        example_2_context_manager()
        example_3_pytorch_training()
        example_4_benchmark()
        example_5_worker_comparison()

        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except FileNotFoundError:
        print("\nError: /tmp/benchmark_v2.tar not found")
        print("Please create a test dataset first:")
        print("  python -c 'from create_test_dataset import create_dataset; create_dataset()'")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
