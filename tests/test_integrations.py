"""
Integration tests for TurboLoader multi-framework support

Tests TensorFlow/Keras and JAX/Flax integrations
"""

import sys
import os
import tempfile
import tarfile
import numpy as np
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

def create_test_tar(num_images=10):
    """Create a small TAR file for testing"""
    tmpdir = tempfile.mkdtemp()
    tar_path = os.path.join(tmpdir, 'test.tar')

    # Create test images
    images_dir = os.path.join(tmpdir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    for i in range(num_images):
        # Create a simple test image (random RGB)
        img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = os.path.join(images_dir, f'img_{i:04d}.npy')
        np.save(img_path, img_data)

    # Create TAR
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_images):
            img_path = os.path.join(images_dir, f'img_{i:04d}.npy')
            tar.add(img_path, arcname=f'img_{i:04d}.npy')

    return tar_path

def test_tensorflow_dataloader():
    """Test TensorFlow/Keras integration"""
    print("\n" + "="*80)
    print("Testing TensorFlow/Keras Integration")
    print("="*80)

    try:
        import tensorflow as tf
        from tensorflow_dataloader import TensorFlowDataLoader, KerasSequence

        tar_path = create_test_tar(20)

        # Test TensorFlowDataLoader
        print("\n[1/3] Testing TensorFlowDataLoader...")
        loader = TensorFlowDataLoader(
            tar_path,
            batch_size=4,
            num_workers=2,
            shuffle=False
        )

        dataset = loader.as_dataset()
        print(f"  Created tf.data.Dataset: {dataset}")

        # Consume a few batches
        batch_count = 0
        for images, labels in dataset.take(3):
            batch_count += 1
            print(f"  Batch {batch_count}: images shape={images.shape}, labels shape={labels.shape}")
            assert images.shape[0] <= 4  # Batch size
            assert len(labels.shape) == 1  # Labels should be 1D

        print(f"  Successfully loaded {batch_count} batches from TensorFlow DataLoader")

        # Test KerasSequence
        print("\n[2/3] Testing KerasSequence...")
        sequence = KerasSequence(
            tar_path,
            batch_size=4,
            num_workers=2
        )

        print(f"  Created KerasSequence with {len(sequence)} batches")

        # Get a few batches
        for i in range(min(3, len(sequence))):
            images, labels = sequence[i]
            print(f"  Batch {i}: images shape={images.shape}, labels shape={labels.shape}")

        print("  Successfully loaded batches from KerasSequence")

        print("\n[3/3] Testing with model.fit()...")
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, None, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("  Created simple Keras model")
        print("  Training for 1 epoch...")

        # Train for 1 epoch
        history = model.fit(sequence, epochs=1, verbose=0)

        print(f"  Training completed! Loss: {history.history['loss'][0]:.4f}")

        print("\n" + "="*80)
        print("TensorFlow/Keras Integration: PASSED ✓")
        print("="*80)

        # Cleanup
        loader.__exit__(None, None, None)
        os.remove(tar_path)

        return True

    except ImportError as e:
        print(f"\nSkipping TensorFlow tests: {e}")
        return None
    except Exception as e:
        print(f"\nTensorFlow tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jax_dataloader():
    """Test JAX/Flax integration"""
    print("\n" + "="*80)
    print("Testing JAX/Flax Integration")
    print("="*80)

    try:
        import jax
        import jax.numpy as jnp
        from jax_dataloader import JAXDataLoader, FlaxDataLoader, prefetch_to_device

        tar_path = create_test_tar(20)

        # Test JAXDataLoader
        print("\n[1/3] Testing JAXDataLoader...")
        loader = JAXDataLoader(
            tar_path,
            batch_size=4,
            num_workers=2,
            device=jax.devices()[0]
        )

        print(f"  Created JAXDataLoader targeting device: {jax.devices()[0]}")

        # Consume a few batches
        batch_count = 0
        for batch in loader:
            if batch_count >= 3:
                break
            batch_count += 1
            print(f"  Batch {batch_count}: image shape={batch['image'].shape}, "
                  f"device={batch['image'].device()}")
            assert isinstance(batch['image'], jnp.ndarray)
            assert batch['image'].shape[0] <= 4  # Batch size

        print(f"  Successfully loaded {batch_count} batches with JAXDataLoader")

        loader.__exit__(None, None, None)

        # Test FlaxDataLoader (if multiple devices available)
        print("\n[2/3] Testing FlaxDataLoader...")

        num_devices = jax.device_count()
        print(f"  Available devices: {num_devices}")

        if num_devices > 1:
            # Use multiple devices
            loader = FlaxDataLoader(
                tar_path,
                batch_size=num_devices * 2,  # 2 per device
                num_workers=2,
                num_devices=num_devices
            )

            batch_count = 0
            for batch in loader:
                if batch_count >= 2:
                    break
                batch_count += 1
                print(f"  Batch {batch_count}: sharded across {num_devices} devices")
                print(f"    Shape: {batch['image'].shape}")  # Should be (num_devices, per_device, ...)

            loader.__exit__(None, None, None)
            print(f"  Successfully loaded {batch_count} batches with multi-device sharding")
        else:
            print("  Skipping multi-device test (only 1 device available)")

        # Test prefetch_to_device
        print("\n[3/3] Testing prefetch_to_device...")
        loader = JAXDataLoader(tar_path, batch_size=4, num_workers=2)
        prefetched = prefetch_to_device(iter(loader), size=2, device=jax.devices()[0])

        batch_count = 0
        for batch in prefetched:
            if batch_count >= 3:
                break
            batch_count += 1
            print(f"  Prefetched batch {batch_count}: shape={batch['image'].shape}")

        print(f"  Successfully prefetched {batch_count} batches")

        loader.__exit__(None, None, None)

        print("\n" + "="*80)
        print("JAX/Flax Integration: PASSED ✓")
        print("="*80)

        # Cleanup
        os.remove(tar_path)

        return True

    except ImportError as e:
        print(f"\nSkipping JAX tests: {e}")
        return None
    except Exception as e:
        print(f"\nJAX tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("\n" + "="*80)
    print("TurboLoader Multi-Framework Integration Tests")
    print("="*80)

    results = {}

    # Run TensorFlow tests
    results['tensorflow'] = test_tensorflow_dataloader()

    # Run JAX tests
    results['jax'] = test_jax_dataloader()

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for name, result in results.items():
        if result is True:
            print(f"  {name.upper()}: PASSED ✓")
        elif result is False:
            print(f"  {name.upper()}: FAILED ✗")
        else:
            print(f"  {name.upper()}: SKIPPED (missing dependencies)")

    print("="*80)

    # Return exit code
    if any(r is False for r in results.values()):
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
