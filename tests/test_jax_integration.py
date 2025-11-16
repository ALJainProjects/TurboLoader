"""
Dedicated JAX/Flax integration tests for TurboLoader

Tests comprehensive JAX and Flax functionality including device placement and sharding.
"""

import sys
import os
import tempfile
import tarfile
import numpy as np
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))


def create_test_tar(num_images=20):
    """Create a small TAR file for testing"""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required for creating test images. Install it with: pip install Pillow")

    tmpdir = tempfile.mkdtemp()
    tar_path = os.path.join(tmpdir, 'test.tar')

    # Create test images
    images_dir = os.path.join(tmpdir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    for i in range(num_images):
        # Create a simple test image (random RGB) as JPEG
        img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_data, mode='RGB')
        img_path = os.path.join(images_dir, f'img_{i:04d}.jpg')
        img.save(img_path, 'JPEG', quality=90)

    # Create TAR
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_images):
            img_path = os.path.join(images_dir, f'img_{i:04d}.jpg')
            tar.add(img_path, arcname=f'img_{i:04d}.jpg')

    return tar_path


def test_jax_dataloader_basic():
    """Test basic JAXDataLoader functionality"""
    print("\n" + "="*80)
    print("Test 1: Basic JAXDataLoader")
    print("="*80)

    try:
        import jax
        import jax.numpy as jnp
        from jax_dataloader import JAXDataLoader

        tar_path = create_test_tar(10)

        loader = JAXDataLoader(
            tar_path,
            batch_size=4,
            num_workers=2,
            shuffle=False
        )

        print(f"  ✓ Created JAXDataLoader")

        # Test iteration
        batch_count = 0
        for images, labels in loader:
            batch_count += 1
            print(f"  ✓ Batch {batch_count}: images={images.shape}, labels={labels.shape}")
            assert isinstance(images, jnp.ndarray) or isinstance(images, np.ndarray)
            assert images.shape[0] <= 4
            assert len(labels.shape) == 1

            if batch_count >= 2:
                break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: Basic JAXDataLoader")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jax_device_placement():
    """Test JAX device placement functionality"""
    print("\n" + "="*80)
    print("Test 2: JAX Device Placement")
    print("="*80)

    try:
        import jax
        import jax.numpy as jnp
        from jax_dataloader import JAXDataLoader

        tar_path = create_test_tar(8)

        # Get available devices
        devices = jax.devices()
        print(f"  ✓ Available JAX devices: {len(devices)}")

        loader = JAXDataLoader(
            tar_path,
            batch_size=2,
            num_workers=1,
            device=devices[0]  # Place on first device
        )

        print(f"  ✓ Created loader with device placement")

        # Get one batch
        for images, labels in loader:
            print(f"  ✓ Batch retrieved: images={images.shape}")

            # Check device placement
            if hasattr(images, 'device'):
                print(f"  ✓ Images on device: {images.device()}")

            break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: JAX Device Placement")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jax_sharding():
    """Test JAX sharding functionality"""
    print("\n" + "="*80)
    print("Test 3: JAX Sharding")
    print("="*80)

    try:
        import jax
        import jax.numpy as jnp
        from jax_dataloader import JAXDataLoader

        tar_path = create_test_tar(16)

        devices = jax.devices()

        if len(devices) < 2:
            print(f"  ⊘ SKIPPED: Need at least 2 devices, found {len(devices)}")
            os.remove(tar_path)
            return None

        loader = JAXDataLoader(
            tar_path,
            batch_size=8,
            num_workers=2,
            devices=devices[:2],  # Use first 2 devices
            shard_data=True
        )

        print(f"  ✓ Created loader with sharding across {len(devices[:2])} devices")

        # Get one batch
        for images, labels in loader:
            print(f"  ✓ Sharded batch retrieved: images={images.shape}")

            # Check if it's sharded
            if hasattr(images, 'sharding'):
                print(f"  ✓ Data is sharded: {images.sharding}")

            break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: JAX Sharding")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flax_training():
    """Test Flax model training with JAXDataLoader"""
    print("\n" + "="*80)
    print("Test 4: Flax Model Training")
    print("="*80)

    try:
        import jax
        import jax.numpy as jnp
        from jax import random
        from flax import linen as nn
        from jax_dataloader import JAXDataLoader

        tar_path = create_test_tar(20)

        # Create simple Flax model
        class SimpleCNN(nn.Module):
            num_classes: int = 10

            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=32, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))  # flatten
                x = nn.Dense(features=self.num_classes)(x)
                return x

        model = SimpleCNN()

        # Initialize model
        rng = random.PRNGKey(0)
        dummy_input = jnp.ones((1, 64, 64, 3))
        params = model.init(rng, dummy_input)

        print("  ✓ Created and initialized Flax model")

        # Create dataloader
        loader = JAXDataLoader(tar_path, batch_size=4, num_workers=2)

        # Simple training step
        def loss_fn(params, images, labels):
            logits = model.apply(params, images)
            # Simple MSE loss for testing
            return jnp.mean((logits - jnp.eye(10)[labels]) ** 2)

        # Get one batch and compute loss
        for images, labels in loader:
            # Convert to JAX arrays if needed
            if not isinstance(images, jnp.ndarray):
                images = jnp.array(images)
            if not isinstance(labels, jnp.ndarray):
                labels = jnp.array(labels)

            loss = loss_fn(params, images, labels)
            print(f"  ✓ Training step completed! Loss: {float(loss):.4f}")
            break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: Flax Model Training")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jax_prefetch():
    """Test JAXDataLoader prefetch functionality"""
    print("\n" + "="*80)
    print("Test 5: JAX Prefetch")
    print("="*80)

    try:
        import jax
        from jax_dataloader import JAXDataLoader

        tar_path = create_test_tar(12)

        loader = JAXDataLoader(
            tar_path,
            batch_size=3,
            num_workers=2,
            prefetch=2  # Enable prefetch
        )

        print("  ✓ Created JAXDataLoader with prefetch=2")

        # Consume batches
        batch_count = 0
        for images, labels in loader:
            batch_count += 1
            if batch_count >= 3:
                break

        print(f"  ✓ Consumed {batch_count} batches with prefetching")

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: JAX Prefetch")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all JAX integration tests"""
    print("\n" + "="*80)
    print("TurboLoader JAX/Flax Integration Tests")
    print("="*80)

    results = {}

    # Run all tests
    results['basic_dataloader'] = test_jax_dataloader_basic()
    results['device_placement'] = test_jax_device_placement()
    results['sharding'] = test_jax_sharding()
    results['flax_training'] = test_flax_training()
    results['prefetch'] = test_jax_prefetch()

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            print(f"  ✓ {name}: PASSED")
        elif result is False:
            print(f"  ✗ {name}: FAILED")
        else:
            print(f"  ⊘ {name}: SKIPPED")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*80)

    # Return exit code
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
