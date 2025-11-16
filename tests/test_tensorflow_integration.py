"""
Dedicated TensorFlow/Keras integration tests for TurboLoader

Tests comprehensive TensorFlow and Keras functionality.
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


def test_tensorflow_dataloader_basic():
    """Test basic TensorFlowDataLoader functionality"""
    print("\n" + "="*80)
    print("Test 1: Basic TensorFlowDataLoader")
    print("="*80)

    try:
        import tensorflow as tf
        from tensorflow_dataloader import TensorFlowDataLoader

        tar_path = create_test_tar(10)

        loader = TensorFlowDataLoader(
            tar_path,
            batch_size=4,
            num_workers=2,
            shuffle=False
        )

        dataset = loader.as_dataset()
        print(f"  ✓ Created tf.data.Dataset: {dataset}")

        # Test iteration
        batch_count = 0
        for images, labels in dataset.take(2):
            batch_count += 1
            print(f"  ✓ Batch {batch_count}: images={images.shape}, labels={labels.shape}")
            assert images.shape[0] <= 4
            assert len(labels.shape) == 1

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: Basic TensorFlowDataLoader")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keras_sequence_basic():
    """Test basic KerasSequence functionality"""
    print("\n" + "="*80)
    print("Test 2: Basic KerasSequence")
    print("="*80)

    try:
        import tensorflow as tf
        from tensorflow_dataloader import KerasSequence

        tar_path = create_test_tar(16)

        sequence = KerasSequence(
            tar_path,
            batch_size=4,
            num_workers=2
        )

        print(f"  ✓ Created KerasSequence with {len(sequence)} batches")

        # Test __len__
        assert len(sequence) > 0
        print(f"  ✓ Length: {len(sequence)} batches")

        # Test __getitem__
        for i in range(min(2, len(sequence))):
            images, labels = sequence[i]
            print(f"  ✓ Batch {i}: images={images.shape}, labels={labels.shape}")
            assert images.shape[0] <= 4

        os.remove(tar_path)

        print("  ✓ PASSED: Basic KerasSequence")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keras_model_training():
    """Test KerasSequence with actual model training"""
    print("\n" + "="*80)
    print("Test 3: Keras Model Training")
    print("="*80)

    try:
        import tensorflow as tf
        from tensorflow_dataloader import KerasSequence

        tar_path = create_test_tar(20)

        sequence = KerasSequence(tar_path, batch_size=4, num_workers=2)

        # Create simple model
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

        print("  ✓ Created and compiled model")

        # Train for 1 epoch
        history = model.fit(sequence, epochs=1, verbose=0)

        print(f"  ✓ Training completed! Loss: {history.history['loss'][0]:.4f}")
        print(f"  ✓ Model trained successfully with TurboLoader data")

        os.remove(tar_path)

        print("  ✓ PASSED: Keras Model Training")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tensorflow_prefetch():
    """Test TensorFlowDataLoader prefetch functionality"""
    print("\n" + "="*80)
    print("Test 4: TensorFlow Prefetch")
    print("="*80)

    try:
        import tensorflow as tf
        from tensorflow_dataloader import TensorFlowDataLoader

        tar_path = create_test_tar(12)

        loader = TensorFlowDataLoader(
            tar_path,
            batch_size=3,
            num_workers=2,
            prefetch=2  # Enable prefetch
        )

        dataset = loader.as_dataset()
        print("  ✓ Created dataset with prefetch=2")

        # Consume batches
        batch_count = 0
        for images, labels in dataset.take(3):
            batch_count += 1

        print(f"  ✓ Consumed {batch_count} batches with prefetching")

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: TensorFlow Prefetch")
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
    """Run all TensorFlow integration tests"""
    print("\n" + "="*80)
    print("TurboLoader TensorFlow/Keras Integration Tests")
    print("="*80)

    results = {}

    # Run all tests
    results['basic_dataloader'] = test_tensorflow_dataloader_basic()
    results['basic_sequence'] = test_keras_sequence_basic()
    results['model_training'] = test_keras_model_training()
    results['prefetch'] = test_tensorflow_prefetch()

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
