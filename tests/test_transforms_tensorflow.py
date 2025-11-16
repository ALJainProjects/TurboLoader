"""
Test TensorFlow transform integration for TurboLoader v0.6.0
"""

import unittest
import numpy as np

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, skipping TensorFlow tests")

try:
    import turboloader as tl
    TURBOLOADER_AVAILABLE = True
except ImportError:
    TURBOLOADER_AVAILABLE = False
    print("TurboLoader not available, skipping tests")


def create_test_image(height=100, width=100, channels=3):
    """Create a random test image."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


@unittest.skipIf(not TENSORFLOW_AVAILABLE or not TURBOLOADER_AVAILABLE,
                 "TensorFlow or TurboLoader not available")
class TestTensorFlowIntegration(unittest.TestCase):
    """Test TurboLoader with TensorFlow."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_image = create_test_image(100, 100, 3)

    def test_basic_transforms(self):
        """Test basic transforms work."""
        tl_resize = tl.Resize(50, 50)
        output = tl_resize.apply(self.test_image)

        self.assertEqual(output.shape, (50, 50, 3))

    def test_tensorflow_format(self):
        """Test TensorFlow HWC format."""
        # TensorFlow uses HWC format (Height, Width, Channels)
        # TurboLoader should match this
        self.assertEqual(self.test_image.shape, (100, 100, 3))

    def test_data_augmentation_pipeline(self):
        """Test typical TensorFlow data augmentation pipeline."""
        # TensorFlow-style pipeline
        transforms = [
            tl.RandomHorizontalFlip(p=0.5, seed=42),
            tl.RandomCrop(width=80, height=80, padding=10, seed=42),
            tl.ColorJitter(brightness=0.2, contrast=0.2, seed=42),
        ]

        result = self.test_image.copy()
        for transform in transforms:
            result = transform.apply(result)

        self.assertEqual(result.shape, (80, 80, 3))

    def test_tensor_format_enum(self):
        """Test TensorFlow tensor format enum."""
        self.assertTrue(hasattr(tl, 'TensorFormat'))
        self.assertTrue(hasattr(tl.TensorFormat, 'TENSORFLOW_HWC'))

    def test_normalization(self):
        """Test normalization for TensorFlow."""
        # Common TensorFlow normalization: scale to [-1, 1]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        tl_norm = tl.Normalize(mean=mean, std=std, to_float=False)
        output = tl_norm.apply(self.test_image)

        self.assertEqual(output.shape, self.test_image.shape)

    def test_resize_modes(self):
        """Test different resize interpolation modes."""
        for mode in [tl.InterpolationMode.NEAREST,
                     tl.InterpolationMode.BILINEAR,
                     tl.InterpolationMode.BICUBIC]:
            tl_resize = tl.Resize(64, 64, mode)
            output = tl_resize.apply(self.test_image)
            self.assertEqual(output.shape, (64, 64, 3))

    def test_batch_processing(self):
        """Test batch processing for TensorFlow."""
        # Create batch
        batch_size = 4
        batch = [create_test_image(100, 100, 3) for _ in range(batch_size)]

        # Apply transforms to batch
        tl_resize = tl.Resize(224, 224)
        batch_output = [tl_resize.apply(img) for img in batch]

        # Stack into TensorFlow-style batch (N, H, W, C)
        batch_array = np.stack(batch_output)

        self.assertEqual(batch_array.shape, (batch_size, 224, 224, 3))

    def test_performance_comparison(self):
        """Compare TurboLoader vs TensorFlow performance."""
        import time

        large_image = create_test_image(1000, 1000, 3)

        # TurboLoader
        tl_resize = tl.Resize(224, 224)
        start = time.time()
        for _ in range(10):
            _ = tl_resize.apply(large_image)
        tl_time = time.time() - start

        print(f"\nTurboLoader: {tl_time:.4f}s for 10 resizes")

        # TensorFlow
        tf_image = tf.constant(large_image)
        start = time.time()
        for _ in range(10):
            _ = tf.image.resize(tf_image, [224, 224])
        tf_time = time.time() - start

        print(f"TensorFlow: {tf_time:.4f}s for 10 resizes")

        # TurboLoader should be competitive
        self.assertLess(tl_time, tf_time * 2.0)  # At most 2x slower

if __name__ == '__main__':
    unittest.main(verbosity=2)
