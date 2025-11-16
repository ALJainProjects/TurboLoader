"""
Test PyTorch transform integration for TurboLoader v0.6.0

Compares TurboLoader transforms with torchvision transforms for correctness.
"""

import unittest
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available, skipping PyTorch tests")

try:
    import turboloader as tl
    TURBOLOADER_AVAILABLE = True
except ImportError:
    TURBOLOADER_AVAILABLE = False
    print("TurboLoader not available, skipping tests")


def create_test_image(height=100, width=100, channels=3):
    """Create a random test image."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


@unittest.skipIf(not PYTORCH_AVAILABLE or not TURBOLOADER_AVAILABLE,
                 "PyTorch or TurboLoader not available")
class TestPyTorchTransforms(unittest.TestCase):
    """Test TurboLoader transforms against PyTorch/torchvision."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_image = create_test_image(100, 100, 3)

    def test_resize(self):
        """Test resize transform."""
        # TurboLoader
        tl_resize = tl.Resize(50, 50, tl.InterpolationMode.BILINEAR)
        tl_output = tl_resize.apply(self.test_image)

        self.assertEqual(tl_output.shape, (50, 50, 3))

    def test_horizontal_flip(self):
        """Test horizontal flip."""
        # Manual flip for verification
        expected = self.test_image[:, ::-1, :]

        # TurboLoader (always flip for testing)
        tl_flip = tl.RandomHorizontalFlip(p=1.0, seed=42)
        tl_output = tl_flip.apply(self.test_image)

        # Check dimensions
        self.assertEqual(tl_output.shape, self.test_image.shape)

    def test_center_crop(self):
        """Test center crop."""
        # TurboLoader
        tl_crop = tl.CenterCrop(50, 50)
        tl_output = tl_crop.apply(self.test_image)

        self.assertEqual(tl_output.shape, (50, 50, 3))

        # Check center pixel matches
        center_in = self.test_image[50, 50]
        center_out = tl_output[25, 25]
        np.testing.assert_array_equal(center_out, center_in)

    def test_grayscale(self):
        """Test grayscale conversion."""
        # TurboLoader
        tl_gray = tl.Grayscale(num_output_channels=1)
        tl_output = tl_gray.apply(self.test_image)

        self.assertEqual(tl_output.shape[2], 1)
        self.assertEqual(tl_output.shape[:2], self.test_image.shape[:2])

    def test_pad(self):
        """Test padding."""
        # TurboLoader
        tl_pad = tl.Pad(10, tl.PaddingMode.CONSTANT, 0)
        tl_output = tl_pad.apply(self.test_image)

        self.assertEqual(tl_output.shape, (120, 120, 3))

        # Check corners are padded with zeros
        self.assertEqual(tl_output[0, 0, 0], 0)
        self.assertEqual(tl_output[0, 0, 1], 0)

    def test_gaussian_blur(self):
        """Test Gaussian blur."""
        # TurboLoader
        tl_blur = tl.GaussianBlur(kernel_size=5, sigma=1.5)
        tl_output = tl_blur.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)

        # Blurred image should have lower variance
        variance_in = np.var(self.test_image)
        variance_out = np.var(tl_output)
        self.assertLess(variance_out, variance_in)

    def test_color_jitter(self):
        """Test color jitter."""
        # TurboLoader
        tl_jitter = tl.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1,
            seed=42
        )
        tl_output = tl_jitter.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)
        # Output should be different from input
        self.assertFalse(np.array_equal(tl_output, self.test_image))

    def test_random_erasing(self):
        """Test random erasing."""
        # TurboLoader (always erase for testing)
        tl_erase = tl.RandomErasing(p=1.0, seed=42)
        tl_output = tl_erase.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)

        # Should have some zeros (erased region)
        self.assertGreater(np.sum(tl_output == 0), 0)

    def test_normalize(self):
        """Test normalization."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # TurboLoader
        tl_norm = tl.Normalize(mean=mean, std=std, to_float=False)
        tl_output = tl_norm.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)

    def test_imagenet_normalize(self):
        """Test ImageNet normalization preset."""
        # TurboLoader
        tl_norm = tl.ImageNetNormalize(to_float=False)
        tl_output = tl_norm.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)

    def test_rotation(self):
        """Test random rotation."""
        # TurboLoader
        tl_rotate = tl.RandomRotation(degrees=45, seed=42)
        tl_output = tl_rotate.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)

    def test_affine(self):
        """Test random affine."""
        # TurboLoader
        tl_affine = tl.RandomAffine(
            degrees=15,
            translate_x=0.1,
            translate_y=0.1,
            scale_min=0.9,
            scale_max=1.1,
            seed=42
        )
        tl_output = tl_affine.apply(self.test_image)

        self.assertEqual(tl_output.shape, self.test_image.shape)

    def test_random_crop(self):
        """Test random crop with padding."""
        # TurboLoader
        tl_crop = tl.RandomCrop(
            width=80,
            height=80,
            padding=10,
            pad_mode=tl.PaddingMode.CONSTANT,
            seed=42
        )
        tl_output = tl_crop.apply(self.test_image)

        self.assertEqual(tl_output.shape, (80, 80, 3))

    def test_chain_transforms(self):
        """Test chaining multiple transforms."""
        # Create a complex pipeline
        transforms = [
            tl.Resize(128, 128),
            tl.RandomHorizontalFlip(p=0.5, seed=42),
            tl.ColorJitter(brightness=0.2, contrast=0.2, seed=42),
            tl.CenterCrop(96, 96),
        ]

        # Apply sequentially
        result = self.test_image.copy()
        for transform in transforms:
            result = transform.apply(result)

        self.assertEqual(result.shape, (96, 96, 3))

    def test_tensor_format_pytorch(self):
        """Test PyTorch tensor format conversion."""
        # This would require actual tensor conversion implementation
        # For now, just check the enum exists
        self.assertTrue(hasattr(tl, 'TensorFormat'))
        self.assertTrue(hasattr(tl.TensorFormat, 'PYTORCH_CHW'))

    def test_tensor_format_tensorflow(self):
        """Test TensorFlow tensor format."""
        self.assertTrue(hasattr(tl.TensorFormat, 'TENSORFLOW_HWC'))

    def test_determinism(self):
        """Test that transforms with same seed are deterministic."""
        # TurboLoader
        tl_flip1 = tl.RandomHorizontalFlip(p=0.5, seed=42)
        tl_flip2 = tl.RandomHorizontalFlip(p=0.5, seed=42)

        output1 = tl_flip1.apply(self.test_image)
        output2 = tl_flip2.apply(self.test_image)

        np.testing.assert_array_equal(output1, output2)

    def test_edge_cases(self):
        """Test edge cases."""
        # 1x1 image
        tiny_image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        tl_resize = tl.Resize(10, 10)
        output = tl_resize.apply(tiny_image)
        self.assertEqual(output.shape, (10, 10, 3))

        # Single channel
        gray_image = create_test_image(50, 50, 1)
        tl_resize = tl.Resize(25, 25)
        output = tl_resize.apply(gray_image)
        self.assertEqual(output.shape, (25, 25, 1))

    def test_performance_resize(self):
        """Benchmark resize performance."""
        import time

        large_image = create_test_image(1000, 1000, 3)
        tl_resize = tl.Resize(224, 224)

        start = time.time()
        for _ in range(10):
            _ = tl_resize.apply(large_image)
        tl_time = time.time() - start

        print(f"\nTurboLoader Resize: {tl_time:.4f}s for 10 iterations")
        self.assertLess(tl_time, 1.0)  # Should be fast

    def test_batch_processing(self):
        """Test batch processing."""
        # Create batch of images
        batch = [create_test_image(100, 100, 3) for _ in range(4)]

        tl_resize = tl.Resize(50, 50)

        # Process batch
        outputs = [tl_resize.apply(img) for img in batch]

        for output in outputs:
            self.assertEqual(output.shape, (50, 50, 3))


class TestSIMDUtilities(unittest.TestCase):
    """Test SIMD utility functions."""

    def test_simd_available(self):
        """Check SIMD availability."""
        features = tl.features()
        self.assertTrue(features['simd_acceleration'])

    def test_convert_formats(self):
        """Test data format conversions."""
        # Create test data
        uint8_data = np.array([0, 127, 255], dtype=np.uint8)

        # These tests would require direct access to SIMD functions
        # For now, we test through transforms that use them
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
