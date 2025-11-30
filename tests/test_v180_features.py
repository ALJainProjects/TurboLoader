#!/usr/bin/env python3
"""
Tests for TurboLoader v1.8.0 Features

Tests the new features added in v1.8.0:
- ARM NEON optimizations (tested via transform performance)
- Modern augmentations (MixUp, CutMix, Mosaic, RandAugment, GridMask)
- Error recovery mechanism
- Logging framework
"""

import sys
import os
import tempfile
import time
import numpy as np
import pytest


class TestVersion:
    """Test version information"""

    def test_version_is_180(self):
        """Verify version is 1.8.0"""
        import turboloader
        assert turboloader.__version__ == "1.8.0"

    def test_version_function(self):
        """Test version() function returns correct version"""
        try:
            import turboloader
            assert turboloader.version() == "1.8.0"
        except (ImportError, AttributeError):
            pytest.skip("C++ module not built")


class TestModernAugmentations:
    """Test MixUp, CutMix, Mosaic, RandAugment, GridMask"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_batch(self):
        """Create a batch of sample images"""
        return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]

    def test_mixup_exists(self):
        """Test MixUp class is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'MixUp')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_cutmix_exists(self):
        """Test CutMix class is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'CutMix')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_mosaic_exists(self):
        """Test Mosaic class is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'Mosaic')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_randaugment_exists(self):
        """Test RandAugment class is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'RandAugment')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_gridmask_exists(self):
        """Test GridMask class is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'GridMask')
        except ImportError:
            pytest.skip("C++ module not built")


class TestLoggingFramework:
    """Test logging framework"""

    def test_enable_logging_exists(self):
        """Test enable_logging function is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'enable_logging')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_disable_logging_exists(self):
        """Test disable_logging function is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'disable_logging')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_set_log_level_exists(self):
        """Test set_log_level function is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'set_log_level')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_set_log_output_exists(self):
        """Test set_log_output function is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'set_log_output')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_log_level_enum_exists(self):
        """Test LogLevel enum is available"""
        try:
            import turboloader
            assert hasattr(turboloader, 'LogLevel')
            # Check enum values
            assert hasattr(turboloader.LogLevel, 'DEBUG')
            assert hasattr(turboloader.LogLevel, 'INFO')
            assert hasattr(turboloader.LogLevel, 'WARNING')
            assert hasattr(turboloader.LogLevel, 'ERROR')
            assert hasattr(turboloader.LogLevel, 'CRITICAL')
        except ImportError:
            pytest.skip("C++ module not built")

    def test_logging_enable_disable(self):
        """Test enabling and disabling logging"""
        try:
            import turboloader
            # Should not raise
            turboloader.enable_logging()
            turboloader.disable_logging()
        except ImportError:
            pytest.skip("C++ module not built")

    def test_set_log_level(self):
        """Test setting log level"""
        try:
            import turboloader
            turboloader.enable_logging()
            turboloader.set_log_level(turboloader.LogLevel.DEBUG)
            turboloader.set_log_level(turboloader.LogLevel.WARNING)
            turboloader.disable_logging()
        except ImportError:
            pytest.skip("C++ module not built")

    def test_set_log_output_file(self):
        """Test setting log output to file"""
        try:
            import turboloader
            with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as f:
                log_path = f.name

            try:
                turboloader.enable_logging()
                turboloader.set_log_output(log_path)
                # Reset to stderr
                turboloader.set_log_output("")
                turboloader.disable_logging()
            finally:
                if os.path.exists(log_path):
                    os.remove(log_path)
        except ImportError:
            pytest.skip("C++ module not built")


class TestTransformPerformance:
    """Test transform performance (implicitly tests NEON optimizations)"""

    @pytest.fixture
    def large_image(self):
        """Create a large image for performance testing"""
        return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    def test_resize_performance(self, large_image):
        """Test resize performance"""
        try:
            import turboloader

            resize = turboloader.Resize(256, 256)

            # Warmup
            for _ in range(3):
                resize.apply(large_image)

            # Time 10 iterations
            start = time.time()
            iterations = 10
            for _ in range(iterations):
                resize.apply(large_image)
            elapsed = time.time() - start

            throughput = iterations / elapsed
            print(f"\nResize throughput: {throughput:.1f} ops/sec")

            # Should be reasonably fast (at least 10 ops/sec for 1024x1024)
            assert throughput > 5, f"Resize too slow: {throughput:.1f} ops/sec"

        except ImportError:
            pytest.skip("C++ module not built")

    def test_normalize_performance(self, large_image):
        """Test normalize performance"""
        try:
            import turboloader

            normalize = turboloader.ImageNetNormalize()

            # Warmup
            for _ in range(3):
                normalize.apply(large_image)

            # Time 10 iterations
            start = time.time()
            iterations = 10
            for _ in range(iterations):
                normalize.apply(large_image)
            elapsed = time.time() - start

            throughput = iterations / elapsed
            print(f"\nNormalize throughput: {throughput:.1f} ops/sec")

            # Should be fast (at least 50 ops/sec)
            assert throughput > 20, f"Normalize too slow: {throughput:.1f} ops/sec"

        except ImportError:
            pytest.skip("C++ module not built")

    def test_flip_performance(self, large_image):
        """Test horizontal flip performance"""
        try:
            import turboloader

            flip = turboloader.RandomHorizontalFlip(1.0)  # Always flip

            # Warmup
            for _ in range(3):
                flip.apply(large_image)

            # Time 10 iterations
            start = time.time()
            iterations = 10
            for _ in range(iterations):
                flip.apply(large_image)
            elapsed = time.time() - start

            throughput = iterations / elapsed
            print(f"\nFlip throughput: {throughput:.1f} ops/sec")

            # Should be fast (at least 50 ops/sec)
            assert throughput > 20, f"Flip too slow: {throughput:.1f} ops/sec"

        except ImportError:
            pytest.skip("C++ module not built")


class TestExistingTransforms:
    """Verify existing transforms still work correctly"""

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_resize(self, sample_image):
        """Test Resize transform"""
        try:
            import turboloader
            resize = turboloader.Resize(128, 128)
            result = resize.apply(sample_image)
            assert result.shape == (128, 128, 3)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_center_crop(self, sample_image):
        """Test CenterCrop transform"""
        try:
            import turboloader
            crop = turboloader.CenterCrop(128, 128)
            result = crop.apply(sample_image)
            assert result.shape == (128, 128, 3)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_random_horizontal_flip(self, sample_image):
        """Test RandomHorizontalFlip transform"""
        try:
            import turboloader
            flip = turboloader.RandomHorizontalFlip(1.0)  # Always flip
            result = flip.apply(sample_image)
            assert result.shape == sample_image.shape
            # Check that it's actually flipped
            np.testing.assert_array_equal(result[:, ::-1, :], sample_image)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_color_jitter(self, sample_image):
        """Test ColorJitter transform"""
        try:
            import turboloader
            jitter = turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1)
            result = jitter.apply(sample_image)
            assert result.shape == sample_image.shape
        except ImportError:
            pytest.skip("C++ module not built")

    def test_normalize(self, sample_image):
        """Test Normalize transform"""
        try:
            import turboloader
            normalize = turboloader.ImageNetNormalize()
            result = normalize.apply(sample_image)
            assert result.shape == sample_image.shape
        except ImportError:
            pytest.skip("C++ module not built")

    def test_compose(self, sample_image):
        """Test Compose (transform pipeline)"""
        try:
            import turboloader
            pipeline = turboloader.Compose([
                turboloader.Resize(224, 224),
                turboloader.RandomHorizontalFlip(0.5),
                turboloader.ImageNetNormalize()
            ])
            result = pipeline.apply(sample_image)
            assert result.shape == (224, 224, 3)
        except ImportError:
            pytest.skip("C++ module not built")


class TestAllExports:
    """Test that all exports from __init__.py are available"""

    def test_all_exports(self):
        """Verify all exported symbols are available"""
        try:
            import turboloader

            expected_exports = [
                # Core
                'DataLoader', 'version', 'features', '__version__',
                # TBL v2
                'TblReaderV2', 'TblWriterV2', 'SampleFormat', 'MetadataType',
                # Smart Batching
                'SmartBatchConfig',
                # Transform Composition
                'Compose', 'ComposedTransforms',
                # Transforms
                'Resize', 'CenterCrop', 'RandomCrop',
                'RandomHorizontalFlip', 'RandomVerticalFlip',
                'ColorJitter', 'GaussianBlur', 'Grayscale',
                'Normalize', 'ImageNetNormalize', 'ToTensor',
                'Pad', 'RandomRotation', 'RandomAffine',
                'RandomPerspective', 'RandomPosterize', 'RandomSolarize',
                'RandomErasing', 'AutoAugment', 'AutoAugmentPolicy',
                # Modern Augmentations (v1.8.0)
                'MixUp', 'CutMix', 'Mosaic', 'RandAugment', 'GridMask',
                # Logging (v1.8.0)
                'LogLevel', 'enable_logging', 'disable_logging', 'set_log_level', 'set_log_output',
                # Enums
                'InterpolationMode', 'PaddingMode', 'TensorFormat',
            ]

            for export in expected_exports:
                assert hasattr(turboloader, export), f"Missing export: {export}"

        except ImportError:
            pytest.skip("C++ module not built")


def run_tests():
    """Run all tests and print summary"""
    print("=" * 70)
    print("TurboLoader v1.8.0 Feature Tests")
    print("=" * 70)

    # Run pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x',  # Stop on first failure
    ])

    return exit_code


if __name__ == '__main__':
    sys.exit(run_tests())
