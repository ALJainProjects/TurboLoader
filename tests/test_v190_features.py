#!/usr/bin/env python3
"""
Tests for TurboLoader v2.0.0 Features

Tests the new features added in v2.0.0:
- Transform Pipe Operator (|)
- HDF5 format support (header only, requires libhdf5)
- TFRecord format support
- Zarr format support
- COCO/Pascal VOC annotation support
- Azure Blob Storage support (header only, requires Azure SDK)
- GPU transforms (header only, requires CUDA)
"""

import sys
import os
import tempfile
import numpy as np
import pytest


class TestVersion:
    """Test version information"""

    def test_version_is_2x(self):
        """Verify version is 2.x.x"""
        import turboloader

        assert turboloader.__version__.startswith("2.")

    def test_version_function(self):
        """Test version() function returns correct version"""
        try:
            import turboloader

            assert turboloader.version().startswith("2.")
        except (ImportError, AttributeError):
            pytest.skip("C++ module not built")


class TestPipeOperator:
    """Test transform pipe operator (|)"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image"""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_two_transforms_pipe(self, sample_image):
        """Test piping two transforms"""
        try:
            import turboloader

            pipeline = turboloader.Resize(224, 224) | turboloader.RandomHorizontalFlip(1.0)

            result = pipeline.apply(sample_image)
            assert result.shape == (224, 224, 3)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_three_transforms_pipe(self, sample_image):
        """Test piping three transforms"""
        try:
            import turboloader

            pipeline = (
                turboloader.Resize(224, 224)
                | turboloader.RandomHorizontalFlip(0.5)
                | turboloader.ImageNetNormalize()
            )

            result = pipeline.apply(sample_image)
            assert result.shape == (224, 224, 3)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_pipe_extends_pipeline(self, sample_image):
        """Test extending pipeline with pipe operator"""
        try:
            import turboloader

            # Create initial pipeline
            pipeline = turboloader.Resize(224, 224) | turboloader.CenterCrop(200, 200)

            # Extend with another transform
            extended = pipeline | turboloader.RandomHorizontalFlip(1.0)

            result = extended.apply(sample_image)
            assert result.shape == (200, 200, 3)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_pipe_vs_compose_equivalent(self, sample_image):
        """Test that pipe operator produces same results as Compose"""
        try:
            import turboloader

            # Using Compose
            compose_pipeline = turboloader.Compose(
                [turboloader.Resize(224, 224), turboloader.CenterCrop(200, 200)]
            )

            # Using pipe operator
            pipe_pipeline = turboloader.Resize(224, 224) | turboloader.CenterCrop(200, 200)

            compose_result = compose_pipeline.apply(sample_image)
            pipe_result = pipe_pipeline.apply(sample_image)

            # Should have same shape
            assert compose_result.shape == pipe_result.shape
        except ImportError:
            pytest.skip("C++ module not built")

    def test_pipeline_len(self, sample_image):
        """Test pipeline length"""
        try:
            import turboloader

            pipeline = (
                turboloader.Resize(224, 224)
                | turboloader.RandomHorizontalFlip(0.5)
                | turboloader.ImageNetNormalize()
            )

            assert len(pipeline) == 3
        except ImportError:
            pytest.skip("C++ module not built")

    def test_pipeline_callable(self, sample_image):
        """Test pipeline is callable"""
        try:
            import turboloader

            pipeline = turboloader.Resize(224, 224) | turboloader.CenterCrop(200, 200)

            # Test __call__
            result = pipeline(sample_image)
            assert result.shape == (200, 200, 3)
        except ImportError:
            pytest.skip("C++ module not built")


class TestNewFeatures:
    """Test new v2.0.0 features are available"""

    def test_features_dict_has_new_features(self):
        """Test features() includes new v2.0.0 features"""
        try:
            import turboloader

            features = turboloader.features()

            assert features["version"].startswith("2.")
            assert "pipe_operator" in features
            assert features["pipe_operator"] == True

            # These features are implemented but may require external libraries
            assert "hdf5_support" in features
            assert "tfrecord_support" in features
            assert "zarr_support" in features
            assert "coco_voc_support" in features
            assert "azure_support" in features
            assert "io_uring" in features
            assert "gpu_transforms" in features

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


class TestModernAugmentations:
    """Test MixUp, CutMix, Mosaic, RandAugment, GridMask"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_mixup_exists(self):
        """Test MixUp class is available"""
        try:
            import turboloader

            assert hasattr(turboloader, "MixUp")
        except ImportError:
            pytest.skip("C++ module not built")

    def test_cutmix_exists(self):
        """Test CutMix class is available"""
        try:
            import turboloader

            assert hasattr(turboloader, "CutMix")
        except ImportError:
            pytest.skip("C++ module not built")

    def test_mosaic_exists(self):
        """Test Mosaic class is available"""
        try:
            import turboloader

            assert hasattr(turboloader, "Mosaic")
        except ImportError:
            pytest.skip("C++ module not built")

    def test_randaugment_exists(self):
        """Test RandAugment class is available"""
        try:
            import turboloader

            assert hasattr(turboloader, "RandAugment")
        except ImportError:
            pytest.skip("C++ module not built")

    def test_gridmask_exists(self):
        """Test GridMask class is available"""
        try:
            import turboloader

            assert hasattr(turboloader, "GridMask")
        except ImportError:
            pytest.skip("C++ module not built")


class TestPipeOperatorWithModernAugmentations:
    """Test pipe operator with modern augmentations"""

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_pipe_with_randaugment(self, sample_image):
        """Test piping with RandAugment"""
        try:
            import turboloader

            pipeline = turboloader.Resize(224, 224) | turboloader.RandAugment(
                num_ops=2, magnitude=9
            )

            result = pipeline.apply(sample_image)
            assert result.shape == (224, 224, 3)
        except ImportError:
            pytest.skip("C++ module not built")

    def test_pipe_with_gridmask(self, sample_image):
        """Test piping with GridMask"""
        try:
            import turboloader

            pipeline = turboloader.Resize(224, 224) | turboloader.GridMask(
                d=0.5, ratio=0.6, p=1.0
            )  # Always apply

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
                "DataLoader",
                "version",
                "features",
                "__version__",
                # TBL v2
                "TblReaderV2",
                "TblWriterV2",
                "SampleFormat",
                "MetadataType",
                # Smart Batching
                "SmartBatchConfig",
                # Transform Composition
                "Compose",
                "ComposedTransforms",
                # Transforms
                "Resize",
                "CenterCrop",
                "RandomCrop",
                "RandomHorizontalFlip",
                "RandomVerticalFlip",
                "ColorJitter",
                "GaussianBlur",
                "Grayscale",
                "Normalize",
                "ImageNetNormalize",
                "ToTensor",
                "Pad",
                "RandomRotation",
                "RandomAffine",
                "RandomPerspective",
                "RandomPosterize",
                "RandomSolarize",
                "RandomErasing",
                "AutoAugment",
                "AutoAugmentPolicy",
                # Modern Augmentations (v1.8.0)
                "MixUp",
                "CutMix",
                "Mosaic",
                "RandAugment",
                "GridMask",
                # Logging (v1.8.0)
                "LogLevel",
                "enable_logging",
                "disable_logging",
                "set_log_level",
                "set_log_output",
                # Enums
                "InterpolationMode",
                "PaddingMode",
                "TensorFormat",
            ]

            for export in expected_exports:
                assert hasattr(turboloader, export), f"Missing export: {export}"

        except ImportError:
            pytest.skip("C++ module not built")


def run_tests():
    """Run all tests and print summary"""
    print("=" * 70)
    print("TurboLoader v2.0.0 Feature Tests")
    print("=" * 70)

    # Run pytest
    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
        ]
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
