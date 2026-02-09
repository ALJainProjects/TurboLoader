#!/usr/bin/env python3
"""
Tests for TurboLoader v2.6.0 new features:
- MemoryEfficientDataLoader class
- create_loader() factory function
"""

import pytest
import numpy as np
import os
import tempfile
import tarfile
from io import BytesIO

# Try to import PIL for creating test images
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_test_tar(num_images=20, width=64, height=48):
    """Create a temporary tar file with test images."""
    if not HAS_PIL:
        pytest.skip("PIL not available for creating test images")

    # Create temp file
    fd, tar_path = tempfile.mkstemp(suffix=".tar")
    os.close(fd)

    with tarfile.open(tar_path, "w") as tar:
        for i in range(num_images):
            # Create a random RGB image
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save to buffer
            buf = BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            # Add to tar
            tarinfo = tarfile.TarInfo(name=f"image_{i:04d}.jpg")
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo, buf)

    return tar_path


@pytest.fixture
def test_tar():
    """Fixture that creates and cleans up a test tar file."""
    tar_path = create_test_tar(num_images=50, width=64, height=48)
    yield tar_path
    os.unlink(tar_path)


class TestMemoryEfficientDataLoader:
    """Tests for the MemoryEfficientDataLoader class."""

    def test_import(self):
        """Test that MemoryEfficientDataLoader can be imported."""
        import turboloader

        assert hasattr(turboloader, "MemoryEfficientDataLoader")

    def test_in_all_exports(self):
        """Test that MemoryEfficientDataLoader is in __all__."""
        import turboloader

        assert "MemoryEfficientDataLoader" in turboloader.__all__

    def test_creation_default(self, test_tar):
        """Test that MemoryEfficientDataLoader can be created with defaults."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
        )
        assert loader is not None
        loader.stop()

    def test_creation_with_memory_budget(self, test_tar):
        """Test creation with explicit memory budget."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=256,
        )
        assert loader is not None
        assert loader.max_memory_mb == 256
        loader.stop()

    def test_auto_tuning_workers_low_memory(self, test_tar):
        """Test that workers are auto-tuned for low memory budget."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=256,
        )
        # Low memory should have fewer workers
        assert loader.num_workers <= 4
        loader.stop()

    def test_auto_tuning_workers_high_memory(self, test_tar):
        """Test that workers are auto-tuned for high memory budget."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=2048,
        )
        # High memory can have more workers
        assert loader.num_workers >= 4
        loader.stop()

    def test_auto_tuning_prefetch(self, test_tar):
        """Test that prefetch_batches is limited for memory efficiency."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=512,
        )
        # Memory efficient loader should have limited prefetch
        assert loader.prefetch_batches <= 2
        loader.stop()

    def test_iteration(self, test_tar):
        """Test that MemoryEfficientDataLoader can iterate over data."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=512,
            output_format="numpy",
        )

        count = 0
        for images, metadata in loader:
            assert isinstance(images, np.ndarray)
            count += images.shape[0]

        assert count == 50  # All images loaded
        loader.stop()

    def test_batch_shape(self, test_tar):
        """Test that batches have correct shape."""
        import turboloader

        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=512,
            output_format="numpy",
        )

        images, _ = next(iter(loader))
        # Shape should be (batch, height, width, channels)
        assert len(images.shape) == 4
        assert images.shape[0] <= 10  # batch size (could be smaller for last batch)
        assert images.shape[3] == 3  # RGB channels
        loader.stop()

    def test_output_format_torch(self, test_tar):
        """Test torch output format if available."""
        import turboloader

        try:
            import torch

            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        # Note: torch output format may not be fully supported in all loaders
        # This test verifies the loader accepts the parameter
        loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=512,
            output_format="torch",
        )

        images, _ = next(iter(loader))
        # Accept either torch tensor or numpy array (implementation dependent)
        assert isinstance(images, (torch.Tensor, np.ndarray))
        loader.stop()


class TestCreateLoaderFactory:
    """Tests for the create_loader() factory function."""

    def test_import(self):
        """Test that create_loader can be imported."""
        import turboloader

        assert hasattr(turboloader, "create_loader")

    def test_in_all_exports(self):
        """Test that create_loader is in __all__."""
        import turboloader

        assert "create_loader" in turboloader.__all__

    def test_create_fast_loader(self, test_tar):
        """Test creating a FastDataLoader via factory."""
        import turboloader

        loader = turboloader.create_loader(
            test_tar,
            loader_type="fast",
            batch_size=10,
        )
        assert isinstance(loader, turboloader.FastDataLoader)
        loader.stop()

    def test_create_memory_efficient_loader(self, test_tar):
        """Test creating a MemoryEfficientDataLoader via factory."""
        import turboloader

        loader = turboloader.create_loader(
            test_tar,
            loader_type="memory_efficient",
            batch_size=10,
            max_memory_mb=512,
        )
        assert isinstance(loader, turboloader.MemoryEfficientDataLoader)
        loader.stop()

    def test_create_standard_loader(self, test_tar):
        """Test creating a DataLoader via factory."""
        import turboloader

        loader = turboloader.create_loader(
            test_tar,
            loader_type="standard",
            batch_size=10,
        )
        assert isinstance(loader, turboloader.DataLoader)
        loader.stop()

    def test_default_is_fast(self, test_tar):
        """Test that default loader_type is 'fast'."""
        import turboloader

        loader = turboloader.create_loader(
            test_tar,
            batch_size=10,
        )
        assert isinstance(loader, turboloader.FastDataLoader)
        loader.stop()

    def test_invalid_loader_type(self, test_tar):
        """Test that invalid loader_type raises error."""
        import turboloader

        with pytest.raises(ValueError):
            turboloader.create_loader(
                test_tar,
                loader_type="invalid_type",
                batch_size=10,
            )

    def test_kwargs_passed_through(self, test_tar):
        """Test that kwargs are passed to the loader."""
        import turboloader

        loader = turboloader.create_loader(
            test_tar,
            loader_type="memory_efficient",
            batch_size=16,
            max_memory_mb=1024,
            output_format="numpy",
        )
        # Verify max_memory_mb was passed through
        assert loader.max_memory_mb == 1024
        loader.stop()


class TestMemoryComparison:
    """Tests comparing memory usage between loader types."""

    def test_memory_efficient_workers_scale_with_budget(self, test_tar):
        """Test that MemoryEfficientDataLoader scales workers with memory budget."""
        import turboloader

        low_mem_loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=256,  # Low memory budget
        )

        high_mem_loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=2048,  # High memory budget
        )

        # Higher memory budget should allow more workers
        assert high_mem_loader.num_workers >= low_mem_loader.num_workers

        low_mem_loader.stop()
        high_mem_loader.stop()

    def test_memory_efficient_prefetch_limited(self, test_tar):
        """Test that MemoryEfficientDataLoader limits prefetch batches."""
        import turboloader

        mem_loader = turboloader.MemoryEfficientDataLoader(
            test_tar,
            batch_size=10,
            max_memory_mb=512,
        )

        # Memory efficient should have limited prefetch (max 2)
        assert mem_loader.prefetch_batches <= 2

        mem_loader.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
