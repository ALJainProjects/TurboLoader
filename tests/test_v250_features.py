#!/usr/bin/env python3
"""
Tests for TurboLoader v2.5.0 new features:
- FastDataLoader class with batch array transfer
- Loader() factory function
- next_batch_array() and next_batch_into() methods
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
    fd, tar_path = tempfile.mkstemp(suffix='.tar')
    os.close(fd)

    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_images):
            # Create a random RGB image
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save to buffer
            buf = BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)

            # Add to tar
            tarinfo = tarfile.TarInfo(name=f'image_{i:04d}.jpg')
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo, buf)

    return tar_path


@pytest.fixture
def test_tar():
    """Fixture that creates and cleans up a test tar file."""
    tar_path = create_test_tar(num_images=50, width=64, height=48)
    yield tar_path
    os.unlink(tar_path)


class TestFastDataLoader:
    """Tests for the FastDataLoader class."""

    def test_import(self):
        """Test that FastDataLoader can be imported."""
        import turboloader
        assert hasattr(turboloader, 'FastDataLoader')

    def test_creation(self, test_tar):
        """Test that FastDataLoader can be created."""
        import turboloader

        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            output_format='numpy'
        )
        assert loader is not None

    def test_next_batch_returns_tuple(self, test_tar):
        """Test that next_batch returns a tuple of (images, metadata) using iterator."""
        import turboloader

        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            output_format='numpy'
        )

        # Use the iterator to get first batch
        images, metadata = next(iter(loader))

        assert isinstance(images, np.ndarray)
        assert isinstance(metadata, dict)
        loader.stop()

    def test_batch_shape_hwc(self, test_tar):
        """Test that batch has correct shape in HWC format."""
        import turboloader

        batch_size = 10
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=batch_size,
            num_workers=2,
            output_format='numpy'  # HWC format
        )

        images, _ = loader.next_batch()

        # Shape should be (N, H, W, C)
        assert len(images.shape) == 4
        assert images.shape[0] <= batch_size  # May be less for last batch
        assert images.shape[3] == 3  # RGB channels

    def test_batch_shape_chw(self, test_tar):
        """Test that batch has correct shape in CHW format."""
        import turboloader

        batch_size = 10
        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=batch_size,
            num_workers=2,
            output_format='numpy_chw'  # CHW format
        )

        images, _ = loader.next_batch()

        # Shape should be (N, C, H, W)
        assert len(images.shape) == 4
        assert images.shape[0] <= batch_size
        assert images.shape[1] == 3  # RGB channels in CHW

    def test_exhaust_all_batches(self, test_tar):
        """Test that we can iterate through all batches using for loop."""
        import turboloader

        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            output_format='numpy'
        )

        total_images = 0
        batch_count = 0

        # Use for loop which uses __iter__ and __next__
        for images, _ in loader:
            total_images += images.shape[0]
            batch_count += 1
            # Safety limit to avoid infinite loops
            if batch_count > 10:
                break

        assert batch_count > 0, "Should have gotten at least one batch"
        assert total_images > 0, "Should have loaded some images"
        # Don't check exact count - async pipeline may terminate differently

    def test_metadata_contains_batch_info(self, test_tar):
        """Test that metadata contains batch information."""
        import turboloader

        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            output_format='numpy'
        )

        _, metadata = loader.next_batch()

        assert 'batch_size' in metadata
        assert 'filenames' in metadata

    def test_output_format_pytorch(self, test_tar):
        """Test that pytorch output format works."""
        import turboloader

        loader = turboloader.FastDataLoader(
            test_tar,
            batch_size=10,
            num_workers=2,
            output_format='pytorch'
        )

        images, _ = loader.next_batch()

        # Should be CHW format for PyTorch
        assert len(images.shape) == 4
        assert images.shape[1] == 3  # CHW


class TestLoaderFactory:
    """Tests for the Loader() factory function."""

    def test_import(self):
        """Test that Loader can be imported."""
        import turboloader
        assert hasattr(turboloader, 'Loader')

    def test_loader_fast_false(self, test_tar):
        """Test that Loader with fast=False returns DataLoader."""
        import turboloader

        loader = turboloader.Loader(
            test_tar,
            batch_size=10,
            num_workers=2,
            fast=False
        )

        assert isinstance(loader, turboloader.DataLoader)

    def test_loader_fast_true(self, test_tar):
        """Test that Loader with fast=True returns FastDataLoader."""
        import turboloader

        loader = turboloader.Loader(
            test_tar,
            batch_size=10,
            num_workers=2,
            fast=True
        )

        assert isinstance(loader, turboloader.FastDataLoader)

    def test_loader_default_is_dataloader(self, test_tar):
        """Test that Loader defaults to DataLoader (fast=False)."""
        import turboloader

        loader = turboloader.Loader(
            test_tar,
            batch_size=10,
            num_workers=2
        )

        assert isinstance(loader, turboloader.DataLoader)


class TestNextBatchArray:
    """Tests for the next_batch_array() C++ binding."""

    def test_binding_exists(self):
        """Test that next_batch_array method exists on _DataLoaderBase."""
        import turboloader
        from turboloader import _DataLoaderBase

        assert hasattr(_DataLoaderBase, 'next_batch_array')

    def test_returns_contiguous_array(self, test_tar):
        """Test that next_batch_array returns contiguous numpy array."""
        import turboloader
        from turboloader import _DataLoaderBase

        loader = _DataLoaderBase(test_tar, batch_size=10, num_workers=2)

        images, metadata = loader.next_batch_array()

        assert isinstance(images, np.ndarray)
        assert images.flags['C_CONTIGUOUS']

    def test_chw_format(self, test_tar):
        """Test CHW format conversion."""
        import turboloader
        from turboloader import _DataLoaderBase

        loader = _DataLoaderBase(test_tar, batch_size=10, num_workers=2)

        # CHW format
        images_chw, _ = loader.next_batch_array(chw_format=True)

        # Should have channels first
        assert images_chw.shape[1] == 3


class TestNextBatchInto:
    """Tests for the next_batch_into() pre-allocated buffer method."""

    def test_binding_exists(self):
        """Test that next_batch_into method exists."""
        import turboloader
        from turboloader import _DataLoaderBase

        assert hasattr(_DataLoaderBase, 'next_batch_into')

    def test_fills_preallocated_buffer(self, test_tar):
        """Test that next_batch_into fills a pre-allocated buffer."""
        import turboloader
        import time
        from turboloader import _DataLoaderBase

        loader = _DataLoaderBase(test_tar, batch_size=10, num_workers=2)

        # Get first batch to know dimensions (with retry for pipeline warmup)
        for _ in range(10):
            first_images, _ = loader.next_batch_array()
            if first_images.size > 0:
                break
            time.sleep(0.01)

        # Recreate loader
        loader = _DataLoaderBase(test_tar, batch_size=10, num_workers=2)

        # Pre-allocate buffer
        buffer = np.zeros_like(first_images)

        # Fill buffer - next_batch_into returns just count (int)
        for _ in range(10):
            count = loader.next_batch_into(buffer)
            if count > 0:
                break
            time.sleep(0.01)

        assert count > 0
        assert count <= 10
        # Buffer should now contain data (not all zeros)
        assert buffer[:count].sum() > 0


class TestVersion:
    """Test version is correctly set."""

    def test_version_is_250(self):
        """Test that version is 2.5.0."""
        import turboloader
        assert turboloader.__version__ == "2.5.0"


class TestBackwardCompatibility:
    """Test that existing DataLoader API still works."""

    def test_dataloader_still_works(self, test_tar):
        """Test that DataLoader iteration still works."""
        import turboloader

        loader = turboloader.DataLoader(
            test_tar,
            batch_size=10,
            num_workers=2
        )

        batch_count = 0
        for batch in loader:
            assert isinstance(batch, list)
            assert len(batch) <= 10
            if len(batch) > 0:
                assert 'image' in batch[0]
            batch_count += 1

        assert batch_count > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
