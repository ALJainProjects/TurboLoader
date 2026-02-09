#!/usr/bin/env python3
"""
Tests for TurboLoader Phase 4: Framework tensor support
Tests next_batch_torch() and next_batch_tf() methods

Tests:
1. PyTorch tensor output (shape, dtype, device)
2. TensorFlow tensor output (shape, dtype)
3. Zero-copy conversion behavior
4. Device placement (CPU/GPU)
5. Dtype conversion
6. Error handling for missing dependencies
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

# Try to import PyTorch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import TensorFlow
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False


def create_test_tar(num_images=20, width=256, height=192):
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
            img.save(buf, format="JPEG", quality=90)
            buf.seek(0)

            # Add to tar
            tarinfo = tarfile.TarInfo(name=f"image_{i:04d}.jpg")
            tarinfo.size = len(buf.getvalue())
            tar.addfile(tarinfo, buf)

    return tar_path


@pytest.fixture
def test_tar():
    """Fixture that creates and cleans up a test tar file."""
    tar_path = create_test_tar(num_images=30, width=256, height=192)
    yield tar_path
    os.unlink(tar_path)


class TestNextBatchTorch:
    """Tests for next_batch_torch() method."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_returns_torch_tensor(self, test_tar):
        """Test that next_batch_torch returns a torch.Tensor."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=10, num_workers=2)

        images, metadata = loader.next_batch_torch()

        assert isinstance(images, torch.Tensor)
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor_shape_chw(self, test_tar):
        """Test that PyTorch tensor has correct CHW shape."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=10, num_workers=2)

        images, _ = loader.next_batch_torch()

        # Should be (N, C, H, W)
        assert len(images.shape) == 4
        assert images.shape[0] <= 10  # batch size
        assert images.shape[1] == 3  # channels first
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor_dtype_float32(self, test_tar):
        """Test default dtype is float32 (normalized)."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, _ = loader.next_batch_torch()

        # Default should be float32 normalized to [0, 1]
        assert images.dtype == torch.float32
        assert images.min() >= 0.0
        assert images.max() <= 1.0
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor_custom_dtype(self, test_tar):
        """Test custom dtype conversion."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        # Request float16
        images, _ = loader.next_batch_torch(dtype=torch.float16)

        assert images.dtype == torch.float16
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor_cpu_device(self, test_tar):
        """Test CPU device placement."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, _ = loader.next_batch_torch(device="cpu")

        assert images.device.type == "cpu"
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    @pytest.mark.skipif(
        not torch.cuda.is_available() if HAS_TORCH else True, reason="CUDA not available"
    )
    def test_torch_tensor_cuda_device(self, test_tar):
        """Test CUDA device placement."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, _ = loader.next_batch_torch(device="cuda")

        assert images.device.type == "cuda"
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_tensor_contiguous(self, test_tar):
        """Test that output tensor is contiguous."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, _ = loader.next_batch_torch()

        assert images.is_contiguous()
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_metadata_preserved(self, test_tar):
        """Test that metadata is returned with tensors."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, metadata = loader.next_batch_torch()

        assert isinstance(metadata, dict)
        assert "filenames" in metadata or "indices" in metadata
        loader.stop()


class TestNextBatchTF:
    """Tests for next_batch_tf() method."""

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_returns_tf_tensor(self, test_tar):
        """Test that next_batch_tf returns a tf.Tensor."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=10, num_workers=2)

        images, metadata = loader.next_batch_tf()

        assert isinstance(images, tf.Tensor)
        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_tensor_shape_hwc(self, test_tar):
        """Test that TensorFlow tensor has correct HWC shape."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=10, num_workers=2)

        images, _ = loader.next_batch_tf()

        # Should be (N, H, W, C)
        assert len(images.shape) == 4
        assert images.shape[0] <= 10  # batch size
        assert images.shape[-1] == 3  # channels last
        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_tensor_dtype_float32(self, test_tar):
        """Test default dtype is float32 (normalized)."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, _ = loader.next_batch_tf()

        # Default should be float32 normalized to [0, 1]
        assert images.dtype == tf.float32
        assert float(tf.reduce_min(images)) >= 0.0
        assert float(tf.reduce_max(images)) <= 1.0
        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_tensor_custom_dtype(self, test_tar):
        """Test custom dtype conversion."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        # Request float16
        images, _ = loader.next_batch_tf(dtype=tf.float16)

        assert images.dtype == tf.float16
        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_metadata_preserved(self, test_tar):
        """Test that metadata is returned with tensors."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        images, metadata = loader.next_batch_tf()

        assert isinstance(metadata, dict)
        loader.stop()


class TestFrameworkIntegration:
    """Integration tests for framework tensor support."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_iteration(self, test_tar):
        """Test iterating through batches with PyTorch tensors."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=10, num_workers=2)

        total_images = 0
        batches = 0

        while not loader.is_finished() and batches < 5:
            images, _ = loader.next_batch_torch()
            total_images += images.shape[0]
            batches += 1

        assert total_images > 0
        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_iteration(self, test_tar):
        """Test iterating through batches with TensorFlow tensors."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=10, num_workers=2)

        total_images = 0
        batches = 0

        while not loader.is_finished() and batches < 5:
            images, _ = loader.next_batch_tf()
            total_images += images.shape[0]
            batches += 1

        assert total_images > 0
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_training_simulation(self, test_tar):
        """Simulate a training loop with PyTorch tensors."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=8, num_workers=2)

        # Simulate simple model (just sum)
        for i in range(3):
            if loader.is_finished():
                break
            images, _ = loader.next_batch_torch()
            # Simulate forward pass
            output = images.sum()
            assert isinstance(output, torch.Tensor)

        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_training_simulation(self, test_tar):
        """Simulate a training loop with TensorFlow tensors."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=8, num_workers=2)

        # Simulate simple model (just sum)
        for i in range(3):
            if loader.is_finished():
                break
            images, _ = loader.next_batch_tf()
            # Simulate forward pass
            output = tf.reduce_sum(images)
            assert isinstance(output, tf.Tensor)

        loader.stop()


class TestEdgeCases:
    """Edge case tests for framework tensor support."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_batch_size_one(self, test_tar):
        """Test with batch size of 1."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=1, num_workers=1)

        images, _ = loader.next_batch_torch()

        assert images.shape[0] == 1
        loader.stop()

    @pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
    def test_tf_batch_size_one(self, test_tar):
        """Test with batch size of 1."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=1, num_workers=1)

        images, _ = loader.next_batch_tf()

        assert images.shape[0] == 1
        loader.stop()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_multiple_batches_sequential(self, test_tar):
        """Test getting multiple batches sequentially."""
        import turboloader

        loader = turboloader.FastDataLoader(test_tar, batch_size=5, num_workers=2)

        batches = []
        for _ in range(4):
            if not loader.is_finished():
                images, _ = loader.next_batch_torch()
                batches.append(images)

        assert len(batches) >= 2
        for batch in batches:
            assert isinstance(batch, torch.Tensor)

        loader.stop()


class TestVersionInfo:
    """Test version info mentions framework support."""

    def test_version_available(self):
        """Test that version string is available."""
        import turboloader

        assert hasattr(turboloader, "__version__")
        assert turboloader.__version__.startswith("2.")

    def test_fastdataloader_has_torch_method(self):
        """Test that FastDataLoader has next_batch_torch method."""
        import turboloader

        assert hasattr(turboloader.FastDataLoader, "next_batch_torch")

    def test_fastdataloader_has_tf_method(self):
        """Test that FastDataLoader has next_batch_tf method."""
        import turboloader

        assert hasattr(turboloader.FastDataLoader, "next_batch_tf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
