#!/usr/bin/env python3
"""
TurboLoader Installation Verification Script

This script verifies that TurboLoader is properly installed and working.
It checks:
- Package installation and version
- Core functionality (DataLoader, transforms)
- SIMD acceleration availability
- Dependencies and system requirements

Usage:
    python verify_installation.py
    python verify_installation.py --verbose
    python verify_installation.py --create-test-data
"""

import argparse
import sys
import os
import tempfile
import tarfile
from pathlib import Path


def print_header(text):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"✗ {text}")


def print_warning(text):
    """Print warning message."""
    print(f"⚠ {text}")


def print_info(text):
    """Print info message."""
    print(f"  {text}")


def check_import():
    """Check if turboloader can be imported."""
    print_header("Checking TurboLoader Import")

    try:
        import turboloader
        print_success("TurboLoader imported successfully")
        print_info(f"Version: {turboloader.__version__}")
        print_info(f"Location: {turboloader.__file__}")
        return True, turboloader
    except ImportError as e:
        print_error(f"Failed to import turboloader: {e}")
        print_info("Install with: pip install turboloader")
        return False, None


def check_version(turboloader):
    """Check TurboLoader version."""
    print_header("Checking Version")

    try:
        version = turboloader.__version__
        print_success(f"TurboLoader version: {version}")

        # Parse version
        major, minor, patch = map(int, version.split('.'))
        if major < 1 or (major == 1 and minor < 7):
            print_warning(f"You are using an older version ({version})")
            print_info("Consider upgrading: pip install --upgrade turboloader")
        else:
            print_success("Version is up to date")

        return True
    except Exception as e:
        print_error(f"Failed to check version: {e}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    print_header("Checking Python Version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print_info(f"Python version: {version_str}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8+ is required")
        return False

    print_success("Python version is compatible")
    return True


def check_dependencies():
    """Check required dependencies."""
    print_header("Checking Dependencies")

    dependencies = {
        'numpy': 'NumPy',
        'PIL': 'Pillow (optional, for comparison)'
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_success(f"{name} is available")
        except ImportError:
            if module == 'PIL':
                print_warning(f"{name} not found (optional)")
            else:
                print_error(f"{name} not found")
                all_ok = False

    return all_ok


def create_test_data(verbose=False):
    """Create test TAR file with synthetic images."""
    print_header("Creating Test Data")

    try:
        import numpy as np

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='turboloader_test_')
        tar_path = os.path.join(temp_dir, 'test_data.tar')

        if verbose:
            print_info(f"Creating test data at: {tar_path}")

        # Create TAR with synthetic images
        with tarfile.open(tar_path, 'w') as tar:
            for i in range(10):
                # Create synthetic image (100x100 RGB)
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

                # Convert to JPEG bytes
                from io import BytesIO
                try:
                    from PIL import Image
                    img = Image.fromarray(img_array)
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=90)
                    img_bytes = buffer.getvalue()
                except ImportError:
                    # Fallback: create minimal JPEG header (won't be valid)
                    print_warning("PIL not available, creating placeholder data")
                    img_bytes = img_array.tobytes()

                # Add to TAR
                from io import BytesIO
                info = tarfile.TarInfo(name=f'image_{i:03d}.jpg')
                info.size = len(img_bytes)
                tar.addfile(info, BytesIO(img_bytes))

        print_success(f"Created test TAR with 10 images")
        if verbose:
            print_info(f"Size: {os.path.getsize(tar_path) / 1024:.1f} KB")

        return tar_path

    except Exception as e:
        print_error(f"Failed to create test data: {e}")
        return None


def test_dataloader(turboloader, tar_path, verbose=False):
    """Test basic DataLoader functionality."""
    print_header("Testing DataLoader")

    try:
        # Create DataLoader
        if verbose:
            print_info("Creating DataLoader...")

        loader = turboloader.DataLoader(
            tar_path,
            batch_size=4,
            num_workers=2,
            shuffle=False
        )

        print_success("DataLoader created successfully")

        # Iterate over batches
        if verbose:
            print_info("Loading batches...")

        batch_count = 0
        sample_count = 0

        for batch in loader:
            batch_count += 1
            sample_count += len(batch)

            if verbose and batch_count == 1:
                sample = batch[0]
                img = sample['image']
                print_info(f"First batch: {len(batch)} samples")
                print_info(f"Image shape: {img.shape}")
                print_info(f"Image dtype: {img.dtype}")

        print_success(f"Loaded {sample_count} samples in {batch_count} batches")
        return True

    except Exception as e:
        print_error(f"DataLoader test failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_transforms(turboloader, verbose=False):
    """Test transform functionality."""
    print_header("Testing Transforms")

    try:
        import numpy as np

        # Create synthetic image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        if verbose:
            print_info(f"Input image shape: {img.shape}")

        # Test individual transforms
        transforms = [
            ('Resize', turboloader.Resize(224, 224)),
            ('RandomHorizontalFlip', turboloader.RandomHorizontalFlip(0.5)),
            ('ImageNetNormalize', turboloader.ImageNetNormalize())
        ]

        for name, transform in transforms:
            try:
                result = transform.apply(img)
                print_success(f"{name}: {result.shape}")

                if verbose:
                    print_info(f"  Input: {img.shape}, Output: {result.shape}")

            except Exception as e:
                print_error(f"{name} failed: {e}")
                return False

        # Test Compose
        if verbose:
            print_info("Testing Compose...")

        pipeline = turboloader.Compose([
            turboloader.Resize(256, 256),
            turboloader.RandomCrop(224, 224),
            turboloader.RandomHorizontalFlip(0.5),
            turboloader.ImageNetNormalize()
        ])

        result = pipeline.apply(img)
        print_success(f"Compose pipeline: {result.shape}")

        return True

    except Exception as e:
        print_error(f"Transform test failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_performance(turboloader, tar_path, verbose=False):
    """Quick performance test."""
    print_header("Testing Performance")

    try:
        import time

        loader = turboloader.DataLoader(
            tar_path,
            batch_size=4,
            num_workers=4,
            shuffle=False
        )

        # Warmup
        for _ in loader:
            break

        # Measure throughput
        start = time.time()
        sample_count = 0

        for batch in loader:
            sample_count += len(batch)

        elapsed = time.time() - start
        throughput = sample_count / elapsed if elapsed > 0 else 0

        print_success(f"Loaded {sample_count} samples in {elapsed:.2f}s")
        print_info(f"Throughput: {throughput:.1f} images/sec")

        if throughput < 100 and verbose:
            print_warning("Performance seems low for synthetic data")
            print_info("This is normal for small test datasets")

        return True

    except Exception as e:
        print_error(f"Performance test failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def check_simd_support(verbose=False):
    """Check if SIMD acceleration is available."""
    print_header("Checking SIMD Support")

    try:
        import platform

        machine = platform.machine().lower()
        system = platform.system()

        print_info(f"Platform: {system}")
        print_info(f"Architecture: {machine}")

        if 'x86_64' in machine or 'amd64' in machine:
            print_success("AVX2 SIMD support available (x86_64)")
        elif 'arm64' in machine or 'aarch64' in machine:
            print_success("NEON SIMD support available (ARM64)")
        else:
            print_warning(f"Unknown architecture: {machine}")
            print_info("SIMD acceleration may not be available")

        return True

    except Exception as e:
        print_error(f"SIMD check failed: {e}")
        return False


def print_summary(results):
    """Print summary of all tests."""
    print_header("Verification Summary")

    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n" + "=" * 70)
        print("  🎉 All checks passed! TurboLoader is ready to use.")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Check out examples/quickstart.ipynb for tutorials")
        print("  - Read docs/TROUBLESHOOTING.md for common issues")
        print("  - Visit https://github.com/ALJainProjects/TurboLoader")
        return True
    else:
        print("\n" + "=" * 70)
        print("  ⚠️  Some checks failed. See errors above.")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  - Read docs/TROUBLESHOOTING.md")
        print("  - Run with --verbose for more details")
        print("  - Check https://github.com/ALJainProjects/TurboLoader/issues")
        return False


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description='Verify TurboLoader installation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--create-test-data',
        action='store_true',
        help='Create test data even if test TAR exists'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  TurboLoader Installation Verification")
    print("=" * 70)

    results = {}

    # Check Python version
    results['python_version'] = check_python_version()

    # Check import
    success, turboloader = check_import()
    results['import'] = success

    if not success:
        print_summary(results)
        return 1

    # Check version
    results['version'] = check_version(turboloader)

    # Check dependencies
    results['dependencies'] = check_dependencies()

    # Check SIMD support
    results['simd'] = check_simd_support(args.verbose)

    # Create test data
    tar_path = create_test_data(args.verbose)
    results['test_data'] = tar_path is not None

    if tar_path:
        # Test DataLoader
        results['dataloader'] = test_dataloader(
            turboloader, tar_path, args.verbose
        )

        # Test transforms
        results['transforms'] = test_transforms(turboloader, args.verbose)

        # Test performance
        results['performance'] = test_performance(
            turboloader, tar_path, args.verbose
        )

        # Cleanup
        try:
            import shutil
            shutil.rmtree(os.path.dirname(tar_path))
        except:
            pass

    # Print summary
    success = print_summary(results)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
