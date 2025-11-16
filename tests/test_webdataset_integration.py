"""
Dedicated WebDataset integration tests for TurboLoader

Tests WebDataset format loading with various file types.
"""

import sys
import os
import tempfile
import tarfile
import json
import numpy as np
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))


def create_webdataset_tar(num_samples=10):
    """
    Create a WebDataset TAR file for testing

    WebDataset format: files grouped by sample ID
    Example: 000000.jpg, 000000.json, 000001.jpg, 000001.json
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required. Install it with: pip install Pillow")

    tmpdir = tempfile.mkdtemp()
    tar_path = os.path.join(tmpdir, 'webdataset.tar')

    # Create sample files
    files_dir = os.path.join(tmpdir, 'files')
    os.makedirs(files_dir, exist_ok=True)

    for i in range(num_samples):
        sample_id = f"{i:06d}"

        # Create JPEG image
        img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_data, mode='RGB')
        img_path = os.path.join(files_dir, f'{sample_id}.jpg')
        img.save(img_path, 'JPEG', quality=90)

        # Create JSON metadata
        metadata = {
            'label': i % 10,
            'caption': f'Sample {i}',
            'category': 'test'
        }
        json_path = os.path.join(files_dir, f'{sample_id}.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f)

        # Create TXT file
        txt_path = os.path.join(files_dir, f'{sample_id}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Description for sample {i}")

    # Create TAR file
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_samples):
            sample_id = f"{i:06d}"
            tar.add(os.path.join(files_dir, f'{sample_id}.jpg'), arcname=f'{sample_id}.jpg')
            tar.add(os.path.join(files_dir, f'{sample_id}.json'), arcname=f'{sample_id}.json')
            tar.add(os.path.join(files_dir, f'{sample_id}.txt'), arcname=f'{sample_id}.txt')

    return tar_path


def test_webdataset_basic():
    """Test basic WebDatasetLoader functionality"""
    print("\n" + "="*80)
    print("Test 1: Basic WebDatasetLoader")
    print("="*80)

    try:
        from webdataset_loader import WebDatasetLoader

        tar_path = create_webdataset_tar(10)

        loader = WebDatasetLoader(
            tar_path,
            batch_size=3,
            num_workers=2,
            shuffle=False
        )

        print(f"  ✓ Created WebDatasetLoader")

        # Test iteration
        batch_count = 0
        total_samples = 0

        for batch in loader:
            batch_count += 1
            total_samples += len(batch)
            print(f"  ✓ Batch {batch_count}: {len(batch)} samples")

            # Verify first sample has expected keys
            if batch:
                sample = batch[0]
                print(f"    Keys: {list(sample.keys())}")
                assert 'jpg' in sample or 'jpeg' in sample

            if batch_count >= 3:
                break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print(f"  ✓ Processed {total_samples} samples in {batch_count} batches")
        print("  ✓ PASSED: Basic WebDatasetLoader")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webdataset_decoder():
    """Test WebDataset decoder functionality"""
    print("\n" + "="*80)
    print("Test 2: WebDataset Decoder")
    print("="*80)

    try:
        from webdataset_loader import WebDatasetLoader, webdataset_decoder

        tar_path = create_webdataset_tar(8)

        loader = WebDatasetLoader(
            tar_path,
            batch_size=2,
            num_workers=1,
            transform=webdataset_decoder
        )

        print(f"  ✓ Created WebDatasetLoader with decoder")

        # Get one batch
        for batch in loader:
            print(f"  ✓ Batch retrieved: {len(batch)} samples")

            if batch:
                sample = batch[0]
                print(f"    Decoded keys: {list(sample.keys())}")

                # Check for expected decoded keys
                if 'image' in sample:
                    print(f"    Image shape: {sample['image'].shape}")
                if 'label' in sample:
                    print(f"    Label: {sample['label']}")
                if 'text' in sample:
                    print(f"    Text length: {len(sample['text'])}")

            break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: WebDataset Decoder")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_webdataset():
    """Test PyTorch WebDataset wrapper"""
    print("\n" + "="*80)
    print("Test 3: PyTorch WebDataset")
    print("="*80)

    try:
        from webdataset_loader import PyTorchWebDataset

        tar_path = create_webdataset_tar(12)

        dataset = PyTorchWebDataset(
            tar_path,
            batch_size=4,
            num_workers=2,
            decode=True
        )

        print(f"  ✓ Created PyTorchWebDataset")

        # Test iteration
        batch_count = 0
        for batch in dataset:
            batch_count += 1
            print(f"  ✓ Batch {batch_count}: {len(batch)} samples")

            if batch_count >= 2:
                break

        dataset.__exit__(None, None, None)
        os.remove(tar_path)

        print(f"  ✓ Consumed {batch_count} batches")
        print("  ✓ PASSED: PyTorch WebDataset")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webdataset_shuffle():
    """Test WebDataset with shuffle"""
    print("\n" + "="*80)
    print("Test 4: WebDataset Shuffle")
    print("="*80)

    try:
        from webdataset_loader import WebDatasetLoader

        tar_path = create_webdataset_tar(20)

        loader = WebDatasetLoader(
            tar_path,
            batch_size=5,
            num_workers=2,
            shuffle=True
        )

        print(f"  ✓ Created WebDatasetLoader with shuffle=True")

        # Get samples
        sample_count = 0
        for batch in loader:
            sample_count += len(batch)
            if sample_count >= 10:
                break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print(f"  ✓ Loaded {sample_count} samples with shuffling")
        print("  ✓ PASSED: WebDataset Shuffle")
        return True

    except ImportError as e:
        print(f"  ⊘ SKIPPED: {e}")
        return None
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webdataset_custom_transform():
    """Test WebDataset with custom transform"""
    print("\n" + "="*80)
    print("Test 5: WebDataset Custom Transform")
    print("="*80)

    try:
        from webdataset_loader import WebDatasetLoader

        tar_path = create_webdataset_tar(8)

        # Custom transform that adds a processed flag
        def custom_transform(sample):
            sample['processed'] = True
            sample['sample_count'] = 1
            return sample

        loader = WebDatasetLoader(
            tar_path,
            batch_size=2,
            num_workers=1,
            transform=custom_transform
        )

        print(f"  ✓ Created WebDatasetLoader with custom transform")

        # Verify transform was applied
        for batch in loader:
            if batch:
                sample = batch[0]
                assert 'processed' in sample
                assert sample['processed'] == True
                print(f"  ✓ Transform applied: processed={sample['processed']}")
            break

        loader.__exit__(None, None, None)
        os.remove(tar_path)

        print("  ✓ PASSED: WebDataset Custom Transform")
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
    """Run all WebDataset integration tests"""
    print("\n" + "="*80)
    print("TurboLoader WebDataset Integration Tests")
    print("="*80)

    results = {}

    # Run all tests
    results['basic_loader'] = test_webdataset_basic()
    results['decoder'] = test_webdataset_decoder()
    results['pytorch_wrapper'] = test_pytorch_webdataset()
    results['shuffle'] = test_webdataset_shuffle()
    results['custom_transform'] = test_webdataset_custom_transform()

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
