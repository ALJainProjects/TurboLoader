# PyPI Release Guide for TurboLoader

Complete step-by-step guide for publishing TurboLoader to PyPI.

## Prerequisites Checklist

Before publishing, ensure:

- [x] Week 1 tasks completed (code cleanup, docs, examples)
- [x] Package builds successfully (`python3 setup.py sdist`)
- [x] All tests pass
- [ ] PyPI account created
- [ ] TestPyPI account created
- [ ] API tokens generated

---

## Step 1: Create PyPI Accounts

### 1.1 Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Fill in registration form
3. Verify email address
4. Enable 2FA (highly recommended)

### 1.2 Create TestPyPI Account

1. Go to https://test.pypi.org/account/register/
2. Fill in registration form (separate from PyPI)
3. Verify email address

### 1.3 Generate API Tokens

**For TestPyPI:**
1. Log in to https://test.pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name: "turboloader-test"
5. Scope: "Entire account" (for first upload)
6. Copy the token (starts with `pypi-`)

**For Production PyPI:**
1. Log in to https://pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name: "turboloader"
5. Scope: "Entire account" (for first upload)
6. Copy the token

### 1.4 Configure .pypirc

Create `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

**Security:** Ensure proper permissions:
```bash
chmod 600 ~/.pypirc
```

---

## Step 2: Install Build Tools

### Option A: Using Virtual Environment (Recommended)

```bash
cd /Users/arnavjain/turboloader

# Create virtual environment
python3 -m venv .venv-publish

# Activate it
source .venv-publish/bin/activate

# Install tools
pip install build twine wheel setuptools

# Verify installation
twine --version
python -m build --version
```

### Option B: User Installation

```bash
pip3 install --user build twine wheel setuptools
```

---

## Step 3: Clean Previous Builds

```bash
cd /Users/arnavjain/turboloader

# Remove old build artifacts
rm -rf build/ dist/ *.egg-info turboloader.egg-info

# Verify clean state
ls dist/  # Should not exist or be empty
```

---

## Step 4: Build the Package

### 4.1 Build Source Distribution and Wheel

```bash
# Using python -m build (modern way)
python3 -m build

# OR using setup.py (fallback)
python3 setup.py sdist bdist_wheel
```

### 4.2 Verify Build Output

```bash
ls -lh dist/
```

Expected files:
```
turboloader-0.2.0.tar.gz       # Source distribution
turboloader-0.2.0-*.whl        # Wheel (platform-specific)
```

### 4.3 Check Package Metadata

```bash
tar -tzf dist/turboloader-0.2.0.tar.gz | head -20
```

Verify it includes:
- `turboloader-0.2.0/turboloader/__init__.py`
- `turboloader-0.2.0/src/` (C++ sources)
- `turboloader-0.2.0/include/` (headers)
- `turboloader-0.2.0/README.md`
- `turboloader-0.2.0/LICENSE`
- `turboloader-0.2.0/setup.py`
- `turboloader-0.2.0/CMakeLists.txt`

---

## Step 5: Validate Package with Twine

```bash
# Check package for errors
twine check dist/*
```

Expected output:
```
Checking dist/turboloader-0.2.0.tar.gz: PASSED
Checking dist/turboloader-0.2.0-*.whl: PASSED
```

Fix any warnings or errors before proceeding.

---

## Step 6: Test Upload to TestPyPI

### 6.1 Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

You'll see:
```
Uploading distributions to https://test.pypi.org/legacy/
Uploading turboloader-0.2.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uploading turboloader-0.2.0-*.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View at:
https://test.pypi.org/project/turboloader/
```

### 6.2 Verify on TestPyPI

Visit: https://test.pypi.org/project/turboloader/

Check:
- Version number correct (0.2.0)
- Description renders properly
- Links work (GitHub, documentation)
- License shows as MIT
- All classifiers present

### 6.3 Test Installation from TestPyPI

```bash
# Create fresh virtual environment
python3 -m venv /tmp/test-turboloader
source /tmp/test-turboloader/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    turboloader

# Test import
python -c "import turboloader; print(turboloader.__version__)"

# Expected: 0.2.0
```

**Note:** `--extra-index-url https://pypi.org/simple/` allows dependencies (like numpy, torch) to be installed from production PyPI.

### 6.4 Test Basic Functionality

```python
python3 << 'EOF'
import sys
sys.path.insert(0, 'build/python')
import turboloader

# Test basic objects exist
config = turboloader.Config()
print(f"✓ Config created: workers={config.num_workers}")

transform_config = turboloader.TransformConfig()
print(f"✓ TransformConfig created: size={transform_config.target_width}x{transform_config.target_height}")

print("✓ All basic tests passed!")
EOF
```

### 6.5 Fix Any Issues

If problems found:
1. Delete the TestPyPI release (Account Settings → Your projects → Manage → Delete)
2. Fix issues locally
3. Increment version to 0.2.1 in `setup.py` and `pyproject.toml`
4. Rebuild and re-upload to TestPyPI

---

## Step 7: Upload to Production PyPI

**⚠️ WARNING: Once uploaded to PyPI, you CANNOT delete or replace a version!**

### 7.1 Final Pre-flight Checks

```bash
# Ensure you're uploading the right version
cat setup.py | grep version=
cat pyproject.toml | grep version

# Verify package contents one more time
twine check dist/*

# Check README renders correctly
python3 -c "import setuptools; print(setuptools.setup.readme)"
```

### 7.2 Upload to Production PyPI

```bash
# Deep breath... this is it!
twine upload dist/*
```

Expected output:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading turboloader-0.2.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uploading turboloader-0.2.0-*.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

View at:
https://pypi.org/project/turboloader/
```

### 7.3 Verify on PyPI

Visit: https://pypi.org/project/turboloader/

Verify everything looks correct!

### 7.4 Test Installation from PyPI

```bash
# Fresh environment
python3 -m venv /tmp/test-pypi-install
source /tmp/test-pypi-install/bin/activate

# Install from PyPI
pip install turboloader

# Verify
python -c "import turboloader; print(turboloader.__version__)"
```

---

## Step 8: Create GitHub Release

### 8.1 Create Git Tag

```bash
cd /Users/arnavjain/turboloader

# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0 - Initial PyPI publication

Features:
- 30-35x speedup over PyTorch DataLoader
- SIMD-optimized transforms (AVX2/AVX-512/NEON)
- Lock-free concurrency
- Zero-copy I/O with mmap
- Drop-in replacement API
"

# Push tag
git push origin v0.2.0
```

### 8.2 Create GitHub Release

1. Go to https://github.com/arnavjain/turboloader/releases/new
2. Select tag: `v0.2.0`
3. Release title: `TurboLoader v0.2.0 - Initial Release`
4. Description:

```markdown
# TurboLoader v0.2.0

🚀 **Initial PyPI Release**

## Highlights

- **30-35x faster** than PyTorch DataLoader on ImageNet
- Drop-in replacement with minimal code changes
- SIMD optimizations (AVX2, AVX-512, NEON)
- Lock-free queue for low-latency data transfer
- Zero-copy I/O with memory-mapped files

## Installation

```bash
pip install turboloader
```

## Quick Start

```python
import turboloader

config = turboloader.Config(num_workers=16, batch_size=256)
pipeline = turboloader.Pipeline(["data.tar"], config)
pipeline.start()

batch = pipeline.next_batch(256)
```

## Benchmark Results

| Dataset | PyTorch | TurboLoader | Speedup |
|---------|---------|-------------|---------|
| ImageNet | 587 img/s | 18,457 img/s | **31.4x** |
| CIFAR-10 | 385 img/s | 12,450 img/s | **32.3x** |

## Documentation

- [README](https://github.com/arnavjain/turboloader/blob/main/README.md)
- [Architecture Deep Dive](https://github.com/arnavjain/turboloader/blob/main/ARCHITECTURE.md)
- [Examples](https://github.com/arnavjain/turboloader/tree/main/examples)
- [Contributing](https://github.com/arnavjain/turboloader/blob/main/CONTRIBUTING.md)

## What's Next

- WebDataset iterator API
- TensorFlow/JAX bindings
- GPU-accelerated JPEG decoding
- Additional transforms

## Contributors

Thank you to everyone who helped make this release possible!

---

**Full Changelog**: https://github.com/arnavjain/turboloader/commits/v0.2.0
```

5. Attach files:
   - Upload `dist/turboloader-0.2.0.tar.gz`
   - Upload wheel file `dist/turboloader-0.2.0-*.whl`

6. Click "Publish release"

---

## Step 9: Verify Everything Works

### 9.1 Check PyPI Badge

Add to README.md:
```markdown
[![PyPI version](https://badge.fury.io/py/turboloader.svg)](https://badge.fury.io/py/turboloader)
[![Downloads](https://pepy.tech/badge/turboloader)](https://pepy.tech/project/turboloader)
```

### 9.2 Test Installation on Clean System

```bash
# Simulate fresh user install
docker run -it --rm python:3.10 bash -c "
    pip install turboloader &&
    python -c 'import turboloader; print(turboloader.__version__)'
"
```

### 9.3 Monitor Initial Downloads

- PyPI downloads: https://pypi.org/project/turboloader/#files
- PePy stats: https://pepy.tech/project/turboloader

---

## Troubleshooting

### Issue: "File already exists"

**Cause:** Version 0.2.0 already uploaded

**Solution:**
- CANNOT re-upload same version to PyPI
- Must increment version (e.g., 0.2.1)
- Update in both `setup.py` and `pyproject.toml`

### Issue: "Invalid distribution file"

**Cause:** Package structure incorrect

**Solution:**
```bash
# Check package contents
tar -tzf dist/turboloader-0.2.0.tar.gz | grep turboloader/

# Ensure turboloader/__init__.py exists
ls -la turboloader/__init__.py
```

### Issue: "Metadata validation failed"

**Cause:** pyproject.toml or setup.py has errors

**Solution:**
```bash
# Validate metadata
python3 setup.py check

# Check for warnings
twine check dist/*
```

### Issue: "Long description rendering failed"

**Cause:** README.md has invalid markdown

**Solution:**
- Test README rendering: https://github.com/pypa/readme_renderer
- Ensure no unsupported HTML

### Issue: "Build fails on install"

**Cause:** C++ compilation errors or missing dependencies

**Solution:**
- Test on fresh system
- Check CMakeLists.txt for dependencies
- Add detailed build instructions to README

---

## Post-Release Tasks

### Immediate (Day 1)

- [ ] Tweet announcement
- [ ] Post to LinkedIn
- [ ] Post to Reddit (r/MachineLearning, r/pytorch)
- [ ] Post to Hacker News
- [ ] Announce on PyTorch forums

### Week 1

- [ ] Monitor GitHub issues
- [ ] Respond to questions
- [ ] Fix critical bugs (release 0.2.1 if needed)
- [ ] Update documentation based on feedback

### Month 1

- [ ] Track download statistics
- [ ] Collect user testimonials
- [ ] Plan next features (v0.3.0)

---

## Version Management

### Versioning Scheme

We use Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR** (0.x.x): Breaking API changes
- **MINOR** (x.2.x): New features, backward compatible
- **PATCH** (x.x.1): Bug fixes, backward compatible

### Release Cadence

- **Patch releases**: As needed for bugs
- **Minor releases**: Every 1-2 months
- **Major releases**: When API is stable (v1.0.0)

### Next Releases

- **v0.2.1**: Bug fixes from initial feedback
- **v0.3.0**: WebDataset iterator API
- **v0.4.0**: TensorFlow/JAX bindings
- **v1.0.0**: Stable API, production-ready

---

## Rollback (Emergency Only)

If critical bug discovered:

1. **Cannot delete from PyPI** - releases are permanent
2. Options:
   - Release patch version (0.2.1) immediately
   - Mark version as "yanked" on PyPI (hides from pip install)

To yank a release:
```bash
# This removes it from default pip install, but doesn't delete
# Users can still install with: pip install turboloader==0.2.0
twine upload --skip-existing --repository pypi dist/*
```

Then mark as yanked in PyPI web interface.

---

## Success Metrics

### Week 1 Goals

- ✅ Package published to PyPI
- 🎯 100+ downloads
- 🎯 10+ GitHub stars
- 🎯 No critical bugs reported

### Month 1 Goals

- 🎯 1,000+ downloads
- 🎯 100+ GitHub stars
- 🎯 3+ community contributions
- 🎯 Listed in PyTorch ecosystem

---

## Quick Reference

```bash
# Complete publication workflow
rm -rf dist/ build/ *.egg-info
python3 setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                         # Production

# Verify
pip install turboloader
python -c "import turboloader; print(turboloader.__version__)"

# Create GitHub release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

---

**Last Updated:** 2025-01-15
**Status:** Ready for publication
**Next Version:** 0.2.0 → 0.3.0 (planned)
