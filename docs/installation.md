<!-- Generated from the project wiki page https://github.com/ALJainProjects/TurboLoader/wiki/Installation — edit there. -->
> Canonical, always-current version: **[Installation](https://github.com/ALJainProjects/TurboLoader/wiki/Installation)** on the wiki.

# Installation

## PyPI (recommended)

```bash
pip install turboloader            # CPU (SIMD) + Metal on Apple Silicon
pip install "turboloader[torch]"   # adds PyTorch for output_format='pytorch' helpers / pinned rings
```

Every release publishes **16 files**: wheels for CPython **3.10, 3.11, 3.12, 3.13, 3.14** on **Linux x86_64** (manylinux_2_27/2_28), **Linux aarch64**, and **macOS arm64** (`macosx_11_0_arm64`), plus a source distribution. Wheels are fully self-contained (libjpeg-turbo, lz4, curl are bundled; the macOS wheel includes the Metal kernels and the AVFoundation/VideoToolbox video path with no FFmpeg dependency).

| Platform | What you get |
|---|---|
| Linux x86_64 / aarch64 | prebuilt wheel, CPU SIMD (AVX2/AVX-512 or NEON) |
| macOS Apple Silicon (11.0+) | prebuilt wheel, CPU NEON + **Metal** GPU paths + **hardware video decode** |
| macOS Intel | prebuilt `macosx_11_0_x86_64` wheel since v2.38 (cross-compiled on the arm64 runner, tested under Rosetta; best-effort leg like arm64 — the sdist always covers it). No Metal path on Intel. |
| Windows | not supported natively — use WSL2 (the CUDA build is developed on WSL2) |
| NVIDIA CUDA | **not in the PyPI wheels** — build from source (below) or use a CUDA-13 wheel from a GitHub Release |

PyTorch is optional: the core loaders, SIMD transforms, numpy and TensorFlow-HWC outputs work without it. `torch` is required for `pin_memory=True`, `TokenDataLoader(device=...)`, `CudaPrefetcher`, and all CUDA loaders.

### Verify what you actually got

```python
import turboloader as t
print(t.__version__)          # e.g. 2.37.0
print(t.features())           # the source of truth for compiled-in capabilities
print(t.metal_available())    # True on the macOS arm64 wheel
print(t.cuda_available())     # True only on a CUDA build
```

`features()` reports honestly: `png_decode` and `webp_decode` are **False** (the shipped image pipeline is JPEG-only), `gpu_transforms`/`nvjpeg_decode` are False on PyPI wheels, `metal_gpu_transforms` is True on Apple Silicon. Cloud/HDF5/TFRecord/Zarr backends were removed in v2.35 and report False.

## CUDA (NVIDIA) — build from source

The CUDA path needs the CUDA toolkit and a GPU at build time, so it cannot ship on PyPI.

```bash
pip install nvidia-nvimgcodec-cu12         # nvImageCodec runtime + header (header is auto-discovered)

CUDA_HOME=/usr/local/cuda \
TURBOLOADER_ENABLE_CUDA=1 \                 # transform + video + resident kernels (cudart)
TURBOLOADER_ENABLE_NVJPEG=1 \               # nvJPEG decoder (CudaImageLoader decode="gpu")
TURBOLOADER_ENABLE_NVIMGCODEC=1 \           # nvImageCodec pipeline (decode="nvimgcodec", beats DALI)
TURBOLOADER_CUDA_ARCH=native \              # required on CUDA 13+; or sm_86 (3090), sm_87 (Orin)
  pip install -e . --no-build-isolation
```

Notes:
- `TURBOLOADER_ENABLE_NVIMGCODEC=1` finds `nvimgcodec.h` inside the installed `nvidia-nvimgcodec-cu12` wheel **of the interpreter you build with** — a fresh venv without that wheel fails with `nvimgcodec.h: No such file` (override with `TURBOLOADER_NVIMGCODEC_INCLUDE=<dir>`).
- Jetson / non-standard layouts: `TURBOLOADER_CUDA_INCLUDE`, `TURBOLOADER_CUDA_LIB`; leave nvJPEG off on Jetson (tegra variant).
- Video on CUDA needs `pip install av` (PyAV) for the default CPU decode backend, and `PyNvVideoCodec` for `decode="nvdec"`.
- After building, `t.cuda_available()` must be True and `t.features()['gpu_transforms']` True — the self-hosted CI gate asserts exactly that.

### Prebuilt CUDA-13 wheels (no compile)

Attached to GitHub Releases (Linux x86_64, CUDA 13.x runtime, nvJPEG + nvImageCodec + video/clip kernels), built and fresh-venv verified on an RTX 3090:

- v2.36.0: `cp310` and `cp312` — https://github.com/ALJainProjects/TurboLoader/releases/tag/v2.36.0
- v2.35.0 / v2.34.1: `cp310`
- **v2.37.0: deferred** — the build machine was offline at release time; they will be attached when it returns (see [Roadmap](https://github.com/ALJainProjects/TurboLoader/wiki/Roadmap)). Nothing in 2.37 changed the CUDA kernels; the 2.36.0 cu13 wheels remain valid.

```bash
pip install https://github.com/ALJainProjects/TurboLoader/releases/download/v2.36.0/turboloader-2.36.0+cu13-cp312-cp312-linux_x86_64.whl
pip install nvidia-nvimgcodec-cu12
```

## Building from source (CPU / Metal)

```bash
git clone https://github.com/ALJainProjects/TurboLoader.git && cd TurboLoader
pip install pybind11 setuptools_scm numpy
pip install -e .            # setup.py builds the extension; Metal is compiled automatically on macOS arm64
```

Requirements: a C++20 compiler (GCC 10+ / Clang 14+ / Apple Clang from a recent Xcode), `libjpeg-turbo`, `lz4`, `libcurl` dev headers (`brew install jpeg-turbo lz4` · `apt install libjpeg-turbo8-dev liblz4-dev libcurl4-openssl-dev`). `libpng`/`libwebp` are **not** needed (removed in v2.35). Opt out of Metal with `TURBOLOADER_ENABLE_METAL=0`. The version comes from the git tag via `setuptools_scm` — a shallow clone without tags will report a `0.1.dev` or stale-looking version; clone with tags or set `SETUPTOOLS_SCM_PRETEND_VERSION`.

Local reproduction of the release wheels: `cibuildwheel --platform macos` (much faster than CI cycles; see [Development and CI](https://github.com/ALJainProjects/TurboLoader/wiki/Development-and-CI)).

## Docker

```dockerfile
FROM python:3.12
RUN pip install turboloader
RUN python -c "import turboloader as t; print(t.__version__, t.features()['simd_acceleration'])"
```

No system libraries are needed for the wheel.
