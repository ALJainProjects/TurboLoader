<!-- Generated from the project wiki page https://github.com/ALJainProjects/TurboLoader/wiki/Troubleshooting-and-FAQ — edit there. -->
> Canonical, always-current version: **[Troubleshooting and FAQ](https://github.com/ALJainProjects/TurboLoader/wiki/Troubleshooting-and-FAQ)** on the wiki.

# Troubleshooting and FAQ

## Installation

**`import turboloader` finds an old version / missing attributes.** Check `python -c "import _turboloader as c; print(c.__file__)"` — a stale extension elsewhere on `sys.path` (an old editable checkout, a `.pth` file) can shadow the installed wheel. `pip install --force-reinstall --no-cache-dir turboloader` and make sure the interpreter is the one you think it is.

**`ValueError: ... needs a CUDA build`** — you're on a PyPI wheel. CUDA loaders need a source build with `TURBOLOADER_ENABLE_CUDA=1` or a cu13 wheel from GitHub Releases ([Installation](https://github.com/ALJainProjects/TurboLoader/wiki/Installation)). Check `turboloader.cuda_available()`.

**Building for CUDA fails with `nvimgcodec.h: No such file`** — `pip install nvidia-nvimgcodec-cu12` into the *build* interpreter; the header is auto-discovered from that wheel.

**Version shows `0.1.devN` or a stale number from source** — `setuptools_scm` needs git tags: clone with tags / `git fetch --tags`, or set `SETUPTOOLS_SCM_PRETEND_VERSION`.

**macOS wheel fails to load / delocate complaints when building wheels** — the video path's framework load commands need `-Wl,-headerpad_max_install_names` (already in `setup.py`; if you fork the build, keep it).

## Data loading

**"Where are my labels?"** A TAR is flat; samples have **no `label` key**. Align your own label array via `meta['indices']` (fast path), `sample['index']` (dict path), or `return_indices=True` (GPU loaders). `PyTorchCompatibleLoader` with a `LabelExtractor` derives labels from folder names, filename patterns, or JSON sidecars if you prefer.

**`ValueError: Conflicting sizes`** — you passed both `image_size=` and a `Resize(...)` transform with different sizes. Pass one (they must agree).

**`ValueError: output_format='pytorch' needs a fixed image size`** — the fast path needs `image_size=` (or a `Resize`) to build one contiguous batch.

**Batches contain another batch's pixels / training diverges after refactoring.** You are holding a zero-copy view past its window — see [Memory and Lifetime Contracts](https://github.com/ALJainProjects/TurboLoader/wiki/Memory-and-Lifetime-Contracts). `.clone()`/`.copy()` or consume before advancing.

**`DataLoader('x.tbl')` raises about `RAW_U8` / mixed sizes / compression.** The training loader serves only uncompressed, uniform RAW files made by `preprocess_to_tbl`; a `.tbl` of JPEGs or an LZ4 file goes through `TblReaderV2` instead. `train_aug=True` on a `.tbl` raises by design (samples are already resized); `TblRawImageLoader(hflip_prob=0.5)` is the one aug available.

**Same order every epoch.** Call `loader.set_epoch(epoch)`.

**Corrupted JPEGs.** The fast path zero-fills the slot of a sample that fails to decode (observable via tests in `tests/test_decode_failures_observable.py`); `cache_decoded=True` drops undelivered rows rather than serving uninitialized memory. Find bad files with the dict path and `sample['filename']`.

## Performance

**More `num_workers` doesn't help.** Expected: the fast path is one process-wide C++ pool, saturated at one worker. Speed comes from staying on the fast path (`output_format='pytorch'`, fused transforms, `train_aug` instead of per-sample Python), `prefetch_batches`, `pin_memory=True`, and — for many epochs — TBL-RAW or a resident loader.

**Throughput lower than the README on my machine.** Source resolution matters (decoding 320px sources is slower than 160px), as do core count, memory bandwidth (the M4 Max numbers are bandwidth-driven), and thermal state. Compare *ratios* from `benchmarks/` on your box, not our absolutes.

**GPU-bound training didn't get faster.** If your step is the bottleneck (e.g. MPS), no loader can help — the pure-GPU floor in `benchmark_e2e_training.py --floor` tells you how much input time is actually exposed.

**Epochs got ~40% slower for every loader.** Torch's default intraop pool fights the decode threads; `torch.set_num_threads(1)` for GPU-bound training.

**`cache_decoded=True` uses a lot of RAM.** It's the whole dataset as float32 (4× uint8); since v2.37 that's built in one allocation (was 3× that at peak). TBL-RAW keeps uint8 in evictable page cache and skips the decode-all pass.

## Video

**"Does TurboLoader support video?"** Yes since v2.34: `MetalVideoLoader` (macOS wheel, no FFmpeg), `CudaVideoLoader` and `VideoDatasetLoader` (CUDA build; `pip install av`). Older docs and FAQs that say otherwise are stale.

**`decode='nvdec'` is slower than `decode='cpu'`.** Under WSL2, NVDEC is virtualization-throttled (130 f/s raw vs 1,453 CPU); that's why `cpu` is the default. On native Linux measure both.

**Encoding test videos with PyAV fails (no x264).** The LGPL PyAV wheels decode H.264 but only *encode* mpeg4; build fresh frames from ndarrays (re-encoding decoded frames hits EINVAL).

**A retained batch of NVDEC frames is all the same frame.** Surface-pool recycling — the loader already copies device-to-device immediately; if you use `PyNvVideoCodec` directly, do the same.

## Platform

- **Windows**: WSL2. **Intel macOS**: sdist build, no Metal. **Jetson**: build with `TURBOLOADER_CUDA_INCLUDE/LIB`, nvJPEG off.
- **macOS `mutex lock failed` at exit with TensorFlow** — a known TF shutdown interaction; exit via `sys.exit(0)` after `gc.collect()`.

## FAQ

**Which capabilities are compiled in?** `turboloader.features()` — the source of truth (JPEG only; `png_decode`/`webp_decode` False; GPU flags reflect the build; `metal_gpu_transforms` True on Apple Silicon).

**Do I need PyTorch?** No for numpy/TF outputs; yes for pinned rings, `device='cuda'` tokens, `CudaPrefetcher`, and CUDA loaders.

**TAR or TBL?** TAR for on-the-fly training with full augmentation; RAW `.tbl` for many-epoch training with a fixed resize(+hflip) recipe ([TBL RAW Preprocessed Pipeline](https://github.com/ALJainProjects/TurboLoader/wiki/TBL-RAW-Preprocessed-Pipeline)).

**Can I use it with Lightning / TensorFlow / JAX?** Yes: wrap the loader in an `IterableDataset` for Lightning (`examples/pytorch_lightning_example.py`); `output_format='tensorflow'` gives HWC float32 for `tf.data.Dataset.from_generator`; numpy batches convert to JAX with `jnp.asarray`.

**Where do I report a wrong number?** Open an issue with the script and your hardware; we publish corrections in the docs rather than hiding them.
