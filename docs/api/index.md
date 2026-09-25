# API reference

- **`DataLoader`** (images: TAR fast path, `.tbl`, dict path; routing to tokens/arrays/map) — [pipeline.md](pipeline.md)
- **Transforms** (24, with signatures) — [transforms.md](transforms.md)
- **TBL-RAW**: `preprocess_to_tbl`, `TblRawImageLoader`, `open_raw_view`, `normalize_u8_batch`, `normalize_u8_gather`, `crop_resize_normalize_u8_gather` — [wiki](https://github.com/ALJainProjects/TurboLoader/wiki/TBL-RAW-Preprocessed-Pipeline)
- **Tokens / arrays / map / WebDataset**: `TokenDataLoader`, `ArrayDataLoader`, `MapDataLoader`, `WebDatasetLoader` — [wiki](https://github.com/ALJainProjects/TurboLoader/wiki/Tokens-and-Arrays)
- **NVIDIA**: `CudaImageLoader`, `CudaResidentLoader` (+ `from_tbl`), `CudaStreamLoader`, `CudaVideoLoader`, `VideoDatasetLoader`, `CudaPrefetcher` — [wiki](https://github.com/ALJainProjects/TurboLoader/wiki/GPU-NVIDIA-CUDA)
- **Apple**: `MetalResidentLoader`, `MetalResidentArrays`, `MetalImageLoader`, `MetalVideoLoader` — [wiki](https://github.com/ALJainProjects/TurboLoader/wiki/GPU-Apple-Metal)
- **Legacy**: `Loader()`, `create_loader()`, `FastDataLoader`, `PyTorchCompatibleLoader` + label extractors — see [Which loader do I use?](https://github.com/ALJainProjects/TurboLoader/wiki/Which-Loader-Do-I-Use)

Every public name is listed in `turboloader.__all__`; `turboloader.features()` reports what the installed build can do.
