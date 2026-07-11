"""CUDA video loader (NVIDIA) — decoded video to GPU-resident training batches.

Two decode backends feed the same fused CUDA kernel (YUV 4:2:0 -> RGB video-range
BT.601/709 + bilinear resize + normalize, the numpy-validated Metal kernel's math):

- ``decode="cpu"`` (default): threaded FFmpeg decode via PyAV on the host, one
  H2D upload per batch, conversion + resize + normalize on the GPU. Frees the
  CPU of all color/resize work and lands batches on-device where training wants
  them. This is the right default because CPU decode is FASTER than NVDEC on
  some stacks — measured on an RTX 3090 under WSL2: PyAV 1,453 f/s vs NVDEC
  130 f/s (video engine virtualization overhead).

- ``decode="nvdec"``: PyNvVideoCodec hardware decode straight to device NV12,
  ZERO host->device traffic. The right choice on native Linux / datacenter GPUs
  where NVDEC runs at spec; measure both on your machine (the benchmark script
  does) and pick.

Yields GPU-resident ``(B, 3, H, W)`` float32 batches (``__cuda_array_interface__``,
adopt zero-copy with ``torch.as_tensor(x, device="cuda")``). A yielded batch is
valid until the NEXT batch (double-buffered) — copy or consume before advancing.
"""

import numpy as np

__all__ = ["CudaVideoLoader"]


class CudaVideoLoader:
    def __init__(
        self,
        path,
        image_size=224,
        batch_size=32,
        *,
        decode="cpu",
        frame_step=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        return_indices=False,
        gpu_id=0,
    ):
        import turboloader as t

        if not getattr(t, "cuda_available", lambda: False)() or not hasattr(
            t, "cuda_video_yuv420_batch"
        ):
            raise RuntimeError("CudaVideoLoader needs a CUDA build with cuda_video_yuv420_batch.")
        if decode not in ("cpu", "nvdec"):
            raise ValueError("decode must be 'cpu' or 'nvdec'")
        self._t = t
        self.path = str(path)
        h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
        self._h, self._w = int(h), int(w)
        self.batch_size = int(batch_size)
        self.decode = decode
        self.frame_step = int(frame_step)
        self.mean = list(mean)
        self.std = list(std)
        self.return_indices = bool(return_indices)
        self.gpu_id = int(gpu_id)

    def _wrap(self, ptr, n):
        from turboloader.cuda_loader import _CudaArray

        return _CudaArray(ptr, (n, 3, self._h, self._w))

    # ------------------------- CPU (PyAV) backend --------------------------
    @staticmethod
    def _copy_frame_i420(frame, dst_row, W, H):
        """Copy a yuv420p frame's planes into a packed I420 row WITHOUT the
        intermediate to_ndarray() allocation: one stride-aware copy per plane
        straight from PyAV's buffers (on slow CPUs the per-frame Python copies,
        not decode, are the bottleneck — measured 181 -> ~2x with this)."""
        ysz, csz = W * H, (W // 2) * (H // 2)
        layout = ((W, H, 0, ysz), (W // 2, H // 2, ysz, csz), (W // 2, H // 2, ysz + csz, csz))
        for i, (w, h, off, size) in enumerate(layout):
            p = frame.planes[i]
            buf = np.frombuffer(p, dtype=np.uint8)
            ls = p.line_size
            if ls == w:
                dst_row[off : off + size] = buf[:size]
            else:  # padded rows: strided view copy
                dst_row[off : off + size].reshape(h, w)[:] = buf[: ls * h].reshape(h, ls)[:, :w]

    def _iter_cpu(self):
        import av
        import torch

        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            W, H = stream.codec_context.width, stream.codec_context.height
            if W % 2 or H % 2:
                raise ValueError("yuv420p video must have even dimensions")
            bt709 = self._bt709(stream, W, H)
            ysz, csz = W * H, (W // 2) * (H // 2)
            fbytes = ysz + 2 * csz

            # Pinned staging (reused across batches): faster H2D, zero per-batch alloc.
            try:
                host_t = torch.empty((self.batch_size, fbytes), dtype=torch.uint8, pin_memory=True)
            except RuntimeError:
                host_t = torch.empty((self.batch_size, fbytes), dtype=torch.uint8)
            host = host_t.numpy()
            batch_rows = 0
            first = -1
            idx = 0
            for frame in container.decode(stream):
                keep = idx % self.frame_step == 0
                if keep:
                    if frame.format.name != "yuv420p":
                        frame = frame.reformat(format="yuv420p")
                    self._copy_frame_i420(frame, host[batch_rows], W, H)
                    if batch_rows == 0:
                        first = idx
                    batch_rows += 1
                idx += 1
                if batch_rows == self.batch_size:
                    yield self._flush_cpu(host_t, batch_rows, W, H, bt709, first, torch)
                    batch_rows = 0
            if batch_rows:
                yield self._flush_cpu(host_t, batch_rows, W, H, bt709, first, torch)

    def _flush_cpu(self, host, n, W, H, bt709, first, torch):
        dev = host[:n].cuda()  # one (pinned, synchronous) H2D per batch
        base = int(dev.data_ptr())
        ysz, csz = W * H, (W // 2) * (H // 2)
        fbytes = ysz + 2 * csz
        y = [base + i * fbytes for i in range(n)]
        cb = [p + ysz for p in y]
        cr = [p + ysz + csz for p in y]
        ptr = self._t.cuda_video_yuv420_batch(
            y,
            cb,
            cr,
            y_stride=W,
            c_stride=W // 2,
            c_px_stride=1,
            src_w=W,
            src_h=H,
            dst_h=self._h,
            dst_w=self._w,
            bt709=bt709,
            mean=self.mean,
            std=self.std,
        )
        del dev  # kernel is synchronous: upload no longer needed
        batch = self._wrap(ptr, n)
        if self.return_indices:
            return batch, np.arange(first, first + n * self.frame_step, self.frame_step)
        return batch

    @staticmethod
    def _bt709(stream, W, H):
        cs = getattr(getattr(stream, "codec_context", None), "colorspace", None)
        name = str(cs).lower() if cs is not None else ""
        if "709" in name:
            return True
        if "601" in name or "170" in name or "470" in name:
            return False
        return H >= 720 or W >= 1280  # untagged: same heuristic as the Metal path

    # ------------------------- NVDEC backend -------------------------------
    def _iter_nvdec(self):
        import PyNvVideoCodec as nvc
        import torch

        demux = nvc.CreateDemuxer(self.path)
        dec = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=demux.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
        )
        # The decoder RECYCLES its surface pool: pointers taken from a frame are
        # only valid until Decode() is called again, no matter how long the Python
        # frame object lives (observed: retaining a batch of frames yields N copies
        # of the LAST surface). Copy each frame device-to-device into our own
        # staging the moment it is produced — still zero host<->device traffic.
        stage = None  # torch cuda uint8 (batch, H*3//2, W)
        rows, first, idx = 0, -1, 0
        W = H = None
        bt709 = False
        for packet in demux:
            for frame in dec.Decode(packet):
                keep = idx % self.frame_step == 0
                if keep:
                    y_v, c_v = self._nv12_plane_views(frame, torch)
                    if W is None:
                        H, W = int(y_v.shape[0]), int(y_v.shape[1])
                        bt709 = H >= 720 or W >= 1280
                        stage = torch.empty(
                            (self.batch_size, H * 3 // 2, W), dtype=torch.uint8, device="cuda"
                        )
                    stage[rows, :H] = y_v
                    stage[rows, H:] = c_v
                    if rows == 0:
                        first = idx
                    rows += 1
                idx += 1
                if rows == self.batch_size:
                    yield self._flush_nvdec(stage, rows, W, H, bt709, first)
                    rows, first = 0, -1
        if rows:
            yield self._flush_nvdec(stage, rows, W, H, bt709, first)

    @staticmethod
    def _nv12_plane_views(frame, torch):
        """Adopt PyNvVideoCodec's frame planes as torch CUDA views: (H, W) luma and
        the interleaved CbCr plane flattened to (H//2, W). Handles both the
        per-plane list API ([Y (H,W,1), CbCr (H/2,W/2,2)]) and single surfaces."""
        surf = frame.cuda()
        if isinstance(surf, (list, tuple)):
            y = torch.as_tensor(surf[0], device="cuda").squeeze(-1)
            c = torch.as_tensor(surf[1], device="cuda")
            return y, c.reshape(c.shape[0], -1)
        s = torch.as_tensor(surf, device="cuda")  # (H*3/2, W)
        H = s.shape[0] * 2 // 3
        return s[:H], s[H:]

    def _flush_nvdec(self, stage, n, W, H, bt709, first):
        base = int(stage.data_ptr())
        fbytes = (H * 3 // 2) * W
        y = [base + i * fbytes for i in range(n)]
        cb = [p + H * W for p in y]  # interleaved CbCr plane after Y in staging
        cr = [p + 1 for p in cb]
        ptr = self._t.cuda_video_yuv420_batch(
            y,
            cb,
            cr,
            y_stride=W,
            c_stride=W,
            c_px_stride=2,
            src_w=W,
            src_h=H,
            dst_h=self._h,
            dst_w=self._w,
            bt709=bt709,
            mean=self.mean,
            std=self.std,
        )
        batch = self._wrap(ptr, n)
        if self.return_indices:
            return batch, np.arange(first, first + n * self.frame_step, self.frame_step)
        return batch

    def __iter__(self):
        return self._iter_cpu() if self.decode == "cpu" else self._iter_nvdec()

    # ------------------------- fused clip sampling --------------------------
    def iter_clips(
        self,
        clip_len,
        *,
        train_aug=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        seed=42,
    ):
        """Yield GPU ``(clip_len, 3, H, W)`` training clips assembled by ONE fused
        kernel launch each: YUV->RGB + the SAME RandomResizedCrop window and
        horizontal flip across every frame of the clip (the standard video-aug
        contract) + resize + normalize. ``train_aug=False`` uses the full frame
        and no flip (deterministic eval clips). Clips are consecutive
        non-overlapping windows honoring ``frame_step``. CPU decode backend.
        A yielded clip is valid until the next clip (double-buffered)."""
        import av
        import torch

        if self.decode != "cpu":
            raise NotImplementedError("iter_clips currently uses the cpu decode backend")
        rng = np.random.default_rng(seed)
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            W, H = stream.codec_context.width, stream.codec_context.height
            bt709 = self._bt709(stream, W, H)
            ysz, csz = W * H, (W // 2) * (H // 2)
            fbytes = ysz + 2 * csz
            try:
                host_t = torch.empty((clip_len, fbytes), dtype=torch.uint8, pin_memory=True)
            except RuntimeError:
                host_t = torch.empty((clip_len, fbytes), dtype=torch.uint8)
            host = host_t.numpy()
            rows, first, idx = 0, -1, 0
            for frame in container.decode(stream):
                if idx % self.frame_step == 0:
                    if frame.format.name != "yuv420p":
                        frame = frame.reformat(format="yuv420p")
                    self._copy_frame_i420(frame, host[rows], W, H)
                    if rows == 0:
                        first = idx
                    rows += 1
                idx += 1
                if rows == clip_len:
                    yield self._flush_clip(
                        host_t, clip_len, W, H, bt709, first, rng, train_aug, scale, ratio, torch
                    )
                    rows = 0
            # tail shorter than clip_len is dropped (standard for clip sampling)

    def _pick_crop(self, W, H, rng, scale, ratio):
        """torchvision RandomResizedCrop-parity sampling (area x log-ratio, 10
        attempts, center fallback), in source pixels."""
        area = W * H
        for _ in range(10):
            target = area * rng.uniform(*scale)
            log_r = rng.uniform(np.log(ratio[0]), np.log(ratio[1]))
            r = np.exp(log_r)
            cw = int(round(np.sqrt(target * r)))
            ch = int(round(np.sqrt(target / r)))
            if 0 < cw <= W and 0 < ch <= H:
                x = rng.integers(0, W - cw + 1)
                y = rng.integers(0, H - ch + 1)
                return float(x), float(y), float(cw), float(ch)
        side = min(W, H)
        return (W - side) / 2.0, (H - side) / 2.0, float(side), float(side)

    def _flush_clip(self, host, t, W, H, bt709, first, rng, train_aug, scale, ratio, torch):
        dev = host[:t].cuda()  # pinned staging tensor
        base = int(dev.data_ptr())
        ysz, csz = W * H, (W // 2) * (H // 2)
        fbytes = ysz + 2 * csz
        y = [base + i * fbytes for i in range(t)]
        cb = [p + ysz for p in y]
        cr = [p + ysz + csz for p in y]
        if train_aug:
            crop = self._pick_crop(W, H, rng, scale, ratio)
            flip = bool(rng.random() < 0.5)
        else:
            crop, flip = (0.0, 0.0, float(W), float(H)), False
        ptr = self._t.cuda_video_clip_yuv420(
            y,
            cb,
            cr,
            y_stride=W,
            c_stride=W // 2,
            c_px_stride=1,
            src_w=W,
            src_h=H,
            dst_h=self._h,
            dst_w=self._w,
            crop=list(crop),
            flip=flip,
            bt709=bt709,
            mean=self.mean,
            std=self.std,
        )
        del dev
        clip = self._wrap(ptr, t)
        if self.return_indices:
            meta = {
                "first_frame_index": first,
                "crop": crop,
                "flip": flip,
            }
            return clip, meta
        return clip
