"""VideoDatasetLoader — labeled clip batches from a DIRECTORY of videos (CUDA).

The single-file loaders (CudaVideoLoader / MetalVideoLoader) stream one video;
training needs random clips from many files with labels. This loader provides
that: ImageFolder-style discovery (``root/class_x/*.mp4``), N worker threads
each decoding clips with PyAV (CPU video decode scales across files — measured
in benchmarks/VIDEO_RESULTS.md), and per-clip GPU assembly through the fused
``cuda_video_clip_yuv420`` kernel: YUV->RGB + ONE RandomResizedCrop window and
horizontal flip applied consistently across every frame of the clip (the
standard video-aug contract) + resize + normalize, one kernel launch per clip.

Yields ``(clips, labels, meta)``:
    clips  -- torch.cuda FloatTensor (batch, clip_len, 3, H, W), freshly
              allocated per batch (no reuse contract for the caller)
    labels -- torch.cuda LongTensor (batch,)
    meta   -- dict with 'paths', 'starts', 'crops', 'flips'

Sampling is uniform over every valid (video, start) window, reproducible per
(seed, epoch) via ``set_epoch`` — the same determinism contract as the other
TurboLoader loaders.
"""

import os
import queue
import threading

import numpy as np

from turboloader.cuda_video import CudaVideoLoader

__all__ = ["VideoDatasetLoader"]

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def _discover(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = []
    for ci, c in enumerate(classes):
        cdir = os.path.join(root, c)
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith(VIDEO_EXTS):
                items.append((os.path.join(cdir, f), ci))
    if not items:
        raise ValueError(f"no videos found under {root} (looked for {VIDEO_EXTS})")
    return items, classes


def _probe(path):
    """(n_frames, W, H, bt709) for one video. Uses container metadata; falls
    back to a counting decode for containers that do not carry a frame count."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        W, H = s.codec_context.width, s.codec_context.height
        n = s.frames or 0
        if not n and s.duration is not None and s.average_rate:
            n = int(s.duration * s.time_base * s.average_rate)
        if not n:
            n = sum(1 for _ in c.decode(s))
        return int(n), int(W), int(H), CudaVideoLoader._bt709(s, W, H)


class VideoDatasetLoader:
    """Random labeled training clips from ``root/<class>/<video>`` on CUDA.

    Args:
        root: dataset directory (one subdirectory per class), or a list of
            ``(path, label)`` pairs.
        clip_len: frames per clip.
        batch_size: clips per batch.
        image_size: output H (or ``(H, W)``).
        frame_step: temporal stride within a clip.
        workers: decoder threads (each holds its own PyAV containers).
        train_aug: RandomResizedCrop + hflip, one window per clip (else full
            frame, no flip — deterministic eval clips).
        steps_per_epoch: batches per epoch (default: enough to cover every
            frame once on average).
        shuffle/seed/set_epoch: reproducible sampling, matching the other
            TurboLoader loaders.
    """

    def __init__(
        self,
        root,
        clip_len=8,
        batch_size=8,
        image_size=112,
        *,
        frame_step=1,
        workers=4,
        train_aug=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        shuffle=True,
        seed=42,
        steps_per_epoch=None,
    ):
        import turboloader as t

        if not getattr(t, "cuda_available", lambda: False)() or not hasattr(
            t, "cuda_video_clip_yuv420"
        ):
            raise RuntimeError(
                "VideoDatasetLoader needs a CUDA build with cuda_video_clip_yuv420 "
                "(single-file alternatives: CudaVideoLoader / MetalVideoLoader)"
            )
        self._t = t
        if isinstance(root, (str, bytes)) or hasattr(root, "__fspath__"):
            self.items, self.classes = _discover(os.fspath(root))
        else:
            self.items = [(str(p), int(y)) for p, y in root]
            self.classes = sorted({y for _, y in self.items})
        h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
        self._h, self._w = int(h), int(w)
        self.clip_len = int(clip_len)
        self.batch_size = int(batch_size)
        self.frame_step = int(frame_step)
        self.workers = int(workers)
        self.train_aug = bool(train_aug)
        self.scale, self.ratio = tuple(scale), tuple(ratio)
        self.mean, self.std = list(mean), list(std)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

        span = self.clip_len * self.frame_step
        self._meta = [_probe(p) for p, _ in self.items]
        usable = np.array([max(0, n - span + 1) for n, _, _, _ in self._meta], dtype=np.int64)
        if not usable.any():
            raise ValueError(f"no video has the {span} frames a clip spans")
        self._usable = usable
        self._cum = np.concatenate([[0], np.cumsum(usable)])
        total_windows = int(self._cum[-1])
        if steps_per_epoch is None:
            total_frames = sum(n for n, _, _, _ in self._meta)
            steps_per_epoch = max(1, total_frames // span // self.batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self._total_windows = total_windows

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    # ------------------------------------------------------------------ plan
    def _plan(self):
        """One epoch of (item_idx, start) samples, uniform over every valid
        window in the dataset (frame-weighted, so long videos are not
        under-sampled the way per-video sampling would)."""
        rng = np.random.default_rng((self.seed, self._epoch))
        n = self.steps_per_epoch * self.batch_size
        if self.shuffle:
            g = rng.integers(0, self._total_windows, size=n)
        else:
            g = (np.arange(n, dtype=np.int64) * (self.clip_len * self.frame_step)) % (
                self._total_windows
            )
        vid = np.searchsorted(self._cum, g, side="right") - 1
        start = g - self._cum[vid]
        return vid.astype(np.int64), start.astype(np.int64)

    # ---------------------------------------------------------------- worker
    def _read_clip(self, cache, item_idx, start, out, no_seek=False):
        """Decode clip_len frames (stride frame_step) from `start` into `out`
        (clip_len, fbytes) uint8 I420 rows. Seeks to the keyframe at/before
        start and walks forward by pts-derived frame index; streams without
        timestamps are re-decoded from frame 0, counting (slow but correct)."""
        import av

        path = self.items[item_idx][0]
        _, W, H, _ = self._meta[item_idx]
        if path not in cache:
            c = av.open(path)
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            cache[path] = (c, s)
        c, s = cache[path]
        tb, rate = s.time_base, s.average_rate
        seeked = not no_seek and start > 0 and tb is not None and rate is not None
        if seeked:
            c.seek(int(start / float(rate) / float(tb)), stream=s, backward=True)
        else:
            c.seek(0)
        span_end = start + self.clip_len * self.frame_step
        rows = 0
        count_idx = 0
        for frame in c.decode(s):
            if frame.pts is not None and tb is not None and rate is not None:
                i = int(round(float(frame.pts * tb) * float(rate)))
            elif seeked:
                # frame index unknowable after a seek -> restart, counting from 0
                return self._read_clip(cache, item_idx, start, out, no_seek=True)
            else:
                i = count_idx
            count_idx += 1
            if i < start or (i - start) % self.frame_step:
                continue
            if i >= span_end:
                break
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")
            CudaVideoLoader._copy_frame_i420(frame, out[rows], W, H)
            rows += 1
            if rows == self.clip_len:
                break
        while 0 < rows < self.clip_len:  # short tail (probe overestimate): repeat last
            out[rows] = out[rows - 1]
            rows += 1
        if rows == 0:
            raise RuntimeError(f"decoded no frames for {path} @ {start}")

    @staticmethod
    def _put_stoppable(q, item, stop):
        """Bounded put that gives up when the consumer is gone (same stop-aware
        pattern as the image path's _prefetched producers — a plain blocking put
        would strand the thread forever on early iterator exit)."""
        while not stop.is_set():
            try:
                q.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self, tasks, results, stop):
        cache = {}
        try:
            while not stop.is_set():
                try:
                    job = tasks.get(timeout=0.25)
                except queue.Empty:
                    continue
                if job is None:
                    break
                task_idx, item_idx, start, crop, flip = job
                _, W, H, _ = self._meta[item_idx]
                fbytes = W * H * 3 // 2
                buf = np.empty((self.clip_len, fbytes), dtype=np.uint8)
                try:
                    self._read_clip(cache, item_idx, int(start), buf)
                    out = ("ok", task_idx, buf, item_idx, int(start), crop, flip)
                except Exception as e:  # surfaced on the consumer thread
                    out = ("err", task_idx, repr(e), None, None, None, None)
                if not self._put_stoppable(results, out, stop):
                    break
        finally:
            for c, _ in cache.values():
                c.close()

    # ------------------------------------------------------------------ iter
    def __iter__(self):
        import torch

        from turboloader._augment import pick_crop

        vid, start = self._plan()
        n_tasks = len(vid)
        rng = np.random.default_rng((self.seed, self._epoch, 1))
        stop = threading.Event()
        tasks = queue.Queue(maxsize=self.workers * 2)
        results = queue.Queue(maxsize=self.batch_size + self.workers)
        threads = [
            threading.Thread(target=self._worker, args=(tasks, results, stop), daemon=True)
            for _ in range(self.workers)
        ]
        for th in threads:
            th.start()

        def feed():
            for ti in range(n_tasks):
                item = int(vid[ti])
                _, W, H, _ = self._meta[item]
                if self.train_aug:
                    crop = pick_crop(W, H, rng, scale=self.scale, ratio=self.ratio)
                    flip = bool(rng.random() < 0.5)
                else:
                    crop, flip = (0.0, 0.0, float(W), float(H)), False
                if not self._put_stoppable(tasks, (ti, item, start[ti], crop, flip), stop):
                    return
            for _ in threads:
                if not self._put_stoppable(tasks, None, stop):
                    return

        # aug decisions are drawn on ONE thread in task order (reproducible);
        # decode completion order is whatever the workers produce — reordered
        # here so batches are deterministic too.
        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()

        # Flat pinned staging: a leading slice of a flat buffer stays contiguous
        # AND pinned for any per-video frame size (a 2-D column slice would need
        # .contiguous(), which silently allocates an UNPINNED copy).
        max_fb = max(W * H * 3 // 2 for _, W, H, _ in self._meta)
        try:
            stage = torch.empty((self.clip_len * max_fb,), dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            stage = torch.empty((self.clip_len * max_fb,), dtype=torch.uint8)
        stage_np = stage.numpy()

        pending = {}
        next_task = 0
        try:
            for _ in range(self.steps_per_epoch):
                clips = torch.empty(
                    (self.batch_size, self.clip_len, 3, self._h, self._w),
                    dtype=torch.float32,
                    device="cuda",
                )
                labels = torch.empty((self.batch_size,), dtype=torch.int64)
                meta = {"paths": [], "starts": [], "crops": [], "flips": []}
                for i in range(self.batch_size):
                    while next_task not in pending:
                        kind, ti, payload, item, st, crop, flip = results.get()
                        if kind == "err":
                            raise RuntimeError(f"video decode failed: {payload}")
                        pending[ti] = (payload, item, st, crop, flip)
                    buf, item, st, crop, flip = pending.pop(next_task)
                    next_task += 1
                    self._assemble(torch, stage, stage_np, buf, item, crop, flip, clips[i])
                    labels[i] = self.items[item][1]
                    meta["paths"].append(self.items[item][0])
                    meta["starts"].append(st)
                    meta["crops"].append(crop)
                    meta["flips"].append(flip)
                yield clips, labels.cuda(), meta
        finally:
            stop.set()  # feeder and workers all use stop-aware puts/gets
            feeder.join(timeout=5)
            for th in threads:
                th.join(timeout=5)

    def _assemble(self, torch, stage, stage_np, buf, item, crop, flip, out):
        """One fused kernel launch: I420 clip -> (T,3,H,W) crop/flip/normalized
        RGB, then copy out of the kernel's double-buffered pool into the batch."""
        _, W, H, bt709 = self._meta[item]
        fbytes = W * H * 3 // 2
        t_frames = self.clip_len
        nbytes = t_frames * fbytes
        stage_np[:nbytes].reshape(t_frames, fbytes)[:] = buf
        dev = stage[:nbytes].cuda().view(t_frames, fbytes)
        base = int(dev.data_ptr())
        ysz, csz = W * H, (W // 2) * (H // 2)
        y = [base + i * fbytes for i in range(t_frames)]
        cb = [p + ysz for p in y]
        cr = [p + ysz + csz for p in y]
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
            flip=bool(flip),
            bt709=bt709,
            mean=self.mean,
            std=self.std,
        )
        from turboloader.cuda_loader import _CudaArray

        view = torch.as_tensor(_CudaArray(ptr, (t_frames, 3, self._h, self._w)), device="cuda")
        out.copy_(view)  # kernel output pool is double-buffered; batch owns a copy
