"""Dekodowanie wideo do kanału Y (luminancja) w pełnej rozdzielczości.

Backendy (kolejność prób ustala ``decode.backends`` w configu):

- ``nvcodec``    PyNVVideoCodec (NVDEC); klatki NV12 od razu w pamięci GPU, bierzemy płaszczyznę Y.
- ``torchcodec`` torchcodec ``VideoDecoder(device="cuda")``; zwraca RGB, Y odtwarzamy
                 (niedokładnie, ±1 DN), więc domyślnie tylko do benchmarku.
- ``torchaudio`` ``torchaudio.io.StreamReader`` z dekoderem ``*_cuvid``; zwraca YUV444, kanał 0 = Y.
- ``pyav``       PyAV na CPU w wątku producenta; transfer na GPU przez pinned memory.
- ``ffmpeg``     ffmpeg w podprocesie (``extractplanes=y``); fallback bez zależności Pythona.

Każdy backend zwraca ``FrameBatch`` z tensorem (GPU) albo tablicą numpy ``[N, H, W] uint8``.
Konstruktor sprawdza tylko zależności; plik otwiera dopiero ``batches()``.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .metadata import VideoMeta

_CUVID = {"avc1": "h264_cuvid", "avc3": "h264_cuvid", "hvc1": "hevc_cuvid", "hev1": "hevc_cuvid",
          "av01": "av1_cuvid", "vp09": "vp9_cuvid", "mp4v": "mpeg4_cuvid"}
# Formaty PyAV, w których płaszczyzna 0 to 8-bitowe Y.
_Y8_FORMATS = {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "nv21", "gray"}


@dataclass
class FrameBatch:
    """Klatki ``[start, start + N)``; ``y``: ``torch.Tensor`` lub ``np.ndarray`` [N, H, W] uint8."""
    start: int
    y: Any
    pict_types: list | None = None   # 'I'/'P'/'B' per klatka, jeśli backend je zna

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def stop(self) -> int:
        return self.start + len(self)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _require_cuda() -> None:
    if not cuda_available():
        raise RuntimeError("backend wymaga CUDA (torch.cuda.is_available() == False)")


class Decoder:
    name = "base"
    gpu_native = False   # dekodowanie sprzętowe na GPU
    exact_luma = True    # False: Y odtwarzane z RGB, a nie wprost z bitstreamu

    def __init__(self, path: str | Path, meta: VideoMeta, *, device: str = "cuda", gpu_id: int = 0,
                 queue_depth: int = 3, full_range: bool = False):
        self.path = Path(path)
        self.meta = meta
        self.gpu_id = int(gpu_id)
        self.queue_depth = max(1, int(queue_depth))
        self.full_range = bool(full_range)
        self.to_gpu = str(device).startswith("cuda") and cuda_available()
        self.device = f"cuda:{self.gpu_id}" if (self.gpu_native or self.to_gpu) else "cpu"
        self.skipped: dict[str, str] = {}   # backendy odrzucone przez open_decoder przed tym
        self._check()

    def _check(self) -> None:
        """Sprawdza zależności (bez otwierania pliku); rzuca wyjątek, jeśli backend niedostępny."""

    def _batches(self, batch_size: int, limit: int | None) -> Iterator[tuple[Any, list | None]]:
        raise NotImplementedError

    def batches(self, batch_size: int, start: int = 0, stop: int | None = None) -> Iterator[FrameBatch]:
        """Klatki z zakresu [start, stop) w paczkach po ``batch_size`` (skrajne mogą być krótsze).
        Dekodowanie jest sekwencyjne od początku pliku; klatki przed ``start`` są pomijane."""
        idx = 0
        gen = self._batches(int(batch_size), stop)
        try:
            for y, picts in gen:
                n = int(y.shape[0])
                lo = max(start - idx, 0)
                hi = n if stop is None else min(stop - idx, n)
                if hi > lo:
                    yield FrameBatch(idx + lo, y[lo:hi], picts[lo:hi] if picts else None)
                idx += n
                if stop is not None and idx >= stop:
                    break
        finally:
            gen.close()

    def smoke_test(self) -> None:
        for _ in self.batches(1, stop=1):
            return
        raise RuntimeError("backend nie zwrócił żadnej klatki")


class NvCodecDecoder(Decoder):
    name = "nvcodec"
    gpu_native = True

    def _check(self) -> None:
        import PyNVVideoCodec  # noqa: F401
        _require_cuda()

    def _batches(self, batch_size, limit):
        import PyNVVideoCodec as nvc
        import torch

        H, W = self.meta.height, self.meta.width
        demux = nvc.CreateDemuxer(filename=str(self.path))
        dec = nvc.CreateDecoder(gpuid=self.gpu_id, codec=demux.GetNvCodecId(), cudacontext=0,
                                cudastream=0, usedevicememory=True)
        buf = torch.empty((batch_size, H, W), dtype=torch.uint8, device=self.device)
        k = done = 0
        for packet in demux:
            for frame in dec.Decode(packet):
                t = torch.from_dlpack(frame)   # NV12: (H*3/2, W), Y w pierwszych H wierszach
                if t.dim() == 3:
                    t = t.squeeze()
                buf[k].copy_(t[:H, :W])        # dekoder używa bufora ponownie → kopia
                k += 1
                done += 1
                if k == batch_size:
                    yield buf, None
                    buf, k = torch.empty_like(buf), 0
                if limit is not None and done >= limit:
                    if k:
                        yield buf[:k], None
                    return
        if k:
            yield buf[:k], None


class TorchCodecDecoder(Decoder):
    name = "torchcodec"
    gpu_native = True
    exact_luma = False

    def _check(self) -> None:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        _require_cuda()

    def _batches(self, batch_size, limit):
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(str(self.path), device=self.device, dimension_order="NCHW")
        n = len(dec) if limit is None else min(len(dec), limit)
        for s in range(0, n, batch_size):
            fb = dec.get_frames_in_range(start=s, stop=min(s + batch_size, n))
            yield rgb_to_luma(fb.data, self.full_range), None


class TorchAudioDecoder(Decoder):
    name = "torchaudio"
    gpu_native = True

    def _check(self) -> None:
        from torchaudio.io import StreamReader  # noqa: F401
        _require_cuda()

    def _batches(self, batch_size, limit):
        from torchaudio.io import StreamReader

        reader = StreamReader(str(self.path))
        reader.add_video_stream(batch_size, decoder=_CUVID.get(self.meta.codec, "h264_cuvid"),
                                hw_accel=self.device)
        for (chunk,) in reader.stream():   # [N, 3, H, W] YUV444 na GPU
            yield chunk[:, 0].contiguous(), None


class PyAVDecoder(Decoder):
    name = "pyav"

    def _check(self) -> None:
        import av  # noqa: F401

    def _frames(self) -> Iterator[tuple[np.ndarray, str | None]]:
        import av

        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                yield luma_plane(frame), pict_type(frame)

    def _batches(self, batch_size, limit):
        return threaded_batches(self, self._frames, batch_size, limit)


class FFmpegPipeDecoder(Decoder):
    name = "ffmpeg"

    def _check(self) -> None:
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("brak ffmpeg w PATH")

    def _frames(self) -> Iterator[tuple[np.ndarray, None]]:
        H, W = self.meta.height, self.meta.width
        cmd = [self.ffmpeg, "-v", "error", "-nostdin", "-i", str(self.path), "-map", "0:v:0",
               "-vf", "extractplanes=y", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
            n = 0
            try:
                while True:
                    raw = proc.stdout.read(H * W)
                    if len(raw) < H * W:
                        break
                    n += 1
                    yield np.frombuffer(raw, np.uint8).reshape(H, W), None
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.stdout.close()
                proc.wait()
            if n == 0 and proc.returncode != 0:
                err.seek(0)
                msg = err.read().decode(errors="replace")[-500:]
                raise RuntimeError(f"ffmpeg zakończył się kodem {proc.returncode}: {msg}")

    def _batches(self, batch_size, limit):
        return threaded_batches(self, self._frames, batch_size, limit)


def luma_plane(frame) -> np.ndarray:
    """Płaszczyzna Y klatki PyAV bez konwersji (widok z uwzględnieniem ``line_size``)."""
    if frame.format.name in _Y8_FORMATS:
        plane = frame.planes[0]
        buf = np.frombuffer(plane, np.uint8)[: plane.height * plane.line_size]
        return buf.reshape(plane.height, plane.line_size)[:, : plane.width]
    return frame.to_ndarray(format="gray")


def pict_type(frame) -> str | None:
    pt = getattr(frame, "pict_type", None)
    if pt is None:
        return None
    name = getattr(pt, "name", None) or str(pt)   # PyAV >= 12: enum; starsze: str
    return name.rsplit(".", 1)[-1]


def rgb_to_luma(rgb, full_range: bool = False):
    """RGB [N,3,H,W] uint8 → Y' BT.709 [N,H,W] uint8 (odwrotność konwersji w dekoderze)."""
    import torch

    y = rgb[:, 0].to(torch.float32).mul_(0.2126)
    y.add_(rgb[:, 1].to(torch.float32), alpha=0.7152)
    y.add_(rgb[:, 2].to(torch.float32), alpha=0.0722)
    if not full_range:
        y.mul_(219.0 / 255.0).add_(16.0)
    return y.round_().clamp_(0, 255).to(torch.uint8)


def threaded_batches(dec: Decoder, frames_fn: Callable[[], Iterator], batch_size: int,
                     limit: int | None) -> Iterator[tuple[Any, list]]:
    """Producent (wątek) dekoduje na CPU i pakuje klatki w paczki; konsument przenosi je
    na GPU. Przy ``dec.to_gpu`` paczki są w pinned memory (szybki, asynchroniczny transfer)."""
    H, W = dec.meta.height, dec.meta.width
    pool: queue.Queue | None = None
    if dec.to_gpu:
        import torch

        pool = queue.Queue()
        for _ in range(dec.queue_depth + 1):
            pool.put(torch.empty((batch_size, H, W), dtype=torch.uint8).pin_memory())
    out: queue.Queue = queue.Queue(maxsize=dec.queue_depth)
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                out.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def new_buffer():
        if pool is None:
            return np.empty((batch_size, H, W), np.uint8)
        while not stop.is_set():
            try:
                return pool.get(timeout=0.1)
            except queue.Empty:
                pass
        return None

    def producer() -> None:
        frames = frames_fn()
        buf = view = None
        k = done = 0
        picts: list = []
        try:
            for arr, pt in frames:
                if stop.is_set():
                    return
                if buf is None:
                    buf = new_buffer()
                    if buf is None:
                        return
                    view = buf.numpy() if pool is not None else buf
                view[k] = arr[:H, :W]
                picts.append(pt)
                k += 1
                done += 1
                if k == batch_size:
                    if not put((buf, k, picts)):
                        return
                    buf, k, picts = None, 0, []
                if limit is not None and done >= limit:
                    break
            if k:
                put((buf, k, picts))
        except BaseException as e:  # przekazujemy błąd do konsumenta
            put(e)
        finally:
            frames.close()
            put(None)

    th = threading.Thread(target=producer, name=f"{dec.name}-producer", daemon=True)
    th.start()
    try:
        while True:
            item = out.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            buf, k, picts = item
            if pool is not None:
                import torch

                y = buf[:k].to(dec.device, non_blocking=True)
                torch.cuda.current_stream(y.device).synchronize()   # bufor wraca do puli dopiero po kopii
                pool.put(buf)
            else:
                y = buf[:k]
            yield y, picts
    finally:
        stop.set()
        th.join(timeout=10)


BACKENDS: dict[str, type[Decoder]] = {
    c.name: c for c in (NvCodecDecoder, TorchCodecDecoder, TorchAudioDecoder, PyAVDecoder, FFmpegPipeDecoder)
}


def make_decoder(name: str, path: str | Path, meta: VideoMeta, dcfg: dict) -> Decoder:
    if name not in BACKENDS:
        raise ValueError(f"nieznany backend {name!r}; dostępne: {list(BACKENDS)}")
    return BACKENDS[name](path, meta, device=dcfg.get("device", "cuda"), gpu_id=dcfg.get("gpu_id", 0),
                          queue_depth=dcfg.get("queue_depth", 3),
                          full_range=dcfg.get("torchcodec_full_range", False))


def open_decoder(path: str | Path, meta: VideoMeta, dcfg: dict,
                 require_exact: bool | None = None) -> Decoder:
    """Pierwszy działający backend z ``dcfg["backends"]`` (sprawdzony dekodowaniem 1 klatki)."""
    if require_exact is None:
        require_exact = bool(dcfg.get("exact_luma_required", True))
    errors: dict[str, str] = {}
    for name in dcfg["backends"]:
        cls = BACKENDS.get(name)
        if cls is None:
            errors[name] = "nieznany backend"
            continue
        if require_exact and not cls.exact_luma:
            errors[name] = "pominięty (decode.exact_luma_required)"
            continue
        try:
            dec = make_decoder(name, path, meta, dcfg)
            dec.smoke_test()
        except Exception as e:  # noqa: BLE001 — próbujemy kolejnego backendu
            errors[name] = f"{type(e).__name__}: {e}"
            continue
        dec.skipped = errors
        return dec
    raise RuntimeError("żaden backend dekodowania nie zadziałał: " + json.dumps(errors, ensure_ascii=False))
