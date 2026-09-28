"""Wycinki wideo wokół toru: klatki (ffmpeg z dokładnym seekiem albo PyAV), klip MP4
z zaznaczonym obiektem i pasek klatek do PDF (≥ 1 s nagrania).

Pamięć: klip ma limit długości (``clip_max_s``) i boku (``clip_max_side_px``; ffmpeg skaluje
kadr), rozciąganie jasności idzie przez tablicę LUT na uint8, a klatki RGB trafiają do
enkodera strumieniowo — tor satelity przez cały kadr 4K nie zapycha RAM-u Colaba.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metadata import VideoMeta


@dataclass
class ClipPlan:
    start: int     # pierwsza klatka
    n: int         # liczba klatek
    x0: int        # kadr w pikselach nagrania
    y0: int
    w: int
    h: int
    out_w: int     # rozmiar po skalowaniu (≤ clip_max_side_px)
    out_h: int

    @property
    def scale(self) -> float:
        """Pikseli nagrania na piksel klipu."""
        return self.w / self.out_w

    def to_clip(self, x, y):
        return (np.asarray(x, float) - self.x0) / self.scale, (np.asarray(y, float) - self.y0) / self.scale


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def plan_clip(frames: np.ndarray, x: np.ndarray, y: np.ndarray, meta: VideoMeta, rcfg: dict) -> ClipPlan:
    """Czas: tor ± ``clip_pad_s``, co najmniej ``clip_min_s``, najwyżej ``clip_max_s`` (wokół
    środka toru). Kadr: bbox toru w tym czasie + ``crop_margin_px``, co najmniej
    ``clip_min_size_px``; wyjście skalowane do boku ≤ ``clip_max_side_px``."""
    fps = meta.fps
    pad = int(round(float(rcfg["clip_pad_s"]) * fps))
    f0, f1 = int(frames.min()) - pad, int(frames.max()) + pad
    need = int(math.ceil(float(rcfg["clip_min_s"]) * fps))
    cap = max(need, int(round(float(rcfg.get("clip_max_s", 10.0)) * fps)))
    if f1 - f0 + 1 < need:
        extra = need - (f1 - f0 + 1)
        f0 -= extra // 2
        f1 += extra - extra // 2
    if f1 - f0 + 1 > cap:
        c = (int(frames.min()) + int(frames.max())) // 2
        f0, f1 = c - cap // 2, c - cap // 2 + cap - 1
    f0, f1 = max(0, f0), min(meta.n_frames - 1, f1)
    sel = (frames >= f0) & (frames <= f1)
    xs, ys = (x[sel], y[sel]) if sel.any() else (x, y)
    m, mins = float(rcfg["crop_margin_px"]), float(rcfg["clip_min_size_px"])

    def span(lo, hi, size):
        if hi - lo < mins:
            c = (lo + hi) / 2
            lo, hi = c - mins / 2, c + mins / 2
        w = min(int(math.ceil(hi - lo)) // 2 * 2, size // 2 * 2)
        lo = int(min(max(0, round(lo)), size - w))
        return lo, w

    x0, w = span(float(xs.min()) - m, float(xs.max()) + m, meta.width)
    y0, h = span(float(ys.min()) - m, float(ys.max()) + m, meta.height)
    s = max(1.0, max(w, h) / float(rcfg.get("clip_max_side_px", 1280)))
    return ClipPlan(f0, f1 - f0 + 1, x0, y0, w, h, _even(w / s), _even(h / s))


def extract_frames(video: Path, meta: VideoMeta, plan: ClipPlan) -> np.ndarray:
    """Klatki Y [n, out_h, out_w] z zakresu planu."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return _extract_ffmpeg(ffmpeg, video, meta, plan)
    return _extract_pyav(video, meta, plan)


def _extract_ffmpeg(ffmpeg: str, video: Path, meta: VideoMeta, p: ClipPlan) -> np.ndarray:
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if p.start > 0:   # dokładny seek: pierwsza klatka o znaczniku ≥ (start − ½)/fps
        cmd += ["-ss", f"{(p.start - 0.5) / meta.fps:.6f}"]
    vf = f"extractplanes=y,crop={p.w}:{p.h}:{p.x0}:{p.y0}"
    if (p.out_w, p.out_h) != (p.w, p.h):
        vf += f",scale={p.out_w}:{p.out_h}:flags=area"
    cmd += ["-i", str(video), "-map", "0:v:0", "-frames:v", str(p.n), "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    size = p.out_w * p.out_h
    k = len(out) // size
    return np.frombuffer(out[:k * size], np.uint8).reshape(k, p.out_h, p.out_w)


def _extract_pyav(video: Path, meta: VideoMeta, p: ClipPlan) -> np.ndarray:
    import av
    from PIL import Image

    from .decode import luma_plane

    out = []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        t0 = float(s.start_time or 0) * tb
        c.seek(int((p.start / meta.fps + t0) / tb), stream=s, backward=True)
        for fr in c.decode(s):
            idx = int(round((float(fr.pts) * tb - t0) * meta.fps))
            if idx < p.start:
                continue
            if idx >= p.start + p.n:
                break
            img = luma_plane(fr)[p.y0:p.y0 + p.h, p.x0:p.x0 + p.w]
            if (p.out_w, p.out_h) != (p.w, p.h):
                img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((p.out_w, p.out_h), Image.BOX))
            out.append(img.copy())
    return np.stack(out) if out else np.empty((0, p.out_h, p.out_w), np.uint8)


def positions_at(frames_needed: np.ndarray, frames: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Pozycja obiektu w dowolnych klatkach: interpolacja w torze, ekstrapolacja liniowa poza nim."""
    fx = np.interp(frames_needed, frames, x)
    fy = np.interp(frames_needed, frames, y)
    if len(frames) >= 2:
        px, py = np.polyfit(frames, x, 1), np.polyfit(frames, y, 1)
        out = (frames_needed < frames.min()) | (frames_needed > frames.max())
        fx = np.where(out, np.polyval(px, frames_needed), fx)
        fy = np.where(out, np.polyval(py, frames_needed), fy)
    return fx, fy


def stretch_stack(frames: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.9) -> np.ndarray:
    """Liniowe rozciągnięcie jasności przez tablicę LUT (bez kopii float całego klipu)."""
    sample = frames[:: max(1, len(frames) // 16), ::2, ::2]
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    hi = max(hi, lo + 1)
    lut = (np.clip((np.arange(256, dtype=np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return lut[frames]


def ring_mask(h: int, w: int, x: float, y: float, radius: float, width: float = 1.0) -> np.ndarray:
    yy, xx = np.ogrid[0:h, 0:w]
    return np.abs(np.hypot(xx - x, yy - y) - radius) < width


def annotate_frame(gray: np.ndarray, x: float, y: float, color, mark: bool, upscale: int = 1,
                   radius: float = 14.0) -> np.ndarray:
    """Klatka szara [h, w] → RGB (powiększona ``upscale``), z okręgiem wokół obiektu."""
    if upscale > 1:
        gray = gray.repeat(upscale, axis=0).repeat(upscale, axis=1)
        x, y, radius = x * upscale + (upscale - 1) / 2, y * upscale + (upscale - 1) / 2, radius * upscale
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if mark:
        rgb[ring_mask(*gray.shape, x, y, radius, max(1.0, upscale * 0.8))] = color
    return rgb


class _Mp4Writer:
    """Strumieniowy zapis MP4: ffmpeg (libx264, a bez niego mpeg4) albo PyAV."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path, self.w, self.h, self.fps = Path(path), w, h, fps
        self.proc = self.container = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            codec = ["-c:v", "libx264", "-crf", "18"] if self._has_x264(ffmpeg) else ["-c:v", "mpeg4", "-q:v", "2"]
            cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.6f}", "-i", "pipe:0", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            import av
            from fractions import Fraction

            self.container = av.open(str(self.path), "w")
            self.stream = self.container.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1001))
            self.stream.width, self.stream.height, self.stream.pix_fmt = w, h, "yuv420p"

    @staticmethod
    def _has_x264(ffmpeg: str) -> bool:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        return "libx264" in out

    def write(self, rgb: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        else:
            import av

            for pkt in self.stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
                self.container.mux(pkt)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="replace")
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg nie zapisał {self.path.name}: {err[-300:]}")
        else:
            for pkt in self.stream.encode():
                self.container.mux(pkt)
            self.container.close()


def write_annotated_mp4(u8: np.ndarray, xs: np.ndarray, ys: np.ndarray, marks: np.ndarray, color, path: Path,
                        fps: float, upscale: int = 1) -> None:
    """Klip MP4 z klatek szarych [n, h, w]; okrąg wokół obiektu w klatkach ``marks``."""
    n, h, w = u8.shape
    H, W = _even(h * upscale), _even(w * upscale)
    writer = _Mp4Writer(path, W, H, fps)
    try:
        for i in range(n):
            rgb = annotate_frame(u8[i], xs[i], ys[i], color, bool(marks[i]), upscale)
            writer.write(np.pad(rgb, ((0, H - rgb.shape[0]), (0, W - rgb.shape[1]), (0, 0)))[:H, :W])
    finally:
        writer.close()


def filmstrip(frames_u8: np.ndarray, plan: ClipPlan, obj_frames: np.ndarray, x: np.ndarray, y: np.ndarray,
              fps: float, rcfg: dict) -> list[dict]:
    """Panele paska klatek: ``filmstrip_panels`` klatek równomiernie w ≥ ``filmstrip_min_s``
    wokół toru, każdy wycięty wokół (inter/ekstrapolowanej) pozycji obiektu."""
    n = len(frames_u8)
    if n == 0:
        return []
    k = int(rcfg["filmstrip_panels"])
    need = int(math.ceil(float(rcfg["filmstrip_min_s"]) * fps))
    lo, hi = int(obj_frames.min()), int(obj_frames.max())
    if hi - lo + 1 < need:
        c = (lo + hi) // 2
        lo, hi = c - need // 2, c - need // 2 + need - 1
    first, last = plan.start, plan.start + n - 1
    if lo < first:        # przesuwamy okno zamiast je obcinać (tor przy początku/końcu nagrania)
        lo, hi = first, hi + (first - lo)
    if hi > last:
        lo, hi = max(first, lo - (hi - last)), last
    sel = np.unique(np.round(np.linspace(lo, hi, k)).astype(int))
    px, py = positions_at(sel.astype(float), obj_frames.astype(float), x, y)
    cx_all, cy_all = plan.to_clip(px, py)
    half = int(rcfg["filmstrip_crop_px"]) // 2
    panels = []
    for f, cx, cy in zip(sel, cx_all, cy_all):
        img = frames_u8[f - plan.start]
        cx = min(max(int(round(cx)), 0), img.shape[1])
        cy = min(max(int(round(cy)), 0), img.shape[0])
        padded = np.pad(img, half)
        panels.append({"frame": int(f), "img": padded[cy:cy + 2 * half, cx:cx + 2 * half],
                       "inside": bool(obj_frames.min() <= f <= obj_frames.max())})
    return panels
