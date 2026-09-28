"""Stack całego nagrania (średnia, maksimum) i statystyki jasności per klatka.

Działa na tensorach GPU (torch) albo tablicach numpy — zależnie od backendu dekodowania.
Średnia posłuży do plate solve (M2), max−średnia pokazuje tory, a statystyki per klatka
wykrywają pulsowanie jasności co GOP (artefakt H.264, sekcja 5.3).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .decode import FrameBatch


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if _is_torch(x) else np.asarray(x)


@dataclass
class StackResult:
    mean: np.ndarray          # [H, W] float32
    max: np.ndarray           # [H, W] uint8
    frames: np.ndarray        # [N] indeksy klatek
    frame_means: np.ndarray   # [N] średnia jasność klatki [DN]
    frame_quantiles: np.ndarray  # [N, Q] kwantyle jasności (podpróbkowane)
    quantiles: tuple[float, ...]
    pict_types: list


class StackAccumulator:
    def __init__(self, subsample: int = 8, quantiles: Sequence[float] = (0.5, 0.999)):
        self.k = max(1, int(subsample))
        self.q = tuple(float(q) for q in quantiles)
        self._sum = self._max = None
        self.n = 0
        self._frames: list[np.ndarray] = []
        self._means: list[np.ndarray] = []
        self._qs: list[np.ndarray] = []
        self._picts: list = []

    def add(self, b: FrameBatch) -> None:
        y = b.y
        n, hw = len(b), int(y.shape[1]) * int(y.shape[2])
        if _is_torch(y):
            import torch

            s = y.sum(0, dtype=torch.int32)   # 255 × 8.4 mln klatek mieści się w int32
            m = y.amax(0)
            means = y.sum((1, 2), dtype=torch.int64).double() / hw
            sub = y[:, ::self.k, ::self.k].reshape(n, -1).float()
            qs = torch.quantile(sub, torch.tensor(self.q, device=sub.device), dim=1).T
            if self._sum is None:
                self._sum, self._max = s, m
            else:
                self._sum += s
                self._max = torch.maximum(self._max, m)
        else:
            s = y.sum(0, dtype=np.int64)
            m = y.max(0)
            means = y.sum((1, 2), dtype=np.int64) / hw
            sub = y[:, ::self.k, ::self.k].reshape(n, -1).astype(np.float32)
            qs = np.quantile(sub, self.q, axis=1).T
            if self._sum is None:
                self._sum, self._max = s, m.copy()
            else:
                self._sum += s
                np.maximum(self._max, m, out=self._max)
        self.n += n
        self._frames.append(np.arange(b.start, b.stop))
        self._means.append(_to_numpy(means).astype(np.float64))
        self._qs.append(_to_numpy(qs).astype(np.float64).reshape(n, len(self.q)))
        self._picts.extend(b.pict_types or [None] * n)

    def result(self) -> StackResult:
        if not self.n:
            raise ValueError("brak klatek w stacku")
        return StackResult(
            mean=(_to_numpy(self._sum).astype(np.float64) / self.n).astype(np.float32),
            max=_to_numpy(self._max).astype(np.uint8),
            frames=np.concatenate(self._frames),
            frame_means=np.concatenate(self._means),
            frame_quantiles=np.concatenate(self._qs),
            quantiles=self.q,
            pict_types=self._picts,
        )


def keyframe_pulse(frame_means: Sequence[float], keyframes: Sequence[int]) -> float:
    """Średnia różnica jasności klatek kluczowych względem sąsiednich klatek niekluczowych [DN].
    Wartość istotnie ≠ 0 oznacza pulsowanie co GOP (artefakt kompresji)."""
    means = np.asarray(frame_means, dtype=np.float64)
    kf = {int(k) for k in keyframes}
    diffs = []
    for k in sorted(kf):
        if not 0 <= k < len(means):
            continue
        nb = [means[j] for j in (k - 1, k + 1) if 0 <= j < len(means) and j not in kf]
        if nb:
            diffs.append(means[k] - np.mean(nb))
    return float(np.mean(diffs)) if diffs else float("nan")


def stretch_u8(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.9) -> np.ndarray:
    """Liniowe rozciągnięcie między percentylami do uint8 (podgląd)."""
    img = np.asarray(img, dtype=np.float32)
    sample = img[::4, ::4] if img.size > 1_000_000 else img
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((img - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def write_png(path: Path, img_u8: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(img_u8).save(path)


def write_frame_stats(path: Path, res: StackResult, fps: float, keyframes: Sequence[int]) -> None:
    kf = {int(k) for k in keyframes}
    qnames = [f"p{q * 100:g}" for q in res.quantiles]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "is_key", "pict_type", "mean_dn", *qnames])
        for i, f in enumerate(res.frames):
            w.writerow([int(f), round(f / fps, 4), int(f in kf), res.pict_types[i] or "",
                        round(float(res.frame_means[i]), 4),
                        *(round(float(v), 3) for v in res.frame_quantiles[i])])
