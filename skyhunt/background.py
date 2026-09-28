"""Model tła i szumu per piksel: mediana i MAD z okna kroczącego (torch, GPU lub CPU).

Tło liczymy raz na blok ``bg_block_frames`` klatek z okna ±``bg_half_window_frames``
(co ``bg_stride``), wyśrodkowanego na bloku. Kamera stoi, gwiazdy dryfują ~0.43 px/s, więc
okno ±1 s rozmywa je o ≤0.4 px, a obiekty szybsze niż ~0.1 px/klatkę nie trafiają do tła.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Block:
    index: int
    start: int      # pierwsza klatka bloku
    stop: int       # za ostatnią klatką bloku
    center: int
    window: tuple[int, ...]   # klatki użyte do mediany

    @property
    def first_needed(self) -> int:
        return min(self.start, self.window[0])

    @property
    def last_needed(self) -> int:
        return max(self.stop - 1, self.window[-1])


def plan_block(b: int, n_frames: int, block: int, half_window: int, stride: int) -> Block:
    start = b * block
    stop = min(start + block, n_frames)
    center = start + (stop - start) // 2
    k = half_window // stride
    window = tuple(f for f in (center + i * stride for i in range(-k, k + 1)) if 0 <= f < n_frames)
    return Block(b, start, stop, center, window)


def n_blocks(n_frames: int, block: int) -> int:
    return math.ceil(n_frames / block) if n_frames > 0 else 0


def median_mad(stack: torch.Tensor, tile_rows: int = 540) -> tuple[torch.Tensor, torch.Tensor]:
    """``stack`` [K, H, W] → (mediana, MAD) [H, W] float32, liczone w pasach wierszy.
    Dla parzystego K torch zwraca niższą z dwóch środkowych wartości (numpy: średnią)."""
    K, H, W = stack.shape
    med = torch.empty((H, W), dtype=torch.float32, device=stack.device)
    mad = torch.empty_like(med)
    for r0 in range(0, H, max(1, int(tile_rows))):
        t = stack[:, r0:r0 + tile_rows].to(torch.float32)
        m = t.median(dim=0).values
        med[r0:r0 + tile_rows] = m
        mad[r0:r0 + tile_rows] = (t - m).abs_().median(dim=0).values
    return med, mad


class FrameRing:
    """Ostatnie klatki w pamięci urządzenia, adresowane numerem klatki (bufor kołowy)."""

    def __init__(self, capacity: int, height: int, width: int, device: str | torch.device):
        self.cap = int(capacity)
        self.buf = torch.empty((self.cap, height, width), dtype=torch.uint8, device=device)
        self.lo = 0   # najstarsza dostępna klatka
        self.hi = 0   # za najnowszą klatką

    def push(self, start: int, frames: torch.Tensor) -> None:
        if start != self.hi:
            raise ValueError(f"oczekiwano klatki {self.hi}, jest {start}")
        n = int(frames.shape[0])
        if self.hi + n - self.lo > self.cap:
            raise RuntimeError(f"bufor klatek przepełniony ({self.hi + n - self.lo} > {self.cap})")
        slots = torch.arange(start, start + n, device=self.buf.device) % self.cap
        self.buf.index_copy_(0, slots, frames.to(self.buf.device, torch.uint8))
        self.hi += n

    def drop_before(self, frame: int) -> None:
        self.lo = max(self.lo, min(int(frame), self.hi))

    def get(self, frames) -> torch.Tensor:
        frames = list(frames)
        bad = [f for f in frames if not self.lo <= f < self.hi]
        if bad:
            raise IndexError(f"klatki {bad[:5]} poza buforem [{self.lo}, {self.hi})")
        slots = torch.tensor([f % self.cap for f in frames], device=self.buf.device)
        return self.buf.index_select(0, slots)
