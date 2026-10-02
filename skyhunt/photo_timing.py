"""Czas zdjęć w sesji z bracketingiem AE.

EXIF X-E3 ma rozdzielczość 1 s (bez SubSec), więc czas jednego zdjęcia znamy z dokładnością
do sekundy. Serie idą jednak z interwałometru w stałym rytmie: start serii k to
T_k = T0 + P·k. Każda seria daje warunek floor(T_k) = t_k (pełna sekunda z EXIF pierwszego
zdjęcia), a wszystkie razem wyznaczają T0 i P dużo dokładniej niż 1 s — jak noniusz, gdy P
nie jest całkowitą liczbą sekund. Czego EXIF nie rozstrzygnie (wspólna faza przy P = 2,000 s),
to i tak wchłonie poprawka zegara Δ z satelitów (F2).

Zdjęcia w serii idą jedno po drugim: otwarcie j-tego = T_k + Σ_{i<j}(T_exp,i + g), gdzie g to
przerwa na odczyt matrycy. g zmierzymy w F2 z kresek satelitów (odstęp w px między końcem
kreski w zdjęciu i a początkiem w i+1, podzielony przez prędkość); do tego czasu wartość
z configu.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Cadence:
    t0: float                 # start serii 0 [s, skala EXIF]
    period: float             # P [s]
    t0_halfwidth: float       # połowa szerokości przedziału dopuszczalnych T0 [s]
    period_halfwidth: float
    regular: bool             # czy jeden rytm wyjaśnia wszystkie serie
    n_sets: int
    n_violations: int = 0     # serie niezgodne z najlepszym rytmem (np. zapchany bufor)

    def to_dict(self) -> dict:
        return asdict(self)


def group_sets(rows: list[dict], frames_per_set: int = 3) -> list[int]:
    """Numer serii dla każdego zdjęcia (wiersze w kolejności zapisu). Z MakerNote: nowa seria,
    gdy numer w serii spada (1-2-3 | 1-2-3); bez niego co ``frames_per_set`` zdjęć."""
    out, k, prev = [], -1, None
    have_seq = all(r.get("sequence_number") for r in rows)
    for i, r in enumerate(rows):
        if have_seq:
            s = int(r["sequence_number"])
            if prev is None or s <= prev:
                k += 1
            prev = s
        else:
            k = i // frames_per_set
        out.append(k)
    return out


def ev_class(exposures: list[float]) -> list[str]:
    """Klasa jasności w serii względem pierwszego zdjęcia: ``ev0``, ``ev+1``, ``ev-1``…"""
    base = exposures[0]
    out = []
    for e in exposures:
        d = int(round(math.log2(e / base))) if base and e else 0
        out.append("ev0" if d == 0 else f"ev{d:+d}")
    return out


def fit_cadence(t_floor: np.ndarray, *, search_s: float = 0.25, step_s: float = 1e-4) -> Cadence:
    """T0, P z warunków t_k ≤ T0 + P·k < t_k + 1 (t_k = pełne sekundy EXIF pierwszych zdjęć serii).
    Dla każdego P z siatki wokół nachylenia prostej przedział dopuszczalnych T0 to
    [max(t_k − P·k), min(t_k + 1 − P·k)); wybieramy środek obszaru dopuszczalnego.
    Gdy żaden P nie pasuje do wszystkich serii (nieregularny rytm), bierzemy P z prostej
    i T0 tak, by łamać jak najmniej warunków."""
    t = np.asarray(t_floor, float)
    n = len(t)
    if n == 0:
        raise ValueError("brak serii")
    if n == 1:
        return Cadence(t[0] + 0.5, float("nan"), 0.5, float("nan"), False, 1)
    k = np.arange(n, dtype=float)
    p_hat = float(np.polyfit(k, t + 0.5, 1)[0])

    def widths(grid):
        lo = (t[None, :] - grid[:, None] * k[None, :]).max(axis=1)
        hi = (t[None, :] + 1 - grid[:, None] * k[None, :]).min(axis=1)
        return lo, hi

    # obszar dopuszczalnych P ma szerokość ~1/n² s: zawężamy siatkę w kilku poziomach
    # (krok 1e-4 → 1e-6 → 1e-8), za każdym razem wokół najmniej łamanego P
    grid = p_hat + np.arange(-search_s, search_s + step_s / 2, step_s)
    step = step_s
    for _ in range(3):
        lo, hi = widths(grid)
        best = float(grid[int(np.argmax(hi - lo))])
        if (hi > lo).sum() >= 50:
            break
        step /= 100
        grid = best + np.arange(-200, 201) * step
    lo, hi = widths(grid)
    ok = hi > lo
    if ok.any():
        ps = grid[ok]
        i = int(np.flatnonzero(ok)[np.argmin(np.abs(ps - np.median(ps)))])
        return Cadence(float((lo[i] + hi[i]) / 2), float(grid[i]), float((hi[i] - lo[i]) / 2),
                       float((ps.max() - ps.min()) / 2 + step / 2), True, n)
    # nieregularnie: P z prostej, T0 = mediana dopuszczalnych przesunięć, liczymy łamane serie
    off = t - p_hat * k
    t0 = float(np.median(off) + 0.5)
    viol = int(np.sum((t0 + p_hat * k < t) | (t0 + p_hat * k >= t + 1)))
    return Cadence(t0, p_hat, 0.5, float("nan"), False, n, viol)


def open_times(set_idx: list[int], exposures: list[float], t_floor: np.ndarray, cad: Cadence,
               gap_s: float) -> np.ndarray:
    """Chwile otwarcia migawki [s, skala EXIF] dla każdego zdjęcia. Start serii z rytmu (gdy
    regularny) albo z EXIF + 0,5 s (środek sekundy); w serii kolejno T_exp + g."""
    set_idx = np.asarray(set_idx, int)
    out = np.empty(len(set_idx))
    regular = cad.regular and math.isfinite(cad.period)
    t_open, prev_k = 0.0, None
    for i, k in enumerate(set_idx):
        if k != prev_k:
            t_open = cad.t0 + cad.period * k if regular else float(t_floor[i]) + 0.5
        else:
            t_open = t_open + float(exposures[i - 1]) + gap_s
        out[i] = t_open
        prev_k = k
    return out
