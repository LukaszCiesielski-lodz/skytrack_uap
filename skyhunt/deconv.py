"""Wyostrzanie zgodne z fizyką: dekonwolucja Richardsona–Lucy z rozmyciem (PSF) zmierzonym na
gwiazdach tego zdjęcia, osobno w kafelkach kadru (przy f/1.0 gwiazdy w rogach to „przecinki”).
Odwracamy rozmycie, które naprawdę wystąpiło — nie dorysowujemy szczegółów. Na luminancji;
kolor zostaje z obrazu wejściowego (szczegół dodawany jednakowo do R, G, B)."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("skyhunt")


def measure_psfs(L: np.ndarray, smask: np.ndarray, grid: tuple[int, int], size: int = 15,
                 per_tile: int = 120) -> np.ndarray:
    """PSF [gy, gx, size, size] (suma 1): mediana wycinków nieprześwietlonych, średnio jasnych gwiazd,
    wycentrowanych z dokładnością do ułamka piksela. Kafelek bez gwiazd dostaje PSF sąsiadów."""
    from scipy import ndimage as ndi

    H, W = L.shape
    gy, gx = grid
    r = size // 2
    from .astrophoto import star_peaks

    psfs = np.full((gy, gx, size, size), np.nan, np.float32)
    peaks, vals = star_peaks(L, smask)
    if not len(vals):
        return psfs
    hi = np.percentile(L, 99.99)
    ok = (vals < 0.5 * hi) & (vals > np.percentile(vals, 25)) & (peaks[:, 0] > r + 1) & (peaks[:, 1] > r + 1) \
        & (peaks[:, 0] < H - r - 2) & (peaks[:, 1] < W - r - 2)
    peaks, vals = peaks[ok], vals[ok]
    yy, xx = np.mgrid[-r - 1:r + 2, -r - 1:r + 2]
    for ty in range(gy):
        for tx in range(gx):
            sel = ((peaks[:, 0] * gy // H) == ty) & ((peaks[:, 1] * gx // W) == tx)
            cuts = []
            for (y, x) in peaks[sel][np.argsort(-vals[sel])][:per_tile]:
                c = L[y - r - 1:y + r + 2, x - r - 1:x + r + 2].astype(np.float64)
                edge = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
                c = c - np.median(edge)
                outer = c.copy()
                outer[r - 2:r + 5, r - 2:r + 5] = 0           # bez rdzenia gwiazdy
                if outer.max() > 0.15 * c.max():               # jasny sąsiad psułby PSF
                    continue
                w = np.clip(c, 0, None)
                if w.sum() <= 0:
                    continue
                cy, cx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
                c = ndi.shift(c, (-cy, -cx), order=1, mode="nearest")[1:-1, 1:-1]
                s = c.sum()
                if s > 0:
                    cuts.append(c / s)
            if len(cuts) >= 8:
                p = np.clip(np.median(np.array(cuts), axis=0), 0, None)
                psfs[ty, tx] = p / p.sum()
    good = np.isfinite(psfs).all(axis=(2, 3))
    if good.any():
        mean = np.nanmean(psfs[good], axis=0)
        psfs[~good] = mean / mean.sum()
    return psfs


def richardson_lucy(d, psf, iters: int):
    """RL na tensorze [1, 1, h, w] (dane dodatnie) z jądrem [size, size]; brzegi: odbicie."""
    import torch

    F = torch.nn.functional
    k = torch.as_tensor(psf, dtype=torch.float32, device=d.device)[None, None]
    kf = torch.flip(k, (2, 3))
    pad = psf.shape[-1] // 2

    def conv(x, ker):
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), ker)

    u = d.clone()
    for _ in range(iters):
        u = u * conv(d / conv(u, k).clamp(min=1e-6), kf)
    return u


def deconvolve(rgb: np.ndarray, smask: np.ndarray, bgmask: np.ndarray, cfg: dict, device: str) -> tuple[np.ndarray, dict]:
    """Luminancja → RL w kafelkach z własnym PSF (zakładka, łączenie wagą), siła ``deconv_strength``."""
    import torch

    from .n2n import noise_sigma

    L = np.nan_to_num(rgb.mean(axis=-1)).astype(np.float32)
    H, W = L.shape
    grid = tuple(int(v) for v in cfg["deconv_grid"])
    psfs = measure_psfs(L, smask, grid, int(cfg["deconv_psf_px"]))
    if not np.isfinite(psfs).all():
        log.warning("dekonwolucja: za mało gwiazd do pomiaru PSF — pomijam")
        return rgb, {"applied": False}
    sig = noise_sigma(L, bgmask)
    ped = 5 * sig - min(float(L.min()), 0.0)                 # RL wymaga danych dodatnich
    gy, gx = grid
    ov = 64
    out = np.zeros_like(L)
    acc = np.zeros_like(L)
    for ty in range(gy):
        for tx in range(gx):
            y0, y1 = ty * H // gy, (ty + 1) * H // gy
            x0, x1 = tx * W // gx, (tx + 1) * W // gx
            Y0, Y1, X0, X1 = max(y0 - ov, 0), min(y1 + ov, H), max(x0 - ov, 0), min(x1 + ov, W)
            d = torch.from_numpy(L[Y0:Y1, X0:X1] + ped)[None, None].to(device)
            u = richardson_lucy(d, psfs[ty, tx], int(cfg["deconv_iters"]))[0, 0].cpu().numpy() - ped
            wy = np.minimum(np.arange(Y1 - Y0) + 1, np.arange(Y1 - Y0)[::-1] + 1).clip(max=ov)
            wx = np.minimum(np.arange(X1 - X0) + 1, np.arange(X1 - X0)[::-1] + 1).clip(max=ov)
            w = np.minimum.outer(wy, wx).astype(np.float32)
            out[Y0:Y1, X0:X1] += u * w
            acc[Y0:Y1, X0:X1] += w
    Ld = out / np.maximum(acc, 1e-6)
    s = float(cfg["deconv_strength"])
    detail = s * (Ld - L)
    fwhm = [float(2.3548 * np.sqrt((p * (np.arange(p.shape[0]) - p.shape[0] // 2)[:, None] ** 2).sum()))
            for p in psfs.reshape(-1, *psfs.shape[2:])]
    log.info("dekonwolucja: PSF w %d×%d kafelkach, FWHM %.1f–%.1f px, %d iteracji, siła %.2f", gx, gy,
             min(fwhm), max(fwhm), int(cfg["deconv_iters"]), s)
    return (rgb + detail[..., None]).astype(np.float32), {"applied": True, "fwhm_px": fwhm, "grid": list(grid)}
