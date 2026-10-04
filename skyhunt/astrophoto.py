"""Zdjęcie astronomiczne z sesji RAW: odwrotność szukania kresek — wszystko, co się rusza, znika,
z reszty powstaje obraz nieba w pełnej rozdzielczości.

Kroki: darki (średnia klatka ciemna per czas naświetlania, odejmowana przed demozaikowaniem) →
pełne RGB z X-Trans (LibRaw, liniowo) → wyrównanie każdego zdjęcia mapami z astrometrii sesji
(te same co stos, ×3) → średnia z odrzucaniem σ w dwóch przejściach (satelity, samoloty, meteory,
promienie kosmiczne) osobno dla każdej klasy jasności → HDR z bracketingu (krótkie czasy tam, gdzie
długie są prześwietlone) → tło: łuna i winietowanie f/1.0 z mediany bloków bez gwiazd i bez
chronionych obiektów (położenie z astrometrii) → balans kolorów na gwiazdach → rozciągnięcie asinh
z zachowaniem koloru → JPEG, TIFF 16 bit, liniowy FITS.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger("skyhunt")

# Duże obiekty, których zewnętrzne części nie mogą trafić do modelu tła:
# nazwa → (RA°, Dec°, oś wielka°, oś mała°, kąt pozycyjny° od północy przez wschód)
DEEP_OBJECTS = {
    "M31": (10.6847, 41.2690, 3.6, 1.3, 35.0),
    "M33": (23.4621, 30.6599, 1.1, 0.7, 23.0),
    "M42": (83.8221, -5.3911, 1.2, 1.2, 0.0),
    "M45": (56.7500, 24.1167, 2.0, 2.0, 0.0),
    "NGC 7000": (314.75, 44.33, 2.2, 1.8, 0.0),
    "NGC 869/884": (35.08, 57.14, 1.2, 0.6, 90.0),
    "M8": (270.92, -24.38, 1.2, 0.7, 0.0),
    "M27": (299.90, 22.72, 0.2, 0.2, 0.0),
}


# ---------------------------------------------------------------- darki i dekodowanie

def _dark_chunk(paths: list[str]) -> dict[float, tuple[np.ndarray, int]]:
    """Proces roboczy: suma klatek ciemnych per czas naświetlania (do głównego procesu idą tylko
    sumy, nie każda klatka — 97 MB na klatkę zapychało RAM Colaba). Cała matryca: marginesy
    „widoczne” bywają różne w darkach i zdjęciach."""
    import rawpy

    from .raw import read_raf_exif

    out: dict[float, tuple[np.ndarray, int]] = {}
    for path in paths:
        exp = float(read_raf_exif(Path(path))["exposure_s"] or 0.0)
        with rawpy.imread(path) as r:
            raw = r.raw_image
            s, n = out.get(exp, (np.zeros(raw.shape, np.float32), 0))
            s += raw
            out[exp] = (s, n + 1)
    return out


def master_darks(folder: Path, out_dir: Path, workers: int) -> dict[float, Path]:
    """Średnia klatka ciemna per czas naświetlania (minus poziom czerni) → ``master_dark_<t>.npy``.
    Raz policzone zostają na Drive i są używane przez kolejne sesje."""
    import multiprocessing as mp

    import rawpy

    from .raw import list_photos

    out_dir.mkdir(parents=True, exist_ok=True)
    have = {float(p.stem.split("_")[-1]): p for p in out_dir.glob("master_dark_full_*.npy")}
    if have:
        return have
    files = list_photos(folder, (".raf",))
    if not files:
        return {}
    with rawpy.imread(str(files[0])) as r:
        black = np.asarray(r.black_level_per_channel, np.float32)[r.raw_colors]
    sums: dict[float, np.ndarray] = {}
    cnt: dict[float, int] = {}
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        chunks = [[str(f) for f in files[k::workers]] for k in range(workers)]
        for part in ex.map(_dark_chunk, [c for c in chunks if c]):
            for exp, (s, n) in part.items():
                if exp in sums and sums[exp].shape != s.shape:
                    log.warning("darki %g s: różne rozmiary matrycy %s i %s — pomijam część", exp, sums[exp].shape,
                                s.shape)
                    continue
                sums[exp] = sums[exp] + s if exp in sums else s
                cnt[exp] = cnt.get(exp, 0) + n
    for exp, s in sums.items():
        p = out_dir / f"master_dark_full_{exp:g}.npy"
        np.save(p, (s / cnt[exp] - black).astype(np.float32))
        have[exp] = p
        log.info("master dark %g s: %d klatek → %s", exp, cnt[exp], p.name)
    return have


def _decode_full(path: str, dark: str | None, cache: str | None):
    """Proces roboczy: RAF (− dark) → liniowe RGB uint16 w orientacji matrycy (bez obrotu z EXIF)."""
    import rawpy

    with rawpy.imread(path) as r:
        if dark:
            d = np.load(dark)
            raw = r.raw_image                                 # cała matryca (jak master dark)
            if d.shape == raw.shape:
                raw[:] = np.clip(raw.astype(np.float32) - d, 0, 65535).astype(np.uint16)
        rgb = r.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1), user_flip=0,
                            output_color=rawpy.ColorSpace.sRGB)
    if cache:
        np.save(cache, rgb)
        return cache
    return rgb


def iter_full(paths: list[Path], darks: list[str | None], caches: list[str | None], workers: int):
    """Zdjęcia po kolei (dekodowane równolegle, kolejka 2 × procesy); z cache, jeśli już jest."""
    import multiprocessing as mp

    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        pending: dict[int, cf.Future] = {}
        nxt = 0
        for i in range(len(paths)):
            while nxt < len(paths) and len(pending) < 2 * workers:
                c = caches[nxt]
                cached = bool(c) and os.path.exists(c)
                pending[nxt] = None if cached else ex.submit(_decode_full, str(paths[nxt]), darks[nxt], c)
                nxt += 1
            fut = pending.pop(i)
            res = caches[i] if fut is None else fut.result()
            yield i, (np.load(res) if isinstance(res, str) else res)


# ---------------------------------------------------------------- wyrównanie w pełnej rozdzielczości

class FullResWarper:
    """Mapy zgrubne etapu ``process`` (węzły w superpikselach 3×3) → siatka próbkowania pełnej
    matrycy. Superpiksel x_s ↔ piksel X = 3·x_s + 1 (środek bloku)."""

    def __init__(self, shape_s: tuple[int, int], shape_f: tuple[int, int], device: str):
        import torch

        self.torch, self.dev = torch, device
        self.h, self.w = shape_s
        self.H, self.W = shape_f
        ys = (torch.arange(self.H, dtype=torch.float32) - 1) / 3 / max(self.h - 1, 1) * 2 - 1
        xs = (torch.arange(self.W, dtype=torch.float32) - 1) / 3 / max(self.w - 1, 1) * 2 - 1
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        self.base = torch.stack([X, Y], dim=-1)[None].to(device)          # [1, H, W, 2]

    def __call__(self, rgb: np.ndarray, xc: np.ndarray, yc: np.ndarray):
        t = self.torch
        F = t.nn.functional
        c = np.stack([(3 * np.asarray(xc) + 1) / (self.W - 1) * 2 - 1, (3 * np.asarray(yc) + 1) / (self.H - 1) * 2 - 1])
        coarse = t.from_numpy(c.astype(np.float32))[None].to(self.dev)
        grid = F.grid_sample(coarse, self.base, mode="bilinear", padding_mode="border", align_corners=True)
        grid = grid[0].permute(1, 2, 0)[None]
        img = t.from_numpy(np.ascontiguousarray(rgb[:self.H, :self.W].transpose(2, 0, 1), dtype=np.float32))
        out = F.grid_sample(img[None].to(self.dev), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        inside = (grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1)
        return t.where(inside[None], out, t.full_like(out, float("nan")))


class ClipStack:
    """Średnia z odrzucaniem σ w dwóch przejściach: (1) średnia i rozrzut każdego piksela (względem
    pierwszego zdjęcia — bez utraty precyzji float32), (2) średnia bez wartości > k·σ od średniej."""

    def __init__(self, k: float):
        self.k = float(k)
        self.ref = self.s1 = self.s2 = self.n = self.mean = self.std = self.cs = self.cn = None

    def add1(self, a) -> None:
        t = __import__("torch")
        if self.ref is None:
            fill = float(t.nanmedian(a[:, ::16, ::16]))
            self.ref = t.nan_to_num(a, nan=fill)
            self.s1, self.s2 = t.zeros_like(a), t.zeros_like(a)
            self.n = t.zeros(a.shape[1:], dtype=t.int16, device=a.device)
        ok = t.isfinite(a).all(dim=0)
        d = t.where(ok[None], a - self.ref, t.zeros_like(a))
        self.s1 += d
        self.s2 += d * d
        self.n += ok.to(t.int16)

    def finish1(self) -> None:
        t = __import__("torch")
        n = self.n.clamp(min=1).float()[None]
        m = self.s1 / n
        self.mean = self.ref + m
        self.std = (self.s2 / n - m * m).clamp(min=0).sqrt()
        self.s1 = self.s2 = self.ref = None
        self.cs, self.cn = t.zeros_like(self.mean), t.zeros_like(self.n)

    def add2(self, a) -> float:
        t = __import__("torch")
        ok = t.isfinite(a).all(dim=0)
        dev = (t.nan_to_num(a) - self.mean).abs() / self.std.clamp(min=1e-3)
        keep = ok & ((dev <= self.k).all(dim=0) | (self.n < 3))
        self.cs += t.where(keep[None], t.nan_to_num(a), t.zeros_like(a))
        self.cn += keep.to(t.int16)
        return float(1 - keep[ok].float().mean()) if bool(ok.any()) else 0.0

    def result(self):
        t = __import__("torch")
        n = self.cn.float()[None]
        return t.where(n > 0, self.cs / n.clamp(min=1), t.full_like(self.cs, float("nan"))), self.cn


# ---------------------------------------------------------------- HDR, tło, kolor, rozciągnięcie

def hdr_merge(means: dict[str, np.ndarray], counts: dict[str, np.ndarray], exposures: dict[str, float],
              sat_frac: float = 0.85) -> np.ndarray:
    """Klasy jasności → jeden obraz w DN/s: średnia ważona n·T (odwrotność wariancji szumu
    fotonowego), bez pikseli prześwietlonych w danej klasie (≥ ``sat_frac`` maksimum klasy)."""
    num = den = None
    shortest = min(exposures, key=exposures.get)
    for c, m in means.items():
        T = float(exposures[c])
        peak = np.nanmax(m.reshape(-1, 3), axis=0)
        ok = np.isfinite(m).all(axis=-1) & (m < sat_frac * peak).all(axis=-1)
        if c == shortest:
            ok |= np.isfinite(m).all(axis=-1)            # najkrótszy czas zawsze (nawet przy przesyceniu)
        w = np.where(ok, counts[c].astype(np.float32) * T, 0.0)[..., None]
        v = np.nan_to_num(m / T)
        num = v * w if num is None else num + v * w
        den = w if den is None else den + w
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)


def block_weights(n: int, nb: int, box: int) -> np.ndarray:
    from .photo_report import _block_weights

    return _block_weights(n, nb, box)


def star_mask(lum: np.ndarray, k: float, ds: int = 4, grow: int = 2) -> np.ndarray:
    """Gwiazdy (i inne jasne drobne źródła): lokalna nadwyżka > k·σ na obrazie zmniejszonym ``ds``×."""
    from scipy.ndimage import binary_dilation, median_filter

    H, W = lum.shape
    h, w = H // ds, W // ds
    small = np.nan_to_num(lum[:h * ds, :w * ds].reshape(h, ds, w, ds).mean(axis=(1, 3)), nan=0.0)
    hp = small - median_filter(small, size=15, mode="nearest")
    sig = 1.4826 * float(np.median(np.abs(hp - np.median(hp))))
    m = binary_dilation(hp > k * max(sig, 1e-9), iterations=grow)
    full = np.repeat(np.repeat(m, ds, axis=0), ds, axis=1)
    out = np.zeros((H, W), bool)
    out[:full.shape[0], :full.shape[1]] = full
    return out


def protect_mask(wcs_s, shape_f: tuple[int, int], names, scale: float) -> tuple[np.ndarray, list[str]]:
    """Elipsy dużych obiektów z ``DEEP_OBJECTS`` rzutowane WCS-em stosu (superpiksele → ×3)."""
    from matplotlib.path import Path as MPath

    from .sky import _world2pix

    H, W = shape_f
    mask = np.zeros((H, W), bool)
    used = []
    pick = DEEP_OBJECTS if names in (None, "auto") else {n: DEEP_OBJECTS[n] for n in names if n in DEEP_OBJECTS}
    ys, xs = np.mgrid[0:H:4, 0:W:4]
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    for name, (ra0, dec0, a, b, pa) in pick.items():
        th = np.linspace(0, 2 * np.pi, 90)
        pa_r = np.radians(pa)
        maj = 0.5 * a * scale * np.cos(th)
        mnr = 0.5 * b * scale * np.sin(th)
        east = maj * np.sin(pa_r) + mnr * np.cos(pa_r)
        north = maj * np.cos(pa_r) - mnr * np.sin(pa_r)
        dec = dec0 + north
        ra = ra0 + east / np.cos(np.radians(dec0))
        x, y = _world2pix(wcs_s, ra, dec)
        X, Y = 3 * np.asarray(x) + 1, 3 * np.asarray(y) + 1
        if not np.isfinite(X).all() or X.max() < 0 or X.min() > W or Y.max() < 0 or Y.min() > H:
            continue
        inside = MPath(np.column_stack([X, Y])).contains_points(pts).reshape(ys.shape)
        if inside.any():
            m = np.repeat(np.repeat(inside, 4, axis=0), 4, axis=1)[:H, :W]
            mask[:m.shape[0], :m.shape[1]] |= m
            used.append(name)
    return mask, used


def background_map(img: np.ndarray, mask: np.ndarray, box: int, smooth_blocks: float) -> np.ndarray:
    """Tło (łuna + winietowanie): mediana w blokach bez pikseli z maski, dziury (obiekty) wypełnione
    z sąsiadów, wygładzone, interpolowane liniowo między środkami bloków."""
    from scipy.ndimage import gaussian_filter

    H, W, C = img.shape
    hb, wb = H // box, W // box
    bg = np.empty((H, W, C), np.float32)
    m = mask[:hb * box, :wb * box].reshape(hb, box, wb, box).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    frac = m.mean(axis=2)
    Wy, Wx = block_weights(H, hb, box), block_weights(W, wb, box)
    for c in range(C):
        a = img[:hb * box, :wb * box, c].reshape(hb, box, wb, box).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
        a = np.where(m, np.nan, a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(a, axis=2)
        med[frac > 0.7] = np.nan
        for _ in range(max(hb, wb)):                     # wypełnianie dziur średnią sąsiadów
            bad = ~np.isfinite(med)
            if not bad.any():
                break
            p = np.pad(med, 1, constant_values=np.nan)
            nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fill = np.nanmean(nb, axis=0)
            med = np.where(bad & np.isfinite(fill), fill, med)
        med = gaussian_filter(np.nan_to_num(med, nan=float(np.nanmedian(med))), smooth_blocks, mode="nearest")
        bg[..., c] = Wy @ med @ Wx.T
    return bg


def star_color_factors(img: np.ndarray, smask: np.ndarray, n_max: int = 400) -> tuple[np.ndarray, int]:
    """Mnożniki R, G, B, po których mediana koloru gwiazd pola jest biała (średnia gwiazda pola
    ma B−V ≈ 0,6, prawie jak Słońce). Fotometria w oknach 7×7 wokół najjaśniejszych pikseli."""
    from scipy import ndimage as ndi

    lum = np.nan_to_num(img.mean(axis=-1))
    lab, n = ndi.label(smask)
    if n == 0:
        return np.ones(3, np.float32), 0
    peaks = ndi.maximum_position(lum, lab, index=np.arange(1, n + 1))
    vals = np.array([lum[p] for p in peaks])
    order = np.argsort(-vals)
    hi = np.nanpercentile(lum, 99.99)
    ratios = []
    for j in order:
        y, x = peaks[j]
        if vals[j] >= 0.9 * hi:                          # najjaśniejsze bywają prześwietlone
            continue
        win = img[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].reshape(-1, 3)
        f = np.nansum(win, axis=0)
        if (f > 0).all():
            ratios.append(f / f[1])
        if len(ratios) >= n_max:
            break
    if len(ratios) < 10:
        return np.ones(3, np.float32), len(ratios)
    med = np.median(np.array(ratios), axis=0)
    return (1.0 / med).astype(np.float32), len(ratios)


def asinh_stretch(img: np.ndarray, bgmask: np.ndarray, target_bg: float, saturation: float,
                  chroma_blur: float) -> np.ndarray:
    """Liniowe RGB (tło ≈ 0) → 0…1: rozciągnięcie asinh luminancji, ten sam mnożnik dla R, G, B
    (kolor gwiazd zostaje), tło na poziomie ``target_bg``, wzmocnienie nasycenia, rozmycie chromy."""
    from scipy.ndimage import gaussian_filter

    img = np.nan_to_num(img)
    L = img.mean(axis=-1)
    bgv = L[bgmask][::7]
    noise = 1.4826 * float(np.median(np.abs(bgv - np.median(bgv)))) if len(bgv) else float(np.std(L))
    black = float(np.median(bgv)) - 2.0 * noise if len(bgv) else 0.0
    hi = float(np.percentile(L, 99.995)) - black
    x0 = 2.0 * noise / max(hi, 1e-9)                     # tło po odjęciu czerni, w skali 0…1

    def f(x, beta):
        return np.arcsinh(beta * x) / np.arcsinh(beta)

    lo_b, hi_b = 1.0, 1e6
    for _ in range(60):                                  # β: tło → target_bg
        mid = math.sqrt(lo_b * hi_b)
        if f(x0, mid) < target_bg:
            lo_b = mid
        else:
            hi_b = mid
    beta = math.sqrt(lo_b * hi_b)
    xl = np.clip((L - black) / max(hi, 1e-9), 0, None)
    gain = np.where(xl > 1e-9, f(xl, beta) / np.maximum(xl, 1e-9), beta / np.arcsinh(beta))
    rgb = (img - black) / max(hi, 1e-9) * gain[..., None]
    lum = rgb.mean(axis=-1, keepdims=True)
    chroma = rgb - lum
    if chroma_blur > 0:
        chroma = np.stack([gaussian_filter(chroma[..., c], chroma_blur) for c in range(3)], axis=-1)
    out = lum + saturation * chroma
    return np.clip(out, 0, 1).astype(np.float32)


def save_outputs(out_dir: Path, name: str, img01: np.ndarray, linear: np.ndarray, crop: tuple[int, int],
                 quality: int) -> list[str]:
    from astropy.io import fits
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    u8 = (img01 * 255 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(out_dir / f"{name}.jpg", quality=quality)
    files.append(f"{name}.jpg")
    H, W = u8.shape[:2]
    cw, ch = min(int(crop[0]), W), min(int(crop[1]), H)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    Image.fromarray(u8[y0:y0 + ch, x0:x0 + cw]).save(out_dir / f"{name}_crop.jpg", quality=quality)
    files.append(f"{name}_crop.jpg")
    try:
        import cv2

        u16 = (img01 * 65535 + 0.5).astype(np.uint16)
        cv2.imwrite(str(out_dir / f"{name}_16bit.tif"), u16[..., ::-1])
        files.append(f"{name}_16bit.tif")
    except ImportError:
        log.warning("brak opencv: bez TIFF 16 bit")
    fits.writeto(out_dir / f"{name}_linear.fits", np.nan_to_num(linear).transpose(2, 0, 1).astype(np.float32),
                 overwrite=True)
    files.append(f"{name}_linear.fits")
    return files


# ---------------------------------------------------------------- całość

def run(folder: Path, cfg: dict, out_root: Path, dark_dir: Path | None = None) -> dict:
    import pandas as pd
    import torch

    from . import photo_stages  # noqa: F401 — rejestracja etapów zdjęć
    from .astrometry import load_wcs
    from .config import config_for_file
    from .pipeline import PHOTO_PIPELINE, StageContext, input_outdir
    from .photo_stages import alignment, stack_device
    from .raw import list_photos

    folder = Path(folder)
    cfg = config_for_file(cfg, folder)
    acfg = cfg["astrophoto"]
    PHOTO_PIPELINE.run(folder, cfg, out_root, only=["astrometry"], log=log)   # EXIF, rytm, plate solve
    outdir = input_outdir(out_root, folder)
    ctx = StageContext(folder, outdir, cfg, None, log)
    ph = pd.read_csv(outdir / "photos.csv")
    al = alignment(ctx, ph)
    workers = max(1, min(int(acfg["workers"]), (os.cpu_count() or 2) - 1))
    dev = stack_device(cfg["photo"]) or "cpu"

    # darki
    dark_map: dict[float, Path] = {}
    if dark_dir is None and str(acfg.get("dark_dir", "auto")) == "auto":
        cands = sorted(p for p in folder.parent.iterdir() if p.is_dir() and "dark" in p.name.lower())
        dark_dir = cands[0] if cands else None
    elif dark_dir is None and acfg.get("dark_dir"):
        dark_dir = Path(acfg["dark_dir"])
    if dark_dir is not None and Path(dark_dir).exists():
        dark_map = master_darks(Path(dark_dir), input_outdir(out_root, Path(dark_dir)), workers)
    classes = [c for c in acfg["classes"] if c in set(ph["ev"])]
    exposures = {c: float(ph.loc[ph["ev"] == c, "exposure_s"].median()) for c in classes}

    def dark_for(t: float) -> str | None:
        hits = [p for e, p in dark_map.items() if abs(e - t) <= 0.02 * t]
        return str(hits[0]) if hits else None

    if dark_map:                                          # rozmiar matrycy darków = zdjęć?
        import rawpy

        first = next(p for p in sorted(folder.iterdir()) if p.suffix.lower() == ".raf")
        with rawpy.imread(str(first)) as r:
            shape = r.raw_image.shape
        bad = [e for e, p in dark_map.items() if np.load(p, mmap_mode="r").shape != shape]
        if bad:
            log.warning("[%s] darki %s mają inny rozmiar matrycy niż zdjęcia %s — pomijam je", folder.name, bad, shape)
            dark_map = {e: p for e, p in dark_map.items() if e not in bad}
    for c in classes:
        log.info("[%s] %s: %d zdjęć po %g s, dark: %s", folder.name, c, int((ph["ev"] == c).sum()), exposures[c],
                 Path(dark_for(exposures[c])).name if dark_for(exposures[c]) else "BRAK")

    # cache zdekodowanych zdjęć na dysku lokalnym (drugie przejście bez dekodowania)
    sel = [int(i) for i in ph.index[ph["ev"].isin(classes)]]
    by_name = {p.name: p for p in list_photos(folder, cfg["input"].get("photo_extensions", [".raf"]))}
    paths = [by_name[ph["file"][i]] for i in sel]
    cache_dir = Path(acfg["cache_dir"]) / folder.name
    caches: list[str | None] = [None] * len(sel)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        need = len(sel) * 160e6
        if shutil.disk_usage(cache_dir).free > need * 1.2:
            caches = [str(cache_dir / f"rgb_{i:05d}.npy") for i in sel]
        else:
            log.warning("[%s] za mało miejsca na cache (%.0f GB) — drugie przejście dekoduje jeszcze raz",
                        folder.name, need / 1e9)
    except OSError:
        pass
    darks = [dark_for(float(ph["exposure_s"][i])) for i in sel]

    warper, stacks = None, {c: ClipStack(float(acfg["clip_sigma"])) for c in classes}
    for p in (1, 2):
        rej = []
        for j, rgb in iter_full(paths, darks, caches, workers):
            i = sel[j]
            if warper is None:
                Hf, Wf = (min(rgb.shape[0], 3 * al.h), min(rgb.shape[1], 3 * al.w))
                warper = FullResWarper((al.h, al.w), (Hf, Wf), dev)
            xc, yc = al.maps(float(ph["tau_mid"][i]))
            a = warper(rgb, xc, yc)
            st = stacks[str(ph["ev"][i])]
            if p == 1:
                st.add1(a)
            else:
                rej.append(st.add2(a))
            if (j + 1) % 30 == 0:
                log.info("[%s] przejście %d: %d/%d zdjęć", folder.name, p, j + 1, len(sel))
        if p == 1:
            for st in stacks.values():
                st.finish1()
        else:
            log.info("[%s] odrzucone piksele (ruchome obiekty, promienie kosmiczne): mediana %.3f%%", folder.name,
                     100 * float(np.median(rej)) if rej else 0.0)
    means, counts = {}, {}
    for c, st in stacks.items():
        m, n = st.result()
        means[c] = m.permute(1, 2, 0).cpu().numpy()
        counts[c] = n.cpu().numpy()
    del stacks
    torch.cuda.empty_cache() if dev != "cpu" else None
    lin = hdr_merge(means, counts, exposures)
    del means

    # tło, winietowanie, kolor
    wcsinfo = ctx.read_json("wcs.json")
    wcs_s = load_wcs(outdir / wcsinfo["reference_wcs"])
    lum = np.nanmean(lin, axis=-1)
    smask = star_mask(lum, float(acfg["bg_star_sigma"]))
    pmask, protected = protect_mask(wcs_s, lum.shape, acfg.get("protect", "auto"), float(acfg["protect_scale"]))
    valid = np.isfinite(lin).all(axis=-1) & (counts[classes[0]] >= max(3, 0.5 * np.median(counts[classes[0]])))
    bgmask = ~smask & ~pmask & valid
    bg = background_map(np.where(valid[..., None], lin, np.nan), ~bgmask, int(acfg["bg_box_px"]),
                        float(acfg["bg_smooth_blocks"]))
    flat = lin - bg
    vign = None
    if acfg.get("flat_from_background", True):
        bl = bg.mean(axis=-1)
        vign = np.clip(bl / np.nanpercentile(bl[valid], 99.5), float(acfg["min_vignetting"]), 1.0)
        flat = flat / vign[..., None]
    factors, n_stars = star_color_factors(flat, smask & valid & ~pmask)
    flat = flat * factors
    flat[~valid] = 0.0
    img01 = asinh_stretch(flat, bgmask, float(acfg["stretch_target_bg"]), float(acfg["saturation"]),
                          float(acfg["chroma_blur_px"]))
    astro = outdir / "astro"
    files = save_outputs(astro, folder.name, img01, flat, tuple(acfg["crop_center"]), int(acfg["jpeg_quality"]))
    info = {"frames": {c: int((ph["ev"] == c).sum()) for c in classes}, "exposures_s": exposures,
            "darks": {c: bool(dark_for(exposures[c])) for c in classes}, "protected": protected,
            "color_factors_rgb": factors.tolist(), "color_stars": n_stars, "shape": list(img01.shape[:2]),
            "vignetting_min": float(np.nanmin(vign[valid])) if vign is not None else None, "files": files}
    (astro / "astro.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("[%s] zdjęcie: %s (chronione przed modelem tła: %s; kolor z %d gwiazd)", folder.name,
             ", ".join(files), ", ".join(protected) or "nic", n_stars)
    return info
