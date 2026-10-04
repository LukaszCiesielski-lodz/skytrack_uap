"""Etapy dla sesji zdjęć RAW (folder z plikami RAF) — osobny rejestr ``PHOTO_PIPELINE``.

F1: ``probe`` (EXIF, serie bracketingu, rytm, czas a priori) → ``frames`` (zdjęcia do plate solve)
→ ``astrometry`` (wspólny rdzeń z wideo) → ``process`` (wszystkie zdjęcia: superpiksele 3×3,
cache luminancji, głęboki stos per klasa jasności wyrównany modelem nieruchomej kamery)
→ ``report`` (mapa na stosie, przebieg sesji). Satelity (kreski) dochodzą w F2.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from .pipeline import PHOTO_PIPELINE, StageContext
from .stages import _site, prior_start, solve_epochs
from .stages import tle as _tle

_EPOCH0 = datetime(1970, 1, 1)


# ---------------------------------------------------------------- sesja: metadane i czas

def _exif_seconds(r: dict) -> float:
    dt = datetime.strptime(r["datetime"][:19], "%Y:%m:%d %H:%M:%S")
    t = (dt - _EPOCH0).total_seconds()
    if r.get("subsec"):
        t += float(f"0.{str(r['subsec']).strip()}")
    return t


def probe_session(folder: Path, cfg: dict) -> dict:
    """EXIF wszystkich zdjęć → serie, klasy jasności, rytm interwałometru, chwile otwarcia
    migawki (τ od otwarcia pierwszego zdjęcia) i czas startu a priori w UTC."""
    from .photo_timing import Cadence, ev_class, fit_cadence, group_sets, open_times
    from .raw import list_photos, read_raf_exif
    from .timing import camera_to_utc

    pcfg, tcfg = cfg["photo"], cfg["time"]
    files = list_photos(folder, cfg["input"].get("photo_extensions", [".raf"]))
    if not files:
        raise RuntimeError(f"brak plików RAF w {folder}")
    rows = [read_raf_exif(f) for f in files]
    if all(r.get("image_count") is not None for r in rows):
        rows.sort(key=lambda r: (_exif_seconds(r), r["image_count"]))
    sets = group_sets(rows, int(pcfg["frames_per_set"]))
    for r, k in zip(rows, sets):
        r["set"] = k
    for k in sorted(set(sets)):
        members = [r for r in rows if r["set"] == k]
        for r, c in zip(members, ev_class([float(m["exposure_s"] or 0) for m in members])):
            r["ev"] = c
    t = np.array([_exif_seconds(r) for r in rows])
    exact = all(r.get("subsec") for r in rows)
    firsts = [i for i, k in enumerate(sets) if i == 0 or sets[i - 1] != k]
    exp = [float(r["exposure_s"] or 0.0) for r in rows]
    if exact:   # ułamki sekund w EXIF: czas wprost, rytm tylko informacyjnie
        p = float(np.median(np.diff(t[firsts]))) if len(firsts) > 1 else float("nan")
        cad = Cadence(float(t[0]), p, 0.005, float("nan"), False, len(firsts))
        t_open = t.copy()
    else:
        cad = fit_cadence(t[firsts])
        t_open = open_times(sets, exp, t, cad, float(pcfg["inter_frame_gap_s"]))
    t_ref = float(t_open[0])
    for r, to, e in zip(rows, t_open, exp):
        r["tau_open"] = float(to - t_ref)
        r["tau_mid"] = float(to - t_ref + e / 2)
    cam_time = _EPOCH0 + timedelta(seconds=t_ref)
    start = camera_to_utc(cam_time, tcfg["camera_tz"], float(tcfg["camera_clock_ahead_s"]))
    isos = sorted({r["iso"] for r in rows if r["iso"]})
    fnums = sorted({r["fnumber"] for r in rows if r["fnumber"]})
    classes = {}
    for r in rows:
        classes.setdefault(r["ev"], set()).add(r["exposure_s"])
    session = {"n_photos": len(rows), "n_sets": len(set(sets)), "duration_s": float(t_open[-1] - t_ref + exp[-1]),
               "cadence": cad.to_dict(), "exif_subsec": exact, "iso": isos, "fnumber": fnums,
               "classes": {c: sorted(v) for c, v in sorted(classes.items())},
               "model": rows[0].get("model"), "first": rows[0]["file"], "last": rows[-1]["file"],
               "sequence_numbers": all(r.get("sequence_number") for r in rows)}
    warn = []
    if len(isos) > 1:
        warn.append(f"zmienne ISO {isos} — ustaw ISO na stałe (bez Auto ISO)")
    if len(fnums) > 1:
        warn.append(f"zmienna przysłona {fnums}")
    if any(len(v) > 1 for v in classes.values()):
        warn.append("klasa jasności ma różne czasy naświetlania")
    if not cad.regular and not exact:
        warn.append(f"nieregularny rytm serii ({cad.n_violations} serii poza rytmem) — czas z EXIF ±0,5 s")
    session["warnings"] = warn
    prior = {"start_utc": start.isoformat(), "sigma_s": float(tcfg["prior_sigma_s"]), "source": "exif",
             "camera_time": cam_time.isoformat(), "candidates_utc": {}}
    return {"session": session, "photos": rows, "time_prior": prior}


@PHOTO_PIPELINE.stage("probe", sections=("time", "role", "photo", "input.photo_extensions"), rev=1)
def probe(ctx: StageContext) -> dict:
    """Metadane zdjęć i czas a priori (EXIF + rytm serii)."""
    import pandas as pd

    res = probe_session(ctx.input_path, ctx.cfg)
    pd.DataFrame(res["photos"]).to_csv(ctx.outdir / "photos.csv", index=False)
    ctx.write_json("meta.json", {"kind": "photos", "session": res["session"], "time_prior": res["time_prior"],
                                 "role": ctx.role})
    s, cad = res["session"], res["session"]["cadence"]
    ctx.log.info("[%s] %d zdjęć w %d seriach, %.0f s, rytm %.4f s (T0 ±%.3f s, %s), klasy %s, ISO %s, start ≈ %s UTC",
                 ctx.input_path.name, s["n_photos"], s["n_sets"], s["duration_s"], cad["period"],
                 cad["t0_halfwidth"], "regularny" if cad["regular"] else "NIEREGULARNY", s["classes"], s["iso"],
                 res["time_prior"]["start_utc"])
    for w in s["warnings"]:
        ctx.log.warning("[%s] %s", ctx.input_path.name, w)
    return {"outputs": ["photos.csv", "meta.json"],
            "metrics": {"n_photos": s["n_photos"], "n_sets": s["n_sets"], "duration_s": s["duration_s"],
                        "period_s": cad["period"], "t0_halfwidth_s": cad["t0_halfwidth"], "regular": cad["regular"]}}


def _photos(ctx: StageContext):
    import pandas as pd

    return pd.read_csv(ctx.outdir / "photos.csv")


# ---------------------------------------------------------------- odczyt zdjęć (równolegle)

def _load_one(path: str, sat_frac: float, cache: str | None):
    """Proces roboczy: RAF → superpiksele (+ zapis luminancji do cache). RGB jako float16,
    żeby mniej danych szło między procesami."""
    from .raw import read_raw, superpixels

    lum, rgb, sat = superpixels(read_raw(Path(path)), sat_frac=sat_frac)
    if cache:
        np.save(cache, np.clip(lum, -65000, 65000).astype(np.float16))   # prześwietlone > zakres float16
    return lum, rgb.astype(np.float16), sat


def decode_workers(pcfg: dict) -> int:
    """Rozpakowanie RAF z kompresją bezstratną (LibRaw) idzie w jednym wątku, ~2–3 s na zdjęcie,
    więc dekodujemy w procesach: co najmniej ``photo.workers``, domyślnie rdzenie − 1."""
    return max(int(pcfg.get("workers", 1)), (os.cpu_count() or 2) - 1, 1)


def load_photos(paths: list[Path], pcfg: dict, cache_files: list | None = None
                ) -> Iterator[tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Zdjęcia po kolei, dekodowane równolegle w procesach (kolejka 2 × liczba procesów)."""
    import multiprocessing as mp

    n = decode_workers(pcfg)
    sat = float(pcfg["sat_frac"])
    caches = cache_files or [None] * len(paths)
    with cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn")) as ex:
        pending: dict[int, cf.Future] = {}
        nxt = 0
        for i in range(len(paths)):
            while nxt < len(paths) and len(pending) < 2 * n:
                pending[nxt] = ex.submit(_load_one, str(paths[nxt]), sat,
                                         str(caches[nxt]) if caches[nxt] else None)
                nxt += 1
            yield i, pending.pop(i).result()


def decode(path: Path, pcfg: dict):
    from .raw import read_raw, superpixels

    return superpixels(read_raw(path), sat_frac=float(pcfg["sat_frac"]))


def robust_sigma(img: np.ndarray) -> float:
    s = img[::7, ::7]
    s = s[np.isfinite(s)]
    med = float(np.median(s))
    return 1.4826 * float(np.median(np.abs(s - med)))


# ---------------------------------------------------------------- zdjęcia do plate solve

def epoch_indices(tau: np.ndarray, ev: list[str], every_s: float, min_epochs: int) -> list[int]:
    """Zdjęcia klasy ev0 rozłożone co ``every_s`` (co najmniej ``min_epochs``, z początkiem i końcem)."""
    cand = np.flatnonzero(np.asarray(ev) == "ev0")
    if len(cand) == 0:
        cand = np.arange(len(tau))
    span = float(tau[cand[-1]] - tau[cand[0]])
    n = max(int(min_epochs), int(span // max(every_s, 1e-6)) + 1)
    targets = np.linspace(tau[cand[0]], tau[cand[-1]], min(n, len(cand)))
    idx = sorted({int(cand[np.argmin(np.abs(tau[cand] - t))]) for t in targets})
    return idx


@PHOTO_PIPELINE.stage("frames", sections=("photo.epoch_every_s", "photo.min_epochs", "photo.sat_frac"),
                      requires=("probe",), rev=1, roles=("sky",))
def frames(ctx: StageContext) -> dict:
    """Wybrane zdjęcia ev0 jako FITS (luminancja superpikseli) do plate solve."""
    from astropy.io import fits

    from .raw import list_photos

    pcfg = ctx.cfg["photo"]
    ph = _photos(ctx)
    idx = epoch_indices(ph["tau_mid"].to_numpy(float), list(ph["ev"]), float(pcfg["epoch_every_s"]),
                        int(pcfg["min_epochs"]))
    folder = ctx.input_path
    by_name = {p.name: p for p in list_photos(folder, ctx.cfg["input"].get("photo_extensions", [".raf"]))}
    edir = ctx.outdir / "epochs"
    edir.mkdir(exist_ok=True)
    out, shape = [], None
    paths = [by_name[ph["file"][i]] for i in idx]
    for j, (lum, _, _) in load_photos(paths, pcfg):
        i = idx[j]
        shape = lum.shape
        name = f"epoch_{i:05d}.fits"
        hdr = fits.Header()
        hdr["PHOTO"] = (int(i), "numer zdjecia w sesji")
        hdr["TAU_S"] = (float(ph["tau_mid"][i]), "srodek naswietlania od otwarcia 1. zdjecia [s]")
        hdr["EXPTIME"] = float(ph["exposure_s"][i])
        fits.writeto(edir / name, lum.astype(np.float32), hdr, overwrite=True)
        out.append({"block": j, "frame": int(i), "tau_s": float(ph["tau_mid"][i]), "file": f"epochs/{name}",
                    "photo": str(ph["file"][i])})
    ctx.write_json("epochs.json", out)
    ctx.write_json("frames.json", {"shape": list(shape), "bin": 3})
    ctx.log.info("[%s] %d zdjęć do plate solve, obraz %d×%d (superpiksele 3×3)", folder.name, len(out),
                 shape[1], shape[0])
    return {"outputs": ["epochs.json", "frames.json"] + [e["file"] for e in out],
            "metrics": {"n_epochs": len(out), "width": shape[1], "height": shape[0]}}


@PHOTO_PIPELINE.stage("astrometry", sections=("astrometry", "site", "camera.nominal_hfov_deg", "camera.sensor_width_mm"),
                      requires=("frames",), rev=1, roles=("sky",))
def astrometry(ctx: StageContext) -> dict:
    """Plate solve zdjęć epok (wspólny rdzeń z wideo)."""
    h, w = ctx.read_json("frames.json")["shape"]
    return solve_epochs(ctx, (int(h), int(w)))


# ---------------------------------------------------------------- głęboki stos

def coarse_grid(shape: tuple[int, int], grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    gx, gy = int(grid[0]), int(grid[1])
    xs, ys = np.linspace(0, w - 1, gx), np.linspace(0, h - 1, gy)
    return np.meshgrid(xs, ys)


def upsample_map(coarse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Wartości na siatce [gy, gx] (węzły od brzegu do brzegu) → pełny obraz [h, w] (dwuliniowo)."""
    from scipy.ndimage import map_coordinates

    gy, gx = coarse.shape
    h, w = shape
    yy = np.linspace(0, gy - 1, h, dtype=np.float32)
    xx = np.linspace(0, gx - 1, w, dtype=np.float32)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(coarse, [Y, X], order=1, mode="nearest").astype(np.float32)


def warp_to_reference(img: np.ndarray, xmap: np.ndarray, ymap: np.ndarray, order: int = 1) -> np.ndarray:
    """Obraz zdjęcia w układzie zdjęcia odniesienia: piksel (x, y) odniesienia ← img(xmap, ymap)."""
    from scipy.ndimage import map_coordinates

    return map_coordinates(img, [ymap, xmap], order=order, mode="constant", cval=np.nan).astype(np.float32)


class PointingModel:
    """Piksel odniesienia → piksel zdjęcia w chwili τ z kamer WSZYSTKICH rozwiązanych epok:
    każda epoka ma własny WCS (dokładny w swojej chwili), między sąsiednimi epokami interpolacja
    liniowa w czasie. Tak powolny ruch statywu (głowica „siada” przy aparacie w zenicie) nie
    rozmywa stosu; jeden WCS z początku zakładałby idealnie nieruchomy aparat."""

    def __init__(self, cameras: list, taus: list[float], ra: np.ndarray, dec: np.ndarray):
        order = np.argsort(taus)
        self.cams = [cameras[i] for i in order]
        self.taus = np.asarray(taus, float)[order]
        self.ra, self.dec = ra, dec

    def coarse(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.cams)
        if n == 1:
            x, y = self.cams[0].pixel(self.ra, self.dec, tau)
            return np.asarray(x, float), np.asarray(y, float)
        lo = int(np.clip(np.searchsorted(self.taus, tau) - 1, 0, n - 2))
        hi = lo + 1
        w = float(np.clip((tau - self.taus[lo]) / max(self.taus[hi] - self.taus[lo], 1e-9), 0.0, 1.0))
        x0, y0 = self.cams[lo].pixel(self.ra, self.dec, tau)
        x1, y1 = self.cams[hi].pixel(self.ra, self.dec, tau)
        return ((1 - w) * np.asarray(x0, float) + w * np.asarray(x1, float),
                (1 - w) * np.asarray(y0, float) + w * np.asarray(y1, float))


def pointing_drift(ref_camera, cameras: list, taus: list[float], ra: float, dec: float) -> list[dict]:
    """Ruch aparatu: gdzie kierunek środka kadru widzi kamera każdej epoki, a gdzie przewiduje
    go kamera odniesienia (nieruchomy aparat) — różnica w pikselach."""
    out = []
    for cam, tau in sorted(zip(cameras, taus), key=lambda p: p[1]):
        xr, yr = ref_camera.pixel(ra, dec, tau)
        xk, yk = cam.pixel(ra, dec, tau)
        out.append({"tau_s": float(tau), "dx_px": float(np.ravel(xk)[0] - np.ravel(xr)[0]),
                    "dy_px": float(np.ravel(yk)[0] - np.ravel(yr)[0])})
    return out


class StackAccumulator:
    """Średnia per piksel z odrzucaniem przejściowych jasnych pikseli (kreski satelitów,
    samoloty): po ``warmup`` zdjęciach próbka > średnia + k·σ zdjęcia nie wchodzi."""

    def __init__(self, shape: tuple[int, int], k: float, warmup: int, channels: int = 0):
        self.sum = np.zeros(shape, np.float32)
        self.n = np.zeros(shape, np.uint16)
        self.k, self.warmup, self.frames = float(k), int(warmup), 0
        self.rgb = np.zeros(shape + (channels,), np.float32) if channels else None

    def add(self, img: np.ndarray, sigma: float, rgb: np.ndarray | None = None) -> np.ndarray:
        ok = np.isfinite(img)
        if self.frames >= self.warmup and sigma > 0:
            mean = np.divide(self.sum, self.n, out=np.zeros_like(self.sum), where=self.n > 0)
            ok &= (self.n == 0) | (img <= mean + self.k * sigma)
        self.sum[ok] += img[ok]
        self.n[ok] += 1
        if self.rgb is not None and rgb is not None:
            okc = ok & np.isfinite(rgb).all(axis=-1)
            self.rgb[okc] += rgb[okc]
        self.frames += 1
        return ok

    def mean(self) -> np.ndarray:
        return np.divide(self.sum, self.n, out=np.full_like(self.sum, np.nan), where=self.n > 0)

    def mean_rgb(self) -> np.ndarray | None:
        if self.rgb is None:
            return None
        return np.divide(self.rgb, self.n[..., None], out=np.full_like(self.rgb, np.nan),
                         where=self.n[..., None] > 0)

    def state(self) -> dict:
        d = {"sum": self.sum, "n": self.n, "frames": np.array(self.frames)}
        if self.rgb is not None:
            d["rgb"] = self.rgb
        return d

    def load(self, d: dict, prefix: str) -> None:
        self.sum, self.n, self.frames = d[f"{prefix}sum"], d[f"{prefix}n"], int(d[f"{prefix}frames"])
        if self.rgb is not None and f"{prefix}rgb" in d:
            self.rgb = d[f"{prefix}rgb"]

    def add_photo(self, lum: np.ndarray, rgb: np.ndarray | None, xc: np.ndarray, yc: np.ndarray,
                  sigma: float) -> float:
        """Wyrównanie (mapy z siatki zgrubnej [gy, gx]) + dodanie; zwraca odsetek odrzuconych pikseli."""
        shape = self.sum.shape
        xmap, ymap = upsample_map(xc, shape), upsample_map(yc, shape)
        aligned = warp_to_reference(lum, xmap, ymap)
        rgb_al = (np.stack([warp_to_reference(rgb[..., k], xmap, ymap) for k in range(3)], axis=-1)
                  if (self.rgb is not None and rgb is not None) else None)
        return float(1 - self.add(aligned, sigma, rgb_al).mean())


class TorchStackAccumulator:
    """To samo co ``StackAccumulator`` na GPU: siatka próbkowania z mapy zgrubnej (interpolacja
    dwuliniowa, węzły od brzegu do brzegu = ``align_corners``), ``grid_sample`` luminancji i RGB
    naraz, sumy i odrzucanie na karcie. Na CPU wyrównanie 4 kanałów 2,7 Mpx trwało ~1 s/zdjęcie."""

    def __init__(self, shape: tuple[int, int], k: float, warmup: int, channels: int = 0, device: str = "cuda"):
        import torch

        self.torch, self.dev = torch, torch.device(device)
        self.shape = tuple(shape)
        self.sum = torch.zeros(self.shape, device=self.dev)
        self.cnt = torch.zeros(self.shape, dtype=torch.int32, device=self.dev)
        self.rgbsum = torch.zeros((3,) + self.shape, device=self.dev) if channels else None
        self.k, self.warmup, self.frames = float(k), int(warmup), 0

    @property
    def rgb(self):
        return self.rgbsum

    @property
    def n(self) -> np.ndarray:
        return self.cnt.cpu().numpy().astype(np.uint16)

    def add_photo(self, lum: np.ndarray, rgb: np.ndarray | None, xc: np.ndarray, yc: np.ndarray,
                  sigma: float) -> float:
        t = self.torch
        F = t.nn.functional
        h, w = self.shape
        norm = np.stack([2 * np.asarray(xc, np.float32) / (w - 1) - 1, 2 * np.asarray(yc, np.float32) / (h - 1) - 1])
        coarse = t.from_numpy(norm.astype(np.float32))[None].to(self.dev)
        grid = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)[None]
        chans = [lum] + ([rgb[..., k] for k in range(3)] if (self.rgbsum is not None and rgb is not None) else [])
        img = t.from_numpy(np.ascontiguousarray(np.stack(chans), dtype=np.float32))[None].to(self.dev)
        out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        a = out[0]
        ok = (grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1) & t.isfinite(a)
        if self.frames >= self.warmup and sigma > 0:
            mean = self.sum / self.cnt.clamp(min=1)
            ok &= (self.cnt == 0) | (a <= mean + self.k * float(sigma))
        okf = ok.float()
        self.sum += t.nan_to_num(a) * okf
        self.cnt += ok.int()
        if self.rgbsum is not None and out.shape[0] == 4:
            self.rgbsum += t.nan_to_num(out[1:]) * okf
        self.frames += 1
        return float(1 - okf.mean().item())

    def mean(self) -> np.ndarray:
        m = (self.sum / self.cnt.clamp(min=1)).cpu().numpy()
        return np.where(self.cnt.cpu().numpy() > 0, m, np.nan).astype(np.float32)

    def mean_rgb(self) -> np.ndarray | None:
        if self.rgbsum is None:
            return None
        m = (self.rgbsum / self.cnt.clamp(min=1)).permute(1, 2, 0).cpu().numpy()
        return np.where(self.cnt.cpu().numpy()[..., None] > 0, m, np.nan).astype(np.float32)

    def state(self) -> dict:
        d = {"sum": self.sum.cpu().numpy(), "n": self.n, "frames": np.array(self.frames)}
        if self.rgbsum is not None:
            d["rgb"] = self.rgbsum.permute(1, 2, 0).cpu().numpy()
        return d

    def load(self, d: dict, prefix: str) -> None:
        t = self.torch
        self.sum = t.from_numpy(np.asarray(d[f"{prefix}sum"], np.float32)).to(self.dev)
        self.cnt = t.from_numpy(np.asarray(d[f"{prefix}n"], np.int32)).to(self.dev)
        self.frames = int(d[f"{prefix}frames"])
        if self.rgbsum is not None and f"{prefix}rgb" in d:
            self.rgbsum = t.from_numpy(np.asarray(d[f"{prefix}rgb"], np.float32)).permute(2, 0, 1).contiguous().to(self.dev)


def stack_device(pcfg: dict) -> str | None:
    """``photo.device``: auto (GPU, jeśli jest) | cuda | cpu."""
    if str(pcfg.get("device", "auto")) == "cpu":
        return None
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else None
    except ImportError:
        return None


class TimeMaps:
    """Mapy zgrubne z ``PointingModel`` liczone w węzłach co ``step_s`` (astropy jest wolne:
    2 wywołania na zdjęcie to ~1 s), między węzłami interpolacja liniowa — obrót nieba i ruch
    statywu są na takim odcinku praktycznie liniowe."""

    def __init__(self, pointing, t_min: float, t_max: float, step_s: float):
        n = max(2, int(np.ceil((t_max - t_min) / max(step_s, 1e-3))) + 1)
        self.t = np.linspace(t_min, t_max, n)
        xs, ys = zip(*(pointing.coarse(float(tt)) for tt in self.t))
        self.x, self.y = np.stack(xs), np.stack(ys)

    def at(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        k = int(np.clip(np.searchsorted(self.t, tau) - 1, 0, len(self.t) - 2))
        w = float(np.clip((tau - self.t[k]) / (self.t[k + 1] - self.t[k]), 0.0, 1.0))
        return (1 - w) * self.x[k] + w * self.x[k + 1], (1 - w) * self.y[k] + w * self.y[k + 1]


def alignment(ctx: StageContext, ph) -> SimpleNamespace:
    """Kamera odniesienia, kamery epok i mapy zgrubne zdjęcie(τ) ← układ odniesienia (stos).
    Piksel układu odniesienia to stały kierunek ICRS: ``camera.icrs(x, y, tau_ref)``."""
    from .astrometry import load_wcs
    from .sky import FixedCamera

    pcfg = ctx.cfg["photo"]
    wcsinfo = ctx.read_json("wcs.json")
    h, w = (int(v) for v in ctx.read_json("frames.json")["shape"])
    ref_epoch = next(e for e in wcsinfo["epochs"] if e["file"] == wcsinfo["reference"])
    tau_ref = float(wcsinfo["reference_tau_s"])
    t_start, site = prior_start(ctx), _site(ctx.cfg)
    camera = FixedCamera(load_wcs(ctx.outdir / wcsinfo["reference_wcs"]), tau_ref, t_start, *site)
    gx, gy = coarse_grid((h, w), tuple(pcfg["stack_grid"]))
    ra_g, dec_g = camera.icrs(gx.ravel(), gy.ravel(), tau_ref)
    solved = [e for e in wcsinfo["epochs"] if e.get("solved")]
    cams = [FixedCamera(load_wcs(ctx.outdir / e["wcs"]), float(e["tau_s"]), t_start, *site) for e in solved]
    taus = [float(e["tau_s"]) for e in solved]
    pointing = PointingModel(cams, taus, ra_g, dec_g)
    tm = TimeMaps(pointing, float(ph["tau_mid"].min()), float(ph["tau_mid"].max()),
                  float(pcfg.get("pointing_step_s", 20.0)))

    def maps(tau: float) -> tuple[np.ndarray, np.ndarray]:
        px, py = tm.at(float(tau))
        return px.reshape(gx.shape), py.reshape(gx.shape)

    return SimpleNamespace(wcsinfo=wcsinfo, h=h, w=w, ref_epoch=ref_epoch, tau_ref=tau_ref, t_start=t_start, site=site,
                           camera=camera, gx=gx, gy=gy, cams=cams, taus=taus, pointing=pointing, tm=tm, maps=maps)


@PHOTO_PIPELINE.stage("process", sections=("photo", "site"), requires=("astrometry",), rev=2, roles=("sky",))
def process(ctx: StageContext) -> dict:
    """Wszystkie zdjęcia: superpiksele, cache luminancji (lokalnie), statystyki, stosy per klasa
    jasności wyrównane do zdjęcia odniesienia astrometrii (model nieruchomej kamery)."""
    import time

    import pandas as pd
    from astropy.io import fits

    from .raw import list_photos

    pcfg = ctx.cfg["photo"]
    ph = _photos(ctx)
    al = alignment(ctx, ph)
    wcsinfo, h, w, ref_epoch, tau_ref = al.wcsinfo, al.h, al.w, al.ref_epoch, al.tau_ref
    camera, gx, gy, cams, taus, tm = al.camera, al.gx, al.gy, al.cams, al.taus, al.tm
    ra_c, dec_c = camera.icrs(w / 2, h / 2, tau_ref)
    drift = pointing_drift(camera, cams, taus, float(np.ravel(ra_c)[0]), float(np.ravel(dec_c)[0]))
    dmax = max((float(np.hypot(d["dx_px"], d["dy_px"])) for d in drift), default=0.0)
    (ctx.log.warning if dmax > 1.0 else ctx.log.info)(
        "[%s] ruch aparatu względem nieruchomego modelu: do %.2f px (uwzględniony w stosie)", ctx.input_path.name, dmax)
    classes = [c for c in pcfg["stack_classes"] if c in set(ph["ev"])]
    dev = stack_device(pcfg)
    if dev:
        acc = {c: TorchStackAccumulator((h, w), float(pcfg["stack_reject_sigma"]), int(pcfg["stack_warmup"]),
                                        channels=3 if c == "ev0" else 0, device=dev) for c in classes}
    else:
        acc = {c: StackAccumulator((h, w), float(pcfg["stack_reject_sigma"]), int(pcfg["stack_warmup"]),
                                   channels=3 if c == "ev0" else 0) for c in classes}
    ckpt = ctx.outdir / "process_ckpt.npz"
    start = 0
    stats: list[dict] = []
    if ckpt.exists():
        try:
            d = dict(np.load(ckpt, allow_pickle=False))
            if int(d["n_photos"]) == len(ph) and str(d["reference"]) == wcsinfo["reference"]:
                for c in classes:
                    acc[c].load(d, f"{c}_")
                start = int(d["next"])
                stats = pd.read_csv(ctx.outdir / "frame_stats_part.csv").to_dict("records") if start else []
                ctx.log.info("[%s] wznawiam stos od zdjęcia %d", ctx.input_path.name, start)
        except Exception as e:  # noqa: BLE001 — uszkodzony punkt kontrolny: od początku
            ctx.log.warning("[%s] punkt kontrolny nieczytelny (%s) — od początku", ctx.input_path.name, e)
            start, stats = 0, []
    cache = Path(pcfg["cache_dir"]) / ctx.outdir.name
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache = None
    by_name = {p.name: p for p in list_photos(ctx.input_path, ctx.cfg["input"].get("photo_extensions", [".raf"]))}
    todo = list(range(start, len(ph)))
    t0 = time.perf_counter()
    every = int(pcfg["checkpoint_every"])
    caches = [cache / f"lum_{i:05d}.npy" for i in todo] if cache is not None else None
    ctx.log.info("[%s] dekodowanie RAF w %d procesach, stos na %s", ctx.input_path.name, decode_workers(pcfg),
                 "GPU" if dev else "CPU")
    loader = load_photos([by_name[ph["file"][i]] for i in todo], pcfg, caches)
    wait = 0.0
    while True:
        tw = time.perf_counter()
        try:
            j, (lum, rgb, sat) = next(loader)
        except StopIteration:
            break
        wait += time.perf_counter() - tw
        i = todo[j]
        rgb = rgb.astype(np.float32)
        sig = robust_sigma(lum)
        c = str(ph["ev"][i])
        row = {"photo": i, "file": ph["file"][i], "ev": c, "tau_mid": float(ph["tau_mid"][i]),
               "bg_median": float(np.median(lum[::7, ::7])), "noise_mad": sig, "n_saturated": int(sat.sum())}
        if c in acc:
            px, py = tm.at(float(ph["tau_mid"][i]))
            xc, yc = px.reshape(gx.shape), py.reshape(gx.shape)
            cy, cx = gx.shape[0] // 2, gx.shape[1] // 2
            row["shift_px"] = float(np.hypot(xc[cy, cx] - gx[cy, cx], yc[cy, cx] - gy[cy, cx]))
            row["rejected_frac"] = acc[c].add_photo(lum, rgb if acc[c].rgb is not None else None, xc, yc, sig)
        stats.append(row)
        done = j + 1
        if done % every == 0 and done < len(todo):
            state = {"n_photos": np.array(len(ph)), "reference": np.array(wcsinfo["reference"]),
                     "next": np.array(i + 1)}
            for cc in classes:
                state.update({f"{cc}_{k}": v for k, v in acc[cc].state().items()})
            np.savez(ckpt, **state)
            pd.DataFrame(stats).to_csv(ctx.outdir / "frame_stats_part.csv", index=False)
            el = time.perf_counter() - t0
            ctx.log.info("[%s] stos: %d/%d zdjęć, %.1f zdj./s (czekanie na odczyt i dekodowanie %.0f%%)",
                         ctx.input_path.name, i + 1, len(ph), done / el, 100 * wait / max(el, 1e-6))
    outputs = []
    info = {"reference_photo": int(ref_epoch["frame"]), "reference_tau_s": tau_ref, "shape": [h, w],
            "cache_dir": str(cache) if cache else None, "classes": {}, "pointing_drift": drift,
            "pointing_drift_max_px": dmax}
    for c in classes:
        mean = acc[c].mean()
        hdr = fits.Header()
        hdr["NPHOTO"] = (int(acc[c].frames), "zdjecia w stosie")
        hdr["TAU_REF"] = (tau_ref, "czas zdjecia odniesienia [s]")
        hdr["EVCLASS"] = c
        name = f"stack_{c.replace('+', 'p').replace('-', 'm')}.fits"
        fits.writeto(ctx.outdir / name, mean.astype(np.float32), hdr, overwrite=True)
        outputs.append(name)
        info["classes"][c] = {"file": name, "frames": int(acc[c].frames), "n_median": float(np.median(acc[c].n)),
                              "exposure_s": float(ph.loc[ph["ev"] == c, "exposure_s"].median())}
        if acc[c].rgb is not None:
            np.save(ctx.outdir / f"stack_rgb_{c}.npy", acc[c].mean_rgb().astype(np.float32))
            outputs.append(f"stack_rgb_{c}.npy")
    pd.DataFrame(stats).to_csv(ctx.outdir / "frame_stats.csv", index=False)
    ctx.write_json("process.json", info)
    for p in (ckpt, ctx.outdir / "frame_stats_part.csv"):
        p.unlink(missing_ok=True)
    rate = len(todo) / max(time.perf_counter() - t0, 1e-6)
    ctx.log.info("[%s] stos gotowy: %s; %.1f zdj./s", ctx.input_path.name,
                 ", ".join(f"{c} {v['frames']} zdj." for c, v in info["classes"].items()), rate)
    return {"outputs": outputs + ["frame_stats.csv", "process.json"],
            "metrics": {"photos": len(ph), "photos_per_s": rate,
                        **{f"frames_{c}": v["frames"] for c, v in info["classes"].items()}}}


# ---------------------------------------------------------------- kreski (F2)

def torch_warp(lum: np.ndarray, xc: np.ndarray, yc: np.ndarray, shape: tuple[int, int], device: str):
    """Zdjęcie → układ odniesienia (tensor na ``device``, NaN poza zdjęciem); siatka jak w
    ``TorchStackAccumulator``."""
    import torch

    F = torch.nn.functional
    h, w = shape
    norm = np.stack([2 * np.asarray(xc, np.float32) / (w - 1) - 1, 2 * np.asarray(yc, np.float32) / (h - 1) - 1])
    coarse = torch.from_numpy(norm.astype(np.float32))[None].to(device)
    grid = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)[None]
    img = torch.from_numpy(np.ascontiguousarray(lum, np.float32))[None, None].to(device)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    inside = (grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1) & torch.isfinite(out)
    return torch.where(inside, out, torch.full_like(out, float("nan")))


def iter_luminance(ctx: StageContext, ph, indices: list[int], pcfg: dict) -> Iterator[tuple[int, np.ndarray]]:
    """Luminancja superpikseli zdjęć ``indices`` po kolei: z cache etapu ``process`` (lokalny dysk),
    a gdy go nie ma (nowa sesja Colaba) — dekodowanie RAF w procesach (z zapisem do cache)."""
    from .raw import list_photos

    cache = Path(pcfg["cache_dir"]) / ctx.outdir.name
    files = [cache / f"lum_{i:05d}.npy" for i in indices]
    if all(f.exists() for f in files):
        for j, f in enumerate(files):
            yield j, np.load(f).astype(np.float32)
        return
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        files = None
    by_name = {p.name: p for p in list_photos(ctx.input_path, ctx.cfg["input"].get("photo_extensions", [".raf"]))}
    for j, (lum, _, _) in load_photos([by_name[ph["file"][i]] for i in indices], pcfg, files):
        yield j, lum


def streak_reject_reason(r: dict | None, dcfg: dict) -> str | None:
    """Dlaczego kandydat nie jest kreską: brak dopasowania, za krótki, S/N z pasów kontrolnych,
    nierówny (jedna plamka zamiast kreski), końce ostrzejsze niż gwiazda (szum) albo rozmyte."""
    if r is None:
        return "bez_dopasowania"
    if r["length"] < float(dcfg["min_len_px"]):
        return "za_krótka"
    if not r["snr"] >= float(dcfg["min_snr"]):
        return "S/N"
    if not r["snr_thirds"] >= float(dcfg["min_snr_thirds"]):
        return "nierówna"
    if not float(dcfg["min_psf_px"]) <= r["psf_sigma"] <= float(dcfg["max_psf_px"]):
        return "końce"
    return None


def photo_classes(ph, pcfg: dict) -> list[str]:
    present = list(dict.fromkeys(ph["ev"]))
    return [c for c in pcfg["stack_classes"] if c in present] + [c for c in present if c not in pcfg["stack_classes"]]


@PHOTO_PIPELINE.stage("streaks", sections=("streaks.detect",), requires=("process",), rev=2, roles=("sky",))
def streaks(ctx: StageContext) -> dict:
    """Kreski na różnicy względem sąsiednich zdjęć tej samej klasy jasności (układ nieba)."""
    import time

    import pandas as pd
    import torch
    from astropy.io import fits
    from scipy.ndimage import binary_dilation

    from .photo_report import flatten
    from .streaks import (block_sigma, components, end_clipped, is_dashed, line_kernels, line_response,
                          merge_collinear, neighbor_median, refine_streak, sample_grid)

    dcfg, pcfg = ctx.cfg["streaks"]["detect"], ctx.cfg["photo"]
    ph = _photos(ctx)
    ph["m"] = ph.groupby("set").cumcount()
    al = alignment(ctx, ph)
    proc = ctx.read_json("process.json")
    dev = stack_device(pcfg) or "cpu"
    kern = line_kernels(int(dcfg["line_len_px"]), int(dcfg["n_angles"]))
    nb, shape = int(dcfg["neighbors"]), (al.h, al.w)
    det = float(dcfg["det_sigma"])
    rows, profiles, stats = [], [], []
    rejected: dict[str, int] = {}
    t0 = time.perf_counter()
    done = 0
    ctx.log.info("[%s] kreski: różnica względem %d sąsiednich zdjęć tej samej klasy, filtr %d kierunków na %s",
                 ctx.input_path.name, 2 * nb, int(dcfg["n_angles"]), "GPU" if dev != "cpu" else "CPU")
    for c in photo_classes(ph, pcfg):
        idx = sorted((int(i) for i in ph.index[ph["ev"] == c]), key=lambda i: float(ph["tau_mid"][i]))
        if len(idx) < 3:
            ctx.log.warning("[%s] klasa %s: za mało zdjęć (%d) na różnicę", ctx.input_path.name, c, len(idx))
            continue
        info = proc["classes"].get(c)
        flat_stack = flatten(fits.getdata(ctx.outdir / info["file"]).astype(np.float32)) if info else None
        relmap = mask = mask_np = None
        buf: dict[int, object] = {}
        loader = iter_luminance(ctx, ph, idx, pcfg)
        nxt = 0
        for k in range(len(idx)):
            while nxt < len(idx) and nxt <= k + nb:
                j, lum = next(loader)
                xc, yc = al.maps(float(ph["tau_mid"][idx[j]]))
                buf[j] = torch_warp(lum, xc, yc, shape, dev)
                nxt += 1
            i = idx[k]
            neigh = [buf[q] for q in range(k - nb, k + nb + 1) if q != k and q in buf]
            med, cnt = neighbor_median(torch.stack(neigh))
            D = torch.where(cnt >= 2, buf[k] - med, torch.full_like(med, float("nan")))
            sub = D[::7, ::7]
            sub = sub[torch.isfinite(sub)]
            if sub.numel() < 100:
                buf.pop(k - nb, None)
                continue
            m0 = sub.median()
            sig = float(1.4826 * (sub - m0).abs().median())
            D = D - m0
            if relmap is None:            # mapa szumu i maska gwiazd raz na klasę
                bs = block_sigma(D.cpu().numpy(), 64)
                sig_ref = float(np.nanmedian(bs))
                relmap = torch.from_numpy(bs / max(sig_ref, 1e-6)).to(dev)
                mask_np = (flat_stack > float(dcfg["star_mask_snr"]) * sig_ref if flat_stack is not None
                           else np.zeros(shape, bool))
                mask_np = binary_dilation(mask_np, iterations=int(dcfg["star_mask_grow_px"]))
                mask = torch.from_numpy(mask_np).to(dev)
                ctx.log.info("[%s] klasa %s: σ różnicy %.1f DN, maska gwiazd %.2f%% kadru", ctx.input_path.name, c,
                             sig_ref, 100 * float(mask_np.mean()))
            z = torch.where(mask | ~torch.isfinite(D), torch.zeros_like(D), D / (relmap * max(sig, 1e-6)))
            z = z.clamp(-10, 50)
            hp = int(dcfg["highpass_px"]) | 1       # zmiany tła między zdjęciami (chmury, łuna) — duża skala
            z = z - torch.nn.functional.avg_pool2d(z[None, None], hp, stride=1, padding=hp // 2,
                                                   count_include_pad=False)[0, 0]
            rmax = line_response(z, kern, dev)
            n_found = 0
            if bool((rmax >= det).any()):
                rm = rmax.cpu().numpy()
                Dn = D.cpu().numpy()
                Dn[mask_np] = np.nan
                blocked = ~np.isfinite(Dn)
                segs = sorted(merge_collinear(components(rm, dcfg), dcfg), key=lambda s: -s["peak"])
                xc, yc = al.maps(float(ph["tau_mid"][i]))
                for s in segs[: int(dcfg["max_per_photo"])]:
                    r = refine_streak(Dn, s["p0"], s["p1"], dcfg)
                    why = streak_reject_reason(r, dcfg)
                    if why:
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    u = (r["b"] - r["a"]) / max(r["length"], 1e-9)
                    xs, ys = [r["a"][0], r["b"][0]], [r["a"][1], r["b"][1]]
                    px, py = sample_grid(xc, shape, xs, ys), sample_grid(yc, shape, xs, ys)
                    sid = len(rows)
                    rows.append({
                        "streak_id": sid, "photo": i, "file": ph["file"][i], "ev": c, "set": int(ph["set"][i]),
                        "m": int(ph["m"][i]), "tau_open": float(ph["tau_open"][i]),
                        "exposure_s": float(ph["exposure_s"][i]),
                        "xa": float(xs[0]), "ya": float(ys[0]), "xb": float(xs[1]), "yb": float(ys[1]),
                        "ya_s": float(py[0]), "yb_s": float(py[1]),
                        "xa_p": float(px[0]), "ya_p": float(py[0]), "xb_p": float(px[1]), "yb_p": float(py[1]),
                        "length_px": r["length"], "angle_deg": float(np.degrees(np.arctan2(u[1], u[0])) % 180),
                        "amp": r["amp"], "psf_sigma": r["psf_sigma"], "snr": r["snr"], "noise": r["noise"],
                        "err_a": r["err_a"], "err_b": r["err_b"], "snr_thirds": r["snr_thirds"], "flux": r["flux"],
                        "clip_a": end_clipped(r["a"], -u, blocked, int(dcfg["edge_px"])),
                        "clip_b": end_clipped(r["b"], u, blocked, int(dcfg["edge_px"])),
                        "n_pieces": len(s["pieces"]),
                        "dashed": is_dashed(s["pieces"], int(dcfg["dash_min_pieces"]), float(dcfg["dash_max_cv"])),
                    })
                    profiles.append(pd.DataFrame({"streak_id": sid, "s_px": r["profile_s"], "flux": r["profile_f"]}))
                    n_found += 1
            stats.append({"photo": i, "ev": c, "sigma": sig, "n_streaks": n_found})
            buf.pop(k - nb, None)
            done += 1
            if done % 150 == 0:
                ctx.log.info("[%s] kreski: %d/%d zdjęć, %d kresek, %.1f zdj./s", ctx.input_path.name, done, len(ph),
                             len(rows), done / max(time.perf_counter() - t0, 1e-6))
    cols = ["streak_id", "photo", "file", "ev", "set", "m", "tau_open", "exposure_s", "xa", "ya", "xb", "yb", "ya_s",
            "yb_s", "xa_p", "ya_p", "xb_p", "yb_p", "length_px", "angle_deg", "amp", "psf_sigma", "snr", "noise",
            "err_a", "err_b", "snr_thirds", "flux", "clip_a", "clip_b", "n_pieces", "dashed"]
    st = pd.DataFrame(rows, columns=cols)
    st.to_parquet(ctx.outdir / "streaks.parquet", index=False)
    (pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(columns=["streak_id", "s_px", "flux"])) \
        .to_parquet(ctx.outdir / "streak_profiles.parquet", index=False)
    pd.DataFrame(stats).to_csv(ctx.outdir / "streak_stats.csv", index=False)
    rate = done / max(time.perf_counter() - t0, 1e-6)
    n_ph = int(st["photo"].nunique()) if len(st) else 0
    ctx.log.info("[%s] kreski: %d na %d zdjęciach (z %d), %.1f zdj./s; odrzuceni kandydaci: %s", ctx.input_path.name,
                 len(st), n_ph, len(ph), rate, ", ".join(f"{k} {v}" for k, v in sorted(rejected.items())) or "brak")
    return {"outputs": ["streaks.parquet", "streak_profiles.parquet", "streak_stats.csv"],
            "metrics": {"streaks": len(st), "photos_with_streaks": n_ph, "photos_per_s": rate,
                        **{f"rejected_{k}": v for k, v in rejected.items()}}}


@PHOTO_PIPELINE.stage("link", sections=("streaks.link", "photo.inter_frame_gap_s"),
                      requires=("streaks",), rev=2, roles=("sky",))
def link(ctx: StageContext) -> dict:
    """Łańcuchy kresek jednego obiektu przez kolejne zdjęcia i przerwa g między zdjęciami serii."""
    import pandas as pd

    from .streaks import fit_gap, link_chains

    lcfg, pcfg = ctx.cfg["streaks"]["link"], ctx.cfg["photo"]
    st = pd.read_parquet(ctx.outdir / "streaks.parquet")
    h = int(ctx.read_json("frames.json")["shape"][0])
    r, g0 = float(lcfg["rolling_shutter_s"]), float(pcfg["inter_frame_gap_s"])
    chains = link_chains(st, lcfg, r, h) if len(st) else []
    timing = fit_gap(st, chains, g0, r, h, lcfg)
    rows = [{"chain_id": cid, "pos": pos, "streak_id": int(st["streak_id"].iloc[k]), "orient": int(o)}
            for cid, ch in enumerate(chains) for pos, (k, o) in enumerate(ch["items"])]
    pd.DataFrame(rows, columns=["chain_id", "pos", "streak_id", "orient"]).to_parquet(ctx.outdir / "chains.parquet",
                                                                                    index=False)
    multi = [c for c in chains if len(c["items"]) >= 2]
    info = {"n_streaks": len(st), "n_chains": len(multi), "n_single": len(chains) - len(multi),
            "chain_lengths": sorted((len(c["items"]) for c in multi), reverse=True), "timing": timing}
    ctx.write_json("link.json", info)
    ctx.log.info("[%s] łańcuchy: %d (najdłuższe %s kresek), pojedyncze kreski: %d", ctx.input_path.name,
                 len(multi), info["chain_lengths"][:5], info["n_single"])
    if timing["fitted"]:
        ctx.log.info("[%s] przerwa między zdjęciami serii g = %.3f ± %.3f s (config %.3f s; %d łańcuchów, "
                     "residua końców %.2f px = %.1f ms; wspólny rozrzut startu serii %.0f ms z %d par)", ctx.input_path.name,
                     timing["g_s"], timing["sigma_s"], g0, timing["n_chains"], timing["rms_px"], timing["rms_ms"],
                     timing.get("set_jitter_ms", float("nan")), timing.get("set_jitter_pairs", 0))
    else:
        ctx.log.info("[%s] przerwa g: bez pomiaru (%s) — zostaje %.3f s z configu", ctx.input_path.name,
                     timing.get("note", "za mało danych"), g0)
    return {"outputs": ["chains.parquet", "link.json"],
            "metrics": {"chains": len(multi), "single": info["n_single"], "gap_s": timing["g_s"],
                        "gap_sigma_s": timing["sigma_s"]}}


PHOTO_PIPELINE.stage("tle", sections=("satellites",), requires=("probe",), rev=2, roles=("sky",))(_tle)


def photo_hint(n_streaks: int, omega: float, curv_arcsec: float, dashed_frac: float, icfg: dict) -> tuple[str, str]:
    """Podpowiedź klasy obiektu ze zdjęć (bez modulacji w czasie — tę daje dopiero F3)."""
    if dashed_frac >= 0.5:
        return "samolot?", "przerywana kreska o równych odstępach (światła pozycyjne)"
    if n_streaks >= 2:
        if (float(icfg["cand_min_deg_s"]) <= omega <= float(icfg["cand_max_deg_s"])
                and curv_arcsec <= float(icfg["cand_max_curv_arcsec"])):
            return "satelita?", f"łańcuch {n_streaks} kresek, {omega:.2f}°/s, tor prosty ({curv_arcsec:.0f}″)"
        return "niesklasyfikowany", f"łańcuch {n_streaks} kresek, {omega:.2f}°/s, krzywizna {curv_arcsec:.0f}″"
    if omega > float(icfg["cand_max_deg_s"]):
        return "meteor?", f"pojedyncza szybka kreska ({omega:.1f}°/s)"
    return "pojedyncza kreska", f"tylko na jednym zdjęciu ({omega:.2f}°/s; kierunek lotu nieznany)"


def timing_from_satellites(observer, catalog, matches: list, skies: dict, pts, height: int) -> dict:
    """Kontrola modelu czasu z zidentyfikowanych satelitów: dla każdego końca kreski różnica
    chwili, w której satelita (TLE) jest w tym miejscu toru, i chwili z modelu. Regresja z osobnym
    wyrazem wolnym dla toru: dt = c_tor + a·(wiersz/H) + b·m → a = poprawka odczytu migawki
    elektronicznej r, b = poprawka przerwy g."""
    X, Y, groups = [], [], []
    for m in matches:
        s = skies.get(m.track_id)
        if s is None:
            continue
        p = pts[pts["track_id"] == m.track_id].sort_values("tau")
        u = observer.topocentric([catalog.satrecs[m.cat_index]], m.delta_s + s.tau)["unit"][0]
        if not np.isfinite(u).all() or s.gc.omega_rad_s <= 0:
            continue
        d = np.angle(np.exp(1j * (s.gc.phase(s.vec) - s.gc.phase(u))))
        dt = d / s.gc.omega_rad_s
        X.append(np.column_stack([p["y_sensor"].to_numpy(float) / height, p["m"].to_numpy(float)]))
        Y.append(dt)
        groups.append(np.full(len(dt), m.track_id))
    out = {"n_tracks": len(X), "rolling_corr_s": float("nan"), "rolling_sigma_s": float("nan"),
           "gap_corr_s": float("nan"), "gap_sigma_s": float("nan"), "rms_ms": float("nan")}
    if not X:
        return out
    X, Y, groups = np.vstack(X), np.concatenate(Y), np.concatenate(groups)
    Xc, Yc = X.copy(), Y.copy()
    for gid in np.unique(groups):
        sel = groups == gid
        Xc[sel] -= X[sel].mean(axis=0)
        Yc[sel] -= Y[sel].mean()
    out["rms_ms"] = float(1000 * np.sqrt(np.mean(Yc ** 2)))
    cols = [k for k in (0, 1) if np.sum(Xc[:, k] ** 2) > (0.05 if k == 0 else 0.5)]
    if not cols or len(Y) - len(np.unique(groups)) - len(cols) < 3:
        return out
    A = Xc[:, cols]
    coef, *_ = np.linalg.lstsq(A, Yc, rcond=None)
    res = Yc - A @ coef
    dof = max(len(Y) - len(np.unique(groups)) - len(cols), 1)
    cov = np.linalg.pinv(A.T @ A) * float(np.sum(res ** 2) / dof)
    for j, k in enumerate(cols):
        key = "rolling" if k == 0 else "gap"
        out[f"{key}_corr_s"], out[f"{key}_sigma_s"] = float(coef[j]), float(np.sqrt(cov[j, j]))
    out["rms_ms"] = float(1000 * np.sqrt(np.mean(res ** 2)))
    return out


@PHOTO_PIPELINE.stage("identify", sections=("identify", "classify", "site", "time", "satellites.ephemeris_dir",
                                            "report.identified_min_confidence", "streaks.link"),
                      requires=("link", "tle"), rev=1, roles=("sky",))
def identify(ctx: StageContext) -> dict:
    """Model czasu kresek → synchronizacja zegara po satelitach (łańcuchy) → NORAD."""
    import pandas as pd

    from .astrometry import load_wcs
    from .report import CONF_RANK
    from .satellites import (SIDEREAL_DEG_S, Observer, identify_tracks, load_catalog, make_track_sky,
                             observer_offset, screen, sunlit_at, sunlit_flags, synchronize)
    from .sky import FixedCamera, field_center, radec_to_vec
    from .streaks import endpoint_rows

    icfg, pcfg, lcfg = ctx.cfg["identify"], ctx.cfg["photo"], ctx.cfg["streaks"]["link"]
    st = pd.read_parquet(ctx.outdir / "streaks.parquet")
    chn = pd.read_parquet(ctx.outdir / "chains.parquet")
    lk = ctx.read_json("link.json")
    meta = ctx.read_json("meta.json")
    wcsinfo = ctx.read_json("wcs.json")
    H, W = (int(v) for v in ctx.read_json("frames.json")["shape"])
    t0, site = prior_start(ctx), _site(ctx.cfg)
    tau_ref = float(wcsinfo["reference_tau_s"])
    camera = FixedCamera(load_wcs(ctx.outdir / wcsinfo["reference_wcs"]), tau_ref, t0, *site)
    timing = lk["timing"]
    use_g = bool(timing.get("fitted")) and float(timing["sigma_s"]) <= float(lcfg["gap_max_sigma_s"]) \
        and not timing.get("at_grid_edge")
    dg = float(timing["dg_s"]) if use_g else 0.0
    r_s = float(lcfg["rolling_shutter_s"])
    by_sid = {int(q["streak_id"]): q for q in st.to_dict("records")}

    def points(tid: int, items: list[tuple[int, int]]) -> list[dict]:
        out = []
        for sid, o in items:
            q = by_sid[sid]
            ends = endpoint_rows(q, o if o else 1, r_s, H, dg)
            ys = (q["ya_s"], q["yb_s"]) if (o if o else 1) > 0 else (q["yb_s"], q["ya_s"])
            for e, (x, y, t, ok), ysen in zip(("in", "out"), ends, ys):
                if ok:      # obcięty koniec (brzeg, maska): położenie nie odpowiada chwili
                    out.append({"track_id": tid, "streak_id": sid, "photo": int(q["photo"]), "m": int(q["m"]),
                                "end": e, "x": x, "y": y, "tau": t, "y_sensor": float(ysen)})
        return out

    tracks: dict[int, dict] = {}
    P, P_rev = [], {}
    for cid, g in chn.groupby("chain_id"):
        g = g.sort_values("pos")
        items = [(int(s), int(o)) for s, o in zip(g["streak_id"], g["orient"])]
        tid = int(cid) + 1
        tracks[tid] = {"items": items, "single": len(items) == 1}
        P += points(tid, items)
        if len(items) == 1:
            P_rev[tid] = points(tid, [(items[0][0], -1)])
    cols = ["track_id", "streak_id", "photo", "m", "end", "x", "y", "tau", "y_sensor"]
    pts = pd.DataFrame(P, columns=cols)
    rev = pd.DataFrame([p for v in P_rev.values() for p in v], columns=cols)

    def sky_points(df):
        if not len(df):
            return df.assign(ra=[], dec=[]), np.empty((0, 3))
        ra, dec = camera.icrs(df["x"].to_numpy(), df["y"].to_numpy(), np.full(len(df), tau_ref))
        return df.assign(ra=ra, dec=dec), camera.apparent_vectors(ra, dec, df["tau"].to_numpy())

    pts = pts.sort_values(["track_id", "tau"]).reset_index(drop=True)
    pts, vec = sky_points(pts)
    rev = rev.sort_values(["track_id", "tau"]).reset_index(drop=True)
    rev, vec_rev = sky_points(rev)

    def skies_of(df, v) -> dict:
        out = {}
        for tid, idx in df.groupby("track_id").indices.items():
            if len(idx) >= 2:
                out[int(tid)] = make_track_sky(int(tid), df["tau"].to_numpy()[idx], v[idx], df["ra"].to_numpy()[idx],
                                               df["dec"].to_numpy()[idx])
        return out

    skies, skies_rev = skies_of(pts, vec), skies_of(rev, vec_rev)
    chain_skies = [s for tid, s in skies.items() if not tracks[tid]["single"]]
    single_skies = [s for tid, s in skies.items() if tracks[tid]["single"]]

    catalog = load_catalog([(ctx.outdir / "gp_elements.csv", "snapshot")], t0)
    observer = Observer(*site, t0)
    _, radius = field_center(camera.wcs, (H, W))
    dur = float(meta["session"]["duration_s"])

    def grid_for(window_s: float):
        step = float(icfg["screen_step_s"])
        offsets = np.arange(-window_s, dur + window_s + step, step)
        ra_c, dec_c = camera.icrs(np.full(len(offsets), W / 2), np.full(len(offsets), H / 2), offsets)
        ctx.log.info("[%s] przesiew katalogu: ±%.0f s, %d chwil × %d obiektów", ctx.input_path.name, window_s,
                     len(offsets), len(catalog))
        return screen(observer, catalog, offsets, radec_to_vec(ra_c, dec_c), radius, icfg,
                      extra_margin_deg=window_s * SIDEREAL_DEG_S)

    sync, grid = synchronize(observer, catalog, chain_skies, grid_for, float(ctx.cfg["time"]["prior_sigma_s"]),
                             float(ctx.cfg["time"]["sync_search_s"]), icfg)
    delta = sync.delta_s
    ctx.log.info("[%s] poprawka zegara Δ = %+.3f ± %.3f s (%s, %s)", ctx.input_path.name, delta, sync.sigma_s,
                 sync.confidence, sync.method)
    offset = None
    try:
        offset = observer_offset(observer, catalog, sync.members, skies) if sync.synced else None
    except Exception as e:  # noqa: BLE001 — diagnostyka nie może zatrzymać identyfikacji
        ctx.log.warning("[%s] przesunięcie obserwatora: %s", ctx.input_path.name, e)
    matches = identify_tracks(observer, catalog, chain_skies, grid, delta, icfg) if grid is not None else {}
    # pojedyncze kreski: oba kierunki lotu, wygrywa lepsze dopasowanie
    flipped = set()
    if grid is not None:
        for s in single_skies:
            a = identify_tracks(observer, catalog, [s], grid, delta, icfg).get(s.track_id) or []
            s_rev = skies_rev.get(s.track_id)
            b = identify_tracks(observer, catalog, [s_rev], grid, delta, icfg).get(s.track_id) if s_rev else None
            b = b or []
            if b and (not a or b[0].rms_deg < a[0].rms_deg):
                matches[s.track_id] = b
                flipped.add(s.track_id)
                skies[s.track_id] = s_rev
            else:
                matches[s.track_id] = a
    if flipped:
        pts = pd.concat([pts[~pts["track_id"].isin(flipped)], rev[rev["track_id"].isin(flipped)]],
                        ignore_index=True).sort_values(["track_id", "tau"]).reset_index(drop=True)
    bests = [m[0] for m in matches.values() if m]
    for m in bests:
        s = skies[m.track_id]
        sel = np.linspace(0, len(s.tau) - 1, min(len(s.tau), 60)).astype(int)
        u = observer.topocentric([catalog.satrecs[m.cat_index]], m.delta_s + s.tau[sel])["unit"][0]
        pra, pdec = camera.astrometric_radec(u, s.tau[sel], m.delta_s)
        m.pred_radec = np.column_stack([pra, pdec]).tolist()
    tau_mid = {s.track_id: s.tau_mid for s in skies.values()}
    eph = sunlit_flags(catalog, bests, observer, ctx.cfg["satellites"]["ephemeris_dir"], tau_mid)
    sunlit_flags(catalog, sync.members + ([sync.reference] if sync.reference else []), observer, None, tau_mid,
                 eph=eph)
    min_rank = CONF_RANK[str(ctx.cfg["report"]["identified_min_confidence"])]
    good = [m for m in bests if CONF_RANK.get(m.confidence, 0) >= max(min_rank, 2) and not tracks[m.track_id]["single"]]
    check = timing_from_satellites(observer, catalog, good, skies, pts, H)
    if check["n_tracks"]:
        ctx.log.info("[%s] kontrola czasu z satelitów (%d torów): residua końców %.1f ms; poprawka odczytu migawki "
                     "%+.3f ± %.3f s, poprawka przerwy g %+.3f ± %.3f s", ctx.input_path.name, check["n_tracks"],
                     check["rms_ms"], check["rolling_corr_s"], check["rolling_sigma_s"], check["gap_corr_s"],
                     check["gap_sigma_s"])

    # tabela końcowa
    cam_sync = camera.with_reference(t0 + timedelta(seconds=delta))
    rows = []
    for tid, tr in sorted(tracks.items()):
        s = skies.get(tid)
        if s is None:
            continue
        p = pts[pts["track_id"] == tid]
        sids = [sid for sid, _ in tr["items"]]
        q = st[st["streak_id"].isin(sids)]
        if tr["single"]:
            omega = float(np.degrees(np.arccos(np.clip(s.vec[0] @ s.vec[-1], -1, 1)))) / max(s.dur_s, 1e-6)
        else:
            omega = s.gc.omega_deg_s
        curv = 0.0 if tr["single"] else s.gc.cross_rms_arcsec
        hint, reason = photo_hint(len(sids), omega, curv, float(q["dashed"].mean()), icfg)
        best = (matches.get(tid) or [None])[0]
        kind = "sat" if best is not None and CONF_RANK.get(best.confidence, 0) >= min_rank else "unid"
        az, alt = cam_sync.radec_altaz(s.ra[[0, -1]], s.dec[[0, -1]], s.tau[[0, -1]])
        rows.append({
            "track_id": tid, "n": len(p), "n_streaks": len(sids), "photo0": int(q["photo"].min()),
            "photo1": int(q["photo"].max()), "kind": kind, "class_hint": hint, "class_reason": reason,
            "dir_ambiguous": bool(tr["single"] and best is None), "dashed_frac": float(q["dashed"].mean()),
            "tau0": float(s.tau[0]), "tau1": float(s.tau[-1]), "tau_mid": s.tau_mid, "dur_s": s.dur_s,
            "utc_start": (t0 + timedelta(seconds=float(s.tau[0]) + delta)).isoformat(),
            "utc_end": (t0 + timedelta(seconds=float(s.tau[-1]) + delta)).isoformat(),
            "omega_deg_s": omega, "curv_arcsec": curv,
            "ra0": float(s.ra[0]), "dec0": float(s.dec[0]), "ra1": float(s.ra[-1]), "dec1": float(s.dec[-1]),
            "az0": float(az[0]), "alt0": float(alt[0]), "az1": float(az[1]), "alt1": float(alt[1]),
            "x0": float(p["x"].iloc[0]), "y0": float(p["y"].iloc[0]), "x1": float(p["x"].iloc[-1]),
            "y1": float(p["y"].iloc[-1]), "peak_snr_median": float(q["snr"].median()),
            "amp_median": float(q["amp"].median()), "f_power": 0.0,
            "norad": best.norad if best else None, "sat_name": best.name if best else None,
            "confidence": best.confidence if best else None, "match_reason": best.reason if best else None,
            "sunlit": best.sunlit if best else None,
        })
    final = pd.DataFrame(rows)
    final.to_parquet(ctx.outdir / "tracks_final.parquet", index=False)
    final.to_csv(ctx.outdir / "tracks_final.csv", index=False)
    pts = pts.assign(frame=pts.groupby("track_id").cumcount())
    pts[["track_id", "frame", "streak_id", "photo", "m", "end", "x", "y", "tau", "y_sensor", "ra", "dec"]] \
        .to_parquet(ctx.outdir / "track_sky.parquet", index=False)
    ctx.write_json("identifications.json", {str(k): [m.to_dict() | ({"pred_radec": getattr(m, "pred_radec", None)}
                                                                     if i == 0 else {}) for i, m in enumerate(v)]
                                            for k, v in matches.items()})

    # satelity przewidziane w kadrze (w chwilach otwartej migawki i między nimi)
    pred_rows = []
    if grid is not None and len(grid.idx):
        step = float(icfg["sample_step_s"])
        taus = np.arange(0, dur + step, step)
        detected = {m.norad for m in bests if CONF_RANK.get(m.confidence, 0) >= min_rank}
        topo = observer.topocentric([catalog.satrecs[i] for i in grid.idx], delta + taus)
        S = len(grid.idx)
        flat = topo["unit"].reshape(-1, 3)
        tt = np.tile(taus, S)
        ok = np.isfinite(flat).all(axis=1)
        px = np.full(len(flat), np.nan)
        py = np.full(len(flat), np.nan)
        if ok.any():
            pra, pdec = camera.astrometric_radec(flat[ok], tt[ok], delta)
            px[ok], py[ok] = camera.pixel(pra, pdec, tt[ok])
        inside = ((px >= 0) & (px < W) & (py >= 0) & (py < H)).reshape(S, len(taus))
        for j in np.flatnonzero(inside.any(axis=1)):
            k = np.flatnonzero(inside[j])
            i = int(grid.idx[j])
            mid = k[len(k) // 2]
            pred_rows.append({"norad": int(catalog.norad[i]), "name": catalog.name[i],
                              "utc_in": (t0 + timedelta(seconds=delta + taus[k[0]])).isoformat(),
                              "utc_out": (t0 + timedelta(seconds=delta + taus[k[-1]])).isoformat(),
                              "range_km": float(topo["dist_km"][j, mid]),
                              "sunlit": sunlit_at(catalog, [i], [delta + taus[mid]], observer, eph)[0],
                              "detected": int(catalog.norad[i]) in detected})
    pd.DataFrame(pred_rows, columns=["norad", "name", "utc_in", "utc_out", "range_km", "sunlit", "detected"]) \
        .to_csv(ctx.outdir / "fov_predicted.csv", index=False)

    gp = ctx.read_json("gp_source.json")
    age = gp.get("age_days_median")
    note = (f"{gp['n_objects']} obiektów ({', '.join(sorted({f['source'] for f in gp['files']}))}); mediana wieku "
            f"elementów {f'{age:.2f} d' if age is not None else '–'}"
            + ("" if gp["spacetrack"] else "; bez Space-Track: niepełne człony rakiet i śmieci"))
    photo_timing = {"cadence": meta["session"]["cadence"], "gap": timing, "gap_used_s": float(timing["g0_s"]) + dg,
                    "gap_from_geometry": use_g, "rolling_shutter_s": r_s, "satellite_check": check}
    ctx.write_json("time_sync.json", {**sync.to_dict(), "start_utc_prior": t0.isoformat(),
                                      "start_utc_synced": (t0 + timedelta(seconds=delta)).isoformat(),
                                      "catalog_note": note, "observer_offset": offset, "photo_timing": photo_timing,
                                      "site": dict(zip(("lat_deg", "lon_deg", "elevation_m"), site))})
    n_sat = int((final["kind"] == "sat").sum()) if len(final) else 0
    ctx.log.info("[%s] zidentyfikowane satelity: %d / %d obiektów (łańcuchy %d, pojedyncze kreski %d); "
                 "przewidziane w kadrze: %d", ctx.input_path.name, n_sat, len(final), len(chain_skies),
                 len(single_skies), len(pred_rows))
    return {"outputs": ["tracks_final.parquet", "tracks_final.csv", "track_sky.parquet", "identifications.json",
                        "fov_predicted.csv", "time_sync.json"],
            "metrics": {"delta_s": delta, "sigma_s": sync.sigma_s, "sync_confidence": sync.confidence,
                        "synced": sync.synced, "tracks": len(final), "satellites": n_sat,
                        "predicted_in_fov": len(pred_rows), "gap_s": photo_timing["gap_used_s"],
                        "rolling_corr_s": check["rolling_corr_s"]}}


@PHOTO_PIPELINE.stage("adsb", sections=("adsb", "site"), requires=("identify",), rev=1, roles=("sky",))
def adsb(ctx: StageContext) -> dict:
    """Niezidentyfikowane obiekty vs trasy samolotów z historii ADS-B (adsb.lol)."""
    import pandas as pd

    from . import adsb as A
    from .astrometry import load_wcs
    from .sky import FixedCamera

    acfg, site = ctx.cfg["adsb"], _site(ctx.cfg)
    sync, meta = ctx.read_json("time_sync.json"), ctx.read_json("meta.json")
    start = datetime.fromisoformat(sync["start_utc_prior"]) + timedelta(seconds=float(sync["delta_s"]))
    end = start + timedelta(seconds=float(meta["session"]["duration_s"]))
    info: dict = {"enabled": str(acfg.get("enabled", "auto")) != "off", "source": "adsb.lol (ODbL 1.0)",
                  "synced": bool(sync.get("synced")), "sources": []}
    matches: list[dict] = []
    if info["enabled"]:
        try:
            parts = []
            for day in A.recording_days(start - timedelta(minutes=5), end + timedelta(minutes=5)):
                df, src = A.day_points(acfg, day, site)
                parts.append(df)
                info["sources"].append(src)
            points = pd.concat(parts, ignore_index=True)
            lo, hi = start.timestamp() - 300, end.timestamp() + 300
            points = points[(points["t"] >= lo) & (points["t"] <= hi)]
            info["n_aircraft"] = int(points["icao"].nunique())
            wcsinfo = ctx.read_json("wcs.json")
            cam = FixedCamera(load_wcs(ctx.outdir / wcsinfo["reference_wcs"]), wcsinfo["reference_tau_s"], start, *site)
            final = pd.read_parquet(ctx.outdir / "tracks_final.parquet")
            pts = pd.read_parquet(ctx.outdir / "track_sky.parquet")
            trs = []
            for tid in (final.loc[final["kind"] != "sat", "track_id"].astype(int) if len(final) else []):
                p = pts[pts["track_id"] == tid].sort_values("tau")
                p = p.iloc[np.unique(np.linspace(0, len(p) - 1, min(len(p), 20)).astype(int))]
                az, alt = cam.radec_altaz(p["ra"].to_numpy(), p["dec"].to_numpy(), p["tau"].to_numpy())
                trs.append({"track_id": tid, "times": start.timestamp() + p["tau"].to_numpy(float),
                            "enu": A.altaz_to_enu(az, alt)})
            matches = A.match_tracks(trs, points, site, acfg) if len(points) else []
        except Exception as e:  # noqa: BLE001 — brak ADS-B nie blokuje raportu
            info["error"] = str(e)
            ctx.log.warning("[%s] ADS-B niedostępne: %s", ctx.input_path.name, e)
    cols = ["track_id", "icao", "reg", "type", "callsign", "sep_deg", "range_km", "alt_m"]
    pd.DataFrame(matches, columns=cols).to_csv(ctx.outdir / "adsb_matches.csv", index=False)
    info["n_matched"] = len(matches)
    ctx.write_json("adsb_source.json", info)
    if info["enabled"] and "error" not in info:
        ctx.log.info("[%s] ADS-B: %d samolotów w promieniu %.0f km, obiekty-samoloty: %s", ctx.input_path.name,
                     info.get("n_aircraft", 0), float(acfg["radius_km"]),
                     ", ".join(f"#{m['track_id']} {m['reg'] or m['icao']} ({m['alt_m']:.0f} m)" for m in matches) or "brak")
    return {"outputs": ["adsb_matches.csv", "adsb_source.json"],
            "metrics": {"aircraft_nearby": info.get("n_aircraft"), "aircraft_tracks": len(matches),
                        "adsb_error": "error" in info}}


@PHOTO_PIPELINE.stage("report", sections=("report", "iod", "classify.periodic_min_power"),
                      requires=("process", "adsb"), rev=3, roles=("sky",))
def report(ctx: StageContext) -> dict:
    """Mapa na głębokim stosie z torami, przebieg sesji, czas z satelitów, obiekty, IOD."""
    import pandas as pd

    from . import iod
    from .photo_report import build

    outputs = build(ctx.outdir, ctx.input_path, ctx.cfg)
    icfg = ctx.cfg["iod"]
    final = pd.read_parquet(ctx.outdir / "tracks_final.parquet")
    air = ctx.outdir / "adsb_matches.csv"
    if air.exists() and "track_id" in final:          # samoloty z ADS-B nie idą do zgłoszeń satelitarnych
        final = final[~final["track_id"].isin(pd.read_csv(air)["track_id"])]
    if "dir_ambiguous" in final:                      # kierunek lotu nieznany → czasy końców niepewne
        final = final[~final["dir_ambiguous"].astype(bool)]
    path, n = iod.export(ctx.outdir / "report", final, pd.read_parquet(ctx.outdir / "track_sky.parquet"),
                         ctx.read_json("time_sync.json"), ctx.read_json("wcs.json"),
                         ctx.read_json("identifications.json"), icfg, ctx.cfg["classify"])
    outputs.append(path.relative_to(ctx.outdir).as_posix())
    ctx.log.info("[%s] raport: %s; IOD: %d pozycji%s", ctx.input_path.name, ", ".join(outputs), n,
                 "  (numer stacji 9999 = nieprzydzielony: ustaw iod.station przed wysłaniem)"
                 if int(icfg["station"]) == 9999 else "")
    return {"outputs": outputs, "metrics": {"pdfs": len([o for o in outputs if o.endswith(".pdf")]),
                                            "iod_positions": n}}
