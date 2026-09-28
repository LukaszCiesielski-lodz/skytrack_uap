"""Plate solving stacków epok przez astrometry.net (``solve-field``), walidacja zgodności epok,
rzeczywiste pole widzenia (rozstrzyga crop 4K).

Indeksy serii 4100 (Tycho-2) 4110–4119 pokrywają pola ~1–30°; pobierane raz do cache na
Drive i sprawdzane sumą MD5. ``solve-field`` nie ma opcji katalogu indeksów, więc
generujemy własny plik konfiguracyjny z ``add_path``.
"""
from __future__ import annotations

import hashlib
import logging
import math
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("skyhunt")


def ensure_index(acfg: dict, opener: Callable = urllib.request.urlopen) -> Path:
    """Pobiera brakujące indeksy i sprawdza MD5. Zwraca katalog indeksów."""
    from .satellites import FetchError, fetch_url

    d = Path(acfg["index_dir"])
    d.mkdir(parents=True, exist_ok=True)
    base = str(acfg["index_url"]).rstrip("/")
    names = [f"index-{int(n)}.fits" for n in acfg["index_series"]]
    missing = [n for n in names if not (d / n).exists()]
    if not missing:
        return d
    sums_path = d / "md5sums.txt"
    if not sums_path.exists():
        try:
            sums_path.write_bytes(fetch_url(f"{base}/md5sums.txt", opener))
        except FetchError as e:
            log.warning("brak md5sums.txt (%s) — indeksy bez weryfikacji MD5", e)
    sums = {}
    for line in (sums_path.read_text().splitlines() if sums_path.exists() else []):
        parts = line.split()
        if len(parts) == 2:
            sums[Path(parts[1]).name] = parts[0]
    for n in missing:
        data = fetch_url(f"{base}/{n}", opener, timeout=600)
        if n in sums and hashlib.md5(data).hexdigest() != sums[n]:
            raise RuntimeError(f"{n}: niezgodna suma MD5")
        (d / n).write_bytes(data)
        log.info("indeks %s: %.1f MB", n, len(data) / 1e6)
    return d


def write_solver_config(index_dir: Path, path: Path) -> Path:
    path.write_text(f"inparallel\nadd_path {index_dir}\nautoindex\n", encoding="utf-8")
    return path


def solve_command(acfg: dict, image: Path, outdir: Path, base: str, config: Path,
                  hint: tuple[float, float] | None, downsample: int) -> list[str]:
    cmd = [str(acfg.get("solve_field", "solve-field")), "--config", str(config), "--overwrite", "--no-plots",
           "--new-fits", "none", "--crpix-center",
           "--scale-units", "degwidth", "--scale-low", str(acfg["scale_low_deg"]),
           "--scale-high", str(acfg["scale_high_deg"]),
           "--tweak-order", str(acfg["tweak_order"]), "--downsample", str(downsample),
           "--cpulimit", str(acfg["cpulimit_s"]), "--objs", str(acfg["max_objs"]),
           "--dir", str(outdir), "--out", base]
    if hint is not None:
        cmd += ["--ra", f"{hint[0]:.5f}", "--dec", f"{hint[1]:.5f}", "--radius", str(acfg["hint_radius_deg"])]
    return cmd + [str(image)]


def solve_epoch(acfg: dict, image: Path, outdir: Path, config: Path, hint) -> Path | None:
    """Zwraca ścieżkę .wcs albo None (po próbie z ``downsample`` i ``retry_downsample``)."""
    base = image.stem
    for ds in dict.fromkeys([int(acfg["downsample"]), int(acfg["retry_downsample"])]):
        cmd = solve_command(acfg, image, outdir, base, config, hint, ds)
        log.info("solve-field: %s (downsample %d)", image.name, ds)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=float(acfg["cpulimit_s"]) + 120)
        wcs = outdir / f"{base}.wcs"
        if (outdir / f"{base}.solved").exists() and wcs.exists():
            return wcs
        log.warning("solve-field bez rozwiązania (%s): %s", image.name, proc.stdout[-300:].strip())
    return None


def solver_available(acfg: dict) -> bool:
    return shutil.which(str(acfg.get("solve_field", "solve-field"))) is not None


def load_wcs(path: Path):
    from astropy.io import fits
    from astropy.wcs import WCS

    return WCS(fits.getheader(path))


def corr_residuals(corr_path: Path) -> dict:
    """Dopasowane gwiazdy z pliku .corr: liczba i RMS [px] (pozycja zmierzona vs katalog)."""
    from astropy.io import fits

    with fits.open(corr_path) as h:
        t = h[1].data
        d = np.hypot(t["field_x"] - t["index_x"], t["field_y"] - t["index_y"])
    return {"n_matched": int(len(d)), "rms_px": float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")}


def fov_from_wcs(wcs, shape: tuple[int, int]) -> dict:
    """Pole widzenia [°] (środki krawędzi) i skala [″/px] w środku kadru."""
    from .sky import angle_deg, radec_to_vec

    H, W = shape
    xs = np.array([0, W - 1, W / 2, W / 2, W / 2, W / 2 + 1])
    ys = np.array([H / 2, H / 2, 0, H - 1, H / 2, H / 2])
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    v = radec_to_vec(ra, dec)
    return {"fov_w_deg": float(angle_deg(v[0], v[1])), "fov_h_deg": float(angle_deg(v[2], v[3])),
            "scale_arcsec_px": float(angle_deg(v[4], v[5])) * 3600,
            "center_ra_deg": float(ra[4]), "center_dec_deg": float(dec[4])}


def epoch_agreement(cameras: list, shape: tuple[int, int], taus: list[float], ref: int = 0) -> float:
    """RMS [px] rozbieżności epok: siatka pikseli → ICRS (epoka k) → piksel w epoce ref,
    z uwzględnieniem obrotu nieba między epokami. Duża wartość = kamera się ruszyła."""
    H, W = shape
    gx, gy = np.meshgrid(np.linspace(0.05 * W, 0.95 * W, 32), np.linspace(0.05 * H, 0.95 * H, 18))
    gx, gy = gx.ravel(), gy.ravel()
    errs = []
    for k, cam in enumerate(cameras):
        if k == ref:
            continue
        ra, dec = cam.icrs(gx, gy, taus[k])
        x, y = cameras[ref].pixel(ra, dec, taus[k])
        errs.append(np.hypot(x - gx, y - gy))
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else 0.0


def crop_verdict(fov_w_deg: float, cfg_camera: dict) -> dict:
    nominal = float(cfg_camera["nominal_hfov_deg"])
    ratio = fov_w_deg / nominal
    focal = float(cfg_camera["sensor_width_mm"]) / (2 * math.tan(math.radians(fov_w_deg / 2)))
    return {"nominal_hfov_deg": nominal, "ratio": ratio,
            "effective_focal_mm_full_width": focal,
            "verdict": "bez cropu" if abs(ratio - 1) < 0.05 else f"crop ~{1 / ratio:.2f}×"}
