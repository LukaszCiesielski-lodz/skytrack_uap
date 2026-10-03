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
from typing import Iterator

import numpy as np

from .pipeline import PHOTO_PIPELINE, StageContext
from .stages import _site, prior_start, solve_epochs

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


@PHOTO_PIPELINE.stage("process", sections=("photo", "site"), requires=("astrometry",), rev=2, roles=("sky",))
def process(ctx: StageContext) -> dict:
    """Wszystkie zdjęcia: superpiksele, cache luminancji (lokalnie), statystyki, stosy per klasa
    jasności wyrównane do zdjęcia odniesienia astrometrii (model nieruchomej kamery)."""
    import time

    import pandas as pd
    from astropy.io import fits

    from .astrometry import load_wcs
    from .raw import list_photos
    from .sky import FixedCamera

    pcfg = ctx.cfg["photo"]
    ph = _photos(ctx)
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
    ra_c, dec_c = camera.icrs(w / 2, h / 2, tau_ref)
    drift = pointing_drift(camera, cams, taus, float(np.ravel(ra_c)[0]), float(np.ravel(dec_c)[0]))
    dmax = max((float(np.hypot(d["dx_px"], d["dy_px"])) for d in drift), default=0.0)
    (ctx.log.warning if dmax > 1.0 else ctx.log.info)(
        "[%s] ruch aparatu względem nieruchomego modelu: do %.2f px (uwzględniony w stosie)", ctx.input_path.name, dmax)
    classes = [c for c in pcfg["stack_classes"] if c in set(ph["ev"])]
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
    ctx.log.info("[%s] dekodowanie RAF w %d procesach", ctx.input_path.name, decode_workers(pcfg))
    for j, (lum, rgb, sat) in load_photos([by_name[ph["file"][i]] for i in todo], pcfg, caches):
        i = todo[j]
        rgb = rgb.astype(np.float32)
        sig = robust_sigma(lum)
        c = str(ph["ev"][i])
        row = {"photo": i, "file": ph["file"][i], "ev": c, "tau_mid": float(ph["tau_mid"][i]),
               "bg_median": float(np.median(lum[::7, ::7])), "noise_mad": sig, "n_saturated": int(sat.sum())}
        if c in acc:
            px, py = pointing.coarse(float(ph["tau_mid"][i]))
            xmap = upsample_map(np.asarray(px, float).reshape(gx.shape), (h, w))
            ymap = upsample_map(np.asarray(py, float).reshape(gx.shape), (h, w))
            row["shift_px"] = float(np.hypot(xmap[h // 2, w // 2] - w // 2, ymap[h // 2, w // 2] - h // 2))
            aligned = warp_to_reference(lum, xmap, ymap)
            rgb_al = (np.stack([warp_to_reference(rgb[..., k], xmap, ymap) for k in range(3)], axis=-1)
                      if acc[c].rgb is not None else None)
            ok = acc[c].add(aligned, sig, rgb_al)
            row["rejected_frac"] = float(1 - ok.mean())
        stats.append(row)
        done = j + 1
        if done % every == 0 and done < len(todo):
            state = {"n_photos": np.array(len(ph)), "reference": np.array(wcsinfo["reference"]),
                     "next": np.array(i + 1)}
            for cc in classes:
                state.update({f"{cc}_{k}": v for k, v in acc[cc].state().items()})
            np.savez(ckpt, **state)
            pd.DataFrame(stats).to_csv(ctx.outdir / "frame_stats_part.csv", index=False)
            ctx.log.info("[%s] stos: %d/%d zdjęć, %.1f zdj./s", ctx.input_path.name, i + 1, len(ph),
                         done / (time.perf_counter() - t0))
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


@PHOTO_PIPELINE.stage("report", sections=("report",), requires=("process",), rev=1, roles=("sky",))
def report(ctx: StageContext) -> dict:
    """F1: mapa na głębokim stosie i przebieg sesji (satelity w F2)."""
    from .photo_report import build

    outputs = build(ctx.outdir, ctx.input_path, ctx.cfg)
    ctx.log.info("[%s] raport: %s", ctx.input_path.name, ", ".join(outputs))
    return {"outputs": outputs, "metrics": {"pdfs": len(outputs)}}
