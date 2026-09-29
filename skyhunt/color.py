"""Kolor torów: odczyt RGB tylko dla klatek z torami, kalibracja na gwiazdach (B−V z katalogu),
pomiar koloru obiektu klatka po klatce i podpowiedź.

Co da się powiedzieć z kamery RGB (3 szerokie pasma, nie widmo):
- równoważny wskaźnik barwy ``bv_eq`` i temperatura ``T_eq`` — położenie na „linii gwiazd”
  wykresu log(R/G) × log(B/G), wyznaczonej z gwiazd w kadrze (to usuwa balans bieli, symulację
  filmu i wzmocnienia kanałów);
- ``green_excess`` — odległość od tej linii w stronę zieleni [dex]: gwiazdy (ciała ~czarne)
  leżą na linii, a emisja liniowa Mg 517 nm / O 557,7 nm (meteory) albo zielone światło
  pozycyjne samolotu odstaje;
- zmiana koloru w czasie (rozbłyski, głowa/ślad meteoru, migające światła).
Skład meteoru w sensie proporcji linii Na/Mg/Fe wymaga siatki dyfrakcyjnej (widma).

Pomiar obiektu: różnica klatki f i średniej z klatek f±k, w których obiekt jest już dalej niż
``diff_min_sep_px`` — gwiazdy (dryf 0,4 px/s) i łuna odejmują się, zostaje obiekt.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .metadata import VideoMeta

log = logging.getLogger("skyhunt")
LOG10E = 1.0 / math.log(10)


# ---------------------------------------------------------------- odczyt RGB

def crop_box(xs, ys, half: float, width: int, height: int) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) z parzystymi offsetami (chroma 4:2:0) obejmujące punkty ± ``half``."""
    x0 = max(0, int(math.floor(float(np.min(xs)) - half)) // 2 * 2)
    y0 = max(0, int(math.floor(float(np.min(ys)) - half)) // 2 * 2)
    x1 = min(width, int(math.ceil(float(np.max(xs)) + half)) + 1)
    y1 = min(height, int(math.ceil(float(np.max(ys)) + half)) + 1)
    w = max(2, (x1 - x0) // 2 * 2)
    h = max(2, (y1 - y0) // 2 * 2)
    return x0, y0, min(w, (width - x0) // 2 * 2), min(h, (height - y0) // 2 * 2)


_HW_OK: dict[str, bool] = {}


def _hwaccel_ok(ffmpeg: str, video: Path) -> bool:
    if ffmpeg not in _HW_OK:
        r = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-hwaccel", "cuda", "-i", str(video),
                            "-frames:v", "1", "-f", "null", "-"], capture_output=True)
        _HW_OK[ffmpeg] = r.returncode == 0
    return _HW_OK[ffmpeg]


def read_rgb(video: Path, meta: VideoMeta, start: int, n: int, crop: tuple[int, int, int, int],
             ccfg: dict) -> Iterator[tuple[int, np.ndarray]]:
    """Klatki ``start … start+n−1`` jako (indeks, RGB [h, w, 3] float32 w 0…1, bez linearyzacji).
    Strumieniowo: w pamięci jest jedna klatka wycinka."""
    start, n = max(0, int(start)), int(n)
    n = min(n, meta.n_frames - start)
    if n <= 0:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and str(ccfg.get("reader", "auto")) != "pyav":
        yield from _rgb_ffmpeg(ffmpeg, video, meta, start, n, crop, ccfg)
    else:
        yield from _rgb_pyav(video, meta, start, n, crop)


def _rgb_ffmpeg(ffmpeg, video, meta, start, n, crop, ccfg):
    x0, y0, w, h = crop
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    hw = str(ccfg.get("ffmpeg_hwaccel", "auto"))
    if hw == "on" or (hw == "auto" and _hwaccel_ok(ffmpeg, video)):
        cmd += ["-hwaccel", "cuda"]
    if start > 0:   # dokładny seek jak w clips: pierwsza klatka o znaczniku ≥ (start − ½)/fps
        cmd += ["-ss", f"{(start - 0.5) / meta.fps:.6f}"]
    vf = f"crop={w}:{h}:{x0}:{y0}"
    matrix = str(ccfg.get("yuv_matrix", "auto"))
    if matrix != "auto":
        vf += f",scale=in_color_matrix={matrix}:out_range=full"
    cmd += ["-i", str(video), "-map", "0:v:0", "-frames:v", str(n), "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"]
    size = w * h * 3 * 2
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got = 0
    try:
        for k in range(n):
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            got += 1
            yield start + k, np.frombuffer(buf, "<u2").reshape(h, w, 3).astype(np.float32) / 65535.0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
    if got == 0 and rc not in (0, -9):
        raise RuntimeError(f"ffmpeg (RGB) nie odczytał klatek {start}…: {err[-300:]}")


def _rgb_pyav(video, meta, start, n, crop):
    import av

    x0, y0, w, h = crop
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        t0 = float(s.start_time or 0) * tb
        c.seek(int((start / meta.fps + t0) / tb), stream=s, backward=True)
        for fr in c.decode(s):
            idx = int(round((float(fr.pts) * tb - t0) * meta.fps))
            if idx < start:
                continue
            if idx >= start + n:
                break
            rgb = fr.to_ndarray(format="rgb48le")[y0:y0 + h, x0:x0 + w]
            yield idx, rgb.astype(np.float32) / 65535.0


def linearize(v: np.ndarray, ccfg: dict) -> np.ndarray:
    """Odwrotność krzywej tonalnej wideo: ``transfer`` = bt709 | linear | liczba (gamma)."""
    tr = ccfg.get("transfer", "bt709")
    v = np.clip(np.asarray(v, np.float32), 0.0, 1.0)
    if tr == "linear":
        return v
    if tr == "bt709":
        return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45)).astype(np.float32)
    return (v ** float(tr)).astype(np.float32)


def delinearize(v: np.ndarray, ccfg: dict) -> np.ndarray:
    tr = ccfg.get("transfer", "bt709")
    v = np.clip(np.asarray(v, np.float32), 0.0, 1.0)
    if tr == "linear":
        return v
    if tr == "bt709":
        return np.where(v < 0.018, 4.5 * v, 1.099 * v ** 0.45 - 0.099).astype(np.float32)
    return (v ** (1.0 / float(tr))).astype(np.float32)


def is_monochrome(rgb: np.ndarray, tol: float = 2e-3) -> bool:
    """Czarno-białe nagranie (symulacja Acros/Monochrome): U = V = 128, więc R = G = B z dokładnością
    do zaokrągleń nawet na gwiazdach; w kolorowym kadrze gwiazdy i szum chromy dają odchyłki."""
    dev = np.abs(rgb - rgb.mean(axis=-1, keepdims=True)).max(axis=-1)
    return float(np.percentile(dev, 99.9)) < tol


def _cut(rgb: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    """Okno (2·half+1)² wokół (cx, cy) z powieleniem brzegu poza wycinkiem."""
    h, w = rgb.shape[:2]
    ys = np.clip(np.arange(cy - half, cy + half + 1), 0, h - 1)
    xs = np.clip(np.arange(cx - half, cx + half + 1), 0, w - 1)
    return rgb[ys[:, None], xs[None, :]]


def _is_false(v) -> bool:
    return v is False or (isinstance(v, np.bool_) and not bool(v))


# ---------------------------------------------------------------- kalibracja na gwiazdach

def bv_to_kelvin(bv):
    """B−V → temperatura [K] (Ballesteros 2012)."""
    bv = np.asarray(bv, float)
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


@dataclass
class ColorCalib:
    """Linia gwiazd: log10(R/G) = ar + br·(B−V), log10(B/G) = ab + bb·(B−V)."""
    ar: float
    br: float
    ab: float
    bb: float
    rms: float = float("nan")
    n: int = 0

    def locus(self, bv):
        bv = np.asarray(bv, float)
        return self.ar + self.br * bv, self.ab + self.bb * bv

    def project(self, r_g, b_g) -> tuple[np.ndarray, np.ndarray]:
        """(bv_eq, green_excess): rzut na linię gwiazd i odległość od niej [dex], „+” = zieleńszy
        (mniej R i mniej B względem G niż gwiazda o tym samym położeniu na linii)."""
        dx, dy = np.asarray(r_g, float) - self.ar, np.asarray(b_g, float) - self.ab
        d2 = self.br ** 2 + self.bb ** 2
        bv = (dx * self.br + dy * self.bb) / d2
        nx, ny = -self.bb, self.br
        if nx + ny < 0:
            nx, ny = -nx, -ny
        norm = math.sqrt(d2)
        return bv, -(dx * nx + dy * ny) / norm

    def to_dict(self) -> dict:
        return {"ar": self.ar, "br": self.br, "ab": self.ab, "bb": self.bb, "rms_dex": self.rms, "n_stars": self.n}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ColorCalib | None":
        if not d or d.get("ar") is None:
            return None
        return cls(float(d["ar"]), float(d["br"]), float(d["ab"]), float(d["bb"]), float(d.get("rms_dex", "nan")),
                   int(d.get("n_stars", 0)))


def _theil_sen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Prosta Theila–Sena [nachylenie, wyraz wolny] — odporna na ~29% odstających."""
    if len(x) > 800:
        k = np.random.default_rng(0).choice(len(x), 800, replace=False)
        x, y = x[k], y[k]
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-6
    s = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return np.array([s, float(np.median(y - s * x))])


def fit_locus(bv, r_g, b_g, *, clip: float = 3.0, iters: int = 5, min_n: int = 5) -> tuple[ColorCalib | None, np.ndarray]:
    """Odporne dopasowanie linii gwiazd: start z prostej Theila–Sena, potem odrzucanie
    > ``clip``·σ (z MAD) i najmniejsze kwadraty. Zwraca (kalibracja albo None, maska użytych gwiazd)."""
    bv, r_g, b_g = (np.asarray(a, float) for a in (bv, r_g, b_g))
    fin = np.isfinite(bv) & np.isfinite(r_g) & np.isfinite(b_g)
    use = fin.copy()
    if use.sum() < min_n:
        return None, use
    pr, pb = _theil_sen(bv[use], r_g[use]), _theil_sen(bv[use], b_g[use])
    for _ in range(iters):
        res = np.hypot(r_g - np.polyval(pr, bv), b_g - np.polyval(pb, bv))
        s = 1.4826 * np.median(res[use]) + 1e-6
        new = fin & (res < clip * s + 0.005)
        if new.sum() < min_n:
            break
        same = np.array_equal(new, use)
        use = new
        pr, pb = np.polyfit(bv[use], r_g[use], 1), np.polyfit(bv[use], b_g[use], 1)
        if same:
            break
    pr = np.polyfit(bv[use], r_g[use], 1)
    pb = np.polyfit(bv[use], b_g[use], 1)
    res = np.hypot(r_g - np.polyval(pr, bv), b_g - np.polyval(pb, bv))
    rms = float(np.sqrt(np.mean(res[use] ** 2)))
    return ColorCalib(float(pr[1]), float(pr[0]), float(pb[1]), float(pb[0]), rms, int(use.sum())), use


def aperture_photometry(patch_lin: np.ndarray, cx: float, cy: float, r: float, r_in: float, r_out: float):
    """Strumień RGB w aperturze kołowej z tłem = mediana pierścienia; (flux[3], σ_flux[3], n_pix)."""
    h, w, _ = patch_lin.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    ap, ann = d <= r, (d >= r_in) & (d <= r_out)
    bg = np.median(patch_lin[ann], axis=0)
    sig = 1.4826 * np.median(np.abs(patch_lin[ann] - bg), axis=0)
    npx = int(ap.sum())
    flux = (patch_lin[ap] - bg).sum(axis=0)
    return flux, sig * math.sqrt(npx * (1 + npx / max(int(ann.sum()), 1))), npx


def calibrate_epoch(video: Path, meta: VideoMeta, epoch: dict, wcs, ccfg: dict) -> dict:
    """Mediana ``calib_frames`` klatek RGB wokół epoki, fotometria gwiazd katalogowych.
    Zwraca {"stars": [...], "monochrome": bool}."""
    from .sky import _world2pix, stars

    cat = stars()
    x, y = _world2pix(wcs, cat["ra"], cat["dec"])
    r, (r_in, r_out) = float(ccfg["aperture_px"]), (float(v) for v in ccfg["annulus_px"])
    half = int(math.ceil(r_out)) + 2
    ok = (np.isfinite(x) & np.isfinite(y) & (cat["mag"] < float(ccfg["calib_max_mag"])) & np.isfinite(cat["bv"])
          & (x > half) & (x < meta.width - 1 - half) & (y > half) & (y < meta.height - 1 - half))
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return {"stars": [], "monochrome": False}
    # izolacja: żadnej innej gwiazdy katalogowej (do mag 6) bliżej niż isolation_px
    fin = np.isfinite(x) & np.isfinite(y)
    allx, ally = x[fin], y[fin]
    iso = float(ccfg["isolation_px"])
    keep = [i for i in idx if np.sum(np.hypot(allx - x[i], ally - y[i]) < iso) <= 1]
    idx = np.asarray(keep, int)
    if len(idx) == 0:
        return {"stars": [], "monochrome": False}
    nf = int(ccfg["calib_frames"])
    f0 = int(epoch["frame"]) - nf // 2
    box = crop_box(x[idx], y[idx], half, meta.width, meta.height)
    patches: list[list[np.ndarray]] = [[] for _ in idx]
    for _, rgb in read_rgb(video, meta, f0, nf, box, ccfg):
        for j, i in enumerate(idx):
            patches[j].append(_cut(rgb, int(round(x[i])) - box[0], int(round(y[i])) - box[1], half))
    first = [p[0] for p in patches if p]
    if first and is_monochrome(np.stack(first)):     # okolice gwiazd: w kolorze zawsze jest chroma
        return {"stars": [], "monochrome": True}
    sat = float(ccfg["sat_level"])
    out = []
    for j, i in enumerate(idx):
        if not patches[j]:
            continue
        med = np.median(np.stack(patches[j]), axis=0)
        cx = half + (x[i] - round(x[i]))
        cy = half + (y[i] - round(y[i]))
        yy, xx = np.mgrid[0:med.shape[0], 0:med.shape[1]]
        saturated = bool(med[np.hypot(xx - cx, yy - cy) <= r].max() >= sat)
        flux, err, _ = aperture_photometry(linearize(med, ccfg), cx, cy, r, r_in, r_out)
        if saturated or np.any(flux <= 0) or flux[1] < float(ccfg["min_snr"]) * err[1]:
            continue
        out.append({"hip": int(cat["hip"][i]), "mag": float(cat["mag"][i]), "bv": float(cat["bv"][i]),
                    "x": float(x[i]), "y": float(y[i]), "R": float(flux[0]), "G": float(flux[1]), "B": float(flux[2]),
                    "r_g": float(np.log10(flux[0] / flux[1])), "b_g": float(np.log10(flux[2] / flux[1])),
                    "epoch": epoch["file"]})
    return {"stars": out, "monochrome": False}


def calibrate(video: Path, meta: VideoMeta, wcsinfo: dict, outdir: Path, ccfg: dict) -> tuple[dict, list[dict]]:
    """Kalibracja z ``calib_epochs`` rozwiązanych epok: linia gwiazd łączna i per epoka, dryf
    balansu bieli, kontrola liniowości (nachylenie log G vs mag ≈ −0,4)."""
    from .astrometry import load_wcs

    solved = [e for e in wcsinfo.get("epochs", []) if e.get("solved")]
    k = int(ccfg["calib_epochs"])
    if len(solved) > k:
        solved = [solved[i] for i in np.unique(np.round(np.linspace(0, len(solved) - 1, k)).astype(int))]
    all_stars, per_epoch, mono = [], [], []
    for e in solved:
        res = calibrate_epoch(video, meta, e, load_wcs(outdir / e["wcs"]), ccfg)
        mono.append(res["monochrome"])
        if res["monochrome"]:
            break
        all_stars += res["stars"]
        s = res["stars"]
        cal, _ = fit_locus([v["bv"] for v in s], [v["r_g"] for v in s], [v["b_g"] for v in s],
                           min_n=int(ccfg["min_stars"]))
        per_epoch.append({"epoch": e["file"], "frame": int(e["frame"]), "n_stars": len(s),
                          **(cal.to_dict() if cal else {})})
    if mono and all(mono):
        return {"monochrome": True, "calibrated": False}, []
    info: dict = {"monochrome": False, "transfer": ccfg.get("transfer", "bt709"), "per_epoch": per_epoch,
                  "n_candidates": len(all_stars)}
    cal, use = fit_locus([v["bv"] for v in all_stars], [v["r_g"] for v in all_stars],
                         [v["b_g"] for v in all_stars], min_n=int(ccfg["min_stars"]))
    for v, u in zip(all_stars, use):
        v["used"] = bool(u)
    info["calibrated"] = cal is not None
    if cal:
        info.update(cal.to_dict())
        good = [v for v in all_stars if v["used"]]
        mags, g = np.array([v["mag"] for v in good]), np.array([v["G"] for v in good])
        info["linearity_slope"] = float(np.polyfit(mags, np.log10(g), 1)[0]) if len(good) >= 3 else None
        ars = [p["ar"] for p in per_epoch if "ar" in p]
        abs_ = [p["ab"] for p in per_epoch if "ab" in p]
        info["wb_drift_dex"] = float(max(np.ptp(ars), np.ptp(abs_))) if len(ars) >= 2 else 0.0
    return info, all_stars


# ---------------------------------------------------------------- kolor toru

def _aperture_mask(h: int, w: int, cx: float, cy: float, u: np.ndarray | None, half_len: float, half_w: float):
    """Prostokąt wzdłuż ruchu (kreska meteoru) albo koło, gdy obiekt stoi."""
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, yy - cy
    if u is None:
        d = np.hypot(dx, dy)
        return d <= half_w, d
    a = dx * u[0] + dy * u[1]
    c = -dx * u[1] + dy * u[0]
    inside = (np.abs(a) <= half_len) & (np.abs(c) <= half_w)
    dist = np.hypot(np.clip(np.abs(a) - half_len, 0, None), np.clip(np.abs(c) - half_w, 0, None))
    return inside, dist


def measure_track_color(video: Path, meta: VideoMeta, frames: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                        along_sigma_px: float, ccfg: dict, *, thumbs: int = 6) -> tuple[list[dict], dict]:
    """Kolor obiektu w klatkach toru (co najwyżej ``track_max_frames``, równomiernie).
    Zwraca (punkty per klatka, miniatury {klatka: RGB uint8})."""
    from .clips import positions_at

    frames = np.asarray(frames, int)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    fps = meta.fps
    v = np.array([np.polyfit(frames, xs, 1)[0], np.polyfit(frames, ys, 1)[0]]) if len(frames) >= 2 else np.zeros(2)
    speed = float(np.hypot(*v))
    u = v / speed if speed > 0.05 else None
    r = float(ccfg["aperture_px"])
    along = float(along_sigma_px) if math.isfinite(float(along_sigma_px)) else 0.0
    half_len = min(r + 2.0 * along, float(ccfg.get("max_half_len_px", 60)))
    margin = 6.0
    hw = int(math.ceil(max(half_len, r) + margin))
    k = int(min(math.ceil(float(ccfg["diff_min_sep_px"]) / max(speed, 1e-3)), int(ccfg["max_diff_frames"])))
    all_f = np.arange(frames.min(), frames.max() + 1)
    m = int(ccfg["track_max_frames"])
    sel = all_f if len(all_f) <= m else np.unique(np.round(np.linspace(all_f[0], all_f[-1], m)).astype(int))
    px, py = positions_at(sel.astype(float), frames.astype(float), xs, ys)
    pos = {int(f): (float(a), float(b)) for f, a, b in zip(sel, px, py)}
    tsel = set(int(f) for f in sel[np.unique(np.round(np.linspace(0, len(sel) - 1, min(thumbs, len(sel)))).astype(int))])
    th = int(ccfg.get("thumb_px", 64)) // 2
    sat = float(ccfg["sat_level"])
    obj: dict[int, np.ndarray] = {}
    bgs: dict[int, list[np.ndarray]] = {}
    sat_flag: dict[int, bool] = {}
    thumbs_out: dict[int, np.ndarray] = {}

    def window(rgb, box, f, half):
        cx, cy = pos[f]
        return _cut(rgb, int(round(cx)) - box[0], int(round(cy)) - box[1], half)

    # porcje kolejnych wybranych klatek o rozpiętości ≤ chunk_frames: mały wycinek, dekodowanie od klatki kluczowej
    chunk = int(ccfg.get("chunk_frames", 60))
    groups, cur = [], [int(sel[0])]
    for f in sel[1:]:
        if f - cur[0] > chunk:
            groups.append(cur)
            cur = []
        cur.append(int(f))
    groups.append(cur)
    for g in groups:
        gx = [pos[f][0] for f in g]
        gy = [pos[f][1] for f in g]
        box = crop_box(gx, gy, max(hw, th) + 1, meta.width, meta.height)
        lo, hi = g[0] - k, g[-1] + k
        gset = set(g)
        for idx, rgb in read_rgb(video, meta, max(0, lo), hi - max(0, lo) + 1, box, ccfg):
            if idx in gset:
                w_ = window(rgb, box, idx, hw)
                obj[idx] = w_
                ap, _ = _aperture_mask(w_.shape[0], w_.shape[1], hw + pos[idx][0] - round(pos[idx][0]),
                                       hw + pos[idx][1] - round(pos[idx][1]), u, half_len, r)
                sat_flag[idx] = bool(w_[ap].max() >= sat) if ap.any() else False
                if idx in tsel:
                    thumbs_out[idx] = (window(rgb, box, idx, th) * 255 + 0.5).clip(0, 255).astype(np.uint8)
            for f in (idx + k, idx - k):
                if f in gset and k > 0:
                    bgs.setdefault(f, []).append(window(rgb, box, f, hw))
    points = []
    for f in sel:
        f = int(f)
        if f not in obj or not bgs.get(f):
            continue
        o = linearize(obj[f], ccfg)
        b = np.mean([linearize(w_, ccfg) for w_ in bgs[f]], axis=0)
        diff = o - b
        cx = hw + pos[f][0] - round(pos[f][0])
        cy = hw + pos[f][1] - round(pos[f][1])
        ap, dist = _aperture_mask(diff.shape[0], diff.shape[1], cx, cy, u, half_len, r)
        ring = dist > 2.0
        if ap.sum() == 0 or ring.sum() < 10:
            continue
        med = np.median(diff[ring], axis=0)
        sig = 1.4826 * np.median(np.abs(diff[ring] - med), axis=0) + 1e-9
        flux = (diff[ap] - med).sum(axis=0)
        err = sig * math.sqrt(ap.sum())
        R, G, B = (float(q) for q in flux)
        eR, eG, eB = (float(q) for q in err)
        ok = R > 0 and G > 0 and B > 0
        points.append({"frame": f, "t_s": f / fps, "R": R, "G": G, "B": B, "eR": eR, "eG": eG, "eB": eB,
                       "snr": G / eG, "saturated": sat_flag.get(f, False),
                       "r_g": math.log10(R / G) if ok else float("nan"),
                       "b_g": math.log10(B / G) if ok else float("nan"),
                       "e_rg": LOG10E * math.hypot(eR / R, eG / G) if ok else float("nan"),
                       "e_bg": LOG10E * math.hypot(eB / B, eG / G) if ok else float("nan")})
    return points, thumbs_out


def _wmean(v: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    w = 1.0 / np.maximum(e, 1e-3) ** 2
    m = float(np.sum(w * v) / np.sum(w))
    return m, float(1.0 / math.sqrt(np.sum(w)))


def summarize(points: list[dict], calib: ColorCalib | None, ccfg: dict) -> dict:
    """Średnie ważone koloru z klatek nieprześwietlonych o SNR ≥ ``min_snr``, zmiana w czasie."""
    out = {"n_frames": len(points), "n_saturated": sum(1 for p in points if p["saturated"]), "n_color": 0,
           "r_g": float("nan"), "b_g": float("nan"), "e_r_g": float("nan"), "e_b_g": float("nan"),
           "bv_eq": float("nan"), "T_eq_K": float("nan"), "green_excess": float("nan"),
           "slope_dex_s": float("nan"), "chi2": float("nan"), "rg_spread": float("nan")}
    use = [p for p in points if not p["saturated"] and p["snr"] >= float(ccfg["min_snr"])
           and math.isfinite(p["r_g"]) and math.isfinite(p["b_g"])]
    out["n_color"] = len(use)
    if not use:
        return out
    rg, brg = np.array([p["r_g"] for p in use]), np.array([p["b_g"] for p in use])
    erg, ebg = np.array([p["e_rg"] for p in use]), np.array([p["e_bg"] for p in use])
    out["r_g"], out["e_r_g"] = _wmean(rg, erg)
    out["b_g"], out["e_b_g"] = _wmean(brg, ebg)
    if len(use) >= 3:
        t = np.array([p["t_s"] for p in use])
        chi = np.sum(((rg - out["r_g"]) / np.maximum(erg, 1e-3)) ** 2 + ((brg - out["b_g"]) / np.maximum(ebg, 1e-3)) ** 2)
        out["chi2"] = float(chi / (2 * len(use) - 2))
        out["rg_spread"] = float(np.std(rg))
        if np.ptp(t) > 0:
            s1 = np.polyfit(t, rg, 1, w=1 / np.maximum(erg, 1e-3))[0]
            s2 = np.polyfit(t, brg, 1, w=1 / np.maximum(ebg, 1e-3))[0]
            out["slope_dex_s"] = float(max(abs(s1), abs(s2)))
    if calib:
        bv, ge = calib.project(out["r_g"], out["b_g"])
        out["bv_eq"], out["green_excess"] = float(bv), float(ge)
        out["T_eq_K"] = float(bv_to_kelvin(np.clip(bv, -0.4, 2.5)))
    return out


def color_hint(s: dict, *, kind: str, class_hint: str, dur_s: float, sunlit_ref: dict | None,
               ccfg: dict) -> tuple[str, str]:
    """Podpowiedź z koloru (zawsze hipoteza) i uzasadnienie liczbowe. Meteor = tor z podpowiedzią
    „meteor?” (szybki, krótki, prosty) z etapu identify."""
    if s["n_color"] == 0:
        why = f"{s['n_saturated']} z {s['n_frames']} klatek prześwietlonych" if s["n_saturated"] else "za słaby sygnał"
        return "brak koloru", why
    T, ge = s["T_eq_K"], s["green_excess"]
    base = (f"T≈{T:.0f} K, B−V≈{s['bv_eq']:.2f}, nadmiar zieleni {ge:+.2f} dex" if math.isfinite(T)
            else f"log R/G {s['r_g']:+.2f}, log B/G {s['b_g']:+.2f} (bez kalibracji)")
    change = (math.isfinite(s["slope_dex_s"]) and math.isfinite(s["chi2"]) and s["chi2"] > float(ccfg["change_chi2"])
              and s["slope_dex_s"] * max(dur_s, 1e-3) > float(ccfg["change_dex"]))
    tail = "; zmiana koloru wzdłuż śladu" if change else ""
    if not math.isfinite(T):
        return "kolor bez kalibracji", base + tail
    if kind == "sat":
        return "Słońce odbite (odniesienie)", base + tail
    if class_hint == "meteor?":
        if ge > float(ccfg["green_thr"]):
            return "zielony nadmiar → Mg / O 557,7 nm?", base + tail
        if T < float(ccfg["warm_T_K"]):
            return "pomarańczowy → Na/Fe?", base + tail
        if T > float(ccfg["hot_T_K"]):
            return "biało-niebieski → szybki, Ca/Mg?", base + tail
        return "biały/żółty meteor", base + tail
    if (s["n_color"] >= 3 and math.isfinite(s["rg_spread"]) and s["rg_spread"] > float(ccfg["nav_spread_dex"])
            and math.isfinite(s["chi2"]) and s["chi2"] > float(ccfg["change_chi2"])):
        return "światła nawigacyjne? (czerwone/zielone)", base + f", rozrzut R/G {s['rg_spread']:.2f} dex" + tail
    if sunlit_ref and math.hypot(s["r_g"] - sunlit_ref["r_g"], s["b_g"] - sunlit_ref["b_g"]) <= float(ccfg["sunlit_tol_dex"]):
        return "oświetlony Słońcem (jak satelity)", base + tail
    if T < float(ccfg["city_T_K"]):
        return "ciepły → łuna miasta (sód)?", base + tail
    if ge > float(ccfg["green_thr"]):
        return "zielony nadmiar (światło zielone?)", base + tail
    return "kolor nietypowy dla Słońca", base + tail


# ---------------------------------------------------------------- etap

SUMMARY_COLS = ["track_id", "n_frames", "n_saturated", "n_color", "r_g", "b_g", "e_r_g", "e_b_g", "bv_eq", "T_eq_K",
                "green_excess", "slope_dex_s", "chi2", "rg_spread", "color_hint", "color_reason"]


def run(outdir: Path, video: Path, meta: VideoMeta, cfg: dict, log_: logging.Logger | None = None) -> dict:
    """Kalibracja + kolor wszystkich torów; zapisuje ``color_calib.json``, ``color_stars.csv``,
    ``track_color.csv``, ``track_color_points.parquet``, ``track_color_thumbs.npz``."""
    import pandas as pd

    from .io import read_json, write_json

    lg = log_ or log
    ccfg = cfg["color"]
    name = video.name
    if str(ccfg.get("enabled", "auto")) == "off":
        info = {"enabled": False}
        write_json(outdir / "color_calib.json", info)
        pd.DataFrame(columns=SUMMARY_COLS).to_csv(outdir / "track_color.csv", index=False)
        return {"outputs": ["color_calib.json", "track_color.csv"], "metrics": {"enabled": False}}
    wcsinfo = read_json(outdir / "wcs.json")
    info, stars_ = calibrate(video, meta, wcsinfo, outdir, ccfg)
    info["enabled"] = True
    pd.DataFrame(stars_).to_csv(outdir / "color_stars.csv", index=False)
    if info.get("monochrome"):
        write_json(outdir / "color_calib.json", info)
        pd.DataFrame(columns=SUMMARY_COLS).to_csv(outdir / "track_color.csv", index=False)
        lg.info("[%s] kolor: nagranie czarno-białe — pomijam", name)
        return {"outputs": ["color_calib.json", "color_stars.csv", "track_color.csv"], "metrics": {"monochrome": True}}
    calib = ColorCalib.from_dict(info) if info.get("calibrated") else None
    if calib is None:
        lg.warning("[%s] kolor: za mało gwiazd do kalibracji (%d) — kolor bez przeliczenia na B−V",
                   name, info.get("n_candidates", 0))
    else:
        slope = info.get("linearity_slope")
        lvl = lg.warning if (slope is not None and abs(slope + 0.4) > float(ccfg["linearity_warn"])) else lg.info
        lvl("[%s] kolor: kalibracja na %d gwiazdach, RMS %.3f dex, liniowość %s (oczek. −0,40), dryf balansu bieli %.3f dex",
            name, calib.n, calib.rms, f"{slope:+.2f}" if slope is not None else "–", info.get("wb_drift_dex", 0.0))
        if info.get("wb_drift_dex", 0.0) > float(ccfg["wb_drift_warn"]):
            lg.warning("[%s] kolor: balans bieli zmieniał się w trakcie nagrania (%.3f dex) — ustaw stały WB w aparacie",
                       name, info["wb_drift_dex"])

    final = pd.read_parquet(outdir / "tracks_final.parquet")
    tp = pd.read_parquet(outdir / "track_points.parquet")
    rows, pts_rows, thumbs = [], [], {}
    for _, t in final.iterrows():
        tid = int(t["track_id"])
        p = tp[tp["track_id"] == tid].sort_values("frame")
        if len(p) < 2:
            continue
        try:
            points, th = measure_track_color(video, meta, p["frame"].to_numpy(), p["x"].to_numpy(), p["y"].to_numpy(),
                                             float(t.get("along_sigma_px", 0.0) or 0.0), ccfg)
        except Exception as e:  # noqa: BLE001 — jeden tor nie blokuje reszty
            lg.warning("[%s] kolor toru #%d nieudany: %s", name, tid, e)
            continue
        s = summarize(points, calib, ccfg)
        rows.append({"track_id": tid, **s, "_kind": str(t.get("kind", "")), "_hint": str(t.get("class_hint", "")),
                     "_dur": float(t.get("dur_s", 0.0)), "_sunlit": t.get("sunlit")})
        pts_rows += [{"track_id": tid, **q} for q in points]
        for f, img in th.items():
            thumbs[f"t{tid}_f{f}"] = img
    # odniesienie: satelity oświetlone przez Słońce
    ref_rows = [r for r in rows if r["_kind"] == "sat" and not _is_false(r["_sunlit"]) and r["n_color"] >= 3]
    sunlit_ref = None
    if ref_rows:
        sunlit_ref = {"r_g": float(np.median([r["r_g"] for r in ref_rows])),
                      "b_g": float(np.median([r["b_g"] for r in ref_rows])), "n": len(ref_rows)}
        if calib:
            bv, ge = calib.project(sunlit_ref["r_g"], sunlit_ref["b_g"])
            sunlit_ref.update(bv_eq=float(bv), green_excess=float(ge), T_eq_K=float(bv_to_kelvin(np.clip(bv, -0.4, 2.5))))
            lo, hi = ccfg["sunlit_bv_expected"]
            if not float(lo) <= bv <= float(hi):
                lg.warning("[%s] kolor: satelity (Słońce odbite) mają B−V %.2f poza oczekiwanym %.1f–%.1f — "
                           "kalibracja koloru niepewna", name, bv, float(lo), float(hi))
    info["sunlit_ref"] = sunlit_ref
    for r in rows:
        r["color_hint"], r["color_reason"] = color_hint(
            r, kind=r["_kind"], class_hint=r["_hint"], dur_s=r["_dur"], sunlit_ref=sunlit_ref, ccfg=ccfg)
    write_json(outdir / "color_calib.json", info)
    pd.DataFrame([{k: r[k] for k in SUMMARY_COLS} for r in rows], columns=SUMMARY_COLS) \
        .to_csv(outdir / "track_color.csv", index=False)
    pd.DataFrame(pts_rows, columns=None if pts_rows else ["track_id", "frame", "t_s", "r_g", "b_g"]) \
        .to_parquet(outdir / "track_color_points.parquet", index=False)
    np.savez_compressed(outdir / "track_color_thumbs.npz", **thumbs)
    n_col = sum(1 for r in rows if r["n_color"] > 0)
    lg.info("[%s] kolor: %d torów z kolorem (z %d), Słońce odbite (satelity): %s", name, n_col, len(rows),
            f"B−V {sunlit_ref.get('bv_eq', float('nan')):.2f} z {sunlit_ref['n']} satelitów" if sunlit_ref else "–")
    return {"outputs": ["color_calib.json", "color_stars.csv", "track_color.csv", "track_color_points.parquet",
                        "track_color_thumbs.npz"],
            "metrics": {"monochrome": False, "stars": calib.n if calib else 0,
                        "calib_rms_dex": calib.rms if calib else None,
                        "linearity_slope": info.get("linearity_slope"), "wb_drift_dex": info.get("wb_drift_dex"),
                        "tracks_colored": n_col,
                        "sunlit_bv": sunlit_ref.get("bv_eq") if sunlit_ref else None}}
