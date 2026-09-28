"""Łączenie detekcji w tory, pomiary torów i podpowiedź klasy.

Port ``link()``, ``analyze()`` i ``classify()`` z ``baseline/skytracks.py`` z poprawkami:
filtr statyczny z warunkiem czasu (wolne obiekty nie znikają), przypisanie globalne
(Hungarian), sklejanie fragmentów toru, progi w pikselach 4K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- filtr statyczny

def static_mask(frame: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float, tcfg: dict) -> np.ndarray:
    """True dla detekcji w komórkach „statycznych” (gwiazdy, hot piksele): w oknie
    ``static_window_s`` komórka ``static_cell_px`` z sąsiadami 3×3 ma ≥ ``static_min_count``
    trafień w ≥ ``static_min_seconds`` różnych sekundach nagrania.

    Liczymy różne sekundy, a nie rozpiętość czasu: przelatujący obiekt siedzi w komórce
    1–2 s, więc jedno przypadkowe trafienie szumu kilka sekund później nie czyni go statycznym."""
    n = len(frame)
    if n == 0:
        return np.zeros(0, bool)
    c = float(tcfg["static_cell_px"])
    chunk = (frame // max(1, int(round(float(tcfg["static_window_s"]) * fps)))).astype(np.int64)
    cx = np.floor(x / c).astype(np.int64) + 2
    cy = np.floor(y / c).astype(np.int64) + 2

    def key(ch, a, b):
        return (ch << 42) | (a << 21) | b

    # każda detekcja „głosuje” na swoją komórkę i 8 sąsiadów
    offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    ekeys = np.concatenate([key(chunk, cx + dx, cy + dy) for dx, dy in offs])
    esec = np.tile((frame // max(1, int(round(fps)))).astype(np.int64), len(offs))
    uniq, inv = np.unique(ekeys, return_inverse=True)
    inv = inv.ravel()
    count = np.bincount(inv, minlength=len(uniq))
    # liczba różnych sekund z trafieniami dla każdej komórki
    pairs = np.unique(inv * (int(esec.max()) + 1) + esec)
    n_sec = np.bincount(pairs // (int(esec.max()) + 1), minlength=len(uniq))
    own = inv[4 * n:5 * n]          # przesunięcie (0, 0) to piąty blok
    return (count[own] >= int(tcfg["static_min_count"])) & (n_sec[own] >= int(tcfg["static_min_seconds"]))


# ---------------------------------------------------------------- łączenie

@dataclass
class _Track:
    idx: list = field(default_factory=list)     # indeksy detekcji
    fr: list = field(default_factory=list)
    xy: list = field(default_factory=list)

    def add(self, i: int, f: int, p) -> None:
        self.idx.append(i)
        self.fr.append(f)
        self.xy.append(p)

    def predict(self, f: int, tcfg: dict) -> tuple[np.ndarray, float]:
        last = np.asarray(self.xy[-1])
        dt = f - self.fr[-1]
        if len(self.fr) < 2:
            return last, float(tcfg["init_gate_px"])
        k = min(len(self.fr), 5)
        fr = np.asarray(self.fr[-k:], float)
        xy = np.asarray(self.xy[-k:], float)
        v = (xy[-1] - xy[0]) / (fr[-1] - fr[0])
        gate = float(tcfg["gate_px"]) + float(tcfg["gate_speed_frac"]) * float(np.hypot(*v)) * dt
        return last + v * dt, gate


def link(frame: np.ndarray, x: np.ndarray, y: np.ndarray, tcfg: dict) -> list[np.ndarray]:
    """Tory jako listy indeksów detekcji (posortowane po klatce). Przypisanie globalne
    (``linear_sum_assignment``) z kosztem = odległość ÷ bramka."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    max_gap = int(tcfg["max_gap_frames"])
    order = np.argsort(frame, kind="stable")
    frames_sorted = frame[order]
    bounds = np.flatnonzero(np.diff(frames_sorted)) + 1
    groups = np.split(order, bounds) if len(order) else []
    active: list[_Track] = []
    done: list[_Track] = []
    for ids in groups:
        f = int(frame[ids[0]])
        keep = []
        for t in active:
            # jednopunktowy „tor” (zwykle szum) żyje jedną klatkę — inaczej jego duża bramka
            # przechwytuje pierwszy punkt nowego, prawdziwego toru
            life = max_gap + 1 if len(t.fr) >= 2 else 1
            (keep if f - t.fr[-1] <= life else done).append(t)
        active = keep
        pts = np.column_stack([x[ids], y[ids]])
        taken = np.zeros(len(ids), bool)
        if active:
            preds = [t.predict(f, tcfg) for t in active]
            centers = np.array([p for p, _ in preds])
            gates = np.array([g for _, g in preds])
            cand = cKDTree(pts).query_ball_point(centers, gates)
            rows = [ti for ti, c in enumerate(cand) if c]
            if rows:
                cols = sorted({j for ti in rows for j in cand[ti]})
                col_pos = {j: k for k, j in enumerate(cols)}
                cost = np.full((len(rows), len(cols)), 1e9)
                for r, ti in enumerate(rows):
                    for j in cand[ti]:
                        cost[r, col_pos[j]] = np.hypot(*(pts[j] - centers[ti])) / max(gates[ti], 1e-9)
                rr, cc = linear_sum_assignment(cost)
                for r, c in zip(rr, cc):
                    if cost[r, c] < 1e8:
                        j = cols[c]
                        active[rows[r]].add(int(ids[j]), f, pts[j])
                        taken[j] = True
        for j in np.flatnonzero(~taken):
            t = _Track()
            t.add(int(ids[j]), f, pts[j])
            active.append(t)
    done.extend(active)
    out = []
    for t in done:
        if len(t.idx) < int(tcfg["min_len"]):
            continue
        a, b = np.asarray(t.xy[0]), np.asarray(t.xy[-1])
        if np.hypot(*(b - a)) < float(tcfg["min_disp_px"]):
            continue
        out.append(np.asarray(t.idx))
    return out


def _fit(frames: np.ndarray, x: np.ndarray, y: np.ndarray):
    fr = frames.astype(float)
    if len(fr) < 2 or fr[-1] == fr[0]:
        return np.zeros(2), np.array([x[0], y[0]])
    px, py = np.polyfit(fr, x, 1), np.polyfit(fr, y, 1)
    return np.array([px[0], py[0]]), np.array([px[1], py[1]])


def merge_fragments(tracks: list[np.ndarray], frame, x, y, tcfg: dict) -> list[np.ndarray]:
    """Skleja fragmenty jednego obiektu przerwane na dłużej niż ``max_gap_frames``:
    koniec A ekstrapolowany do startu B mieści się w ``merge_gate_px`` i kierunki
    różnią się o ≤ ``merge_angle_deg``."""
    gate, max_gap = float(tcfg["merge_gate_px"]), int(tcfg["merge_gap_frames"])
    max_ang = math.radians(float(tcfg["merge_angle_deg"]))
    tracks = [np.asarray(t) for t in tracks]
    merged = True
    while merged:
        merged = False
        tracks.sort(key=lambda t: frame[t[0]])
        fits = [_fit(frame[t], x[t], y[t]) for t in tracks]
        for a in range(len(tracks)):
            va, _ = fits[a]
            ea = tracks[a][-1]
            for b in range(len(tracks)):
                if a == b:
                    continue
                sb = tracks[b][0]
                gap = int(frame[sb] - frame[ea])
                if not 1 <= gap <= max_gap:
                    continue
                vb, _ = fits[b]
                na, nb = np.hypot(*va), np.hypot(*vb)
                if na == 0 or nb == 0:
                    continue
                ang = math.acos(float(np.clip(va @ vb / (na * nb), -1, 1)))
                pred = np.array([x[ea], y[ea]]) + va * gap
                if ang <= max_ang and np.hypot(*(pred - [x[sb], y[sb]])) <= gate:
                    tracks[a] = np.concatenate([tracks[a], tracks[b]])
                    del tracks[b]
                    merged = True
                    break
            if merged:
                break
    return tracks


# ---------------------------------------------------------------- pomiary

def periodicity(frames: np.ndarray, flux: np.ndarray, fps: float, tcfg: dict) -> tuple[float, float]:
    """(częstotliwość piku [Hz], moc piku ÷ mediana widma) jasności toru; NaN przy krótkich torach."""
    if len(frames) < int(tcfg["fft_min_frames"]):
        return float("nan"), 0.0
    grid = np.arange(frames[0], frames[-1] + 1)
    fl = np.interp(grid, frames, flux)
    k = int(tcfg["fft_detrend_frames"])
    trend = np.convolve(np.pad(fl, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(fl)]
    det = fl - trend
    spec = np.abs(np.fft.rfft(det * np.hanning(len(det))))
    freq = np.fft.rfftfreq(len(det), 1 / fps)
    spec[freq < float(tcfg["fft_min_hz"])] = 0
    if not spec.any():
        return float("nan"), 0.0
    j = int(np.argmax(spec))
    return float(freq[j]), float(spec[j] / (np.median(spec[spec > 0]) + 1e-9))


def measure_track(frames, x, y, flux, cxx, cxy, cyy, *, fps: float, width: int, height: int,
                  star_sigma_px: float | None, nominal_hfov_deg: float, tcfg: dict) -> dict:
    fr = frames.astype(float)
    v, p0 = _fit(frames, x, y)
    speed = float(np.hypot(*v))   # px/klatkę
    resid = np.hypot(p0[0] + v[0] * fr - x, p0[1] + v[1] * fr - y)
    nvec = np.array([-v[1], v[0]]) / (speed + 1e-9)
    cross = np.sqrt(np.clip(nvec[0] ** 2 * cxx + 2 * nvec[0] * nvec[1] * cxy + nvec[1] ** 2 * cyy, 0, None))
    f_peak, f_power = periodicity(frames, flux, fps, tcfg)
    m = float(tcfg["edge_margin_px"])

    def inside(px, py):
        return bool(m < px < width - 1 - m and m < py < height - 1 - m)

    return {
        "frame0": int(frames[0]), "frame1": int(frames[-1]), "n": int(len(frames)),
        "dur_s": float((frames[-1] - frames[0]) / fps),
        "x0": float(x[0]), "y0": float(y[0]), "x1": float(x[-1]), "y1": float(y[-1]),
        "vx_px_frame": float(v[0]), "vy_px_frame": float(v[1]), "speed_px_frame": speed,
        "deg_s_nominal": speed * fps * nominal_hfov_deg / width,
        "curv_px": float(np.sqrt(np.mean(resid ** 2))),
        "cross_sigma_px": float(np.median(cross)),
        "cross_ratio": float(np.median(cross) / star_sigma_px) if star_sigma_px else float("nan"),
        "flux_mean": float(np.mean(flux)), "flux_cv": float(np.std(flux) / (abs(np.mean(flux)) + 1e-9)),
        "f_peak_hz": f_peak, "f_power": f_power,
        "f_alias_hz": float(fps - f_peak) if math.isfinite(f_peak) else float("nan"),
        "starts_inside": inside(x[0], y[0]), "ends_inside": inside(x[-1], y[-1]),
    }


def classify_hint(deg_s: float, dur_s: float, curv_px: float, cross_ratio: float, f_peak_hz: float,
                  f_power: float, fps: float, ccfg: dict, peak_snr: float = float("nan")) -> tuple[str, str]:
    """Podpowiedź klasy dla toru bez dopasowania do katalogu: (klasa, uzasadnienie).
    Szerokość nie świadczy o nieostrości, gdy obiekt jest jasny (prześwietlenie rozlewa obraz)."""
    periodic = f_power > float(ccfg["periodic_min_power"])
    straight = curv_px < float(ccfg["straight_max_curv_px"])
    bright = math.isfinite(peak_snr) and peak_snr >= float(ccfg.get("bright_peak_snr", math.inf))
    near = math.isfinite(cross_ratio) and cross_ratio > float(ccfg["near_cross_ratio"]) and not bright
    a_lo, a_hi = ccfg["aircraft_f_hz"]
    s_lo, s_hi = ccfg["satellite_deg_s"]
    if deg_s > float(ccfg["meteor_min_deg_s"]) and dur_s < float(ccfg["meteor_max_dur_s"]) and straight:
        return "meteor?", f"szybki ({deg_s:.2f}°/s), krótki ({dur_s:.2f} s), prosty"
    if periodic and a_lo <= f_peak_hz <= a_hi and deg_s < float(ccfg["meteor_min_deg_s"]):
        return "samolot?", f"błyski {f_peak_hz:.2f} Hz (światła stroboskopowe)"
    if near or (periodic and f_peak_hz > a_hi) or not straight:
        why = []
        if near:
            why.append(f"nieostry: szerokość {cross_ratio:.1f}× gwiazdy")
        if periodic and f_peak_hz > a_hi:
            why.append(f"modulacja {f_peak_hz:.1f} Hz lub alias {fps - f_peak_hz:.1f} Hz")
        if not straight:
            why.append(f"tor zakrzywiony ({curv_px:.1f} px)")
        return "bliski obiekt?", "; ".join(why)
    if straight and s_lo <= deg_s <= s_hi:
        return "satelita?", f"prosty, {deg_s:.2f}°/s, bez dopasowania w katalogu"
    return "niesklasyfikowany", f"{deg_s:.2f}°/s, krzywizna {curv_px:.1f} px"


def build_tracks(det: dict, *, fps: float, width: int, height: int, star_sigma_px: float | None,
                 nominal_hfov_deg: float, tcfg: dict) -> tuple[list[dict], dict, dict]:
    """Detekcje → (lista torów z pomiarami, punkty torów jako kolumny, statystyki)."""
    frame, x, y = det["frame"], det["x"], det["y"]
    static = static_mask(frame, x, y, fps, tcfg)
    keep = np.flatnonzero(~static)
    tracks = link(frame[keep], x[keep], y[keep], tcfg)
    tracks = [keep[t] for t in tracks]
    tracks = merge_fragments(tracks, frame, x, y, tcfg)
    tracks.sort(key=lambda t: (frame[t[0]], x[t[0]]))
    rows, pts = [], {k: [] for k in ("track_id", *det.keys())}
    for tid, t in enumerate(tracks, 1):
        t = t[np.argsort(frame[t], kind="stable")]
        row = measure_track(frame[t], x[t], y[t], det["flux"][t], det["cxx"][t], det["cxy"][t], det["cyy"][t],
                            fps=fps, width=width, height=height, star_sigma_px=star_sigma_px,
                            nominal_hfov_deg=nominal_hfov_deg, tcfg=tcfg)
        rows.append({"track_id": tid, **row})
        pts["track_id"].append(np.full(len(t), tid))
        for k in det:
            pts[k].append(det[k][t])
    points = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in pts.items()}
    stats = {"detections": int(len(frame)), "static": int(static.sum()), "tracks": len(rows)}
    return rows, points, stats
