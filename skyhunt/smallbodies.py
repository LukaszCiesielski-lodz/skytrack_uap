"""Planetoidy, komety i NEO w kadrze.

Lista znanych małych ciał w polu widzenia pochodzi z JPL Small-Body Identification API
(``sb_ident``: pozycje z numerycznego całkowania orbit, jasność V, ruch w ″/h). Planetoida pasa
głównego przesuwa się przez 5 minut o ~0,1 px (24,7″/px), więc w nagraniu wygląda jak gwiazda:
detektor torów jej nie zobaczy. Mierzymy ją w stosie wszystkich klatek, w oknie przesuwanym
razem z obrotem nieba (i ruchem własnym obiektu). Zasięg takiego stosu jest o kilka magnitudo
głębszy niż pojedynczej klatki.

„Wykryta” = źródło o SNR ≥ progu w przewidzianym miejscu, z jasnością zgodną z przewidywaną,
bez jaśniejszej gwiazdy tła (Gaia DR3 z VizieR) w aperturze. Punkt zerowy jasności z gwiazd
katalogowych z tego samego stosu, z poprawką na winietowanie (ZP = a + b·r²).

Do JPL wysyłamy współrzędne obserwatora zaokrąglone do ``site_round_deg`` (~10 km): paralaksa
nawet bliskiej NEO (0,01 au) zmienia się przez to o ~1″, a dokładne miejsce zostaje prywatne.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

_COMET = re.compile(r"^(\d+[PCDXI](/|-|$)|[PCDXI]/)")


# ---------------------------------------------------------------- JPL sb_ident

def ra_hms(ra_deg: float) -> str:
    """RA [°] → ``hh-mm-ss.ss`` (format parametrów sb_ident)."""
    total = round((float(ra_deg) % 360.0) / 15.0 * 3600.0, 2) % 86400.0
    hh = int(total // 3600)
    mm = int((total - hh * 3600) // 60)
    ss = total - hh * 3600 - mm * 60
    return f"{hh:02d}-{mm:02d}-{ss:05.2f}"


def dec_dms(dec_deg: float) -> str:
    """Dec [°] → ``dd-mm-ss.s``; ujemne z przedrostkiem ``M`` (wymóg sb_ident)."""
    total = round(abs(float(dec_deg)) * 3600.0, 1)
    dd = int(total // 3600)
    mm = int((total - dd * 3600) // 60)
    ss = total - dd * 3600 - mm * 60
    return f"{'M' if dec_deg < 0 else ''}{dd:02d}-{mm:02d}-{ss:04.1f}"


def parse_ra(s: str) -> float:
    """``22:39:31.27`` → [°]."""
    h, m, sec = (float(v) for v in str(s).strip().split(":"))
    return 15.0 * (h + m / 60 + sec / 3600)


def parse_dec(s: str) -> float:
    """``+11 43'40.4"`` / ``-05 12'03"`` → [°]."""
    t = str(s).strip()
    sign = -1.0 if t.startswith("-") else 1.0
    nums = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", t)]
    while len(nums) < 3:
        nums.append(0.0)
    return sign * (nums[0] + nums[1] / 60 + nums[2] / 3600)


def query_params(center_ra: float, center_dec: float, radius_deg: float, obs_utc: datetime,
                 site: tuple[float, float, float], scfg: dict, *, neo: bool = False) -> dict:
    """Parametry zapytania: prostokąt RA/Dec opisany na kole o promieniu ``radius_deg`` (kadr jest
    obrócony względem osi RA/Dec), położenie obserwatora zaokrąglone do ``site_round_deg``."""
    rnd = float(scfg.get("site_round_deg", 0.1))
    lat, lon, elev = site
    q = lambda v: round(round(float(v) / rnd) * rnd, 6)  # noqa: E731
    dec_hw = min(float(radius_deg), 89.0)
    edge = abs(float(center_dec)) + dec_hw
    ra_hw = 180.0 if edge >= 89.0 else min(180.0, dec_hw / math.cos(math.radians(edge)))
    p = {"lat": q(lat), "lon": q(lon), "alt": round(float(elev) / 1000.0, 1),
         "obs-time": obs_utc.strftime("%Y-%m-%d_%H:%M:%S"),
         "fov-ra-center": ra_hms(center_ra), "fov-dec-center": dec_dms(center_dec),
         "fov-ra-hwidth": round(ra_hw, 3), "fov-dec-hwidth": round(dec_hw, 3),
         "two-pass": "true", "suppress-first-pass": "true", "mag-required": "true",
         "vmag-lim": float(scfg["neo_vmag_lim"] if neo else scfg["vmag_lim"])}
    if neo:
        p["sb-group"] = "neo"
    return p


def fetch_sbident(params: dict, scfg: dict) -> dict:
    url = str(scfg.get("api_url", "https://ssd-api.jpl.nasa.gov/sb_ident.api"))
    full = url + "?" + urllib.parse.urlencode(params, safe="-_:")
    req = urllib.request.Request(full, headers={"User-Agent": "skyhunt (github.com/LukaszCiesielski-lodz/skytrack_uap)"})
    try:
        with urllib.request.urlopen(req, timeout=float(scfg.get("timeout_s", 120))) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:   # 400: szczegóły w treści odpowiedzi
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"JPL sb_ident HTTP {e.code}: {body}") from e


def parse_sbident(js: dict, *, is_neo: bool = False) -> list[dict]:
    """Wiersze ``data_second_pass`` (albo ``data_first_pass``) → słowniki. Kolumny po nagłówkach,
    bo pierwszy przebieg ma dodatkowe kolumny błędów. Tempo ruchu w ″/h (dRA·cosδ, dDec) —
    w dokumentacji API nagłówek bywa podpisany „deg/s”, ale wartości to ″/h."""
    fields = js.get("fields_second") or js.get("fields_first") or []
    rows = js.get("data_second_pass") or js.get("data_first_pass") or []

    def col(*keys: str) -> int | None:
        for i, f in enumerate(fields):
            if all(k.lower() in f.lower() for k in keys):
                return i
        return None

    i_name, i_ra, i_dec = col("object name"), col("astrometric ra"), col("astrometric dec")
    i_v, i_rr, i_dr = col("visual magnitude"), col("ra rate"), col("dec rate")
    i_era, i_edec = col("error ra"), col("error dec")
    out = []
    if None in (i_name, i_ra, i_dec):
        return out
    for r in rows:
        def num(i):
            try:
                return float(r[i]) if i is not None else float("nan")
            except (TypeError, ValueError, IndexError):
                return float("nan")
        name = str(r[i_name]).strip()
        out.append({"name": name, "ra": parse_ra(r[i_ra]), "dec": parse_dec(r[i_dec]), "vmag": num(i_v),
                    "ra_rate": num(i_rr), "dec_rate": num(i_dr),
                    "err_arcsec": max(num(i_era), num(i_edec)) if i_era is not None else float("nan"),
                    "kind": "kometa" if _COMET.match(name) else "planetoida", "is_neo": bool(is_neo)})
    return out


def merge_bodies(main: list[dict], neo: list[dict]) -> list[dict]:
    """Lista główna + NEO (ta sama nazwa → flaga NEO), posortowane od najjaśniejszych."""
    by = {b["name"]: dict(b) for b in main}
    for b in neo:
        if b["name"] in by:
            by[b["name"]]["is_neo"] = True
        else:
            by[b["name"]] = dict(b, is_neo=True)
    return sorted(by.values(), key=lambda b: (b["vmag"] if math.isfinite(b["vmag"]) else 99.0))


def body_radec(b: dict, dt_s) -> tuple[np.ndarray, np.ndarray]:
    """Pozycja ``dt_s`` [s] od chwili zapytania: ruch liniowy (minuty nagrania)."""
    dt_h = np.asarray(dt_s, float) / 3600.0
    rr = b["ra_rate"] if math.isfinite(b["ra_rate"]) else 0.0
    dr = b["dec_rate"] if math.isfinite(b["dec_rate"]) else 0.0
    dec = b["dec"] + dr * dt_h / 3600.0
    ra = b["ra"] + rr * dt_h / 3600.0 / max(math.cos(math.radians(b["dec"])), 1e-6)
    return np.mod(ra, 360.0), dec


def rate_arcsec_h(b: dict) -> float:
    return math.hypot(b["ra_rate"] if math.isfinite(b["ra_rate"]) else 0.0,
                      b["dec_rate"] if math.isfinite(b["dec_rate"]) else 0.0)


# ---------------------------------------------------------------- gwiazdy tła (Gaia, VizieR)

def parse_vizier_tsv(text: str, mag_col: str = "Gmag") -> list[tuple[float, float, float]]:
    """Odpowiedź VizieR asu-tsv → [(RA, Dec, mag)]; komentarze, jednostki i kreski pomijane."""
    out, header = [], None
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        if header is None:
            header = parts
            continue
        try:
            row = dict(zip(header, parts))
            out.append((float(row["RA_ICRS"]), float(row["DE_ICRS"]), float(row[mag_col])))
        except (KeyError, ValueError):
            continue
    return out


def gaia_neighbors(ra: float, dec: float, radius_arcsec: float, mag_max: float, scfg: dict) -> list:
    """Gwiazdy Gaia DR3 (G ≤ ``mag_max``) w promieniu od pozycji obiektu."""
    url = str(scfg.get("gaia_url", "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"))
    params = {"-source": "I/355/gaiadr3", "-c": f"{ra:.6f} {dec:+.6f}", "-c.rs": f"{radius_arcsec:.1f}",
              "-out": "RA_ICRS,DE_ICRS,Gmag", "Gmag": f"<{mag_max:.2f}", "-out.max": "200"}
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "skyhunt"})
    with urllib.request.urlopen(req, timeout=float(scfg.get("timeout_s", 120))) as r:
        return parse_vizier_tsv(r.read().decode("utf-8", "replace"))


# ---------------------------------------------------------------- stos przesuwany za obiektem

@dataclass
class WindowStack:
    """Suma okien ``(2·hw+1)²`` śledzących pozycję obiektu; ``frac`` — przesunięcia podpikselowe
    pozycji względem środka okna (z wagą liczby klatek), czyli gdzie leży obiekt w stosie."""
    hw: int
    total: np.ndarray | None = None
    n: int = 0
    frac: list = field(default_factory=list)

    def __post_init__(self):
        if self.total is None:
            self.total = np.zeros((2 * self.hw + 1, 2 * self.hw + 1), np.float64)

    @property
    def mean(self) -> np.ndarray | None:
        return self.total / self.n if self.n else None

    @property
    def center(self) -> tuple[float, float]:
        if not self.frac:
            return float(self.hw), float(self.hw)
        f = np.asarray(self.frac, float)
        w = f[:, 2]
        return float(self.hw + np.sum(f[:, 0] * w) / w.sum()), float(self.hw + np.sum(f[:, 1] * w) / w.sum())


def stack_windows(batches: Iterable, pos_at: Callable[[np.ndarray], np.ndarray], n_targets: int, hw: int,
                  frame_shape: tuple[int, int], to_linear: Callable[[np.ndarray], np.ndarray]) -> list[WindowStack]:
    """Jedno przejście po paczkach klatek ``(start, y[N, H, W])`` (numpy albo torch).
    ``pos_at(klatki)`` → pozycje [K, len(klatki), 2] (x, y). W paczce okno stoi w miejscu
    pozycji ze środka paczki (ruch nieba ~0,4 px/s, paczka 32 klatek ≈ 1 s → ≤ 0,2 px)."""
    H, W = frame_shape
    stacks = [WindowStack(hw) for _ in range(n_targets)]
    size = 2 * hw + 1
    for start, y in batches:
        n = int(y.shape[0])
        pos = np.asarray(pos_at(np.array([start + (n - 1) / 2.0])), float)[:, 0, :]
        for k in range(n_targets):
            x, yy = pos[k]
            if not (math.isfinite(x) and math.isfinite(yy)):
                continue
            cx, cy = int(round(x)), int(round(yy))
            x0, y0 = cx - hw, cy - hw
            if x0 < 0 or y0 < 0 or x0 + size > W or y0 + size > H:
                continue
            w = y[:, y0:y0 + size, x0:x0 + size]
            if hasattr(w, "cpu"):
                w = w.cpu().numpy()
            stacks[k].total += to_linear(np.asarray(w)).sum(axis=0)
            stacks[k].n += n
            stacks[k].frac.append((x - cx, yy - cy, n))
    return stacks


def y_to_linear(y: np.ndarray, transfer: str | float = "bt709", full_range: bool = False) -> np.ndarray:
    """Y (8 bit) → światło liniowe 0…1 (odwrotność krzywej tonalnej; zakres 16–235 albo 0–255)."""
    from .color import linearize

    v = np.asarray(y, np.float32)
    v = v / 255.0 if full_range else (v - 16.0) / 219.0
    return linearize(v, {"transfer": transfer})


# ---------------------------------------------------------------- fotometria

def photometry(img: np.ndarray, cx: float, cy: float, r: float, r_in: float, r_out: float) -> dict:
    """Apertura kołowa, tło = mediana pierścienia, σ piksela z MAD pierścienia."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    ap, ann = d <= r, (d >= r_in) & (d <= r_out)
    bg = float(np.median(img[ann]))
    sig = float(1.4826 * np.median(np.abs(img[ann] - bg))) + 1e-12
    npx, nann = int(ap.sum()), int(ann.sum())
    flux = float((img[ap] - bg).sum())
    err = sig * math.sqrt(npx * (1 + npx / max(nann, 1)))
    return {"flux": flux, "err": err, "snr": flux / err, "bg": bg, "sigma": sig, "peak": float(img[ap].max())}


def centroid(img: np.ndarray, cx: float, cy: float, search: float, box: int = 2) -> tuple[float, float]:
    """Najjaśniejszy piksel w promieniu ``search`` od (cx, cy), potem środek ciężkości w oknie
    ±``box`` (po odjęciu mediany obrazu)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.hypot(xx - cx, yy - cy) <= search
    if not m.any():
        return cx, cy
    py, px = np.unravel_index(int(np.argmax(np.where(m, img, -np.inf))), img.shape)
    y0, y1, x0, x1 = max(py - box, 0), min(py + box + 1, h), max(px - box, 0), min(px + box + 1, w)
    win = np.clip(img[y0:y1, x0:x1] - np.median(img), 0, None)
    s = win.sum()
    if s <= 0:
        return float(px), float(py)
    gy, gx = np.mgrid[y0:y1, x0:x1]
    return float((gx * win).sum() / s), float((gy * win).sum() / s)


def fit_zero_point(r2: np.ndarray, mags: np.ndarray, fluxes: np.ndarray, *, clip: float = 3.0):
    """ZP(r) = a + b·r² (winietowanie) z gwiazd o znanej jasności; odporne (3σ z MAD).
    Zwraca (a, b, rms, n) albo None przy < 4 gwiazdach."""
    r2, mags, fluxes = (np.asarray(v, float) for v in (r2, mags, fluxes))
    ok = np.isfinite(r2) & np.isfinite(mags) & np.isfinite(fluxes) & (fluxes > 0)
    if ok.sum() < 4:
        return None
    zp = mags + 2.5 * np.log10(np.where(ok, fluxes, 1.0))
    use = ok.copy()
    for _ in range(5):
        b, a = np.polyfit(r2[use], zp[use], 1)
        res = zp - (a + b * r2)
        s = 1.4826 * np.median(np.abs(res[use])) + 1e-3
        new = ok & (np.abs(res) <= clip * s)
        if new.sum() < 4 or np.array_equal(new, use):
            break
        use = new
    b, a = np.polyfit(r2[use], zp[use], 1)
    rms = float(np.sqrt(np.mean((zp[use] - (a + b * r2[use])) ** 2)))
    return float(a), float(b), rms, int(use.sum())


def mag_from_flux(flux: float, zp: float) -> float:
    return zp - 2.5 * math.log10(flux) if flux > 0 else float("nan")


def status(r: dict, scfg: dict) -> str:
    """Werdykt dla zmierzonego obiektu (liczby są w tabeli raportu)."""
    v, m, lim = r["vmag"], r["mag_meas"], r["mag_lim"]
    blend = r.get("blend_mag", float("nan"))
    found = math.isfinite(r["snr"]) and r["snr"] >= float(scfg["detect_snr"])
    if math.isfinite(blend) and math.isfinite(v) and blend <= v + float(scfg["blend_dmag"]):
        return f"zlewa się z gwiazdą (G {blend:.1f})"
    if found and (not math.isfinite(m) or not math.isfinite(v) or abs(m - v) <= float(scfg["mag_tol"])):
        return "wykryta"
    if found:
        return f"źródło, jasność niezgodna ({m:.1f} vs {v:.1f})"
    if math.isfinite(lim) and math.isfinite(v) and v > lim:
        return f"za słaba (zasięg {lim:.1f} mag)"
    return "niewykryta"


# ---------------------------------------------------------------- szybkie NEO jako tory

def _mid_radec(t: dict) -> tuple[float, float]:
    from .sky import radec_to_vec, vec_to_radec

    v = radec_to_vec(t["ra0"], t["dec0"]) + radec_to_vec(t["ra1"], t["dec1"])
    ra, dec = vec_to_radec(v / np.linalg.norm(v))
    return float(ra), float(dec)


def match_tracks(tracks: list[dict], bodies: list[dict], t_query: datetime, t0: datetime, delta_s: float,
                 scfg: dict) -> list[dict]:
    """Niezidentyfikowany tor ↔ małe ciało: pozycja w środku toru ≤ ``track_match_deg`` od
    przewidywanej i tempo ruchu zgodne co do rzędu (×0,5…2). Dotyczy tylko bardzo bliskich NEO
    (tor wymaga ≥ ~0,1 px/klatkę, czyli dziesiątek tysięcy ″/h)."""
    from .sky import angle_deg, radec_to_vec

    tol = float(scfg.get("track_match_deg", 0.2))
    out = []
    for t in tracks:
        dur_h = max(float(t["tau1"]) - float(t["tau0"]), 1e-3) / 3600.0
        a, b = radec_to_vec(t["ra0"], t["dec0"]), radec_to_vec(t["ra1"], t["dec1"])
        track_rate = float(angle_deg(a, b)) * 3600.0 / dur_h
        mid = radec_to_vec(*_mid_radec(t))
        t_mid = t0 + timedelta(seconds=delta_s + (float(t["tau0"]) + float(t["tau1"])) / 2)
        for body in bodies:
            rate = rate_arcsec_h(body)
            if rate <= 0 or not 0.5 <= track_rate / rate <= 2.0:
                continue
            ra, dec = body_radec(body, (t_mid - t_query).total_seconds())
            sep = float(angle_deg(mid, radec_to_vec(float(ra), float(dec))))
            if sep <= tol:
                out.append({"track_id": int(t["track_id"]), "name": body["name"], "is_neo": body["is_neo"],
                            "sep_deg": sep, "track_rate": track_rate, "body_rate": rate})
    return out


# ---------------------------------------------------------------- etap

COLS = ["name", "kind", "is_neo", "vmag", "ra", "dec", "ra_rate", "dec_rate", "rate_arcsec_h", "motion_px",
        "x", "y", "in_frame", "measured", "snr", "mag_meas", "mag_lim", "offset_arcsec", "blend_mag", "status",
        "stamp", "stamp_cx", "stamp_cy", "note"]
TRACK_COLS = ["track_id", "name", "is_neo", "sep_deg", "track_rate", "body_rate"]


def run(outdir: Path, video: Path, meta, cfg: dict, *, camera, clock, t0: datetime, delta_s: float,
        site: tuple[float, float, float], open_batches: Callable[[], Iterable],
        log_: logging.Logger | None = None) -> dict:
    """Zapytanie JPL, stos okien (obiekty + gwiazdy do punktu zerowego), fotometria, werdykt."""
    import pandas as pd

    from .io import read_json, write_json
    from .sky import stars

    lg = log_ or log
    scfg = cfg["smallbodies"]
    name = video.name
    H, W = meta.height, meta.width
    dur = meta.n_frames / meta.fps
    tau_mid = dur / 2
    t_query = t0 + timedelta(seconds=delta_s + tau_mid)
    wcsinfo = read_json(outdir / "wcs.json")
    scale = float((wcsinfo.get("fov") or {}).get("scale_arcsec_px") or 24.7)
    ra_c, dec_c = camera.icrs(W / 2, H / 2, tau_mid)
    ra_c, dec_c = float(np.atleast_1d(ra_c)[0]), float(np.atleast_1d(dec_c)[0])
    radius = math.hypot(W, H) / 2 * scale / 3600 + float(scfg.get("fov_margin_deg", 1.0))

    info: dict = {"t_query_utc": t_query.isoformat(), "center_ra": ra_c, "center_dec": dec_c,
                  "radius_deg": radius, "source": "JPL Small-Body Identification API (sb_ident)"}
    main = fetch_sbident(query_params(ra_c, dec_c, radius, t_query, site, scfg), scfg)
    neo = fetch_sbident(query_params(ra_c, dec_c, radius, t_query, site, scfg, neo=True), scfg)
    info["api_version"] = (main.get("signature") or {}).get("version")
    info["site_sent"] = {k: (main.get("summary") or {}).get(k) for k in ("lat", "lon", "alt")}
    bodies = merge_bodies(parse_sbident(main), parse_sbident(neo, is_neo=True))

    rows = []
    for b in bodies:
        ra, dec = body_radec(b, 0.0)
        x, y = camera.pixel(ra, dec, tau_mid)
        x, y = float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0])
        inside = bool(0 <= x < W and 0 <= y < H)
        rows.append({**b, "rate_arcsec_h": rate_arcsec_h(b), "motion_px": rate_arcsec_h(b) * dur / 3600.0 / scale,
                     "x": x, "y": y, "in_frame": inside, "measured": False, "snr": float("nan"),
                     "mag_meas": float("nan"), "mag_lim": float("nan"), "offset_arcsec": float("nan"),
                     "blend_mag": float("nan"), "stamp": "", "stamp_cx": float("nan"), "stamp_cy": float("nan"),
                     "note": "", "status": "nie mierzono (limit max_targets)" if inside else "poza kadrem"})
    in_frame = [r for r in rows if r["in_frame"]]
    targets = in_frame[:int(scfg["max_targets"])]          # rows są od najjaśniejszych
    lg.info("[%s] małe ciała (JPL): %d w zapytaniu, %d w kadrze, mierzę %d", name, len(rows), len(in_frame),
            len(targets))

    # gwiazdy do punktu zerowego: z katalogu, niezasycone, z dala od brzegów (okno jedzie z niebem)
    cat = stars()
    lo_m, hi_m = (float(v) for v in scfg["zp_mag_range"])
    sx, sy = camera.pixel(cat["ra"], cat["dec"], tau_mid)
    sx, sy = np.asarray(sx, float), np.asarray(sy, float)
    hw = int(scfg["window_px"])
    edge = hw + 200
    ok = ((cat["mag"] >= lo_m) & (cat["mag"] <= hi_m) & np.isfinite(sx) & np.isfinite(sy)
          & (sx > edge) & (sx < W - edge) & (sy > edge) & (sy < H - edge))
    zp_idx = np.flatnonzero(ok)
    if len(zp_idx) > int(scfg["zp_max_stars"]):
        zp_idx = zp_idx[np.argsort(cat["mag"][zp_idx])[:int(scfg["zp_max_stars"])]]

    # pozycje na siatce czasu (astropy raz), między węzłami interpolacja liniowa
    K, nt, ns = len(targets) + len(zp_idx), len(targets), len(zp_idx)
    n_grid = max(2, int(math.ceil(dur / float(scfg.get("pos_step_s", 2.0)))) + 1)
    fgrid = np.linspace(0, meta.n_frames - 1, n_grid)
    tgrid = np.asarray(clock.tau(fgrid, np.full(n_grid, H / 2.0)), float)
    grid = np.full((K, n_grid, 2), np.nan)
    dt0 = (t0 + timedelta(seconds=delta_s) - t_query).total_seconds()
    for k, b in enumerate(targets):
        ra, dec = body_radec(b, dt0 + tgrid)
        x, y = camera.pixel(ra, dec, tgrid)
        grid[k, :, 0], grid[k, :, 1] = x, y
    if ns:
        x, y = camera.pixel(np.repeat(cat["ra"][zp_idx], n_grid), np.repeat(cat["dec"][zp_idx], n_grid),
                            np.tile(tgrid, ns))
        grid[nt:, :, 0] = np.asarray(x, float).reshape(ns, n_grid)
        grid[nt:, :, 1] = np.asarray(y, float).reshape(ns, n_grid)

    def pos_at(frames: np.ndarray) -> np.ndarray:
        out = np.empty((K, len(frames), 2))
        for k in range(K):
            out[k, :, 0] = np.interp(frames, fgrid, grid[k, :, 0])
            out[k, :, 1] = np.interp(frames, fgrid, grid[k, :, 1])
        return out

    tr = cfg.get("color", {}).get("transfer", "bt709")
    full_range = bool(scfg.get("y_full_range", False))
    stacks = stack_windows(open_batches(), pos_at, K, hw, (H, W), lambda v: y_to_linear(v, tr, full_range)) if K else []
    r_ap = float(scfg["aperture_px"])
    r_in, r_out = (float(v) for v in scfg["annulus_px"])
    sat = float(scfg.get("zp_sat_linear", 0.95))

    # punkt zerowy z winietowaniem: ZP(r) = a + b·r², r w połówkach szerokości kadru
    def r2_of(x: float, y: float) -> float:
        return ((x - W / 2) ** 2 + (y - H / 2) ** 2) / (W / 2) ** 2

    zr2, zm, zf, zerr = [], [], [], []
    for j, i in enumerate(zp_idx):
        st = stacks[nt + j]
        if st.mean is None:
            continue
        ph = photometry(st.mean, *st.center, r_ap, r_in, r_out)
        if ph["peak"] >= sat or ph["snr"] < 10:
            continue
        zr2.append(r2_of(float(sx[i]), float(sy[i])))
        zm.append(float(cat["mag"][i]))
        zf.append(ph["flux"])
        zerr.append(ph["err"])
    zp = fit_zero_point(np.array(zr2), np.array(zm), np.array(zf))
    snr_det = float(scfg["detect_snr"])
    if zp:
        info.update(zp_a=zp[0], zp_b=zp[1], zp_rms=zp[2], zp_stars=zp[3])
        err_med = float(np.median(zerr))
        info["mag_lim_center"] = zp[0] - 2.5 * math.log10(snr_det * err_med) if err_med > 0 else None
        lg.info("[%s] punkt zerowy z %d gwiazd (RMS %.2f mag), zasięg stosu (%.0fσ, środek kadru): %s mag",
                name, zp[3], zp[2], snr_det, f"{info['mag_lim_center']:.1f}" if info["mag_lim_center"] else "–")
    else:
        lg.warning("[%s] za mało gwiazd do punktu zerowego — bez jasności zmierzonych", name)

    stamps = {}
    for k, r in enumerate(targets):
        st = stacks[k]
        if st.mean is None:
            r["status"] = "przy krawędzi (okno poza kadrem)"
            continue
        cx, cy = st.center
        ph = photometry(st.mean, cx, cy, r_ap, r_in, r_out)
        mx, my = centroid(st.mean, cx, cy, float(scfg["search_px"]))
        r.update(measured=True, snr=ph["snr"], offset_arcsec=math.hypot(mx - cx, my - cy) * scale,
                 stamp=f"b{k}", stamp_cx=cx, stamp_cy=cy)
        stamps[f"b{k}"] = st.mean.astype(np.float32)
        if zp:
            zpk = zp[0] + zp[1] * r2_of(r["x"], r["y"])
            r["mag_meas"] = mag_from_flux(ph["flux"], zpk)
            r["mag_lim"] = zpk - 2.5 * math.log10(snr_det * ph["err"]) if ph["err"] > 0 else float("nan")
        try:
            v = r["vmag"] if math.isfinite(r["vmag"]) else 15.0
            nb = gaia_neighbors(r["ra"], r["dec"], (r_ap + 1) * scale, v + float(scfg["blend_dmag"]), scfg)
            r["blend_mag"] = min((m for _, _, m in nb), default=float("nan"))
        except Exception as e:  # noqa: BLE001 — bez VizieR: bez sprawdzenia gwiazd tła
            r["note"] = f"gwiazdy tła niesprawdzone ({type(e).__name__})"
        r["status"] = status(r, scfg)

    matches = []
    tf = outdir / "tracks_final.parquet"
    if tf.exists() and bodies:
        final = pd.read_parquet(tf)
        matches = match_tracks(final[final["kind"] != "sat"].to_dict("records"), bodies, t_query, t0, delta_s, scfg)

    pd.DataFrame(rows, columns=COLS).to_csv(outdir / "smallbodies.csv", index=False)
    pd.DataFrame(matches, columns=TRACK_COLS).to_csv(outdir / "smallbodies_tracks.csv", index=False)
    np.savez_compressed(outdir / "smallbodies_stamps.npz", **stamps)
    det = [r["name"] for r in rows if r["status"] == "wykryta"]
    info.update(n_query=len(rows), n_in_frame=len(in_frame), n_measured=sum(1 for r in rows if r["measured"]),
                n_detected=len(det), track_matches=len(matches))
    write_json(outdir / "smallbodies.json", info)
    lg.info("[%s] małe ciała: wykryte %s; tory NEO: %d", name, ", ".join(det) or "brak", len(matches))
    return {"outputs": ["smallbodies.csv", "smallbodies_tracks.csv", "smallbodies_stamps.npz", "smallbodies.json"],
            "metrics": {"in_frame": len(in_frame), "detected": len(det), "track_matches": len(matches),
                        "mag_lim_center": info.get("mag_lim_center")}}


def empty_outputs(outdir: Path, info: dict) -> dict:
    """Puste wyniki (wyłączone / brak sieci): raport powstaje i tak."""
    import pandas as pd

    from .io import write_json

    pd.DataFrame(columns=COLS).to_csv(outdir / "smallbodies.csv", index=False)
    pd.DataFrame(columns=TRACK_COLS).to_csv(outdir / "smallbodies_tracks.csv", index=False)
    write_json(outdir / "smallbodies.json", info)
    return {"outputs": ["smallbodies.csv", "smallbodies_tracks.csv", "smallbodies.json"],
            "metrics": {k: v for k, v in info.items() if k in ("enabled", "error")}}
