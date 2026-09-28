"""Satelity: snapshot elementów orbit (CelesTrak / Space-Track, CSV/OMM), propagacja SGP4,
synchronizacja zegara kamery po przelotach i identyfikacja torów (NORAD).

Elementy trzymamy w CSV/OMM, nie TLE: od 07.2026 nowe obiekty mają numery NORAD ≥ 100000,
których format TLE nie obejmuje. Numer NORAD bierzemy z wiersza CSV.

Czas: ``offset`` to sekundy od ``t_ref`` (wstępny start nagrania). Punkt toru z chwili
nagrania τ odpowiada chwili fizycznej t_ref + Δ + τ; Δ = poprawka zegara.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Callable

import numpy as np

from .sky import GreatCircle, angle_deg, fit_great_circle

log = logging.getLogger("skyhunt")

EARTH_RADIUS_KM = 6378.137
SIDEREAL_DEG_S = 360.0 / 86164.0905
_SNAP = re.compile(r"^(?P<src>celestrak|spacetrack)_(?P<group>.+)_(?P<stamp>\d{8}T\d{4}Z)\.csv$")
SPACETRACK_LOGIN = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_QUERY = ("https://www.space-track.org/basicspacedata/query/class/gp_history/EPOCH/{a}--{b}"
                    "/orderby/NORAD_CAT_ID%20asc/format/csv")


# ---------------------------------------------------------------- cache elementów orbit

class FetchError(RuntimeError):
    pass


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")


def list_snapshots(cache_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(cache_dir).glob("*.csv")):
        m = _SNAP.match(p.name)
        if m:
            fetched = datetime.strptime(m["stamp"], "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)
            out.append({"path": p, "source": m["src"], "group": m["group"], "fetched": fetched})
    return out


def fetch_url(url: str, opener: Callable = urllib.request.urlopen, timeout: float = 120, data=None) -> bytes:
    """Pobranie bez ponawiania: CelesTrak blokuje IP po serii błędów (301/403/404)."""
    try:
        with (opener(url, data=data, timeout=timeout) if data is not None else opener(url, timeout=timeout)) as r:
            status = getattr(r, "status", 200)
            if status != 200:
                raise FetchError(f"HTTP {status} dla {url.split('?')[0]}")
            return r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} dla {url.split('?')[0]} — nie ponawiam") from None


def select_celestrak(scfg: dict, t_rec: datetime, *, now: datetime | None = None,
                     opener: Callable = urllib.request.urlopen) -> list[dict]:
    """Dla każdej grupy: najwcześniejszy snapshot pobrany po nagraniu; w razie braku —
    pobranie (świeże nagranie, brak pobrania w ostatnich ``min_refetch_h``) albo najnowszy
    wcześniejszy snapshot z ostrzeżeniem."""
    now = now or datetime.now(timezone.utc)
    cache = Path(scfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    snaps = [s for s in list_snapshots(cache) if s["source"] == "celestrak"]
    age_days = (now - t_rec).total_seconds() / 86400
    fetch_ok = bool(scfg.get("auto_fetch", True)) and age_days < float(scfg["max_elements_age_days"])
    chosen = []
    for g in scfg["celestrak_groups"]:
        mine = sorted((s for s in snaps if s["group"] == g), key=lambda s: s["fetched"])
        after = [s for s in mine if s["fetched"] >= t_rec]
        if after:
            chosen.append(after[0])
            continue
        recent = [s for s in mine if (now - s["fetched"]).total_seconds() < float(scfg["min_refetch_h"]) * 3600]
        if fetch_ok and not recent:
            url = f"{scfg['celestrak_url']}?GROUP={urllib.parse.quote(g)}&FORMAT=csv"
            try:
                data = fetch_url(url, opener)
            except (FetchError, OSError) as e:
                log.warning("CelesTrak %s: %s; dalsze pobieranie wstrzymane", g, e)
                fetch_ok = False
            else:
                path = cache / f"celestrak_{g}_{_stamp(now)}.csv"
                path.write_bytes(data)
                chosen.append({"path": path, "source": "celestrak", "group": g, "fetched": now})
                continue
        if mine:
            log.warning("CelesTrak %s: brak snapshotu po nagraniu, używam %s", g, mine[-1]["path"].name)
            chosen.append(mine[-1])
        else:
            log.warning("CelesTrak %s: brak danych", g)
    return chosen


def fetch_spacetrack(scfg: dict, t_rec: datetime, *, now: datetime | None = None,
                     opener_factory: Callable | None = None) -> dict | None:
    """Historia elementów z Space-Track (jedno zapytanie, cache na Drive). Dane logowania
    wyłącznie z ``$SPACETRACK_USER`` / ``$SPACETRACK_PASSWORD``; nigdy nie trafiają do logów."""
    user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASSWORD")
    if str(scfg.get("spacetrack", "auto")) == "off":
        return None
    if not (user and pw):
        log.info("Space-Track: brak $SPACETRACK_USER/$SPACETRACK_PASSWORD (komórka Secrets) — tylko CelesTrak")
        return None
    a = (t_rec - timedelta(days=float(scfg["history_days_before"]))).date()
    b = (t_rec + timedelta(days=float(scfg["history_days_after"]) + 1)).date()
    group = f"gp-history-{a}--{b}"
    cache = Path(scfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    for s in list_snapshots(cache):
        if s["source"] == "spacetrack" and s["group"] == group:
            return s
    opener = (opener_factory or (lambda: urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar()))))()
    body = urllib.parse.urlencode({"identity": user, "password": pw}).encode()
    try:
        fetch_url(SPACETRACK_LOGIN, opener.open, data=body)
        data = fetch_url(SPACETRACK_QUERY.format(a=a, b=b), opener.open, timeout=900)
    except (FetchError, OSError) as e:
        log.warning("Space-Track: %s", str(e).replace(pw, "***").replace(user, "***"))
        return None
    now = now or datetime.now(timezone.utc)
    path = cache / f"spacetrack_{group}_{_stamp(now)}.csv"
    path.write_bytes(data)
    return {"path": path, "source": "spacetrack", "group": group, "fetched": now}


# ---------------------------------------------------------------- katalog

def _epoch(s: str) -> datetime:
    s = s.strip().replace("Z", "")
    if "." in s:   # fromisoformat w starszych Pythonach wymaga 3 lub 6 cyfr ułamka
        head, frac = s.split(".", 1)
        s = f"{head}.{(frac + '000000')[:6]}"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@dataclass
class Catalog:
    satrecs: list
    norad: np.ndarray
    name: list
    object_id: list
    epoch: list
    source: list

    def __len__(self) -> int:
        return len(self.satrecs)


def load_catalog(files: list[tuple[Path, str]], t_rec: datetime) -> Catalog:
    """Wczytuje CSV/OMM; dla każdego NORAD wybiera elementy z epoką najbliższą nagraniu."""
    from sgp4 import omm
    from sgp4.api import Satrec

    best: dict[int, tuple] = {}
    for path, source in files:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for fields in omm.parse_csv(fh):
                try:
                    nid = int(fields["NORAD_CAT_ID"])
                    ep = _epoch(fields["EPOCH"])
                except (KeyError, ValueError):
                    continue
                d = abs((ep - t_rec).total_seconds())
                if nid not in best or d < best[nid][0]:
                    best[nid] = (d, fields, fields.get("SOURCE") or source, ep)
    sats, norad, names, oids, epochs, sources = [], [], [], [], [], []
    for nid in sorted(best):
        _, fields, source, ep = best[nid]
        sat = Satrec()
        try:
            omm.initialize(sat, fields)
        except (KeyError, ValueError):
            continue
        sats.append(sat)
        norad.append(nid)
        names.append(fields.get("OBJECT_NAME", "").strip())
        oids.append(fields.get("OBJECT_ID", "").strip())
        epochs.append(ep)
        sources.append(source)
    return Catalog(sats, np.array(norad, dtype=np.int64), names, oids, epochs, sources)


def write_catalog_csv(files: list[tuple[Path, str]], catalog: Catalog, out: Path) -> None:
    """Zamrożona kopia użytych wierszy (odtwarzalność wyniku)."""
    from sgp4 import omm

    keep = set(int(n) for n in catalog.norad)
    rows, header = {}, None
    for path, source in files:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for fields in omm.parse_csv(fh):
                try:
                    nid, ep = int(fields["NORAD_CAT_ID"]), _epoch(fields["EPOCH"])
                except (KeyError, ValueError):
                    continue
                i = int(np.searchsorted(catalog.norad, nid))
                if nid in keep and catalog.epoch[i] == ep:
                    rows[nid] = {**fields, "SOURCE": source}
                    header = header or list(rows[nid].keys())
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, header or ["NORAD_CAT_ID"], extrasaction="ignore")
        w.writeheader()
        for nid in sorted(rows):
            w.writerow(rows[nid])


# ---------------------------------------------------------------- propagacja

class Observer:
    """Obserwator na Ziemi + propagacja SGP4 do kierunków topocentrycznych (GCRS)."""

    def __init__(self, lat_deg: float, lon_deg: float, elevation_m: float, t_ref: datetime):
        from sgp4.api import jday
        from skyfield.api import load, wgs84

        self.ts = load.timescale()
        self.topos = wgs84.latlon(lat_deg, lon_deg, elevation_m=elevation_m)
        self.t_ref = t_ref.astimezone(timezone.utc)
        r = self.t_ref
        self.jd0, self.fr0 = jday(r.year, r.month, r.day, r.hour, r.minute, r.second + r.microsecond * 1e-6)

    def times(self, offsets):
        r = self.t_ref
        return self.ts.utc(r.year, r.month, r.day, r.hour, r.minute,
                           r.second + r.microsecond * 1e-6 + np.atleast_1d(np.asarray(offsets, float)))

    def jd_fr(self, offsets) -> tuple[np.ndarray, np.ndarray]:
        fr = self.fr0 + np.atleast_1d(np.asarray(offsets, float)) / 86400.0
        k = np.floor(fr)
        return self.jd0 + k, fr - k

    def up_vectors(self, offsets) -> np.ndarray:
        p = self.topos.at(self.times(offsets)).position.km.T
        return p / np.linalg.norm(p, axis=-1, keepdims=True)

    def topocentric(self, satrecs: list, offsets) -> dict:
        """Kierunki [S, T, 3] (geometryczne, GCRS), odległości [km], promień geocentryczny, błędy SGP4."""
        from sgp4.api import SatrecArray
        from skyfield.sgp4lib import TEME

        offsets = np.atleast_1d(np.asarray(offsets, float))
        jd, fr = self.jd_fr(offsets)
        err, r, _ = SatrecArray(list(satrecs)).sgp4(jd, fr)
        t = self.times(offsets)
        rot = TEME.rotation_at(t)            # GCRS → TEME, [3, 3, T]
        if rot.ndim == 2:
            rot = rot[..., None]
        r_gcrs = np.einsum("jit,stj->sti", rot, r)   # transpozycja: TEME → GCRS
        obs = self.topos.at(t).position.km.T         # [T, 3]
        d = r_gcrs - obs[None]
        dist = np.linalg.norm(d, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = d / dist[..., None]
        unit[err != 0] = np.nan
        return {"unit": unit, "dist_km": dist, "r_km": np.linalg.norm(r_gcrs, axis=-1), "err": err}


# ---------------------------------------------------------------- tor na niebie

@dataclass
class TrackSky:
    track_id: int
    tau: np.ndarray       # czas nagrania punktów [s] (z przesunięciem ekspozycji)
    vec: np.ndarray       # kierunki pozorne [N, 3]
    ra: np.ndarray
    dec: np.ndarray
    gc: GreatCircle       # czas względem tau_mid
    tau_mid: float

    @property
    def u_mid(self) -> np.ndarray:
        return self.gc.position(0.0)

    @property
    def dur_s(self) -> float:
        return float(self.tau[-1] - self.tau[0])


def make_track_sky(track_id: int, tau: np.ndarray, vec: np.ndarray, ra, dec) -> TrackSky:
    tau_mid = float(0.5 * (tau[0] + tau[-1]))
    return TrackSky(track_id, np.asarray(tau, float), np.asarray(vec, float), np.asarray(ra), np.asarray(dec),
                    fit_great_circle(vec, tau - tau_mid), tau_mid)


# ---------------------------------------------------------------- przesiew i przejścia

@dataclass
class SatGrid:
    idx: np.ndarray       # indeksy w katalogu
    offsets: np.ndarray   # [T] s od t_ref
    unit: np.ndarray      # [S, T, 3]


def screen(observer: Observer, catalog: Catalog, offsets: np.ndarray, field_vecs: np.ndarray,
           radius_deg: float, icfg: dict, extra_margin_deg: float = 0.0, chunk: int = 400) -> SatGrid:
    """Satelity, które w siatce czasów przechodzą w pobliżu kadru nad horyzontem."""
    lim = radius_deg + float(icfg["fov_margin_deg"]) + extra_margin_deg
    up = observer.up_vectors(offsets)
    keep_idx, keep_unit = [], []
    for c0 in range(0, len(catalog), chunk):
        topo = observer.topocentric(catalog.satrecs[c0:c0 + chunk], offsets)
        u = topo["unit"]
        with np.errstate(invalid="ignore"):
            near = angle_deg(u, field_vecs[None]) <= lim
            alt = 90.0 - angle_deg(u, up[None])
            ok = near & (alt >= float(icfg["min_alt_deg"])) & (topo["err"] == 0)
        hit = np.flatnonzero(ok.any(axis=1))
        keep_idx.append(c0 + hit)
        keep_unit.append(u[hit])
    idx = np.concatenate(keep_idx) if keep_idx else np.empty(0, int)
    unit = np.concatenate(keep_unit) if keep_unit else np.empty((0, len(offsets), 3))
    return SatGrid(idx, np.asarray(offsets, float), unit)


def closest_approach(unit: np.ndarray, offsets: np.ndarray, u: np.ndarray, t_lo: float = -np.inf,
                     t_hi: float = np.inf) -> dict:
    """Dla każdego satelity (siatka [S, T, 3]): najmniejsza odległość łuku ruchu od kierunku
    ``u`` w oknie [t_lo, t_hi], czas tego zbliżenia, prędkość kątowa i kierunek ruchu."""
    a, b = unit[:, :-1], unit[:, 1:]
    dt = np.diff(offsets)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.cross(a, b)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        s = n @ u
        p = u[None, None] - s[..., None] * n
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        inside = (np.sum(np.cross(a, p) * n, -1) >= 0) & (np.sum(np.cross(p, b) * n, -1) >= 0)
        d_line = np.degrees(np.arcsin(np.clip(np.abs(s), 0, 1)))
        d_end = np.minimum(angle_deg(a, u[None, None]), angle_deg(b, u[None, None]))
        dist = np.where(inside, d_line, d_end)
    seg_ok = (offsets[1:] >= t_lo) & (offsets[:-1] <= t_hi)
    dist = np.where(np.isfinite(dist) & seg_ok[None], dist, np.inf)
    k = np.argmin(dist, axis=1)
    rows = np.arange(len(unit))
    ak, bk, pk, nk = a[rows, k], b[rows, k], p[rows, k], n[rows, k]
    seg = angle_deg(ak, bk)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.clip(angle_deg(ak, pk) / seg, 0, 1)
    return {"dist_deg": dist[rows, k], "t": offsets[k] + frac * dt[k], "speed_deg_s": seg / dt[k],
            "direction": np.cross(nk, pk)}


@dataclass
class Match:
    track_id: int
    cat_index: int
    norad: int
    name: str
    object_id: str
    epoch: str
    source: str
    delta_s: float          # δ: chwila fizyczna punktu = t_ref + δ + τ
    rms_deg: float
    cross_deg: float
    along_arcsec: float
    speed_ratio: float
    dir_deg: float
    range_km: float
    height_km: float
    sat_omega_deg_s: float
    sunlit: bool | None = None
    confidence: str = ""
    reason: str = ""
    ambiguous_with: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fine_align(observer: Observer, catalog: Catalog, i: int, track: TrackSky, delta0: float,
               half_range: float) -> Match | None:
    """Doprecyzowanie δ (minimum RMS odległości punktów toru od satelity) i metryki dopasowania."""
    from scipy.optimize import minimize_scalar

    sat = [catalog.satrecs[i]]

    def positions(delta):
        return observer.topocentric(sat, delta + track.tau)

    def rms(delta):
        u = positions(delta)["unit"][0]
        a = angle_deg(u, track.vec)
        return float(np.sqrt(np.nanmean(a ** 2))) if np.isfinite(a).any() else 1e9

    res = minimize_scalar(rms, bounds=(delta0 - half_range, delta0 + half_range), method="bounded",
                          options={"xatol": 1e-3})
    delta = float(res.x)
    topo = positions(delta)
    su = topo["unit"][0]
    if not np.isfinite(su).all():
        return None
    t_rel = track.tau - track.tau_mid
    sgc = fit_great_circle(su, t_rel)
    along = np.angle(np.exp(1j * (track.gc.phase(su) - track.gc.phase(track.vec))))
    m = len(su) // 2
    return Match(
        track_id=track.track_id, cat_index=int(i), norad=int(catalog.norad[i]), name=catalog.name[i],
        object_id=catalog.object_id[i], epoch=catalog.epoch[i].isoformat(), source=catalog.source[i],
        delta_s=delta, rms_deg=float(res.fun),
        cross_deg=float(np.sqrt(np.mean(track.gc.cross_offset_deg(su) ** 2))),
        along_arcsec=float(np.degrees(np.sqrt(np.mean(along ** 2))) * 3600),
        speed_ratio=float(sgc.omega_rad_s / track.gc.omega_rad_s) if track.gc.omega_rad_s else float("nan"),
        dir_deg=float(angle_deg(track.gc.tangent(0.0), sgc.tangent(0.0))),
        range_km=float(topo["dist_km"][0, m]), height_km=float(topo["r_km"][0, m] - EARTH_RADIUS_KM),
        sat_omega_deg_s=sgc.omega_deg_s,
    )


def candidate_pairs(grid: SatGrid, track: TrackSky, icfg: dict, t_lo: float, t_hi: float,
                    cross_tol: float, speed_tol: float, dir_tol: float, chunk: int = 256) -> list[tuple[int, float]]:
    """Wstępne pary (indeks w katalogu, δ₀) z siatki przesiewu."""
    out = []
    omega = track.gc.omega_deg_s
    # tylko fragment siatki w oknie czasu, satelity porcjami (pamięć przy oknie ±2 h)
    lo, hi = t_lo + track.tau_mid, t_hi + track.tau_mid
    k0 = max(0, int(np.searchsorted(grid.offsets, lo)) - 1)
    k1 = min(len(grid.offsets), int(np.searchsorted(grid.offsets, hi)) + 1)
    if k1 - k0 < 2:
        return out
    offs = grid.offsets[k0:k1]
    for s0 in range(0, len(grid.idx), chunk):
        ca = closest_approach(grid.unit[s0:s0 + chunk, k0:k1], offs, track.u_mid, lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (ca["dist_deg"] <= cross_tol) \
                & (np.abs(ca["speed_deg_s"] / omega - 1) <= speed_tol) \
                & (angle_deg(ca["direction"], track.gc.tangent(0.0)[None]) <= dir_tol)
        out += [(int(grid.idx[s0 + j]), float(ca["t"][j] - track.tau_mid)) for j in np.flatnonzero(ok)]
    return out


# ---------------------------------------------------------------- synchronizacja czasu

@dataclass
class SyncResult:
    synced: bool
    delta_s: float
    sigma_s: float
    confidence: str
    method: str
    reference: dict | None
    members: list = field(default_factory=list)       # dopasowania potwierdzające Δ
    window_s: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def observer_offset(observer: Observer, catalog: Catalog, members: list, skies: dict) -> dict | None:
    """Przesunięcie obserwatora [km] z paralaksy torów, które potwierdziły Δ: błędne
    współrzędne dają boczne odchylenie ∝ 1/odległość satelity. Model liniowy:
    u_zmierz − u_przew ≈ −(I − u uᵀ) d / r, rozwiązany metodą najmniejszych kwadratów."""
    A, e = [], []
    for m in members:
        m = m if isinstance(m, dict) else m.to_dict()
        s = skies.get(int(m["track_id"]))
        if s is None:
            continue
        k = len(s.tau) // 2
        topo = observer.topocentric([catalog.satrecs[int(m["cat_index"])]], [float(m["delta_s"]) + s.tau[k]])
        u, dist = topo["unit"][0, 0], float(topo["dist_km"][0, 0])
        if not np.isfinite(u).all():
            continue
        A.append(-(np.eye(3) - np.outer(u, u)) / dist)
        e.append(s.vec[k] - u)
    if len(A) < 2:          # tory (każdy daje 2 niezależne równania; niewiadome: 3)
        return None
    A, e = np.vstack(A), np.concatenate(e)
    d, *_ = np.linalg.lstsq(A, e, rcond=None)
    up = observer.up_vectors([0.0])[0]
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)

    def rms_arcsec(r):
        return float(np.degrees(np.sqrt(np.mean(np.sum(r.reshape(-1, 3) ** 2, axis=1)))) * 3600)

    n_km, e_km = float(d @ north), float(d @ east)
    return {"north_km": n_km, "east_km": e_km, "up_km": float(d @ up), "horizontal_km": float(np.hypot(n_km, e_km)),
            "rms_before_arcsec": rms_arcsec(e), "rms_after_arcsec": rms_arcsec(e - A @ d), "n_tracks": len(A) // 3}


def is_sync_candidate(track: TrackSky, icfg: dict) -> bool:
    return (track.gc.cross_rms_arcsec <= float(icfg["cand_max_curv_arcsec"])
            and float(icfg["cand_min_deg_s"]) <= track.gc.omega_deg_s <= float(icfg["cand_max_deg_s"])
            and track.dur_s >= float(icfg["cand_min_dur_s"]))


def _sigma_delta(m: Match, track: TrackSky, npts: int, icfg: dict) -> float:
    omega = max(track.gc.omega_deg_s, 1e-6)
    meas = (m.along_arcsec / 3600.0) / omega / math.sqrt(max(npts, 1))
    v_km_s = math.radians(omega) * m.range_km
    tle = float(icfg["tle_along_track_sigma_km"]) / max(v_km_s, 1e-6)
    return math.sqrt(meas ** 2 + tle ** 2)


def synchronize(observer: Observer, catalog: Catalog, tracks: list[TrackSky], grid_for: Callable,
                prior_sigma_s: float, search_s: float, icfg: dict) -> tuple[SyncResult, SatGrid | None]:
    """Poprawka zegara Δ z przelotów satelitów.

    Kandydaci: tory proste o prędkości LEO. Dla każdego okna (± window_sigma·σ, potem
    ± sync_search_s) szukamy par tor–satelita; Δ uznajemy, gdy ≥ ``min_tracks`` różnych
    torów daje zgodne δ (± ``cluster_s``). Wartość Δ bierzemy z pierwszego (najwcześniejszego
    w nagraniu) zidentyfikowanego satelity (``reference: first``) — tak ustalił użytkownik —
    a zgodność pozostałych jest kontrolą."""
    cands = [t for t in tracks if is_sync_candidate(t, icfg)]
    windows = [float(icfg["window_sigma"]) * prior_sigma_s]
    if search_s > windows[0]:
        windows.append(float(search_s))
    last_grid = None
    for W in windows:
        grid = grid_for(W)
        last_grid = grid
        pairs: list[Match] = []
        for tr in cands:
            for i, d0 in candidate_pairs(grid, tr, icfg, -W, W, float(icfg["pre_tol_deg"]),
                                         float(icfg["speed_tol"]), float(icfg["dir_tol_deg"])):
                m = fine_align(observer, catalog, i, tr, d0, float(icfg["refine_half_range_s"]))
                if m and m.cross_deg <= float(icfg["pre_tol_deg"]) and abs(m.speed_ratio - 1) <= float(icfg["speed_tol"]):
                    pairs.append(m)
        if not pairs:
            continue
        best_cluster, best_key = [], None
        for p in pairs:
            members: dict[int, Match] = {}
            for q in pairs:
                if abs(q.delta_s - p.delta_s) <= float(icfg["cluster_s"]):
                    if q.track_id not in members or q.rms_deg < members[q.track_id].rms_deg:
                        members[q.track_id] = q
            n_sat = len({m.norad for m in members.values()})
            key = (min(len(members), n_sat), -float(np.mean([m.rms_deg for m in members.values()])))
            if best_key is None or key > best_key:
                best_key, best_cluster = key, list(members.values())
        by_id = {t.track_id: t for t in tracks}
        n_ok = int(best_key[0])
        if n_ok >= int(icfg["min_tracks"]):
            ref = _reference(best_cluster, by_id, str(icfg["reference"]))
            conf, method = "high", f"zgodność {n_ok} torów (±{icfg['cluster_s']} s)"
        else:
            ref = min(pairs, key=lambda m: m.rms_deg)
            best_cluster = [ref]
            conf, method = "low", "pojedynczy tor — brak potwierdzenia drugim satelitą"
        delta = ref.delta_s if str(icfg["reference"]) != "median" else float(np.median([m.delta_s for m in best_cluster]))
        sigma = _sigma_delta(ref, by_id[ref.track_id], len(by_id[ref.track_id].tau), icfg)
        return SyncResult(True, delta, sigma, conf, method, ref.to_dict(),
                          [m.to_dict() for m in sorted(best_cluster, key=lambda m: by_id[m.track_id].tau[0])],
                          W), grid
    return SyncResult(False, 0.0, prior_sigma_s, "none", "brak dopasowań", None, [],
                      windows[-1], "czas z metadanych (nie zsynchronizowano)"), last_grid


def _reference(cluster: list[Match], by_id: dict, mode: str) -> Match:
    if mode == "best":
        return min(cluster, key=lambda m: m.rms_deg)
    if mode == "median":
        ds = np.array([m.delta_s for m in cluster])
        return cluster[int(np.argmin(np.abs(ds - np.median(ds))))]
    return min(cluster, key=lambda m: by_id[m.track_id].tau[0])   # "first"


# ---------------------------------------------------------------- identyfikacja

def identify_tracks(observer: Observer, catalog: Catalog, tracks: list[TrackSky], grid: SatGrid,
                    delta_s: float, icfg: dict) -> dict[int, list[Match]]:
    """Dla każdego toru: dopasowania przy ustalonym Δ, posortowane od najlepszego.
    Pierwsze na liście ma ``confidence`` i ``reason``; niejednoznaczne są oznaczone."""
    tol_t = float(icfg["dt_tol_s"])
    out: dict[int, list[Match]] = {}
    for tr in tracks:
        found = []
        for i, d0 in candidate_pairs(grid, tr, icfg, delta_s - tol_t, delta_s + tol_t,
                                     max(float(icfg["cross_tol_deg"]) * 3, 0.3), float(icfg["speed_tol"]),
                                     float(icfg["dir_tol_deg"])):
            m = fine_align(observer, catalog, i, tr, d0, float(icfg["refine_half_range_s"]))
            if m is None:
                continue
            if (m.cross_deg <= float(icfg["cross_tol_deg"]) and abs(m.delta_s - delta_s) <= tol_t
                    and abs(m.speed_ratio - 1) <= float(icfg["id_speed_tol"])
                    and m.dir_deg <= float(icfg["id_dir_tol_deg"])):
                found.append(m)
        found.sort(key=lambda m: m.rms_deg)
        if found:
            best = found[0]
            amb = [m for m in found[1:] if m.rms_deg <= best.rms_deg * float(icfg["ambiguity_ratio"])]
            best.ambiguous_with = [m.norad for m in amb]
            tight = best.cross_deg <= float(icfg["cross_tol_deg"]) / 2 and abs(best.delta_s - delta_s) <= tol_t / 2
            best.confidence = "low" if amb else ("high" if tight else "medium")
            best.reason = (f"residuum poprzeczne {best.cross_deg * 3600:.0f}″, δ−Δ = {best.delta_s - delta_s:+.2f} s, "
                           f"prędkość ×{best.speed_ratio:.3f}, kierunek {best.dir_deg:.2f}°"
                           + (f"; niejednoznaczne z NORAD {best.ambiguous_with}" if amb else ""))
        out[tr.track_id] = found
    return out


def load_ephemeris(ephemeris_dir: str):
    """de421 (Słońce, Ziemia) z cache na Drive; brak sieci / pliku → None."""
    try:
        from skyfield.api import Loader

        return Loader(ephemeris_dir)("de421.bsp")
    except Exception as e:  # noqa: BLE001
        log.warning("oświetlenie satelitów pominięte: %s", e)
        return None


def sunlit_at(catalog: Catalog, indices, offsets, observer: Observer, eph) -> list:
    """Czy satelita ``catalog[i]`` był oświetlony przez Słońce w chwili t_ref + offset (None bez efemerydy)."""
    if eph is None:
        return [None] * len(indices)
    from skyfield.api import EarthSatellite

    out = []
    for i, off in zip(indices, offsets):
        sat = EarthSatellite.from_satrec(catalog.satrecs[int(i)], observer.ts)
        out.append(bool(sat.at(observer.times([float(off)])).is_sunlit(eph)[0]))
    return out


def sunlit_flags(catalog: Catalog, matches: list, observer: Observer, ephemeris_dir: str | None,
                 taus: dict[int, float], eph=None):
    """Uzupełnia ``sunlit`` dopasowań (obiekty ``Match`` albo słowniki z ``to_dict``); zwraca efemerydę."""
    eph = eph if eph is not None else load_ephemeris(ephemeris_dir)
    get = lambda m, k: m[k] if isinstance(m, dict) else getattr(m, k)  # noqa: E731
    flags = sunlit_at(catalog, [get(m, "cat_index") for m in matches],
                      [get(m, "delta_s") + taus[get(m, "track_id")] for m in matches], observer, eph)
    for m, f in zip(matches, flags):
        if isinstance(m, dict):
            m["sunlit"] = f
        else:
            m.sunlit = f
    return eph
