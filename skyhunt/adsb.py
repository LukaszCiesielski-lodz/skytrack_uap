"""Samoloty z historii ADS-B: dopasowanie torów do tras samolotów przelatujących nad miejscem
obserwacji.

Źródło: archiwum adsb.lol (licencja ODbL 1.0, https://www.adsb.lol/docs/open-data/historical/).
Każdy dzień to wydanie na GitHubie (``adsblol/globe_history_<rok>``, tag
``v<RRRR.MM.DD>-planes-readsb-prod-0``) z archiwum tar podzielonym na części; w środku pliki
``traces/../trace_full_<hex>.json`` (gzip). Format trasy: readsb README-json, „trace jsons”:
``timestamp`` + lista ``[dt_s, lat, lon, alt_ft|"ground"|null, gs_kt, track, flags, vrate,
aircraft|null, source, geom_alt_ft, …]``.

Archiwum dnia ma kilka GB, więc czytamy je strumieniowo (bez zapisu na dysk) i zapisujemy na
Drive tylko punkty w promieniu ``radius_km`` od miejsca obserwacji (cache na kolejne nagrania).
OpenSky REST nie nadaje się: stany samolotów tylko do 1 h wstecz.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import math
import tarfile
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("skyhunt")

RELEASE_API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
WGS84_A, WGS84_E2 = 6378137.0, 6.69437999014e-3
FT = 0.3048
COLUMNS = ["icao", "reg", "type", "callsign", "t", "lat", "lon", "alt_m", "gs_kt", "track_deg"]


# ---------------------------------------------------------------- geometria

def ecef(lat_deg, lon_deg, h_m) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(n + h_m) * np.cos(lat) * np.cos(lon), (n + h_m) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + h_m) * np.sin(lat)], axis=-1)


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Wiersze: wschód, północ, góra (geodezyjnie) w ECEF."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([[-math.sin(lon), math.cos(lon), 0.0],
                     [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
                     [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]])


def topocentric(site: tuple[float, float, float], lat, lon, alt_m) -> tuple[np.ndarray, np.ndarray]:
    """Kierunki ENU (jednostkowe, [N, 3]) i odległości [km] od obserwatora do punktów."""
    d = ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt_m, float)) - ecef(*site)
    enu = d @ enu_basis(site[0], site[1]).T
    r = np.linalg.norm(enu, axis=-1)
    return enu / r[..., None], r / 1000.0


def altaz_to_enu(az_deg, alt_deg) -> np.ndarray:
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)


# ---------------------------------------------------------------- archiwum adsb.lol

class _Concat(io.RawIOBase):
    """Kolejne części archiwum (.tar.aa, .tar.ab, …) jako jeden strumień."""

    def __init__(self, urls: list[str], opener: Callable):
        self.urls, self.opener, self.cur = list(urls), opener, None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while True:
            if self.cur is None:
                if not self.urls:
                    return 0
                self.cur = self.opener(self.urls.pop(0), timeout=600)
            n = self.cur.readinto(b)
            if n:
                return n
            self.cur.close()
            self.cur = None


def release_parts(day: date, instances, opener: Callable = urllib.request.urlopen) -> tuple[str, list[str]]:
    """(tag, adresy części) pierwszego dostępnego wydania dnia."""
    from .satellites import FetchError, fetch_url

    for inst in instances:
        tag = f"v{day:%Y.%m.%d}-planes-readsb-{inst}"
        try:
            rel = json.loads(fetch_url(RELEASE_API.format(year=day.year, tag=tag), opener))
        except FetchError as e:
            log.info("adsb.lol %s: %s", tag, e)
            continue
        urls = sorted(a["browser_download_url"] for a in rel.get("assets", []) if ".tar" in a.get("name", ""))
        if urls:
            return tag, urls
    raise FileNotFoundError(f"brak wydania adsb.lol dla {day:%Y-%m-%d}")


def scan_traces(stream, site: tuple[float, float, float], radius_km: float, t_lo: float, t_hi: float) -> list[dict]:
    """Punkty tras w promieniu ``radius_km`` i oknie czasu [t_lo, t_hi] (unix) ze strumienia tar."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(site[0])), 0.05))
    rows = []
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for m in tar:
            if not m.isfile() or "trace_full_" not in m.name:
                continue
            raw = tar.extractfile(m).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            ts, tr = float(d.get("timestamp") or 0), d.get("trace") or []
            if not tr or ts + tr[-1][0] < t_lo or ts + tr[0][0] > t_hi:
                continue
            callsign = None
            for e in tr:
                if len(e) > 8 and isinstance(e[8], dict) and e[8].get("flight"):
                    callsign = str(e[8]["flight"]).strip()
                t = ts + float(e[0])
                if t < t_lo or t > t_hi or e[1] is None or e[2] is None:
                    continue
                if abs(e[1] - site[0]) > dlat or abs(e[2] - site[1]) > dlon:
                    continue
                alt = e[10] if len(e) > 10 and isinstance(e[10], (int, float)) else e[3]
                if not isinstance(alt, (int, float)):
                    continue                                   # na ziemi albo brak wysokości
                rows.append({"icao": d.get("icao"), "reg": d.get("r"), "type": d.get("t"), "callsign": callsign,
                             "t": t, "lat": float(e[1]), "lon": float(e[2]), "alt_m": float(alt) * FT,
                             "gs_kt": e[4], "track_deg": e[5]})
    return rows


def cache_path(cache_dir: Path, day: date, site, radius_km: float) -> Path:
    key = hashlib.sha1(f"{site[0]:.2f},{site[1]:.2f},{radius_km:.0f}".encode()).hexdigest()[:10]
    return Path(cache_dir) / f"adsb_{day:%Y%m%d}_{key}.parquet"


def day_points(acfg: dict, day: date, site, *, opener: Callable = urllib.request.urlopen):
    """Punkty samolotów z całego dnia UTC w promieniu od miejsca (cache na Drive)."""
    import pandas as pd

    path = cache_path(Path(acfg["cache_dir"]), day, site, float(acfg["radius_km"]))
    if path.exists():
        return pd.read_parquet(path), {"cache": path.name}
    tag, urls = release_parts(day, acfg["instances"], opener)
    log.info("ADS-B: czytam %s (%d części, strumieniowo — kilka minut)", tag, len(urls))
    t_lo = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    stream = io.BufferedReader(_Concat(urls, opener), buffer_size=1 << 20)
    rows = scan_traces(stream, site, float(acfg["radius_km"]), t_lo - 600, t_lo + 86400 + 600)
    df = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df, {"release": tag, "cache": path.name}


# ---------------------------------------------------------------- dopasowanie

def aircraft_directions(points, site, times: np.ndarray, max_gap_s: float) -> dict[str, tuple]:
    """Dla każdego samolotu: kierunki ENU i odległości [km] w chwilach ``times`` (interpolacja
    liniowa między punktami trasy; NaN poza trasą albo w dziurze > ``max_gap_s``)."""
    out = {}
    for icao, g in points.groupby("icao"):
        g = g.sort_values("t")
        tt = g["t"].to_numpy()
        if tt[0] > times[-1] or tt[-1] < times[0]:
            continue
        k = np.clip(np.searchsorted(tt, times), 1, len(tt) - 1)
        ok = (times >= tt[0]) & (times <= tt[-1]) & ((tt[k] - tt[k - 1]) <= max_gap_s)
        if not ok.any():
            continue
        lat = np.interp(times, tt, g["lat"].to_numpy())
        lon = np.interp(times, tt, g["lon"].to_numpy())
        alt = np.interp(times, tt, g["alt_m"].to_numpy())
        u, r = topocentric(site, lat, lon, alt)
        u[~ok], r[~ok], alt[~ok] = np.nan, np.nan, np.nan

        def first(col):
            s = g[col].dropna()
            return None if s.empty else s.iloc[0]

        out[str(icao)] = (u, r, alt, {"reg": first("reg"), "type": first("type"), "callsign": first("callsign")})
    return out


def match_tracks(tracks: list[dict], points, site, acfg: dict) -> list[dict]:
    """``tracks``: {track_id, times (unix), enu [N,3]} → najlepsze dopasowanie samolotu dla toru."""
    res = []
    lim = float(acfg["match_deg"])
    for tr in tracks:
        times, v = tr["times"], tr["enu"]
        best = None
        for icao, (u, r, alt, info) in aircraft_directions(points, site, times, float(acfg["max_gap_s"])).items():
            ok = np.isfinite(u).all(axis=1)
            if ok.sum() < max(3, 0.5 * len(times)):
                continue
            sep = np.degrees(np.arccos(np.clip(np.sum(u[ok] * v[ok], axis=1), -1, 1)))
            med = float(np.median(sep))
            if med <= lim and (best is None or med < best["sep_deg"]):
                best = {"track_id": tr["track_id"], "icao": icao, **info, "sep_deg": med,
                        "range_km": float(np.nanmedian(r)), "alt_m": float(np.nanmedian(alt))}
        if best:
            res.append(best)
    return res


def recording_days(start: datetime, end: datetime) -> list[date]:
    days, d = [], start.astimezone(timezone.utc).date()
    while d <= end.astimezone(timezone.utc).date():
        days.append(d)
        d += timedelta(days=1)
    return days
