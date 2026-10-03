"""Niebo: konstelacje i gwiazdy (d3-celestial), model kamery nieruchomej względem Ziemi,
kierunki pozorne i dopasowanie wielkiego koła do toru.

Model kamery: piksel widzi stały kierunek Alt/Az. Mając WCS z epoki τ₀, pozycję ICRS
piksela w chwili τ liczymy: WCS → ICRS → AltAz(Tc+τ₀) → ten sam Alt/Az w chwili Tc+τ → ICRS.
To obrót wokół bieguna daty o kąt obrotu Ziemi (τ−τ₀); błąd zegara Tc się znosi, więc
pozycje torów na niebie nie zależą od (jeszcze nieznanej) poprawki czasu.
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "d3celestial"
ARCSEC = 3600.0


# ---------------------------------------------------------------- dane d3-celestial

@functools.lru_cache(maxsize=None)
def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _ra(lon: float) -> float:
    return float(lon) % 360.0     # d3-celestial: lon w [−180, 180], RA = lon mod 360


@functools.lru_cache(maxsize=1)
def constellation_lines() -> dict[str, list[np.ndarray]]:
    """Id konstelacji (np. ``Cyg``) → łamane [N, 2] (RA, Dec) w stopniach."""
    out: dict[str, list[np.ndarray]] = {}
    for feat in _load("constellations.lines.json")["features"]:
        out[feat["id"]] = [np.array([[_ra(lon), lat] for lon, lat in line]) for line in feat["geometry"]["coordinates"]]
    return out


@functools.lru_cache(maxsize=1)
def constellation_names() -> dict[str, dict]:
    """Id → {"name": łacińska nazwa, "label": (RA, Dec) miejsca etykiety}."""
    out = {}
    for feat in _load("constellations.json")["features"]:
        p = feat["properties"]
        lon, lat = (p.get("display") or feat["geometry"]["coordinates"])[:2]
        out[feat["id"]] = {"name": p.get("name", feat["id"]), "label": (_ra(lon), float(lat))}
    return out


@functools.lru_cache(maxsize=1)
def stars() -> dict[str, np.ndarray]:
    """Gwiazdy do mag 6: hip, ra, dec, mag, bv (wskaźnik barwy B−V; NaN, gdy brak)."""
    feats = _load("stars.6.json")["features"]

    def bv(f) -> float:
        try:
            return float(f["properties"].get("bv"))
        except (TypeError, ValueError):
            return float("nan")

    return {
        "hip": np.array([int(f["id"]) for f in feats]),
        "ra": np.array([_ra(f["geometry"]["coordinates"][0]) for f in feats]),
        "dec": np.array([float(f["geometry"]["coordinates"][1]) for f in feats]),
        "mag": np.array([float(f["properties"]["mag"]) for f in feats]),
        "bv": np.array([bv(f) for f in feats]),
    }


@functools.lru_cache(maxsize=1)
def star_names() -> dict[int, str]:
    return {int(k): v["name"] for k, v in _load("starnames.json").items() if v.get("name")}


def resolve_star(name: str) -> tuple[float, float]:
    """Nazwa gwiazdy (``Deneb``) albo ``HIP 102098`` → (RA, Dec) [°]."""
    key = name.strip().lower()
    if key.startswith("hip"):
        hip = int(key[3:].strip())
    else:
        hits = [h for h, n in star_names().items() if n.lower() == key]
        if not hits:
            raise KeyError(f"nieznana gwiazda: {name!r}")
        hip = hits[0]
    s = stars()
    i = np.flatnonzero(s["hip"] == hip)
    if not len(i):
        raise KeyError(f"brak pozycji gwiazdy HIP {hip} w katalogu do mag 6")
    return float(s["ra"][i[0]]), float(s["dec"][i[0]])


# ---------------------------------------------------------------- wektory

def radec_to_vec(ra, dec) -> np.ndarray:
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def vec_to_radec(v) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0, np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))


def angle_deg(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def densify(radec: np.ndarray, step_deg: float) -> np.ndarray:
    """Łamana (RA, Dec) zagęszczona po wielkich kołach co ``step_deg``."""
    v = radec_to_vec(radec[:, 0], radec[:, 1])
    out = [v[0]]
    for a, b in zip(v[:-1], v[1:]):
        ang = math.radians(float(angle_deg(a, b)))
        n = max(1, int(math.ceil(math.degrees(ang) / step_deg)))
        for k in range(1, n + 1):
            t = k / n
            if ang < 1e-12:
                p = b
            else:
                p = (math.sin((1 - t) * ang) * a + math.sin(t * ang) * b) / math.sin(ang)
            out.append(p)
    ra, dec = vec_to_radec(np.array(out))
    return np.column_stack([ra, dec])


# ---------------------------------------------------------------- rzutowanie na obraz

def _world2pix(wcs, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    try:
        x, y = wcs.all_world2pix(ra, dec, 0, quiet=True)
    except Exception:  # noqa: BLE001 — brak zbieżności SIP daleko poza kadrem
        x, y = wcs.wcs_world2pix(ra, dec, 0)
    return np.asarray(x, float), np.asarray(y, float)


def field_center(wcs, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """(wektor środka kadru, promień do narożnika [°])."""
    H, W = shape
    xs = np.array([W / 2, 0, W - 1, 0, W - 1])
    ys = np.array([H / 2, 0, 0, H - 1, H - 1])
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    v = radec_to_vec(ra, dec)
    return v[0], float(angle_deg(v[0], v[1:]).max())


def project_polyline(wcs, shape, radec: np.ndarray, *, center: np.ndarray, densify_deg: float,
                     max_sep_deg: float = 60.0, margin_px: float = 200.0) -> list[np.ndarray]:
    """Łamana na niebie → kawałki [N, 2] w pikselach (tylko w kadrze + margines)."""
    H, W = shape
    pts = densify(radec, densify_deg)
    near = angle_deg(radec_to_vec(pts[:, 0], pts[:, 1]), center) <= max_sep_deg
    x, y = _world2pix(wcs, pts[:, 0], pts[:, 1])
    ok = near & np.isfinite(x) & np.isfinite(y) & (x > -margin_px) & (x < W + margin_px) \
        & (y > -margin_px) & (y < H + margin_px)
    pieces, cur = [], []
    for i in range(len(pts)):
        if ok[i]:
            cur.append((x[i], y[i]))
        elif cur:
            pieces.append(np.array(cur))
            cur = []
    if cur:
        pieces.append(np.array(cur))
    return [p for p in pieces if len(p) >= 2]


def constellation_overlay(wcs, shape, densify_deg: float = 0.5) -> list[dict]:
    """Konstelacje widoczne w kadrze: [{"id", "name", "pieces": [...], "label_xy": (x, y)}]."""
    H, W = shape
    center, radius = field_center(wcs, shape)
    names = constellation_names()
    out = []
    for cid, lines in constellation_lines().items():
        pieces = [p for line in lines for p in project_polyline(
            wcs, shape, line, center=center, densify_deg=densify_deg, max_sep_deg=radius + 30)]
        inside = [p[(p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)] for p in pieces]
        inside = np.concatenate(inside) if inside else np.empty((0, 2))
        if not len(inside):
            continue
        info = names.get(cid, {"name": cid, "label": None})
        lx = ly = None
        if info["label"] is not None:
            lx, ly = _world2pix(wcs, np.array([info["label"][0]]), np.array([info["label"][1]]))
            lx, ly = float(lx[0]), float(ly[0])
        if lx is None or not (0 <= lx < W and 0 <= ly < H):
            lx, ly = inside.mean(axis=0)
        out.append({"id": cid, "name": info["name"], "pieces": pieces, "label_xy": (float(lx), float(ly))})
    return out


def named_stars_overlay(wcs, shape, max_mag: float, always: list[str] = ()) -> list[dict]:
    """Nazwane gwiazdy w kadrze: jaśniejsze niż ``max_mag`` oraz wymienione w ``always``."""
    H, W = shape
    s, names = stars(), star_names()
    always_l = {a.strip().lower() for a in always}
    sel = [i for i, h in enumerate(s["hip"]) if h in names
           and (s["mag"][i] <= max_mag or names[h].lower() in always_l)]
    if not sel:
        return []
    sel = np.array(sel)
    center, radius = field_center(wcs, shape)
    near = angle_deg(radec_to_vec(s["ra"][sel], s["dec"][sel]), center) <= radius + 5
    sel = sel[near]
    x, y = _world2pix(wcs, s["ra"][sel], s["dec"][sel])
    return [{"name": names[int(s["hip"][i])], "mag": float(s["mag"][i]), "x": float(xi), "y": float(yi)}
            for i, xi, yi in zip(sel, x, y) if 0 <= xi < W and 0 <= yi < H]


# ---------------------------------------------------------------- kamera nieruchoma

def _astropy_setup() -> None:
    from astropy.utils import iers

    # Bez sieci astropy nie ma aktualnych IERS-A; różnica UT1−UTC (<1 s) i tak się
    # znosi w modelu kamery, więc zamiast błędu wystarczy ostrzeżenie.
    iers.conf.iers_degraded_accuracy = "warn"


class FixedCamera:
    """Kamera nieruchoma względem Ziemi, skalibrowana WCS-em z epoki ``tau0_s``."""

    def __init__(self, wcs, tau0_s: float, t_ref: datetime, lat_deg: float, lon_deg: float, elevation_m: float):
        import astropy.units as u
        from astropy.coordinates import EarthLocation
        from astropy.time import Time

        _astropy_setup()
        self.wcs = wcs
        self.tau0 = float(tau0_s)
        self.t_ref = Time(t_ref)
        self.location = EarthLocation.from_geodetic(lon_deg * u.deg, lat_deg * u.deg, elevation_m * u.m)

    def _altaz(self, tau):
        import astropy.units as u
        from astropy.coordinates import AltAz

        return AltAz(obstime=self.t_ref + np.asarray(tau, float) * u.s, location=self.location, pressure=0)

    def altaz(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(Az, Alt) [°] piksela — geometryczne, bez refrakcji; stałe w czasie."""
        c = self.wcs.pixel_to_world(np.asarray(x, float), np.asarray(y, float)).transform_to(self._altaz(self.tau0))
        return c.az.deg, c.alt.deg

    def icrs(self, x, y, tau) -> tuple[np.ndarray, np.ndarray]:
        """(RA, Dec) ICRS [°] piksela (x, y) w chwili nagrania ``tau`` [s]."""
        from astropy.coordinates import ICRS, SkyCoord

        x, y, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(tau, float))
        aa = self.wcs.pixel_to_world(x, y).transform_to(self._altaz(self.tau0))
        c = SkyCoord(az=aa.az, alt=aa.alt, frame=self._altaz(tau)).transform_to(ICRS())
        return c.ra.deg, c.dec.deg

    def radec_altaz(self, ra, dec, tau) -> tuple[np.ndarray, np.ndarray]:
        """(Az, Alt) [°] kierunku ICRS w chwili ``tau`` — geometryczne, bez refrakcji."""
        import astropy.units as u
        from astropy.coordinates import ICRS, SkyCoord

        ra, dec, tau = np.broadcast_arrays(np.asarray(ra, float), np.asarray(dec, float), np.asarray(tau, float))
        aa = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=ICRS()).transform_to(self._altaz(tau))
        return aa.az.deg, aa.alt.deg

    def pixel(self, ra, dec, tau) -> tuple[np.ndarray, np.ndarray]:
        """Odwrotność ``icrs``: (RA, Dec) w chwili ``tau`` → piksel."""
        import astropy.units as u
        from astropy.coordinates import ICRS, SkyCoord

        ra, dec, tau = np.broadcast_arrays(np.asarray(ra, float), np.asarray(dec, float), np.asarray(tau, float))
        aa = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=ICRS()).transform_to(self._altaz(tau))
        c0 = SkyCoord(az=aa.az, alt=aa.alt, frame=self._altaz(np.full(tau.shape, self.tau0))).transform_to(ICRS())
        return _world2pix(self.wcs, c0.ra.deg, c0.dec.deg)

    def apparent_vectors(self, ra, dec, tau, delta_s: float = 0.0) -> np.ndarray:
        """ICRS (astrometryczne, jak katalog gwiazd) → kierunki pozorne [N, 3] w GCRS
        obserwatora w chwili t_ref + delta + tau (aberracja). Porównywalne z geometrycznym
        topocentrycznym wektorem satelity ze Skyfield."""
        import astropy.units as u
        from astropy.coordinates import GCRS, ICRS, SkyCoord

        tau = np.asarray(tau, float)
        t = self.t_ref + (tau + delta_s) * u.s
        pos, vel = self.location.get_gcrs_posvel(t)
        c = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg, frame=ICRS())
        g = c.transform_to(GCRS(obstime=t, obsgeoloc=pos, obsgeovel=vel))
        return radec_to_vec(g.ra.deg, g.dec.deg)

    def astrometric_radec(self, vecs, tau, delta_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Odwrotność ``apparent_vectors``: kierunki w GCRS obserwatora (np. satelity ze Skyfield)
        → (RA, Dec) w układzie katalogu gwiazd, zgodnym z WCS (bez tego ~20″ przesunięcia)."""
        import astropy.units as u
        from astropy.coordinates import GCRS, ICRS, SkyCoord

        ra, dec = vec_to_radec(np.asarray(vecs, float))
        t = self.t_ref + (np.asarray(tau, float) + delta_s) * u.s
        pos, vel = self.location.get_gcrs_posvel(t)
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=GCRS(obstime=t, obsgeoloc=pos, obsgeovel=vel))
        c = c.transform_to(ICRS())
        return c.ra.deg, c.dec.deg

    def with_reference(self, t_ref: datetime) -> "FixedCamera":
        """Ten sam WCS, inny czas odniesienia (po synchronizacji zegara)."""
        cam = object.__new__(FixedCamera)
        cam.__dict__.update(self.__dict__)
        from astropy.time import Time

        cam.t_ref = Time(t_ref)
        return cam


# ---------------------------------------------------------------- wielkie koło

@dataclass
class GreatCircle:
    pole: np.ndarray          # biegun koła (ruch przeciwnie do wskazówek wokół bieguna)
    e1: np.ndarray
    e2: np.ndarray
    phi0: float               # kąt [rad] w t = 0
    omega_rad_s: float
    cross_rms_arcsec: float
    along_rms_arcsec: float

    @property
    def omega_deg_s(self) -> float:
        return math.degrees(self.omega_rad_s)

    def position(self, t) -> np.ndarray:
        phi = self.phi0 + self.omega_rad_s * np.asarray(t, float)
        return np.cos(phi)[..., None] * self.e1 + np.sin(phi)[..., None] * self.e2

    def tangent(self, t) -> np.ndarray:
        phi = self.phi0 + self.omega_rad_s * np.asarray(t, float)
        return -np.sin(phi)[..., None] * self.e1 + np.cos(phi)[..., None] * self.e2

    def cross_offset_deg(self, v) -> np.ndarray:
        return np.degrees(np.arcsin(np.clip(np.asarray(v) @ self.pole, -1, 1)))

    def phase(self, v) -> np.ndarray:
        v = np.asarray(v)
        return np.arctan2(v @ self.e2, v @ self.e1)


def fit_great_circle(vecs: np.ndarray, t: np.ndarray) -> GreatCircle:
    """Wielkie koło i ruch jednostajny po nim: biegun z SVD, kąt(t) dopasowany liniowo."""
    vecs, t = np.asarray(vecs, float), np.asarray(t, float)
    _, _, vt = np.linalg.svd(vecs, full_matrices=False)
    pole = vt[-1] / np.linalg.norm(vt[-1])

    def basis(p):
        e1 = vecs[0] - (vecs[0] @ p) * p
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(p, e1)

    e1, e2 = basis(pole)
    phi = np.unwrap(np.arctan2(vecs @ e2, vecs @ e1))
    b, a = np.polyfit(t, phi, 1) if len(t) > 1 else (0.0, phi[0])
    if b < 0:   # orientacja: ruch dodatni
        pole = -pole
        e1, e2 = basis(pole)
        phi = np.unwrap(np.arctan2(vecs @ e2, vecs @ e1))
        b, a = np.polyfit(t, phi, 1) if len(t) > 1 else (0.0, phi[0])
    cross = np.arcsin(np.clip(vecs @ pole, -1, 1))
    along = phi - (a + b * t)
    return GreatCircle(pole, e1, e2, float(a), float(b),
                       float(np.degrees(np.sqrt(np.mean(cross ** 2)))) * ARCSEC,
                       float(np.degrees(np.sqrt(np.mean(along ** 2)))) * ARCSEC)
