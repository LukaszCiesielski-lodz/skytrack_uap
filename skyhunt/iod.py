"""Eksport obserwacji pozycyjnych w formacie IOD (Interactive Orbit Determination) —
standard amatorskiej sieci obserwatorów satelitów (SeeSat-L, satobs.org).

Rekord (kolumny 1-based): 1-15 obiekt (NORAD + oznaczenie COSPAR), 17-20 numer stacji,
22 warunki, 24-40 data i czas UTC (YYYYMMDDHHMMSSsss), 42-43 niepewność czasu (MX),
45 format kątów (1: RA/Dec HHMMSSs+DDMMSS), 46 epoka (5 = J2000), 48-61 RA/Dec,
63-64 niepewność pozycji (MX; w formacie 1 w sekundach łuku), 66 zachowanie optyczne.
MX oznacza M·10^(X−8).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def mx(value: float) -> str:
    """Niepewność → dwie cyfry MX (zaokrąglenie w górę, zakres 1e-8 … 9e1)."""
    if not math.isfinite(value) or value <= 0:
        return "  "
    x = int(math.floor(math.log10(value))) + 8
    m = int(math.ceil(value / 10 ** (x - 8) - 1e-9))
    if m >= 10:
        m, x = 1, x + 1
    x = min(max(x, 0), 9)
    return f"{min(m, 9)}{x}"


def radec_format1(ra_deg: float, dec_deg: float) -> str:
    """RA/Dec w formacie 1: ``HHMMSSs+DDMMSS`` (14 znaków, kolumny 48-61)."""
    t = round((ra_deg % 360.0) / 15.0 * 36000)          # dziesiąte części sekundy czasu
    t %= 24 * 36000
    hh, rem = divmod(t, 36000)
    mm, ss10 = divmod(rem, 600)
    sign = "-" if dec_deg < 0 else "+"
    a = round(abs(dec_deg) * 3600)
    dd, rem = divmod(a, 3600)
    dm, ds = divmod(rem, 60)
    return f"{hh:02d}{mm:02d}{ss10:03d}{sign}{min(dd, 90):02d}{dm:02d}{ds:02d}"


def object_field(norad: int | None, cospar: str | None, unknown: str) -> str:
    """Kolumny 1-15: ``NNNNN YY LLLPPP``; nieznany obiekt → ``unknown`` z configu."""
    if norad is None or not (0 < int(norad) <= 99999) or not cospar or "-" not in cospar:
        return f"{unknown:<15.15}"
    year, rest = cospar.split("-", 1)
    launch, piece = rest[:3], rest[3:]
    return f"{int(norad):05d} {year[-2:]} {launch:>3.3}{piece:<3.3}"


def record(obj: str, station: int, condition: str, t: datetime, dt_unc_s: float, ra_deg: float, dec_deg: float,
           pos_unc_arcsec: float, behaviour: str = " ") -> str:
    ms = int(round(t.microsecond / 1000))
    if ms == 1000:
        t, ms = t.replace(microsecond=0) + timedelta(seconds=1), 0
    stamp = t.strftime("%Y%m%d%H%M%S") + f"{ms:03d}"
    return (f"{obj:<15.15} {int(station):04d} {condition[:1]} {stamp} {mx(dt_unc_s)} 15 "
            f"{radec_format1(ra_deg, dec_deg)} {mx(pos_unc_arcsec)} {behaviour[:1]}").rstrip()


def export(outdir: Path, final, points, sync: dict, wcsinfo: dict, identifications: dict, icfg: dict,
           classify_cfg: dict) -> tuple[Path, int]:
    """``iod.txt``: po 3 pozycje (początek, środek, koniec) dla zidentyfikowanych satelitów oraz
    niezidentyfikowanych torów spełniających kryteria jakości z sekcji ``iod``."""
    t0 = datetime.fromisoformat(sync["start_utc_prior"])
    delta = float(sync.get("delta_s") or 0.0)
    dt_unc = max(float(sync.get("sigma_s") or 0.0), 0.001)
    scale = float((wcsinfo.get("fov") or {}).get("scale_arcsec_px") or 25.0)
    epoch_rms = float(wcsinfo.get("epoch_rms_px") or 0.5)
    pos_unc = scale * math.hypot(epoch_rms, float(icfg["centroid_px"]))
    station, cond, unknown = int(icfg["station"]), str(icfg["condition"]), str(icfg["unknown_object"])
    periodic = float(classify_cfg["periodic_min_power"])
    lines = []
    for _, t in final.sort_values("tau0").iterrows():
        tid = int(t["track_id"])
        is_sat = t["kind"] == "sat"
        if not is_sat and not (int(t["n"]) >= int(icfg["unid_min_points"])
                               and float(t["curv_arcsec"]) <= float(icfg["unid_max_curv_arcsec"])
                               and float(t.get("peak_snr_median", np.nan)) >= float(icfg["unid_min_snr"])):
            continue
        if is_sat and not icfg.get("include_identified", True):
            continue
        p = points[points["track_id"] == tid].sort_values("frame")
        if len(p) < 3:
            continue
        cospar = None
        if is_sat:
            best = (identifications.get(str(tid)) or [{}])[0]
            cospar = best.get("object_id")
        obj = object_field(int(t["norad"]) if is_sat else None, cospar, unknown)
        beh = "R" if float(t.get("f_power", 0) or 0) > periodic else " "
        for k in (0, len(p) // 2, len(p) - 1):
            q = p.iloc[k]
            lines.append(record(obj, station, cond, t0 + timedelta(seconds=delta + float(q["tau"])), dt_unc,
                                float(q["ra"]), float(q["dec"]), pos_unc, beh))
    path = outdir / "iod.txt"
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return path, len(lines)
