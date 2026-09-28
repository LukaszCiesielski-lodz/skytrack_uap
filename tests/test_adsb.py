"""ADS-B (adsb.lol): geometria, strumieniowe czytanie archiwum tar z części, cache, dopasowanie torów.
Bez sieci: atrapa API GitHuba i plików wydania."""
import gzip
import io
import json
import tarfile
from datetime import date, datetime, timezone

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from skyhunt import adsb as A  # noqa: E402

SITE = (51.72, 19.58, 210.0)
DAY = date(2026, 9, 27)
T0 = datetime(2026, 9, 27, 18, 30, tzinfo=timezone.utc).timestamp()


def test_topocentric_geometry():
    u, r = A.topocentric(SITE, [SITE[0]], [SITE[1]], [SITE[2] + 10000])
    assert r[0] == pytest.approx(10.0, abs=0.01)
    assert np.degrees(np.arcsin(u[0, 2])) == pytest.approx(90, abs=0.01)
    north = SITE[0] + 10.0 / 111.25                         # ~10 km na północ, 10 km wyżej
    u, r = A.topocentric(SITE, [north], [SITE[1]], [SITE[2] + 10000])
    az = np.degrees(np.arctan2(u[0, 0], u[0, 1])) % 360
    alt = np.degrees(np.arcsin(u[0, 2]))
    assert (az < 0.5 or az > 359.5) and alt == pytest.approx(45, abs=0.5)
    assert np.allclose(A.altaz_to_enu([0.0, 90.0], [0.0, 90.0]), [[0, 1, 0], [0, 0, 1]], atol=1e-12)


def trace(icao, lat0, lon0, dlat_s, t_start, n=60, alt_ft=33000, gz=True):
    d = {"icao": icao, "r": f"SP-{icao[:3].upper()}", "t": "B738", "timestamp": t_start,
         "trace": [[i * 5.0, lat0 + dlat_s * i * 5.0, lon0, alt_ft, 450.0, 0.0, 0, 0,
                    {"flight": "LOT123  "} if i == 0 else None, "adsb_icao", alt_ft + 200] for i in range(n)]}
    raw = json.dumps(d).encode()
    return gzip.compress(raw) if gz else raw


def tar_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def archive():
    # samolot A przelatuje nad miejscem z południa na północ; B daleko (Berlin); plik innego typu
    return tar_bytes({
        "traces/aa/trace_full_4b1805.json": trace("4b1805", SITE[0] - 0.3, SITE[1], 0.002, T0 - 150),
        "traces/bb/trace_full_3c6444.json": trace("3c6444", 52.5, 13.4, 0.001, T0 - 150, gz=False),
        "heatmap/37.bin.ttf": b"\x00" * 64,
    })


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


class Opener:
    def __init__(self, data: bytes):
        half = len(data) // 2
        self.parts = {"https://x/v2026.09.27-planes-readsb-prod-0.tar.aa": data[:half],
                      "https://x/v2026.09.27-planes-readsb-prod-0.tar.ab": data[half:]}
        self.urls = []

    def __call__(self, url, data=None, timeout=None):
        self.urls.append(url)
        if "api.github.com" in url:
            if "prod-0" not in url:
                import urllib.error
                raise urllib.error.HTTPError(url, 404, "nie ma", {}, None)
            assets = [{"name": u.rsplit("/", 1)[1], "browser_download_url": u} for u in reversed(list(self.parts))]
            return Resp(json.dumps({"assets": assets}).encode())
        return Resp(self.parts[url])


def acfg(tmp_path):
    return {"cache_dir": str(tmp_path), "instances": ["prod-0"], "radius_km": 150, "max_gap_s": 120,
            "match_deg": 1.5, "enabled": "auto"}


def test_day_points_streams_parts_and_caches(tmp_path):
    op = Opener(archive())
    df, src = A.day_points(acfg(tmp_path), DAY, SITE, opener=op)
    assert src["release"] == "v2026.09.27-planes-readsb-prod-0"
    assert set(df["icao"]) == {"4b1805"}                             # Berlin poza promieniem
    assert df["callsign"].iloc[0] == "LOT123" and df["alt_m"].iloc[0] == pytest.approx(33200 * 0.3048)
    n_calls = len(op.urls)
    df2, src2 = A.day_points(acfg(tmp_path), DAY, SITE, opener=op)   # drugi raz z cache
    assert len(op.urls) == n_calls and "cache" in src2 and len(df2) == len(df)


def test_match_track_to_aircraft(tmp_path):
    df, _ = A.day_points(acfg(tmp_path), DAY, SITE, opener=Opener(archive()))
    times = T0 + np.linspace(-5, 5, 15)
    lat = np.interp(times, df["t"], df["lat"])
    u, _ = A.topocentric(SITE, lat, np.full_like(lat, SITE[1]), np.interp(times, df["t"], df["alt_m"]))
    far = A.altaz_to_enu(np.full(15, 90.0), np.full(15, 30.0))
    res = A.match_tracks([{"track_id": 7, "times": times, "enu": u},
                          {"track_id": 8, "times": times, "enu": far}], df, SITE, acfg(tmp_path))
    assert [m["track_id"] for m in res] == [7]
    assert res[0]["icao"] == "4b1805" and res[0]["sep_deg"] < 0.01 and res[0]["callsign"] == "LOT123"
    assert res[0]["alt_m"] == pytest.approx(33200 * 0.3048, rel=1e-6)


def test_recording_days_crosses_midnight():
    a = datetime(2026, 9, 27, 23, 58, tzinfo=timezone.utc)
    b = datetime(2026, 9, 28, 0, 3, tzinfo=timezone.utc)
    assert A.recording_days(a, b) == [date(2026, 9, 27), date(2026, 9, 28)]
