"""Synchronizacja czasu i identyfikacja NORAD na syntetycznych torach z prawdziwych elementów ISS
(publiczne TLE z README pakietu sgp4) i satelitów-wabików. Bez sieci."""
import csv
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("sgp4")
pytest.importorskip("skyfield")
pytest.importorskip("scipy")

from skyhunt.satellites import (Observer, identify_tracks, load_catalog, make_track_sky,  # noqa: E402
                                screen, synchronize)

ISS_L1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_L2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
SITE = (51.718042, 19.582748, 210.0)
FPS = 24.0


def omm_rows():
    from sgp4.api import Satrec
    from sgp4.exporter import export_omm

    base = Satrec.twoline2rv(ISS_L1, ISS_L2)
    rows = [export_omm(base, "ISS (ZARYA)")]
    # wabiki: ta sama orbita przesunięta o 2° anomalii (~30 s przed ISS) oraz orbity odległe
    for norad, d_ma, d_raan in ((90010, 2.0, 0.0), (90001, 90.0, 0.0), (90002, 0.0, 40.0), (100500, 180.0, 90.0)):
        r = dict(rows[0])
        r["NORAD_CAT_ID"] = norad
        r["OBJECT_NAME"] = f"DECOY {norad}"
        r["OBJECT_ID"] = f"2099-{norad % 1000:03d}A"
        r["MEAN_ANOMALY"] = (float(r["MEAN_ANOMALY"]) + d_ma) % 360
        r["RA_OF_ASC_NODE"] = (float(r["RA_OF_ASC_NODE"]) + d_raan) % 360
        rows.append(r)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture(scope="module")
def high_pass():
    """Czas górowania wysokiego przelotu ISS nad miejscem obserwacji (blisko epoki elementów)."""
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    sat = EarthSatellite(ISS_L1, ISS_L2, "ISS", ts)
    topos = wgs84.latlon(SITE[0], SITE[1], elevation_m=SITE[2])
    t0 = ts.utc(2019, 12, 9, 16)
    times, events = sat.find_events(topos, t0, ts.utc(2019, 12, 13), altitude_degrees=45.0)
    culm = [t for t, e in zip(times, events) if e == 1]
    assert culm, "brak wysokiego przelotu ISS w oknie testowym"
    return culm[0].utc_datetime().replace(tzinfo=timezone.utc)


def synthetic_track(observer_true, satrec, t_start_offset, dur_s, track_id, noise_arcsec=7.0, seed=0):
    """Tor zmierzony: kierunki satelity w chwilach fizycznych T_true + τ (+ szum)."""
    rng = np.random.default_rng(seed)
    tau = t_start_offset + np.arange(0, dur_s, 2 / FPS)
    u = observer_true.topocentric([satrec], tau)["unit"][0]
    u = u + rng.normal(0, np.radians(noise_arcsec / 3600), size=u.shape)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return tau, u


def build(tmp_path, pass_time, prior_error_s, with_iss=True):
    rows = omm_rows()
    if not with_iss:
        rows = rows[1:]
    path = write_csv(tmp_path / "gp.csv", rows)
    t_true = pass_time - timedelta(seconds=3)            # start nagrania (prawdziwy)
    prior = t_true + timedelta(seconds=prior_error_s)    # start z metadanych (błędny)
    catalog_all = load_catalog([(write_csv(tmp_path / "all.csv", omm_rows()), "test")], prior)
    obs_true = Observer(*SITE, t_true)
    i_iss = int(np.flatnonzero(catalog_all.norad == 25544)[0])
    i_dec = int(np.flatnonzero(catalog_all.norad == 90010)[0])
    tau1, u1 = synthetic_track(obs_true, catalog_all.satrecs[i_iss], 0.0, 6.0, 1)
    # drugi tor: wabik na tej samej orbicie (2° anomalii dalej) — widoczny w innym miejscu nieba
    tau2, u2 = synthetic_track(obs_true, catalog_all.satrecs[i_dec], 1.0, 5.0, 2, seed=1)
    tracks = [make_track_sky(1, tau1, u1, np.zeros(len(tau1)), np.zeros(len(tau1))),
              make_track_sky(2, tau2, u2, np.zeros(len(tau2)), np.zeros(len(tau2)))]
    catalog = load_catalog([(path, "test")], prior)
    observer = Observer(*SITE, prior)
    return catalog, observer, tracks


def grid_factory(observer, catalog, icfg, tracks):
    def grid_for(window_s):
        offsets = np.arange(-window_s, 10 + window_s + 2, 2.0)
        center = tracks[0].u_mid
        return screen(observer, catalog, offsets, np.repeat(center[None], len(offsets), 0), 85.0, icfg)
    return grid_for


def test_sync_and_identify_two_tracks(tmp_path, high_pass, cfg):
    icfg = cfg["identify"]
    catalog, observer, tracks = build(tmp_path, high_pass, prior_error_s=37.0)
    sync, grid = synchronize(observer, catalog, tracks, grid_factory(observer, catalog, icfg, tracks), 60.0, 7200.0, icfg)
    assert sync.synced and sync.confidence == "high"
    assert sync.delta_s == pytest.approx(-37.0, abs=0.1)
    assert sync.reference["track_id"] == 1                     # „pierwszy” = najwcześniejszy w nagraniu
    assert {m["norad"] for m in sync.members} == {25544, 90010}
    ids = identify_tracks(observer, catalog, tracks, grid, sync.delta_s, icfg)
    assert ids[1][0].norad == 25544 and ids[1][0].confidence in ("high", "medium")
    assert ids[2][0].norad == 90010
    assert ids[1][0].height_km == pytest.approx(420, abs=60)


def test_single_track_gives_low_confidence(tmp_path, high_pass, cfg):
    icfg = cfg["identify"]
    catalog, observer, tracks = build(tmp_path, high_pass, prior_error_s=-20.0)
    sync, _ = synchronize(observer, catalog, tracks[:1], grid_factory(observer, catalog, icfg, tracks), 60.0, 7200.0,
                          icfg)
    assert sync.synced and sync.confidence == "low"
    assert sync.delta_s == pytest.approx(20.0, abs=0.1)
    assert sync.reference["norad"] == 25544


def test_decoys_only_do_not_match(tmp_path, high_pass, cfg):
    icfg = {**cfg["identify"], "window_sigma": 1}
    catalog, observer, tracks = build(tmp_path, high_pass, prior_error_s=10.0, with_iss=False)
    sync, grid = synchronize(observer, catalog, tracks[:1], grid_factory(observer, catalog, icfg, tracks), 20.0, 20.0,
                             icfg)
    # tor ISS nie może zostać przypisany żadnemu wabikowi
    assert not sync.synced and sync.delta_s == 0.0
    ids = identify_tracks(observer, catalog, tracks[:1], grid, sync.delta_s, icfg)
    assert ids[1] == []


def test_omm_keeps_large_norad(tmp_path):
    catalog = load_catalog([(write_csv(tmp_path / "gp.csv", omm_rows()), "test")],
                           datetime(2019, 12, 9, tzinfo=timezone.utc))
    assert 100500 in set(catalog.norad.tolist())
    assert 25544 in set(catalog.norad.tolist())
    assert catalog.name[list(catalog.norad).index(25544)] == "ISS (ZARYA)"
