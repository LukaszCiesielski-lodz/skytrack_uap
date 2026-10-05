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


def test_median_delta_per_satellite_without_classfd():
    from skyhunt.satellites import Match, median_delta

    def m(tid, norad, d, src="spacetrack"):
        return Match(tid, 0, norad, "x", "", "", src, d, 0.01, 0.01, 1.0, 1.0, 0.0, 500.0, 450.0, 0.8)

    cluster = [m(1, 100, 4.95), m(2, 101, 4.40), m(3, 102, 4.50), m(4, 103, 4.60),
               m(5, 200, 5.00, "classfd"), m(6, 200, 5.05, "classfd"), m(7, 200, 5.02, "classfd"),
               m(8, 104, 4.45), m(9, 104, 4.47)]                  # NORAD 104: pocięty na 2 tory
    med, sig, n = median_delta(cluster)
    assert n == 5 and med == pytest.approx(4.50) and 0 < sig < 0.2
    med2, _, n2 = median_delta(cluster[4:7] + cluster[:1])         # mało publicznych: classfd też liczony
    assert n2 == 2 and med2 == pytest.approx((4.95 + 5.02) / 2)


def test_observer_offset_from_parallax(tmp_path, high_pass):
    """Tory nagrane ~2,2 km od współrzędnych w configu (jak DSCF4641): paralaksa odtwarza przesunięcie."""
    from skyhunt.satellites import observer_offset

    catalog = load_catalog([(write_csv(tmp_path / "gp.csv", omm_rows()), "test")], high_pass)
    t0 = high_pass - timedelta(seconds=3)
    obs_true = Observer(SITE[0] + 0.019, SITE[1] - 0.010, SITE[2], t0)   # ≈ 2,11 km N, 0,69 km W
    observer = Observer(*SITE, t0)
    members, skies = [], {}
    for tid, norad, start in ((1, 25544, 0.0), (2, 90010, 1.0)):
        i = int(np.flatnonzero(catalog.norad == norad)[0])
        tau, u = synthetic_track(obs_true, catalog.satrecs[i], start, 5.0, tid, seed=tid)
        skies[tid] = make_track_sky(tid, tau, u, np.zeros(len(tau)), np.zeros(len(tau)))
        members.append({"track_id": tid, "cat_index": i, "delta_s": 0.0})
    off = observer_offset(observer, catalog, members, skies)
    assert off["north_km"] == pytest.approx(0.019 * 111.25, abs=0.25)
    assert off["east_km"] == pytest.approx(-0.010 * 111.32 * np.cos(np.radians(SITE[0])), abs=0.25)
    assert off["rms_after_arcsec"] < 0.2 * off["rms_before_arcsec"]


def test_sunlit_flags_fill_dicts_and_matches(monkeypatch):
    import skyhunt.satellites as S

    seen = {}
    real_sunlit_at = S.sunlit_at

    def fake(catalog, indices, offsets, observer, eph):
        seen["args"] = (list(indices), list(offsets))
        return [True, False]

    monkeypatch.setattr(S, "sunlit_at", fake)
    members = [{"track_id": 1, "cat_index": 5, "delta_s": 2.0}, {"track_id": 2, "cat_index": 7, "delta_s": 2.0}]
    eph = object()
    assert S.sunlit_flags(None, members, None, None, {1: 10.0, 2: 20.0}, eph=eph) is eph
    assert [m["sunlit"] for m in members] == [True, False]
    assert seen["args"] == ([5, 7], [12.0, 22.0])
    assert real_sunlit_at(None, [1, 2], [0.0, 1.0], None, None) == [None, None]


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


class _VecCamera:
    """Kamera testowa: RA/Dec w tabeli punktów to wprost współrzędne kątowe wektorów kierunku."""

    @staticmethod
    def apparent_vectors(ra, dec, tau):
        ra, dec = np.radians(ra), np.radians(dec)
        return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _split_pass(tmp_path, high_pass, cfg, offset_arcsec=0.0):
    """Przelot ISS jako dwa tory (0–3 s i 4–7 s, przerwa jak brakująca seria zdjęć); drugi tor
    opcjonalnie przesunięty w poprzek o ``offset_arcsec`` (inny obiekt obok toru)."""
    import pandas as pd

    icfg = cfg["identify"]
    catalog, observer, _ = build(tmp_path, high_pass, prior_error_s=0.0)
    obs_true = Observer(*SITE, high_pass - timedelta(seconds=3))
    i_iss = int(np.flatnonzero(catalog.norad == 25544)[0])
    rows, skies = [], {}
    for tid, (t_a, dur) in ((1, (0.0, 3.0)), (2, (4.0, 3.0))):
        tau, u = synthetic_track(obs_true, catalog.satrecs[i_iss], t_a, dur, tid, seed=tid)
        if tid == 2 and offset_arcsec:
            pole = make_track_sky(tid, tau, u, tau, tau).gc.pole
            u = u + pole * np.radians(offset_arcsec / 3600)
            u /= np.linalg.norm(u, axis=1, keepdims=True)
        ra, dec = np.degrees(np.arctan2(u[:, 1], u[:, 0])), np.degrees(np.arcsin(u[:, 2]))
        skies[tid] = make_track_sky(tid, tau, u, ra, dec)
        rows += [{"track_id": tid, "tau": t, "ra": r, "dec": d} for t, r, d in zip(tau, ra, dec)]
    grid = grid_factory(observer, catalog, icfg, list(skies.values()))(5.0)
    matches = identify_tracks(observer, catalog, list(skies.values()), grid, 0.0, icfg)
    tracks = {1: {"items": [(10, 1), (11, 1)], "single": False}, 2: {"items": [(12, 1), (13, 1)], "single": False}}
    return tracks, skies, matches, pd.DataFrame(rows), observer, catalog, grid, icfg


def test_merge_same_norad_joins_split_pass(tmp_path, high_pass, cfg):
    from skyhunt.photo_stages import merge_same_norad

    tracks, skies, matches, pts, observer, catalog, grid, icfg = _split_pass(tmp_path, high_pass, cfg)
    assert matches[1][0].norad == matches[2][0].norad == 25544
    pts = merge_same_norad(tracks, skies, matches, pts, _VecCamera(), observer, catalog, grid, 0.0, icfg,
                           cfg["streaks"]["link"])
    assert list(tracks) == [1] and tracks[1]["items"] == [(10, 1), (11, 1), (12, 1), (13, 1)]
    assert list(matches) == [1] and matches[1][0].norad == 25544
    assert set(pts["track_id"]) == {1} and skies[1].dur_s == pytest.approx(7.0, abs=0.1)


def test_merge_same_norad_flags_offset_track(tmp_path, high_pass, cfg):
    from skyhunt.photo_stages import merge_same_norad

    tracks, skies, matches, pts, observer, catalog, grid, icfg = _split_pass(tmp_path, high_pass, cfg, 250.0)
    assert matches[2][0].norad == 25544 and matches[2][0].confidence == "medium"
    merge_same_norad(tracks, skies, matches, pts, _VecCamera(), observer, catalog, grid, 0.0, icfg,
                     cfg["streaks"]["link"])
    assert set(tracks) == {1, 2}
    assert matches[1][0].same_norad_as == [2] and matches[2][0].same_norad_as == [1]
    assert "ten sam NORAD co #2" in matches[1][0].reason


def test_point_residuals_show_offset(tmp_path, high_pass, cfg):
    from skyhunt.satellites import point_residuals

    _, skies, matches, _, observer, catalog, _, _ = _split_pass(tmp_path, high_pass, cfg, 250.0)
    cross1, along1 = point_residuals(observer, catalog, matches[1][0], skies[1])
    cross2, _ = point_residuals(observer, catalog, matches[2][0], skies[2])
    assert np.nanmedian(np.abs(cross1)) < 30 and np.nanmedian(np.abs(along1)) < 0.05
    assert np.nanmedian(np.abs(cross2)) == pytest.approx(250, abs=40)


def test_light_state_and_shadow_crossing(monkeypatch):
    import skyhunt.satellites as S

    assert S.light_state([True, True, True]) == "oświetlony"
    assert S.light_state([False, False, False]) == "w cieniu"
    assert S.light_state([True, True, False]) == "wchodzi w cień"
    assert S.light_state([False, True, True]) == "wychodzi z cienia"
    assert S.light_state([True, False, True]) == "na granicy cienia"
    assert S.light_state([True, None, True]) is None

    edge = 1004.3          # granica cienia w chwili δ + τ
    monkeypatch.setattr(S, "sunlit_at", lambda cat, idx, offs, obs, eph: [o < edge for o in offs])
    m = S.Match(1, 0, 1, "x", "", "", "test", 1000.0, 0.01, 0.01, 1.0, 1.0, 0.0, 500.0, 450.0, 0.8)
    tau = np.linspace(0.0, 10.0, 11)
    track = make_track_sky(1, tau, np.tile([1.0, 0.0, 0.0], (11, 1)) + np.c_[np.zeros(11), tau * 1e-3, np.zeros(11)],
                           tau, tau)
    S.track_light(None, m, track, None, object())
    assert m.light == "wchodzi w cień" and m.shadow_tau == pytest.approx(4.3, abs=0.02)
