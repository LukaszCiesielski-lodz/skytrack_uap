"""Filtr statyczny, łączenie w tory, sklejanie fragmentów, pomiary i podpowiedź klasy."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from skyhunt.tracks import build_tracks, classify_hint, flock_groups, periodicity, static_mask  # noqa: E402

FPS = 24.0


def make_dets(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(100):                       # A: szybki, 3.16 px/klatkę
        rows.append((f, 100 + 3 * f, 200 + f))
    for f in range(240):                       # B: wolny, 0.5 px/klatkę przez 10 s
        rows.append((f, 500 + 0.5 * f, 300))
    for f in range(50, 151):                   # C: z przerwą 4 klatek (fragmenty do sklejenia)
        if not 90 <= f <= 93:
            rows.append((f, 1100 - 2 * (f - 50), 100 + 2 * (f - 50)))
    for f in range(240):
        if rng.random() < 0.6:                 # migocząca gwiazda
            rows.append((f, 50 + rng.normal(0, 0.3), 50 + rng.normal(0, 0.3)))
        rows.append((f, 1000, 700))            # gorący piksel
        for _ in range(3):                     # szum
            rows.append((f, rng.uniform(0, 1200), rng.uniform(0, 800)))
    a = np.array(rows, float)
    n = len(a)
    return {"frame": a[:, 0].astype(np.int64), "x": a[:, 1], "y": a[:, 2], "flux": np.full(n, 100.0),
            "sum_snr": np.full(n, 20.0), "peak_snr": np.full(n, 8.0), "npix": np.full(n, 9.0),
            "cxx": np.ones(n), "cxy": np.zeros(n), "cyy": np.ones(n)}


def test_static_mask(cfg):
    det = make_dets()
    st = static_mask(det["frame"], det["x"], det["y"], FPS, cfg["tracks"])
    star = (np.abs(det["x"] - 50) < 2) & (np.abs(det["y"] - 50) < 2)
    hot = (det["x"] == 1000) & (det["y"] == 700)
    slow = det["y"] == 300
    assert st[star].all() and st[hot].all()
    assert not st[slow].any()


def test_build_tracks(cfg):
    rows, points, stats = build_tracks(make_dets(), fps=FPS, width=1280, height=800, star_sigma_px=1.0,
                                       nominal_hfov_deg=26.4, tcfg=cfg["tracks"])
    speeds = sorted(r["speed_px_frame"] for r in rows)
    assert len(rows) == 3, rows
    assert speeds == pytest.approx([0.5, np.hypot(2, 2), np.hypot(3, 1)], rel=0.01)
    c = next(r for r in rows if abs(r["speed_px_frame"] - np.hypot(2, 2)) < 0.1)
    assert c["n"] == 97 and c["frame0"] == 50 and c["frame1"] == 150   # fragmenty sklejone
    assert all(r["curv_px"] < 0.5 for r in rows)
    assert set(np.unique(points["track_id"])) == {1, 2, 3}
    assert stats["static"] > 200


def test_weak_short_dropped_and_long_gap_merged(cfg):
    rows = []
    for f in range(10, 17):                    # A: 7 kropek SNR 5,5 — łańcuch szumu
        rows.append((f, 100 + 5 * f, 100, 5.5))
    for f in range(30, 42):                    # B: słaby, ale 12 punktów — zostaje
        rows.append((f, 400 + 5 * (f - 30), 300, 6.0))
    for f in range(60, 66):                    # C: krótki, ale jasny (meteor/błysk) — zostaje
        rows.append((f, 700 + 5 * (f - 60), 500, 20.0))
    for f in [*range(100, 150), *range(190, 240)]:   # D: jasny, przerwa 40 klatek (jak tor #9)
        rows.append((f, 100 + 3 * (f - 100), 600 + (f - 100), 50.0))
    streak = []
    for f in range(300, 307):                  # E: słaby meteor — 7 kresek 20 px/klatkę, σ wzdłuż 6 px
        streak.append((f, 100 + 20 * (f - 300), 50, 5.5))
    a = np.array(rows + streak, float)
    n = len(a)
    cxx = np.ones(n)
    cxx[len(rows):] = 36.0
    det = {"frame": a[:, 0].astype(np.int64), "x": a[:, 1], "y": a[:, 2], "flux": np.full(n, 100.0),
           "sum_snr": a[:, 3] * 3, "peak_snr": a[:, 3], "npix": np.full(n, 9.0),
           "cxx": cxx, "cxy": np.zeros(n), "cyy": np.ones(n)}
    out, _, stats = build_tracks(det, fps=30.0, width=1280, height=800, star_sigma_px=1.0,
                                 nominal_hfov_deg=26.4, tcfg=cfg["tracks"])
    assert sorted((r["frame0"], r["n"]) for r in out) == [(30, 12), (60, 6), (100, 100), (300, 7)]
    assert stats["weak_dropped"] == 1
    meteor = next(r for r in out if r["frame0"] == 300)
    assert meteor["along_sigma_px"] == pytest.approx(6.0) and meteor["streak_ratio"] == pytest.approx(6.0)


def test_periodicity_and_alias(cfg):
    frames = np.arange(96)
    flux = 100 + 30 * np.sin(2 * np.pi * 5.2 * frames / FPS)
    f, power = periodicity(frames, flux, FPS, cfg["tracks"])
    assert f == pytest.approx(5.2, abs=0.26) and power > 6
    assert FPS - f == pytest.approx(18.8, abs=0.26)


def test_kinematics_straight_vs_turning():
    from skyhunt.tracks import kinematics

    f = np.arange(0, 120)
    k = kinematics(f, 100 + 5.0 * f, 200 + 2.0 * f, FPS)                       # jednostajny, prosty
    assert k["accel_px_s2"] < 1e-6 and k["speed_cv"] < 1e-6 and k["turn_deg"] < 1e-6
    ang = np.radians(90 * f / f[-1])                                          # skręt o 90° ze stałą prędkością
    xs, ys = np.cumsum(5 * np.cos(ang)), np.cumsum(5 * np.sin(ang))
    k = kinematics(f, xs, ys, FPS)
    assert 70 < k["turn_deg"] < 95 and k["speed_cv"] < 0.05 and k["accel_px_s2"] > 10
    t = f / FPS                                                               # hamowanie
    k = kinematics(f, 300 * t - 40 * t ** 2, 0 * t, FPS)
    assert k["accel_px_s2"] == pytest.approx(80, rel=0.01) and k["speed_cv"] > 0.1


def test_classify_hint(cfg):
    c = cfg["classify"]
    assert classify_hint(4.0, 0.8, 1.0, 1.0, np.nan, 0, FPS, c)[0] == "meteor?"
    assert classify_hint(0.9, 6.0, 0.5, 1.0, np.nan, 0, FPS, c)[0] == "satelita?"
    assert classify_hint(0.5, 10.0, 0.5, 1.0, 1.0, 20, FPS, c)[0] == "samolot?"
    label, reason = classify_hint(4.4, 3.0, 26.0, 2.2, 5.2, 20, FPS, c)       # jak tor #69 z DSCF4641
    assert label == "ptak?" and "18.8" in reason and "nieostry" in reason
    # bez istotnej modulacji (moc 3 jak u satelitów) zakrzywiony tor to nie „ptak?”
    assert classify_hint(3.4, 6.0, 20.8, 1.6, 11.2, 3.2, FPS, c)[0] == "bliski obiekt?"
    # tor #58 z DSCF4641 (STARLINK-2112): prosty, 0,77°/s, szeroki tylko przez prześwietlenie
    assert classify_hint(0.77, 7.8, 0.5, 2.46, 5.0, 3, FPS, c)[0] == "bliski obiekt?"
    assert classify_hint(0.77, 7.8, 0.5, 2.46, 5.0, 3, FPS, c, peak_snr=200.0)[0] == "satelita?"


def test_flock_groups_parallel_tracks(cfg):
    """Przelot ptaków: 3 równoległe tory ~2°/s na różnych liniach + fragment jednego z nich;
    wolny tor (satelita) i tor prostopadły nie należą do grupy."""
    cc = cfg["classify"]
    tid = [1, 2, 3, 4, 5, 6]
    deg_s = [2.0, 1.9, 2.1, 2.0, 0.6, 2.0]
    x0 = [100, 100, 100, 1500, 100, 2000]
    y0 = [500, 800, 1100, 505, 300, 100]
    x1 = [1200, 1200, 1200, 2500, 1200, 2000]
    y1 = [520, 820, 1120, 525, 320, 1500]
    g = flock_groups(tid, deg_s, x0, y0, x1, y1, cc)
    assert set(g) == {1, 2, 3, 4}
    assert g[1][0] == 3 and g[4][0] == 3          # fragment #4 leży na linii #1: liczy się raz
    assert g[2][1] == pytest.approx(2.0, abs=0.1)
    assert flock_groups(tid[:2], deg_s[:2], x0[:2], y0[:2], x1[:2], y1[:2], cc) == {}
