"""Kreski na zdjęciach (F2): filtr odcinkowy, końce, sklejanie, łańcuchy i przerwa w serii —
na syntetycznych obrazach i torach, bez prawdziwych plików RAF."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")


def render_streak(shape, a, b, flux_per_px, psf=0.8):
    """Kreska: gęstość liniowa ``flux_per_px`` wzdłuż odcinka a–b, rozmyta PSF (Gauss)."""
    from scipy.special import erf

    h, w = shape
    Y, X = np.mgrid[0:h, 0:w].astype(float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.hypot(*(b - a))
    u = (b - a) / L
    s = (X - a[0]) * u[0] + (Y - a[1]) * u[1]
    d = -(X - a[0]) * u[1] + (Y - a[1]) * u[0]
    k = 1 / (math.sqrt(2) * psf)
    along = 0.5 * (erf(s * k) - erf((s - L) * k))
    return flux_per_px * along * np.exp(-d ** 2 / (2 * psf ** 2)) / (math.sqrt(2 * math.pi) * psf)


def test_line_kernels_unit_noise_response():
    from skyhunt.streaks import line_kernels

    k = line_kernels(11, 16)
    assert k.shape == (16, 11, 11)
    assert np.allclose((k ** 2).sum(axis=(1, 2)), 1.0, atol=1e-5)
    assert k[0, 5].sum() > 0.9 * k[0].sum()       # 0°: poziomy odcinek przez środek


def test_detect_and_refine_streak_endpoints(cfg):
    pytest.importorskip("torch")
    from skyhunt.streaks import components, line_kernels, line_response, merge_collinear, refine_streak

    dcfg = cfg["streaks"]["detect"]
    rng = np.random.default_rng(3)
    shape = (160, 240)
    a, b = (40.3, 50.6), (170.8, 95.2)
    img = rng.normal(0, 1, shape) + render_streak(shape, a, b, 6.0)
    img += render_streak(shape, (200, 30), (200.01, 30.01), 400.0, psf=1.0)   # jasny punkt (resztka gwiazdy)
    rmax = line_response(img.astype(np.float32), line_kernels(int(dcfg["line_len_px"]), int(dcfg["n_angles"]))).numpy()
    segs = merge_collinear(components(rmax, dcfg), dcfg)
    assert len(segs) == 1                       # punkt dał okrągłą plamę — odrzucony
    r = refine_streak(img.astype(np.float32), segs[0]["p0"], segs[0]["p1"], dcfg)
    ends = sorted([tuple(r["a"]), tuple(r["b"])])
    assert np.hypot(*(np.array(ends[0]) - a)) < 1.0 and np.hypot(*(np.array(ends[1]) - b)) < 1.0
    assert r["amp"] == pytest.approx(6.0, rel=0.25) and r["snr"] > 15 and r["snr_thirds"] > 5 and r["n_control"] >= 8
    assert 0.5 < r["psf_sigma"] < 1.3
    assert r["err_a"] < 1.0 and r["err_b"] < 1.0


def test_correlated_noise_is_not_a_streak(cfg):
    """Szum po wyrównaniu zdjęć jest skorelowany: S/N kreski liczone z pasów obok nie może z niego
    robić kresek (stary wzór dawał tu S/N 9–15)."""
    from scipy.ndimage import gaussian_filter

    from skyhunt.streaks import refine_streak

    dcfg = cfg["streaks"]["detect"]
    rng = np.random.default_rng(11)
    img = gaussian_filter(rng.normal(0, 1, (200, 300)), 0.8).astype(np.float32)
    snrs = []
    for k in range(6):
        p0 = np.array([60.0 + 20 * k, 60.0 + 10 * k])
        r = refine_streak(img, p0, p0 + np.array([18.0, 6.0 * (k - 2)]), dcfg)
        if r is not None:
            snrs.append(r["snr"])
    assert snrs and max(snrs) < float(dcfg["min_snr"])


def test_merge_pieces_and_dashed():
    from skyhunt.streaks import is_dashed, merge_collinear

    dcfg = {"merge_angle_deg": 3, "merge_perp_px": 2.0, "merge_gap_px": 50}
    segs = [{"p0": np.array([10.0 + 40 * i, 20.0 + 4 * i]), "p1": np.array([30.0 + 40 * i, 22.0 + 4 * i]),
             "peak": 6.0, "pieces": [(0.0, 20.1)]} for i in range(4)]
    segs.append({"p0": np.array([10.0, 120.0]), "p1": np.array([60.0, 120.0]), "peak": 9.0, "pieces": [(0.0, 50.0)]})
    out = merge_collinear(segs, dcfg)
    assert len(out) == 2
    dashed = next(s for s in out if len(s["pieces"]) == 4)
    assert dashed["p0"][0] == pytest.approx(10.0) and dashed["p1"][0] == pytest.approx(150.0)
    assert is_dashed(dashed["pieces"], 3, 0.35)
    assert not is_dashed([(0, 20), (25, 30), (90, 100)], 3, 0.35)


def test_neighbor_median_with_nan_and_streak():
    torch = pytest.importorskip("torch")
    from skyhunt.streaks import neighbor_median

    v = torch.tensor([[1.0, 5.0], [2.0, float("nan")], [100.0, 7.0], [3.0, float("nan")]])[..., None]
    med, n = neighbor_median(v)
    assert med[:, 0].tolist() == pytest.approx([2.5, 6.0])     # kreska (100) odrzucona; 2 wartości → średnia
    assert n[:, 0].tolist() == [4, 2]


def test_end_clipped_and_sample_grid():
    from skyhunt.streaks import end_clipped, sample_grid

    blocked = np.zeros((50, 80), bool)
    blocked[:, 70:] = True
    assert end_clipped(np.array([68.0, 25.0]), np.array([1.0, 0.0]), blocked, 3)
    assert not end_clipped(np.array([40.0, 25.0]), np.array([1.0, 0.0]), blocked, 3)
    assert end_clipped(np.array([1.0, 25.0]), np.array([-1.0, 0.0]), blocked, 3)     # brzeg kadru
    coarse = np.array([[0.0, 10.0], [100.0, 110.0]])                 # węzły w rogach obrazu 80×50
    assert sample_grid(coarse, (51, 81), [40.0], [25.0])[0] == pytest.approx(55.0)


def synthetic_session(g_true=0.12, g0=0.10, n_sets=6, swap_seed=5, jitter_s=0.0, drop_set=None):
    """Dwa obiekty przez 6 serii bracketingu (0,5 s / 1 s / 0,25 s, co 4 s) + jedna samotna kreska.
    Czasy otwarcia w tabeli z nominalną przerwą g0, prawdziwe kreski z g_true."""
    import pandas as pd

    rng = np.random.default_rng(swap_seed)
    T = [0.5, 1.0, 0.25]
    objs = [lambda t: (100 + 40 * t + 0.08 * t * t, 300 + 10 * t), lambda t: (1500 - 30 * t, 100 + 25 * t)]
    rows = []
    for k in range(n_sets):
        S = 4.0 * k + 0.3 + (rng.normal(0, jitter_s) if jitter_s else 0.0)   # prawdziwy start (interwałometr)
        S_nom = 4.0 * k + 0.3
        for m in range(3):
            true_open = S + sum(T[:m]) + m * g_true
            nominal = S_nom + sum(T[:m]) + m * g0
            for o, f in enumerate(objs):
                a = np.array(f(true_open)) + rng.normal(0, 0.1, 2)
                b = np.array(f(true_open + T[m])) + rng.normal(0, 0.1, 2)
                swap = bool(rng.integers(0, 2))
                if drop_set is not None and o == 0 and k == drop_set:
                    continue
                if swap:
                    a, b = b, a
                rows.append({"photo": 3 * k + m, "set": k, "tau_open": nominal, "exposure_s": T[m], "m": m,
                             "xa": a[0], "ya": a[1],
                             "xb": b[0], "yb": b[1], "ya_s": a[1], "yb_s": b[1], "clip_a": False, "clip_b": False,
                             "snr": 30.0, "err_a": 0.1, "err_b": 0.1, "obj": o, "swapped": swap})
    rows.append({"photo": 7, "set": 2, "tau_open": 0.0, "exposure_s": 1.0, "m": 1, "xa": 900, "ya": 900, "xb": 950, "yb": 960,
                 "ya_s": 900, "yb_s": 960, "clip_a": False, "clip_b": False, "snr": 12.0, "err_a": 0.2, "err_b": 0.2,
                 "obj": 9, "swapped": False})
    df = pd.DataFrame(rows).sort_values("photo", kind="stable").reset_index(drop=True)
    df["streak_id"] = np.arange(len(df))
    return df


def test_link_chains_and_gap_from_geometry(cfg):
    from skyhunt.streaks import fit_gap, link_chains

    lcfg = cfg["streaks"]["link"]
    st = synthetic_session()
    chains = link_chains(st, lcfg, 0.0, 1333)
    sizes = sorted(len(c["items"]) for c in chains)
    assert sizes == [1, 18, 18]
    for c in chains:
        objs = {int(st["obj"][k]) for k, _ in c["items"]}
        assert len(objs) == 1                                         # bez mieszania obiektów
        if len(c["items"]) > 1:
            for k, o in c["items"]:                                   # kierunek lotu odtworzony
                assert o == (-1 if st["swapped"][k] else 1)
    fit = fit_gap(st, chains, 0.10, 0.0, 1333, lcfg)
    assert fit["fitted"] and fit["n_chains"] == 2
    assert fit["g_s"] == pytest.approx(0.12, abs=0.003)
    assert fit["sigma_s"] < 0.005 and fit["rms_px"] < 0.5


def test_photo_hint():
    from skyhunt.photo_stages import photo_hint

    icfg = {"cand_min_deg_s": 0.2, "cand_max_deg_s": 3.0, "cand_max_curv_arcsec": 150}
    assert photo_hint(12, 0.7, 20, 0.0, icfg)[0] == "satelita?"
    assert photo_hint(5, 0.4, 20, 0.8, icfg)[0] == "samolot?"
    assert photo_hint(1, 8.0, 0, 0.0, icfg)[0] == "meteor?"
    assert photo_hint(1, 0.7, 0, 0.0, icfg)[0] == "pojedyncza kreska"


def test_merge_chain_broken_by_missing_set(cfg):
    from skyhunt.streaks import link_chains

    st = synthetic_session(drop_set=2)            # obiekt 0 niewidoczny przez całą serię (> max_gap_s)
    chains = link_chains(st, cfg["streaks"]["link"], 0.0, 1333)
    assert sorted(len(c["items"]) for c in chains) == [1, 15, 18]
    assert all(len({int(st["obj"][k]) for k, _ in c["items"]}) == 1 for c in chains)


def test_set_jitter_estimate(cfg):
    from skyhunt.streaks import fit_gap, link_chains

    lcfg = cfg["streaks"]["link"]
    calm = synthetic_session()
    fit = fit_gap(calm, link_chains(calm, lcfg, 0.0, 1333), 0.10, 0.0, 1333, lcfg)
    assert fit["set_jitter_pairs"] > 0 and fit["set_jitter_ms"] < 10
    shaky = synthetic_session(jitter_s=0.05)      # start serii losowo ±50 ms, wspólny dla obu obiektów
    fit = fit_gap(shaky, link_chains(shaky, lcfg, 0.0, 1333), 0.10, 0.0, 1333, lcfg)
    assert fit["set_jitter_ms"] > 15


def test_streak_reject_reason(cfg):
    from skyhunt.photo_stages import streak_reject_reason

    dcfg = cfg["streaks"]["detect"]
    good = {"length": 40.0, "snr": 20.0, "snr_thirds": 8.0, "psf_sigma": 1.0}
    assert streak_reject_reason(good, dcfg) is None
    assert streak_reject_reason(None, dcfg) == "bez_dopasowania"
    assert streak_reject_reason(good | {"psf_sigma": 0.3}, dcfg) == "końce"        # szum: brzeg na granicy fitu
    assert streak_reject_reason(good | {"snr_thirds": 0.5}, dcfg) == "nierówna"
    assert streak_reject_reason(good | {"snr": float("nan")}, dcfg) == "S/N"
