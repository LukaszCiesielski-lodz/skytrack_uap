"""Planetoidy/NEO: format zapytania JPL sb_ident, parsowanie odpowiedzi, stos okien przesuwanych
za obiektem, fotometria z punktem zerowym (winietowanie), werdykty, dopasowanie szybkiego NEO
do toru. Bez sieci: odpowiedzi JPL i VizieR są wklejone."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from skyhunt import smallbodies as SB

JPL = {"signature": {"version": "1.1"},
       "summary": {"lat": "51.7", "lon": "19.6", "alt": "0.2"},
       "fields_second": ["Object name", "Astrometric RA (hh:mm:ss)", "Astrometric Dec (dd mm'ss\")",
                         "Dist. from center RA (\")", "Dist. from center Dec (\")", "Dist. from center Norm (\")",
                         "Visual magnitude (V)", "RA rate (\"/h)", "Dec rate (\"/h)"],
       "data_second_pass": [["198 Ampella (A879 LA)", "22:39:31.27", "+11 43'40.4\"", "1.E5", "-5.E4", "1.4E5",
                             "10.7", "-1.550E+01", "-1.812E+01"],
                            ["C/2025 X1 (Test)", "20:00:00.00", "-05 30'00.0\"", "0", "0", "0", "11.9",
                             "3.0E+01", "0"]]}
NEO = {"fields_first": ["Object name", "Astrometric RA (hh:mm:ss)", "Astrometric Dec (dd mm'ss\")",
                        "Dist. from center RA (\")", "Dist. from center Dec (\")", "Dist. from center Norm (\")",
                        "Visual magnitude (V)", "RA rate (\"/h)", "Dec rate (\"/h)", "Est. error RA (\")",
                        "Est. error Dec (\")"],
       "data_first_pass": [["99942 Apophis (2004 MN4)", "19:50:00.00", "+08 50'00.0\"", "0", "0", "0", "15.5",
                            "1.2E+03", "-3.0E+02", "2.0", "3.5"],
                           ["198 Ampella (A879 LA)", "22:39:31.27", "+11 43'40.4\"", "0", "0", "0", "10.7",
                            "-1.550E+01", "-1.812E+01", "0.1", "0.1"]]}


def test_sexagesimal_format_and_parse():
    assert SB.ra_hms(339.880) == "22-39-31.20"
    assert SB.ra_hms(359.99999) == "00-00-00.00"
    assert SB.dec_dms(-5.5) == "M05-30-00.0" and SB.dec_dms(11.7279) == "11-43-40.4"
    assert SB.parse_ra("22:39:31.27") == pytest.approx(339.88029, abs=1e-5)
    assert SB.parse_dec("+11 43'40.4\"") == pytest.approx(11.72789, abs=1e-5)
    assert SB.parse_dec("-05 30'00.0\"") == pytest.approx(-5.5)
    assert SB.parse_dec("-00 12'00\"") == pytest.approx(-0.2)


def test_query_params_rounding_and_width():
    scfg = {"site_round_deg": 0.1, "vmag_lim": 13.0, "neo_vmag_lim": 16.0}
    t = datetime(2026, 10, 1, 19, 0, 0, tzinfo=timezone.utc)
    p = SB.query_params(299.0, 45.0, 16.0, t, (51.737064, 19.572815, 221.0), scfg)
    assert p["lat"] == pytest.approx(51.7) and p["lon"] == pytest.approx(19.6) and p["alt"] == pytest.approx(0.2)
    assert p["obs-time"] == "2026-10-01_19:00:00" and p["fov-dec-hwidth"] == pytest.approx(16.0)
    assert p["fov-ra-hwidth"] > 16.0 / np.cos(np.radians(45))      # szerzej w RA (zbieżność południków)
    assert "sb-group" not in p and p["vmag-lim"] == 13.0
    n = SB.query_params(299.0, 80.0, 16.0, t, (51.7, 19.6, 221.0), scfg, neo=True)
    assert n["sb-group"] == "neo" and n["vmag-lim"] == 16.0 and n["fov-ra-hwidth"] == 180.0


def test_parse_and_merge_bodies():
    main = SB.parse_sbident(JPL)
    assert [b["name"] for b in main] == ["198 Ampella (A879 LA)", "C/2025 X1 (Test)"]
    a = main[0]
    assert a["vmag"] == 10.7 and a["ra_rate"] == pytest.approx(-15.5) and a["kind"] == "planetoida"
    assert main[1]["kind"] == "kometa" and main[1]["dec"] == pytest.approx(-5.5)
    neo = SB.parse_sbident(NEO, is_neo=True)
    assert neo[0]["err_arcsec"] == pytest.approx(3.5)
    allb = SB.merge_bodies(main, neo)
    assert [b["name"] for b in allb][0] == "198 Ampella (A879 LA)"    # najjaśniejsze pierwsze
    by = {b["name"]: b for b in allb}
    assert by["198 Ampella (A879 LA)"]["is_neo"] and by["99942 Apophis (2004 MN4)"]["is_neo"]
    assert not by["C/2025 X1 (Test)"]["is_neo"]
    ra, dec = SB.body_radec(by["99942 Apophis (2004 MN4)"], 3600.0)
    assert dec == pytest.approx(SB.parse_dec("+08 50'00\"") - 300 / 3600)
    assert (ra - 297.5) * np.cos(np.radians(8.83)) == pytest.approx(1200 / 3600, rel=0.01)


def test_stack_windows_follows_target_and_gains_snr():
    rng = np.random.default_rng(3)
    H, W, n = 80, 120, 64
    path = lambda f: (40.0 + 0.3 * f, 30.0 + 0.1 * f)                        # noqa: E731
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for f in range(n):
        x, y = path(f)
        frames.append(10.0 + 0.8 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2 ** 2)) + rng.normal(0, 1.0, (H, W)))
    frames = np.array(frames)
    batches = [(s, frames[s:s + 8]) for s in range(0, n, 8)]

    def pos_at(fr):
        return np.array([[path(float(f)) for f in fr]])

    st = SB.stack_windows(batches, pos_at, 1, 10, (H, W), lambda v: np.asarray(v, float))[0]
    assert st.n == n
    ph = SB.photometry(st.mean, *st.center, 3, 5, 8)
    single = SB.photometry(frames[0][20:41, 30:51], 10.0, 10.0, 3, 5, 8)
    assert ph["snr"] > 5                                   # pojedyncza klatka: SNR ~1
    assert ph["sigma"] < 0.3 * single["sigma"]             # szum średniej z 64 klatek: ~1/8
    cx, cy = SB.centroid(st.mean, *st.center, 2.5)
    assert abs(cx - st.center[0]) < 0.8 and abs(cy - st.center[1]) < 0.8


def test_zero_point_with_vignetting():
    rng = np.random.default_rng(0)
    r2 = rng.uniform(0, 1.2, 30)
    mags = rng.uniform(4, 6, 30)
    zp_true = 12.0 - 0.8 * r2                       # brzegi ciemniejsze
    flux = 10 ** (-0.4 * (mags - zp_true)) * (1 + rng.normal(0, 0.02, 30))
    flux[:3] *= 5                                   # gwiazdy z sąsiadem w aperturze
    a, b, rms, n = SB.fit_zero_point(r2, mags, flux)
    assert a == pytest.approx(12.0, abs=0.05) and b == pytest.approx(-0.8, abs=0.08) and n >= 25 and rms < 0.05
    assert SB.mag_from_flux(10 ** (-0.4 * (9.0 - 12.0)), 12.0) == pytest.approx(9.0)
    assert SB.fit_zero_point([0, 1], [5, 5], [1, 1]) is None


def test_status_verdicts():
    cfg = {"detect_snr": 5, "mag_tol": 1.0, "blend_dmag": 1.5}
    base = {"vmag": 10.7, "mag_meas": 10.9, "mag_lim": 11.8, "snr": 9.0, "blend_mag": float("nan")}
    assert SB.status(base, cfg) == "wykryta"
    assert SB.status({**base, "blend_mag": 11.5}, cfg).startswith("zlewa się z gwiazdą")
    assert SB.status({**base, "mag_meas": 8.5}, cfg).startswith("źródło, jasność niezgodna")
    assert SB.status({**base, "snr": 2.0, "vmag": 12.4}, cfg) == "za słaba (zasięg 11.8 mag)"
    assert SB.status({**base, "snr": 2.0, "vmag": 11.0}, cfg) == "niewykryta"


def test_parse_vizier_tsv():
    text = ("#\n# VizieR Astronomical Server\n#Column\tRA_ICRS\n\nRA_ICRS\tDE_ICRS\tGmag\n"
            "deg\tdeg\tmag\n-----------\t-----------\t-------\n"
            "339.881234\t+11.728001\t11.203\n339.879000\t+11.729500\t 9.874\n")
    stars = SB.parse_vizier_tsv(text)
    assert stars == [(339.881234, 11.728001, 11.203), (339.879, 11.7295, 9.874)]


def test_match_fast_neo_to_track():
    t0 = datetime(2026, 10, 1, 19, 0, 0, tzinfo=timezone.utc)
    rate = 100000.0                                  # ″/h ≈ 28″/s: bardzo bliski przelot
    body = {"name": "2026 AB", "ra": 300.0, "dec": 20.0, "vmag": 12.0, "ra_rate": rate, "dec_rate": 0.0,
            "is_neo": True}
    cosd = np.cos(np.radians(20.0))
    d_ra = rate * 10 / 3600 / 3600 / cosd            # 10 s toru
    track = {"track_id": 7, "tau0": 95.0, "tau1": 105.0, "ra0": 300.0 - d_ra / 2, "dec0": 20.0,
             "ra1": 300.0 + d_ra / 2, "dec1": 20.0}
    m = SB.match_tracks([track], [body], t0 + timedelta(seconds=100), t0, 0.0, {"track_match_deg": 0.2})
    assert len(m) == 1 and m[0]["track_id"] == 7 and m[0]["sep_deg"] < 0.01
    slow = dict(body, ra_rate=20.0)                  # planetoida pasa głównego: tempo niezgodne
    assert SB.match_tracks([track], [slow], t0 + timedelta(seconds=100), t0, 0.0, {"track_match_deg": 0.2}) == []
