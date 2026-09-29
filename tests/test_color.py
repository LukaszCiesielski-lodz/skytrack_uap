"""Kolor: odczyt RGB z wycinka, linearyzacja, linia gwiazd (B−V), pomiar koloru toru z odjęciem
tła z sąsiednich klatek, wykrywanie nagrań czarno-białych, podpowiedzi."""
import math

import numpy as np
import pytest

from skyhunt import color as C
from skyhunt.metadata import probe_video
from tests.conftest import SYN_H, SYN_W, color_dot_position


def test_linearize_roundtrip():
    v = np.linspace(0, 1, 101, dtype=np.float32)
    for tr in ("bt709", 2.2, "linear"):
        cc = {"transfer": tr}
        assert np.allclose(C.delinearize(C.linearize(v, cc), cc), v, atol=1e-4)
    assert C.linearize(np.float32(0.5), {"transfer": "bt709"}) == pytest.approx(0.2598, abs=1e-3)


def test_bv_to_kelvin_sun_and_monotonic():
    assert C.bv_to_kelvin(0.65) == pytest.approx(5780, abs=60)
    t = C.bv_to_kelvin(np.linspace(-0.3, 2.0, 20))
    assert np.all(np.diff(t) < 0)


def test_fit_locus_robust_and_projection():
    rng = np.random.default_rng(1)
    bv = rng.uniform(-0.1, 1.6, 60)
    rg = 0.10 + 0.40 * bv + rng.normal(0, 0.01, 60)
    bg = -0.05 - 0.60 * bv + rng.normal(0, 0.01, 60)
    rg[:12] += 0.5                                         # 20% odstających (np. gwiazda w łunie)
    cal, use = C.fit_locus(bv, rg, bg, min_n=15)
    assert cal.ar == pytest.approx(0.10, abs=0.02) and cal.br == pytest.approx(0.40, abs=0.02)
    assert cal.ab == pytest.approx(-0.05, abs=0.02) and cal.bb == pytest.approx(-0.60, abs=0.02)
    assert not use[:12].any() and cal.rms < 0.03
    b, ge = cal.project(*cal.locus(1.0))
    assert b == pytest.approx(1.0, abs=1e-6) and ge == pytest.approx(0.0, abs=1e-6)
    x, y = cal.locus(0.6)
    b, ge = cal.project(x - 0.1, y - 0.1)                   # mniej R i mniej B niż gwiazda = zieleńszy
    assert ge > 0.05 and b == pytest.approx(0.6, abs=0.1)
    assert C.ColorCalib.from_dict(cal.to_dict()).bb == pytest.approx(cal.bb)


def test_aperture_photometry_gaussian():
    yy, xx = np.mgrid[0:25, 0:25]
    g = np.exp(-((xx - 12.3) ** 2 + (yy - 11.8) ** 2) / (2 * 1.2 ** 2))
    img = 0.1 + np.stack([2 * g, 4 * g, 1 * g], axis=-1) * 0.05
    flux, err, n = C.aperture_photometry(img.astype(np.float32), 12.3, 11.8, 4, 7, 11)
    assert flux[1] / flux[0] == pytest.approx(2.0, rel=0.02) and flux[1] / flux[2] == pytest.approx(4.0, rel=0.02)
    assert n > 40


def test_read_rgb_and_monochrome(synthetic_color_video, synthetic_video):
    meta = probe_video(synthetic_color_video)
    cc = {"ffmpeg_hwaccel": "off"}
    frames = list(C.read_rgb(synthetic_color_video, meta, 10, 3, (0, 0, SYN_W, SYN_H), cc))
    assert [f for f, _ in frames] == [10, 11, 12]
    _, rgb = frames[0]
    assert rgb.shape == (SYN_H, SYN_W, 3)
    x, y = color_dot_position(10)
    px = rgb[y - 1, x - 1]
    assert px[1] > px[0] + 0.2 and px[1] > px[2] + 0.2          # zielona kropka
    bg = rgb[5, 90]
    assert abs(bg[0] - bg[1]) < 0.03 and abs(bg[2] - bg[1]) < 0.03
    assert not C.is_monochrome(rgb)
    # wycinek z parzystym offsetem
    _, sub = next(iter(C.read_rgb(synthetic_color_video, meta, 10, 1, (x - 6, y - 6, 12, 12), cc)))
    assert np.allclose(sub, rgb[y - 6:y + 6, x - 6:x + 6], atol=0.02)
    meta_bw = probe_video(synthetic_video)
    _, gray = next(iter(C.read_rgb(synthetic_video, meta_bw, 5, 1, (0, 0, SYN_W, SYN_H), cc)))
    assert C.is_monochrome(gray)


def test_measure_track_color_green_dot(synthetic_color_video, cfg):
    meta = probe_video(synthetic_color_video)
    cc = {**cfg["color"], "ffmpeg_hwaccel": "off"}
    frames = np.arange(10, 31)
    p = np.array([color_dot_position(i) for i in frames], float) - 0.5   # środek kropki 4×4
    points, thumbs = C.measure_track_color(synthetic_color_video, meta, frames, p[:, 0], p[:, 1], 0.0, cc)
    assert len(points) >= 15 and not any(q["saturated"] for q in points)
    cal = C.ColorCalib(0.0, 0.5, 0.0, -0.8, 0.02, 30)
    s = C.summarize(points, cal, cc)
    assert s["n_color"] >= 15
    assert s["r_g"] < -0.3 and s["b_g"] < -0.3                   # zielony: mało R i B względem G
    assert s["green_excess"] > 0.2
    # gwiazda obok toru nie zaburza koloru (odjęcie tła z klatek f±k): rozrzut mały
    assert s["rg_spread"] < 0.1
    assert len(thumbs) >= 1 and next(iter(thumbs.values())).dtype == np.uint8


def test_color_hints(cfg):
    cc = cfg["color"]
    base = {"n_color": 10, "n_frames": 12, "n_saturated": 2, "r_g": -0.1, "b_g": -0.3, "bv_eq": 0.6,
            "T_eq_K": 5900.0, "green_excess": 0.0, "slope_dex_s": 0.0, "chi2": 1.0, "rg_spread": 0.02}
    h = lambda s, **kw: C.color_hint({**base, **s}, **{"kind": "unid", "class_hint": "meteor?", "dur_s": 0.5,  # noqa: E731
                                                        "sunlit_ref": None, "ccfg": cc, **kw})[0]
    assert h({"green_excess": 0.12}).startswith("zielony nadmiar → Mg")
    assert h({"T_eq_K": 3000.0}).startswith("pomarańczowy")
    assert h({"T_eq_K": 9500.0}).startswith("biało-niebieski")
    ref = {"r_g": -0.1, "b_g": -0.3}
    assert h({}, class_hint="satelita?", sunlit_ref=ref) == "oświetlony Słońcem (jak satelity)"
    assert h({"T_eq_K": 2500.0, "r_g": 0.3}, class_hint="bliski obiekt?", sunlit_ref=ref).startswith("ciepły")
    nav = {"rg_spread": 0.4, "chi2": 20.0, "n_color": 40}
    assert h(nav, class_hint="samolot?", blinking=True).startswith("światła nawigacyjne")
    # bez migania rozrzut koloru ciepłego, wolnego obiektu (ptak w łunie) to nie światła samolotu
    assert h({**nav, "T_eq_K": 2800.0, "r_g": 0.3}, class_hint="bliski obiekt?").startswith("ciepły")
    assert h({}, kind="sat") == "Słońce odbite (odniesienie)"
    assert h({"n_color": 0}) == "brak koloru"
    _, why = C.color_hint({**base, "slope_dex_s": 0.5, "chi2": 10.0}, kind="unid", class_hint="meteor?", dur_s=0.8,
                          sunlit_ref=None, ccfg=cc)
    assert "zmiana koloru" in why


def test_crop_box_even_and_inside():
    x0, y0, w, h = C.crop_box([101.3, 150.0], [51.0, 60.2], 10, 3840, 2160)
    assert x0 % 2 == 0 and y0 % 2 == 0 and w % 2 == 0 and h % 2 == 0
    assert x0 <= 91.3 and x0 + w >= 160 and y0 <= 41 and y0 + h >= 70
    x0, y0, w, h = C.crop_box([3835.0], [2158.0], 10, 3840, 2160)
    assert x0 + w <= 3840 and y0 + h <= 2160
    assert math.isfinite(float(w))
