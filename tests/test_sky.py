"""Konstelacje, gwiazdy, model kamery nieruchomej, kierunki pozorne, wielkie koło."""
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("astropy")

from skyhunt.satellites import SIDEREAL_DEG_S  # noqa: E402
from skyhunt.sky import (FixedCamera, _world2pix, angle_deg, constellation_overlay, densify,  # noqa: E402
                         fit_great_circle, named_stars_overlay, radec_to_vec, resolve_star, vec_to_radec)

SHAPE = (2160, 3840)
T0 = datetime(2026, 9, 27, 18, 26, 8, tzinfo=timezone.utc)
SITE = (51.718042, 19.582748, 210.0)


def tan_wcs(ra, dec, shape=SHAPE, scale_arcsec=24.75):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    w.wcs.radesys = "ICRS"
    return w


def test_resolve_deneb():
    ra, dec = resolve_star("Deneb")
    assert ra == pytest.approx(310.358, abs=0.01) and dec == pytest.approx(45.280, abs=0.01)
    assert resolve_star("HIP 102098") == (ra, dec)
    with pytest.raises(KeyError):
        resolve_star("Nibiru")


def test_cygnus_in_frame_centered_on_deneb():
    ra, dec = resolve_star("Deneb")
    w = tan_wcs(ra, dec)
    x, y = _world2pix(w, np.array([ra]), np.array([dec]))
    assert x[0] == pytest.approx(SHAPE[1] / 2 - 0.5, abs=0.01) and y[0] == pytest.approx(SHAPE[0] / 2 - 0.5, abs=0.01)
    over = {c["id"]: c for c in constellation_overlay(w, SHAPE)}
    assert "Cyg" in over and over["Cyg"]["name"] == "Cygnus"
    assert sum(len(p) for p in over["Cyg"]["pieces"]) >= 10
    names = {s["name"] for s in named_stars_overlay(w, SHAPE, 2.0, ["Deneb"])}
    assert "Deneb" in names


def test_densify_on_great_circle():
    pts = densify(np.array([[0.0, 0.0], [10.0, 0.0]]), 0.5)
    assert len(pts) == 21 and np.allclose(pts[:, 1], 0, atol=1e-9)
    wrap = densify(np.array([[359.0, 10.0], [1.0, 10.0]]), 0.5)
    assert (np.abs(((wrap[:, 0] + 180) % 360) - 180) <= 1.0 + 1e-9).all()   # przez RA = 0, nie dookoła


def test_fixed_camera_follows_sky_rotation():
    ra, dec = resolve_star("Deneb")
    cam = FixedCamera(tan_wcs(ra, dec), 0.0, T0, *SITE)
    ra0, dec0 = cam.icrs(1920, 1080, 0.0)
    ra1, dec1 = cam.icrs(1920, 1080, 100.0)
    d_ra = ((ra1 - ra0 + 180) % 360) - 180
    assert d_ra == pytest.approx(100 * SIDEREAL_DEG_S, abs=0.01)
    assert abs(dec1 - dec0) * 3600 < 15          # obrót wokół bieguna daty (≈0.36° od bieguna ICRS)
    x, y = cam.pixel(ra1, dec1, 100.0)
    assert float(x) == pytest.approx(1920, abs=0.02) and float(y) == pytest.approx(1080, abs=0.02)
    # błąd zegara odniesienia się znosi
    cam2 = FixedCamera(tan_wcs(ra, dec), 0.0, T0.replace(minute=27), *SITE)
    ra2, dec2 = cam2.icrs(1920, 1080, 100.0)
    assert angle_deg(radec_to_vec(ra1, dec1), radec_to_vec(ra2, dec2)) * 3600 < 1.0


def test_apparent_vectors_aberration():
    ra, dec = resolve_star("Deneb")
    cam = FixedCamera(tan_wcs(ra, dec), 0.0, T0, *SITE)
    v = cam.apparent_vectors(np.array([ra]), np.array([dec]), np.array([0.0]))
    sep = float(angle_deg(v[0], radec_to_vec(ra, dec))) * 3600
    assert 1.0 < sep < 21.5


def test_altaz_near_zenith_for_deneb():
    ra, dec = resolve_star("Deneb")
    cam = FixedCamera(tan_wcs(ra, dec), 0.0, T0, *SITE)
    az, alt = cam.altaz(1919.5, 1079.5)
    assert 78 < float(alt) < 86     # Deneb ~0.5 h przed górowaniem, blisko zenitu


def test_great_circle_fit():
    pole = radec_to_vec(40.0, 30.0)
    e1 = np.cross(pole, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(pole, e1)
    t = np.linspace(-2, 2, 49)
    phi = np.radians(0.9) * t
    pts = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    gc = fit_great_circle(pts, t)
    assert gc.omega_deg_s == pytest.approx(0.9, rel=1e-6) and gc.cross_rms_arcsec < 1e-6
    back = fit_great_circle(pts[::-1], t)    # ruch w drugą stronę → ta sama prędkość dodatnia
    assert back.omega_deg_s == pytest.approx(0.9, rel=1e-6)
    assert angle_deg(back.position(0.0), pts[24]) < 1e-6
    ra, dec = vec_to_radec(gc.pole)
    assert angle_deg(radec_to_vec(ra, dec), gc.pole) < 1e-9
