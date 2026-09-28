"""Astrometria: komenda solve-field, indeksy (atrapa pobierania), .corr, pole widzenia, zgodność
epok; opcjonalnie prawdziwy plate solve syntetycznego pola wokół Deneba (marker ``solver``)."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")

from skyhunt.astrometry import (corr_residuals, crop_verdict, ensure_index, epoch_agreement,  # noqa: E402
                                fov_from_wcs, solve_command, solve_epoch, solver_available, write_solver_config)
from skyhunt.sky import FixedCamera, resolve_star  # noqa: E402
from tests.test_sky import SHAPE, SITE, T0, tan_wcs  # noqa: E402


def test_solve_command_has_hint_and_scale(cfg, tmp_path):
    a = cfg["astrometry"]
    cmd = solve_command(a, tmp_path / "e.fits", tmp_path, "e", tmp_path / "s.cfg", resolve_star("Deneb"), 2)
    s = " ".join(cmd)
    assert "--ra 310.35" in s and "--dec 45.28" in s and "--radius 20" in s
    assert "--scale-units degwidth" in s and "--scale-low 12" in s and "--scale-high 32" in s
    assert "--tweak-order 3" in s and "--downsample 2" in s and cmd[-1].endswith("e.fits")
    assert "--no-remove-lines" in s and "--uniformize 0" in s   # bez pomocników w Pythonie (NumPy 2)
    cfgfile = write_solver_config(tmp_path / "idx", tmp_path / "s.cfg")
    assert f"add_path {tmp_path / 'idx'}" in cfgfile.read_text()


def test_solve_epoch_retries_without_hint(cfg, tmp_path, monkeypatch):
    import subprocess
    import types

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=255, stdout="", stderr="brak rozwiązania")

    monkeypatch.setattr(subprocess, "run", fake_run)
    a = cfg["astrometry"]
    assert solve_epoch(a, tmp_path / "e.fits", tmp_path, tmp_path / "s.cfg", resolve_star("Deneb")) is None
    assert ["--ra" in c for c in calls] == [True, True, False]   # downsample 2, 4 z podpowiedzią, potem bez


def test_ensure_index_checks_md5(tmp_path):
    good = b"FITS-A"
    sums = f"{hashlib.md5(good).hexdigest()}  index-4119.fits\n{'0' * 32}  index-4118.fits\n".encode()

    class Resp:
        def __init__(self, d):
            self.d, self.status = d, 200

        def read(self):
            return self.d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(url, timeout=None):
        return Resp(sums if url.endswith("md5sums.txt") else good)

    acfg = {"index_dir": str(tmp_path), "index_url": "https://x/4100", "index_series": [4119]}
    assert (ensure_index(acfg, opener) / "index-4119.fits").read_bytes() == good
    with pytest.raises(RuntimeError):
        ensure_index({**acfg, "index_series": [4118]}, opener)


def test_corr_residuals(tmp_path):
    from astropy.io import fits

    cols = [fits.Column(name=n, format="D", array=np.array(v, float)) for n, v in
            (("field_x", [0, 10]), ("field_y", [0, 10]), ("index_x", [3, 10]), ("index_y", [4, 10]))]
    p = tmp_path / "e.corr"
    fits.BinTableHDU.from_columns(cols).writeto(p)
    r = corr_residuals(p)
    assert r["n_matched"] == 2 and r["rms_px"] == pytest.approx(np.sqrt(25 / 2))


def test_fov_and_crop(cfg):
    fov = fov_from_wcs(tan_wcs(*resolve_star("Deneb")), SHAPE)
    half = np.radians(1919.5 * 24.75 / 3600)   # rzut gnomoniczny: kąt = 2·atan(pół szerokości w płaszczyźnie)
    assert fov["fov_w_deg"] == pytest.approx(2 * np.degrees(np.arctan(half)), abs=0.01)
    assert fov["scale_arcsec_px"] == pytest.approx(24.75, rel=0.01)
    assert crop_verdict(26.4, cfg["camera"])["verdict"] == "bez cropu"
    v = crop_verdict(26.4 / 1.18, cfg["camera"])
    assert v["verdict"].startswith("crop") and v["ratio"] == pytest.approx(1 / 1.18, rel=1e-3)


def _wcs_at(cam0, tau):
    """WCS epoki τ zgodny z obrotem nieba (dopasowany do punktów kamery z epoki 0)."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points

    gx, gy = np.meshgrid(np.linspace(0, SHAPE[1] - 1, 12), np.linspace(0, SHAPE[0] - 1, 8))
    ra, dec = cam0.icrs(gx.ravel(), gy.ravel(), tau)
    # punkt styczności = obrócony punkt styczności epoki 0 (inaczej TAN nie jest dokładny)
    x_t, y_t = cam0.wcs.wcs.crpix - 1
    tangent = SkyCoord(*cam0.icrs(x_t, y_t, tau), unit="deg")
    return fit_wcs_from_points((gx.ravel(), gy.ravel()), SkyCoord(ra, dec, unit="deg"), proj_point=tangent,
                               projection="TAN")


def test_epoch_agreement_detects_camera_motion():
    ra, dec = resolve_star("Deneb")
    cam0 = FixedCamera(tan_wcs(ra, dec), 0.0, T0, *SITE)
    cam1 = FixedCamera(_wcs_at(cam0, 60.0), 60.0, T0, *SITE)
    assert epoch_agreement([cam0, cam1], SHAPE, [0.0, 60.0]) < 0.1
    moved = tan_wcs(ra, dec)
    moved.wcs.crpix = [moved.wcs.crpix[0] + 5, moved.wcs.crpix[1]]
    cam_moved = FixedCamera(_wcs_at(FixedCamera(moved, 0.0, T0, *SITE), 60.0), 60.0, T0, *SITE)
    assert epoch_agreement([cam0, cam_moved], SHAPE, [0.0, 60.0]) == pytest.approx(5, abs=0.3)


@pytest.mark.solver
def test_real_solve_of_synthetic_field(cfg, tmp_path):
    """Syntetyczne pole gwiazd (katalog do mag 6) wokół Deneba → solve-field → środek i skala."""
    import os

    from astropy.io import fits

    from skyhunt.sky import _world2pix, stars

    acfg = dict(cfg["astrometry"])
    acfg["index_dir"] = os.environ.get("SKYHUNT_ASTROMETRY_INDEX", acfg["index_dir"])
    if not solver_available(acfg) or not all((Path(acfg["index_dir"]) / f"index-{n}.fits").exists()
                                             for n in acfg["index_series"]):
        pytest.skip("brak solve-field lub indeksów")
    ra0, dec0 = resolve_star("Deneb")
    truth = tan_wcs(ra0 + 3, dec0 - 2)
    s = stars()
    x, y = _world2pix(truth, s["ra"], s["dec"])
    ok = (x > 5) & (x < SHAPE[1] - 5) & (y > 5) & (y < SHAPE[0] - 5)
    img = np.random.default_rng(0).normal(100, 3, SHAPE).astype(np.float32)
    yy, xx = np.mgrid[-6:7, -6:7]
    for xi, yi, m in zip(x[ok], y[ok], s["mag"][ok]):
        cx, cy = int(round(xi)), int(round(yi))
        if 6 <= cx < SHAPE[1] - 6 and 6 <= cy < SHAPE[0] - 6:
            img[cy - 6:cy + 7, cx - 6:cx + 7] += 3e4 * 10 ** (-0.4 * m) * np.exp(
                -((xx + cx - xi) ** 2 + (yy + cy - yi) ** 2) / (2 * 1.5 ** 2))
    fits.writeto(tmp_path / "epoch.fits", img)
    solver_cfg = write_solver_config(Path(acfg["index_dir"]), tmp_path / "s.cfg")
    wcs_path = solve_epoch(acfg, tmp_path / "epoch.fits", tmp_path, solver_cfg, (ra0, dec0))
    assert wcs_path is not None
    from skyhunt.astrometry import load_wcs

    fov = fov_from_wcs(load_wcs(wcs_path), SHAPE)
    assert fov["center_ra_deg"] == pytest.approx(ra0 + 3, abs=0.05)
    assert fov["center_dec_deg"] == pytest.approx(dec0 - 2, abs=0.05)
    assert fov["scale_arcsec_px"] == pytest.approx(24.75, rel=0.01)
