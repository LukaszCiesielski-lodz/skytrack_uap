"""Raport PDF na syntetycznych wynikach etapów: satelita (zielony, NORAD, załącznik MP4) i obiekt
niezidentyfikowany (czerwony, °/s, UTC), strona zbiorcza."""
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pypdf")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("astropy")

from skyhunt.clips import filmstrip, plan_clip  # noqa: E402
from skyhunt.metadata import probe_video  # noqa: E402
from skyhunt.report import build  # noqa: E402
from skyhunt.sky import _world2pix, resolve_star, stars  # noqa: E402
from tests.conftest import SYN_N, dot_position  # noqa: E402
from tests.test_sky import tan_wcs  # noqa: E402


def track_row(tid, kind, n, hint="satelita?"):
    return {"track_id": tid, "kind": kind, "n": n, "tau0": 0.02, "tau1": 1.645, "tau_mid": 0.83, "dur_s": 1.625,
            "omega_deg_s": 0.912, "ra0": 310.1, "dec0": 45.2, "ra1": 311.0, "dec1": 45.9, "az0": 120.0, "alt0": 80.0,
            "az1": 121.0, "alt1": 81.0, "curv_arcsec": 12.0, "starts_inside": True, "ends_inside": False,
            "class_hint": hint, "class_reason": "prosty", "cross_ratio": 1.1, "f_peak_hz": float("nan"),
            "f_alias_hz": float("nan"), "f_power": 0.0}


@pytest.fixture
def outdir(tmp_path, synthetic_video):
    from astropy.io import fits

    meta = probe_video(synthetic_video)
    out = tmp_path / "out" / synthetic_video.stem
    (out / "epochs").mkdir(parents=True)
    (out / "astrometry").mkdir()
    wcs = tan_wcs(*resolve_star("Deneb"), shape=(meta.height, meta.width), scale_arcsec=0.25 * 3600)
    img = np.random.default_rng(0).normal(100, 3, (meta.height, meta.width)).astype(np.float32)
    s = stars()
    x, y = _world2pix(wcs, s["ra"], s["dec"])
    ok = (x >= 0) & (x < meta.width - 1) & (y >= 0) & (y < meta.height - 1) & (s["mag"] < 4)
    img[np.round(y[ok]).astype(int), np.round(x[ok]).astype(int)] += 400
    fits.writeto(out / "epochs/epoch_000020.fits", img)
    fits.PrimaryHDU(header=wcs.to_header()).writeto(out / "astrometry/epoch_000020.wcs")
    (out / "wcs.json").write_text(json.dumps({
        "epochs": [{"file": "epochs/epoch_000020.fits", "frame": 20, "tau_s": 0.83, "solved": True,
                    "wcs": "astrometry/epoch_000020.wcs"}],
        "n_epochs": 1, "n_solved": 1, "reference_tau_s": 0.83, "epoch_rms_px": 0.0,
        "fov": {"fov_w_deg": 24.0, "fov_h_deg": 16.0, "scale_arcsec_px": 900.0}, "crop": {"verdict": "bez cropu"}}))

    frames = np.arange(SYN_N)
    p1 = np.array([dot_position(i) for i in frames], float)
    p2 = np.column_stack([85 - 1.5 * frames, 10 + 1.0 * frames])
    rows = []
    for tid, p in ((1, p1), (2, p2)):
        ra, dec = wcs.all_pix2world(p[:, 0], p[:, 1], 0)
        rows.append(pd.DataFrame({"track_id": tid, "frame": frames, "x": p[:, 0], "y": p[:, 1],
                                  "tau": frames / 24 + 0.02, "ra": ra, "dec": dec}))
    pd.concat(rows).to_parquet(out / "track_sky.parquet", index=False)
    pd.DataFrame([track_row(1, "sat", SYN_N), track_row(2, "unid", SYN_N, "bliski obiekt?")]) \
        .to_parquet(out / "tracks_final.parquet", index=False)
    ra1, dec1 = wcs.all_pix2world(p1[::5, 0] + 0.5, p1[::5, 1], 0)
    best = {"track_id": 1, "cat_index": 0, "norad": 25544, "name": "ISS (ZARYA)", "object_id": "1998-067A",
            "epoch": "2026-09-27T12:00:00+00:00", "source": "celestrak", "delta_s": -36.98, "rms_deg": 0.01,
            "cross_deg": 0.004, "along_arcsec": 9.0, "speed_ratio": 1.0, "dir_deg": 0.1, "range_km": 480.0,
            "height_km": 421.0, "sat_omega_deg_s": 0.91, "sunlit": True, "confidence": "high", "reason": "test",
            "ambiguous_with": [], "pred_radec": np.column_stack([ra1, dec1]).tolist()}
    (out / "identifications.json").write_text(json.dumps({"1": [best], "2": []}))
    (out / "time_sync.json").write_text(json.dumps({
        "synced": True, "delta_s": -37.0, "sigma_s": 0.05, "confidence": "high", "method": "test",
        "reference": best, "members": [best], "start_utc_prior": "2026-09-27T18:26:08+00:00",
        "catalog_note": "test"}))
    pd.DataFrame([{"norad": 44713, "name": "STARLINK-1007", "utc_in": "2026-09-27T18:26:10+00:00",
                   "utc_out": "2026-09-27T18:26:20+00:00", "range_km": 700.0, "sunlit": None, "detected": False}]) \
        .to_csv(out / "fov_predicted.csv", index=False)
    dark = tmp_path / "out" / "dark_frames"
    dark.mkdir()
    (dark / "darkstats.json").write_text(json.dumps({"file": "dark_frames.MOV", "tracks_per_hour": 2.0,
                                                     "tracks": 0, "duration_s": 320.0}))
    return out, meta


def _text(path):
    from pypdf import PdfReader

    r = PdfReader(str(path))
    return r, "".join(p.extract_text() or "" for p in r.pages)


def test_report_pdfs(outdir, synthetic_video, cfg):
    out, meta = outdir
    outputs = build(out, synthetic_video, meta, cfg)
    assert outputs[0] == "report/summary.pdf"
    assert "report/objects/sat_25544_t1.pdf" in outputs and "report/objects/unid_t2.pdf" in outputs
    assert all((out / o).exists() for o in outputs)

    r, text = _text(out / "report/objects/sat_25544_t1.pdf")
    assert len(r.pages) >= 2
    assert "25544" in text and "UTC" in text and "°/s" in text and "ISS" in text
    if "report/clips/t1.mp4" in outputs:
        assert "t1.mp4" in r.attachments

    r, text = _text(out / "report/objects/unid_t2.pdf")
    assert "niezidentyfikowany" in text and "°/s" in text and "Początek" in text and "Koniec" in text
    assert "18:25:31" in text          # 18:26:08 − 37 s + 0.02 s

    r, text = _text(out / "report/summary.pdf")
    assert len(r.pages) >= 5 and "Δ" in text and "dark_frames.MOV" in text


def test_report_color_page(outdir, synthetic_video, cfg):
    out, meta = outdir
    (out / "color_calib.json").write_text(json.dumps({
        "enabled": True, "monochrome": False, "calibrated": True, "ar": 0.1, "br": 0.4, "ab": -0.05, "bb": -0.6,
        "rms_dex": 0.03, "n_stars": 40, "n_candidates": 52, "linearity_slope": -0.38, "wb_drift_dex": 0.01,
        "sunlit_ref": {"r_g": 0.35, "b_g": -0.45, "n": 5, "bv_eq": 0.65, "green_excess": 0.0, "T_eq_K": 5780.0}}))
    bv = np.linspace(0, 1.5, 20)
    pd.DataFrame({"bv": bv, "r_g": 0.1 + 0.4 * bv, "b_g": -0.05 - 0.6 * bv, "used": True, "mag": 4.0}) \
        .to_csv(out / "color_stars.csv", index=False)
    summ = {"track_id": 2, "n_frames": 30, "n_saturated": 3, "n_color": 25, "r_g": 0.1, "b_g": -0.4, "e_r_g": 0.01,
            "e_b_g": 0.01, "bv_eq": 0.4, "T_eq_K": 6700.0, "green_excess": 0.12, "slope_dex_s": 0.0, "chi2": 1.0,
            "rg_spread": 0.02, "color_hint": "zielony nadmiar → Mg / O 557,7 nm?", "color_reason": "test"}
    pd.DataFrame([summ]).to_csv(out / "track_color.csv", index=False)
    f = np.arange(30)
    pd.DataFrame({"track_id": 2, "frame": f, "t_s": f / 24, "r_g": 0.1, "b_g": -0.4, "e_rg": 0.01, "e_bg": 0.01,
                  "snr": 20.0, "saturated": f < 3}).to_parquet(out / "track_color_points.parquet", index=False)
    np.savez_compressed(out / "track_color_thumbs.npz", t2_f5=np.full((64, 64, 3), 90, np.uint8),
                        t2_f20=np.full((64, 64, 3), 120, np.uint8))
    build(out, synthetic_video, meta, cfg)
    r, text = _text(out / "report/objects/unid_t2.pdf")
    from skyhunt.color import bv_to_kelvin

    # tabela parametrów pokazuje T z B−V (kelvin_text), nie T_eq_K z pliku
    assert len(r.pages) >= 3 and "Nadmiar zieleni" in text and f"{float(bv_to_kelvin(0.4)):.0f} K" in text \
        and "dyfrakcyjnej" in text
    r, text = _text(out / "report/summary.pdf")
    assert "kalibracja na gwiazdach" in text and "Słońce odbite" in text


def test_filmstrip_spans_one_second(synthetic_video, cfg):
    from skyhunt.clips import extract_frames

    meta = probe_video(synthetic_video)
    frames = np.arange(10, 14)            # tor krótszy niż 1 s
    p = np.array([dot_position(i) for i in frames], float)
    plan = plan_clip(frames, p[:, 0], p[:, 1], meta, cfg["report"])
    assert plan.n >= 24
    fr = extract_frames(synthetic_video, meta, plan)
    assert len(fr) == plan.n
    panels = filmstrip(fr, plan, frames, p[:, 0], p[:, 1], meta.fps, cfg["report"])
    assert (panels[-1]["frame"] - panels[0]["frame"] + 1) >= 24
    assert all(pn["img"].shape == (160, 160) for pn in panels)
