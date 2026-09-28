"""Format IOD: kodowanie MX, RA/Dec (format 1), kolumny rekordu, wybór torów do eksportu."""
from datetime import datetime, timezone

import pytest

from skyhunt.iod import export, mx, object_field, radec_format1, record


@pytest.mark.parametrize("value, code", [(0.001, "15"), (0.05, "56"), (0.1, "17"), (0.14, "27"), (1.0, "18"),
                                         (20.0, "29"), (90.0, "99")])
def test_mx(value, code):
    assert mx(value) == code


def test_radec_and_object_fields():
    assert radec_format1(310.357979, 45.280339) == "2041259+451649"          # Deneb
    assert radec_format1(0.0, -0.5) == "0000000-003000"
    assert object_field(47391, "2021-005AU", "99999 00 000UNK") == "47391 21 005AU "
    assert object_field(None, None, "99999 00 000UNK") == "99999 00 000UNK"
    assert object_field(123456, "2026-151G", "99999 00 000UNK") == "99999 00 000UNK"   # > 5 cyfr nie mieści się


def test_record_columns():
    t = datetime(2026, 9, 27, 18, 27, 19, 270000, tzinfo=timezone.utc)
    r = record("47391 21 005AU ", 9999, "G", t, 0.14, 310.357979, 45.280339, 12.3, "R")
    assert r == "47391 21 005AU  9999 G 20260927182719270 27 15 2041259+451649 29 R"
    assert r[16:20] == "9999" and r[21] == "G" and r[23:40] == "20260927182719270"   # kolumny 17-20, 22, 24-40
    assert r[44:46] == "15" and r[47:61] == "2041259+451649" and r[62:64] == "29"    # 45-46, 48-61, 63-64


def test_export_selects_tracks(tmp_path, cfg):
    pd = pytest.importorskip("pandas")
    final = pd.DataFrame([
        {"track_id": 1, "kind": "sat", "norad": 47391.0, "n": 140, "curv_arcsec": 5.0, "peak_snr_median": 200.0,
         "f_power": 3.0, "tau0": 44.0},
        {"track_id": 2, "kind": "unid", "norad": None, "n": 80, "curv_arcsec": 30.0, "peak_snr_median": 25.0,
         "f_power": 1.0, "tau0": 0.4},
        {"track_id": 3, "kind": "unid", "norad": None, "n": 6, "curv_arcsec": 90.0, "peak_snr_median": 5.5,
         "f_power": 1.0, "tau0": 10.0},     # szum: za krótki
    ])
    pts = pd.DataFrame([{"track_id": tid, "frame": f, "tau": f / 24.0, "ra": 310.0 + f * 0.01, "dec": 45.0}
                        for tid in (1, 2, 3) for f in range(10)])
    sync = {"start_utc_prior": "2026-09-27T18:26:08+00:00", "delta_s": 26.85, "sigma_s": 0.14}
    wcsinfo = {"fov": {"scale_arcsec_px": 24.7}, "epoch_rms_px": 0.44}
    ids = {"1": [{"object_id": "2021-005AU"}]}
    path, n = export(tmp_path, final, pts, sync, wcsinfo, ids, cfg["iod"], cfg["classify"])
    lines = path.read_text(encoding="ascii").splitlines()
    assert n == 6 and len(lines) == 6
    assert all(ln.startswith("99999 00 000UNK") for ln in lines[:3])       # tor 2 (wcześniejszy) pierwszy
    assert all(ln.startswith("47391 21 005AU ") for ln in lines[3:])
    assert lines[0][23:40] == "20260927182634850"                          # t0 + Δ + τ(klatka 0)
