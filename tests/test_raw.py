"""Zdjęcia RAW: EXIF z JPEG-a w RAF i MakerNote Fuji (sztuczny plik), superpiksele X-Trans,
serie bracketingu i rytm interwałometru z pełnych sekund EXIF (noniusz)."""
import io
import struct

import numpy as np
import pytest

from skyhunt import photo_timing as PT
from skyhunt import raw as R


def makernote(**vals) -> bytes:
    """MakerNote Fuji: FUJIFILM + offset IFD (12) + wpisy little-endian."""
    tags = {"sequence_number": (0x1101, 3), "image_count": (0x1438, 4), "auto_bracketing": (0x1100, 3),
            "shutter_type": (0x1050, 3)}
    entries = [(0x1000, 2, 4, b"FINE")] + [(tags[k][0], tags[k][1], 1, v) for k, v in vals.items()]
    body = struct.pack("<H", len(entries))
    for tag, typ, cnt, v in entries:
        if typ == 2:
            body += struct.pack("<HHI", tag, typ, cnt) + v
        elif typ == 3:
            body += struct.pack("<HHIH", tag, typ, cnt, v) + b"\0\0"
        else:
            body += struct.pack("<HHII", tag, typ, cnt, v)
    return b"FUJIFILM" + struct.pack("<I", 12) + body + struct.pack("<I", 0)


def fake_raf(path, when="2026:10:02 21:00:01", exposure=(1, 2), seq=2, count=17261):
    """RAF: nagłówek z offsetem i długością JPEG-a (bajty 84–92) + JPEG z EXIF (Pillow)."""
    from PIL import Image, TiffImagePlugin

    ex = Image.Exif()
    ex[0x010F] = "FUJIFILM"
    ex[0x0110] = "X-E3"
    ex[0x8769] = {0x9003: when, 0x829A: TiffImagePlugin.IFDRational(*exposure), 0x8827: 800,
                  0x829D: TiffImagePlugin.IFDRational(1, 1),
                  0x927C: makernote(sequence_number=seq, image_count=count, auto_bracketing=1, shutter_type=1)}
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (40, 40, 40)).save(buf, "JPEG", exif=ex.tobytes())
    jpg = buf.getvalue()
    head = (R.RAF_MAGIC + b" 0201FF383501").ljust(84, b"\0") + struct.pack(">II", 148, len(jpg))
    path.write_bytes(head.ljust(148, b"\0") + jpg + b"\0" * 64)
    return path


def test_fuji_makernote():
    mn = makernote(sequence_number=3, image_count=17262, auto_bracketing=1, shutter_type=1)
    d = R.parse_fuji_makernote(mn)
    assert d == {"sequence_number": 3, "image_count": 17262, "auto_bracketing": 1, "shutter_type": 1}
    assert R.parse_fuji_makernote(b"garbage") == {}


def test_read_raf_exif(tmp_path):
    pytest.importorskip("PIL")
    p = fake_raf(tmp_path / "DSCF0001.RAF")
    e = R.read_raf_exif(p)
    assert e["datetime"] == "2026:10:02 21:00:01" and e["subsec"] is None
    assert e["exposure_s"] == pytest.approx(0.5) and e["iso"] == 800 and e["fnumber"] == pytest.approx(1.0)
    assert e["model"] == "X-E3" and e["sequence_number"] == 2 and e["image_count"] == 17261
    (tmp_path / "x.RAF").write_bytes(b"not a raf file" * 10)
    with pytest.raises(ValueError):
        R.read_raf_exif(tmp_path / "x.RAF")


def test_superpixels_xtrans():
    pat = R.xtrans_pattern()
    H, W = 12, 18
    colors = np.tile(pat, (H // 6, W // 6))
    signal = np.array([100.0, 200.0, 50.0])
    cfa = (1000 + signal[colors]).astype(np.uint16)
    cfa[4, 7] = 16383                                     # nasycony piksel w bloku (1, 2)
    frame = R.RawFrame(cfa, colors, np.array([1000.0, 1000.0, 1000.0]), 16383.0)
    lum, rgb, sat = R.superpixels(frame)
    assert lum.shape == (4, 6) and rgb.shape == (4, 6, 3)
    assert lum[0, 0] == pytest.approx(2 * 100 + 5 * 200 + 2 * 50)
    assert np.allclose(rgb[0, 0], signal)
    assert sat[1, 2] and sat.sum() == 1
    # każdy blok 3×3 wzoru: 2 R, 5 G, 2 B
    for by in range(2):
        for bx in range(2):
            b = pat[3 * by:3 * by + 3, 3 * bx:3 * bx + 3]
            assert [(b == c).sum() for c in range(3)] == [2, 5, 2]


def test_group_sets_and_classes():
    rows = [{"sequence_number": s} for s in (1, 2, 3, 1, 2, 3, 1, 2)]
    assert PT.group_sets(rows) == [0, 0, 0, 1, 1, 1, 2, 2]
    assert PT.group_sets([{}] * 7, 3) == [0, 0, 0, 1, 1, 1, 2]
    assert PT.ev_class([0.5, 1.0, 0.25]) == ["ev0", "ev+1", "ev-1"]


def test_cadence_vernier_recovers_phase():
    t0, p = 1000.37, 2.973
    k = np.arange(120)
    floors = np.floor(t0 + p * k)
    cad = PT.fit_cadence(floors)
    assert cad.regular and cad.n_sets == 120
    assert cad.t0 == pytest.approx(t0, abs=0.03) and cad.period == pytest.approx(p, abs=5e-4)
    assert cad.t0_halfwidth < 0.05
    bad = floors.copy()
    bad[60:] += 2                                          # zapchany bufor: przeskok rytmu
    assert not PT.fit_cadence(bad).regular
    one = PT.fit_cadence(np.array([5.0]))
    assert one.t0 == 5.5 and not one.regular


def test_open_times_within_set():
    cad = PT.Cadence(100.2, 3.0, 0.01, 0.001, True, 2)
    sets = [0, 0, 0, 1, 1, 1]
    exp = [0.5, 1.0, 0.25, 0.5, 1.0, 0.25]
    t = PT.open_times(sets, exp, np.floor(np.array([100.2, 100.8, 101.9, 103.2, 103.8, 104.9])), cad, 0.1)
    assert np.allclose(t, [100.2, 100.8, 101.9, 103.2, 103.8, 104.9])


def test_rhythm_integer_clock_with_slips():
    """Start serii na pełnej sekundzie zegara aparatu, czasem sekunda dłużej (zapis na kartę) —
    jak s3_deneb_0410: rytm „nieregularny” dla noniusza, ale czasy z EXIF + stała faza są dobre."""
    d = np.full(399, 4.0)
    d[[49, 59, 69]] += 1                                   # przeskoki przed seriami 50, 60, 70
    t = 1000.0 + np.r_[0, np.cumsum(d)]
    assert not PT.fit_cadence(t).regular
    rh = PT.rhythm(t)
    assert rh["mode"] == "integer_clock" and rh["step_s"] == 4 and rh["slip_sets"] == [50, 60, 70]
    assert "pełnej sekundzie" in PT.rhythm_text(rh, 11.0) and "potwierdzone" in PT.rhythm_text(rh, 11.0)
    assert "do potwierdzenia" in PT.rhythm_text(rh, None)

    floors = np.floor(1000.37 + 2.973 * np.arange(120))
    assert PT.rhythm(floors)["mode"] == "vernier"
    rng = np.random.default_rng(3)
    noisy = np.floor(1000.0 + np.r_[0, np.cumsum(rng.uniform(3.0, 6.0, 199))])
    assert PT.rhythm(noisy)["mode"] == "irregular"
    assert "NIEREGULARNY" in PT.rhythm_text(PT.rhythm(noisy))
