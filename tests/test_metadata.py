"""Parser atomów MP4/MOV: pliki budowane bajt po bajcie + plik z PyAV + prawdziwe nagranie."""
from datetime import datetime

import pytest

from skyhunt.metadata import probe_video
from tests.conftest import SYN_GOP, SYN_H, SYN_N, SYN_W


def box(typ: bytes, payload: bytes, large: bool = False) -> bytes:
    if large:
        return (1).to_bytes(4, "big") + typ + (16 + len(payload)).to_bytes(8, "big") + payload
    return (8 + len(payload)).to_bytes(4, "big") + typ + payload


def u32(v: int) -> bytes:
    return v.to_bytes(4, "big")


def mac_seconds(dt: datetime) -> int:
    return int((dt - datetime(1904, 1, 1)).total_seconds())


def make_mov(*, created: datetime, n=7680, timescale=24000, delta=1001, gop=24, ctts_offsets=None,
             mvhd_v1=False, moov_last=False, exif="2026:09:27 20:36:08", codec=b"avc1",
             w=3840, h=2160, large_mdat=False) -> bytes:
    ct = mac_seconds(created)
    dur = n * delta
    if mvhd_v1:
        mvhd = box(b"mvhd", b"\x01\x00\x00\x00" + ct.to_bytes(8, "big") + ct.to_bytes(8, "big")
                   + u32(timescale) + dur.to_bytes(8, "big") + bytes(80))
    else:
        mvhd = box(b"mvhd", b"\x00" * 4 + u32(ct) + u32(ct) + u32(timescale) + u32(dur) + bytes(80))
    mdhd = box(b"mdhd", b"\x00" * 4 + u32(ct) + u32(ct) + u32(timescale) + u32(dur) + bytes(4))
    hdlr = box(b"hdlr", b"\x00" * 4 + b"\x00" * 4 + b"vide" + bytes(12) + b"Video\x00")
    entry_body = bytes(6) + (1).to_bytes(2, "big") + bytes(16) + w.to_bytes(2, "big") + h.to_bytes(2, "big") + bytes(50)
    stsd = box(b"stsd", b"\x00" * 4 + u32(1) + box(codec, entry_body))
    stts = box(b"stts", b"\x00" * 4 + u32(1) + u32(n) + u32(delta))
    keys = list(range(0, n, gop))
    stss = box(b"stss", b"\x00" * 4 + u32(len(keys)) + b"".join(u32(k + 1) for k in keys))
    stbl_children = stsd + stts + stss
    if ctts_offsets is not None:
        stbl_children += box(b"ctts", b"\x00" * 4 + u32(len(ctts_offsets))
                             + b"".join(u32(1) + u32(o) for o in ctts_offsets))
    stbl = box(b"stbl", stbl_children)
    minf = box(b"minf", box(b"vmhd", bytes(12)) + stbl)
    mdia = box(b"mdia", mdhd + hdlr + minf)
    tkhd = box(b"tkhd", bytes(84))
    # ścieżka audio przed wideo — parser musi ją pominąć
    audio = box(b"trak", box(b"mdia", box(b"hdlr", bytes(8) + b"soun" + bytes(12))))
    udta = box(b"udta", box(b"MVTG", b"FUJIFILM DIGITAL CAMERA X-E3\x00" + exif.encode() + b"\x00"))
    moov = box(b"moov", mvhd + audio + box(b"trak", tkhd + mdia) + udta)
    ftyp = box(b"ftyp", b"qt  " + u32(0) + b"qt  ")
    mdat = box(b"mdat", bytes(64), large=large_mdat)
    return ftyp + (mdat + moov if moov_last else moov + mdat)


def test_fuji_like_mov(tmp_path):
    p = tmp_path / "a.MOV"
    p.write_bytes(make_mov(created=datetime(2026, 9, 27, 20, 41, 53)))
    m = probe_video(p)
    assert (m.width, m.height, m.codec) == (3840, 2160, "avc1")
    assert (m.fps_num, m.fps_den) == (24000, 1001)
    assert m.n_frames == 7680
    assert m.duration_s == pytest.approx(320.32)
    assert m.keyframes[:3] == [0, 24, 48] and len(m.keyframes) == 320
    assert m.gop_length == 24
    assert m.has_bframes is False
    assert m.mvhd_creation == "2026-09-27T20:41:53"
    assert m.maker_datetime == "2026-09-27T20:36:08"
    assert (m.make, m.model) == ("FUJIFILM", "X-E3")


@pytest.mark.parametrize("kw", [dict(mvhd_v1=True), dict(moov_last=True), dict(large_mdat=True, moov_last=True)])
def test_layout_variants(tmp_path, kw):
    p = tmp_path / "b.mp4"
    p.write_bytes(make_mov(created=datetime(2026, 1, 2, 3, 4, 5), n=100, gop=10, **kw))
    m = probe_video(p)
    assert m.n_frames == 100 and m.keyframes == list(range(0, 100, 10))
    assert m.mvhd_creation == "2026-01-02T03:04:05"


def test_bframes_detected(tmp_path):
    p = tmp_path / "c.mp4"
    p.write_bytes(make_mov(created=datetime(2026, 1, 1), n=10, ctts_offsets=[0, 2002, 0]))
    assert probe_video(p).has_bframes is True
    p.write_bytes(make_mov(created=datetime(2026, 1, 1), n=10, ctts_offsets=[0, 0]))
    assert probe_video(p).has_bframes is False


def test_not_a_video(tmp_path):
    p = tmp_path / "x.mov"
    p.write_bytes(b"\x00\x00\x00\x10free" + bytes(8))
    with pytest.raises(ValueError):
        probe_video(p)


def test_pyav_written_file(synthetic_video):
    m = probe_video(synthetic_video)
    assert (m.width, m.height) == (SYN_W, SYN_H)
    assert m.n_frames == SYN_N
    assert m.fps == pytest.approx(24.0)
    assert m.keyframes[0] == 0 and all(k % SYN_GOP == 0 for k in m.keyframes)


@pytest.mark.video
def test_real_video(real_video):
    m = probe_video(real_video)
    assert m.width > 0 and m.height > 0 and m.n_frames > 0
    assert m.keyframes == sorted(m.keyframes) and m.keyframes[0] == 0
    assert m.n_frames == pytest.approx(m.duration_s * m.fps, abs=2)
