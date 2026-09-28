from datetime import datetime, timezone

import pytest

from skyhunt.metadata import VideoMeta
from skyhunt.timing import camera_start, camera_to_utc, time_prior


def meta(**kw) -> VideoMeta:
    base = dict(path="x.MOV", size_bytes=1, codec="avc1", width=3840, height=2160, fps_num=24000,
                fps_den=1001, n_frames=7680, duration_s=320.32,
                mvhd_creation="2026-09-27T20:41:53", maker_datetime="2026-09-27T20:36:08")
    base.update(kw)
    return VideoMeta(**base)


TCFG = {"camera_tz": "Europe/Warsaw", "camera_clock_ahead_s": 600, "start_source": "maker_datetime",
        "prior_sigma_s": 60}


def test_dscf4641_prior():
    # zegar +10 min, CEST (UTC+2): 20:36:08 → 20:26:08 lokalnie → 18:26:08 UTC
    p = time_prior(meta(), TCFG)
    assert p.start_utc == "2026-09-27T18:26:08+00:00"
    assert p.source == "maker_datetime" and p.sigma_s == 60
    # mvhd to ~24 s po końcu nagrania — kandydat widoczny do porównania
    assert p.candidates_utc["mvhd_creation_minus_duration"].startswith("2026-09-27T18:26:32")


def test_winter_time_offset():
    t = camera_to_utc(datetime(2026, 12, 1, 20, 0, 0), "Europe/Warsaw", 0)
    assert t == datetime(2026, 12, 1, 19, 0, 0, tzinfo=timezone.utc)


def test_fallback_source_inflates_sigma():
    p = time_prior(meta(maker_datetime=None), TCFG)
    assert p.source == "mvhd_creation" and p.sigma_s == 180


def test_no_timestamp():
    with pytest.raises(ValueError):
        time_prior(meta(maker_datetime=None, mvhd_creation=None), TCFG)


def test_camera_start_sources():
    m = meta()
    assert camera_start(m, "mvhd_creation_minus_duration") == datetime(2026, 9, 27, 20, 36, 32, 680000)
    assert camera_start(m, "nieznane") is None
