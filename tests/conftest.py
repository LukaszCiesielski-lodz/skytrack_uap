from __future__ import annotations

import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Syntetyczne wideo: tło 40 DN, jasna kropka 3×3 (220 DN) poruszająca się o (2, 0.5) px/klatkę.
SYN_W, SYN_H, SYN_N, SYN_GOP = 96, 64, 40, 10


def dot_position(i: int) -> tuple[int, int]:
    return 10 + 2 * i, 20 + i // 2   # (x, y)


def _encode(path: Path, codec: str) -> None:
    import av

    with av.open(str(path), mode="w") as c:
        c.metadata["creation_time"] = "2026-09-27T18:00:00.000000Z"   # → mvhd
        s = c.add_stream(codec, rate=24)
        s.width, s.height, s.pix_fmt = SYN_W, SYN_H, "yuv420p"
        if codec == "libx264":
            s.options = {"crf": "10", "x264-params": f"keyint={SYN_GOP}:min-keyint={SYN_GOP}:bframes=0:scenecut=0"}
        else:
            s.codec_context.gop_size = SYN_GOP
            s.bit_rate = 4_000_000
        for i in range(SYN_N):
            yuv = np.full((SYN_H * 3 // 2, SYN_W), 128, np.uint8)
            yuv[:SYN_H] = 40
            x, y = dot_position(i)
            yuv[y - 1:y + 2, x - 1:x + 2] = 220
            fr = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
            fr.pts, fr.time_base = i, Fraction(1, 24)
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


# Kolorowe wideo: zielona kropka 4×4 (Y, Cb, Cr) = (150, 80, 70) — RGB ≈ (63, 222, 59), bez
# prześwietlenia — na parzystych współrzędnych (chroma 4:2:0) i nieruchoma niebieskawa „gwiazda”
# obok toru (ma zniknąć przy odjęciu tła).
GREEN_YUV = (150, 80, 70)
STAR_XY, STAR_YUV = (40, 34), (150, 170, 100)


def color_dot_position(i: int) -> tuple[int, int]:
    return 8 + 2 * i, 20 + 2 * (i // 4)


def _put(yp, up, vp, x, y, yuv):
    yp[y - 2:y + 2, x - 2:x + 2] = yuv[0]
    up[y // 2 - 1:y // 2 + 1, x // 2 - 1:x // 2 + 1] = yuv[1]
    vp[y // 2 - 1:y // 2 + 1, x // 2 - 1:x // 2 + 1] = yuv[2]


def _encode_color(path: Path, codec: str) -> None:
    import av

    with av.open(str(path), mode="w") as c:
        c.metadata["creation_time"] = "2026-09-28T18:25:00.000000Z"
        s = c.add_stream(codec, rate=24)
        s.width, s.height, s.pix_fmt = SYN_W, SYN_H, "yuv420p"
        if codec == "libx264":
            s.options = {"crf": "4", "x264-params": f"keyint={SYN_GOP}:min-keyint={SYN_GOP}:bframes=0:scenecut=0"}
        else:
            s.codec_context.gop_size = SYN_GOP
            s.bit_rate = 8_000_000
        for i in range(SYN_N):
            yp = np.full((SYN_H, SYN_W), 40, np.uint8)
            up = np.full((SYN_H // 2, SYN_W // 2), 128, np.uint8)
            vp = up.copy()
            _put(yp, up, vp, *STAR_XY, STAR_YUV)
            _put(yp, up, vp, *color_dot_position(i), GREEN_YUV)
            yuv = np.concatenate([yp, up.reshape(SYN_H // 4, SYN_W), vp.reshape(SYN_H // 4, SYN_W)])
            fr = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
            fr.pts, fr.time_base = i, Fraction(1, 24)
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture(scope="session")
def synthetic_color_video(tmp_path_factory) -> Path:
    av = pytest.importorskip("av")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    path = tmp_path_factory.mktemp("video_color") / f"synthetic_color_{codec}.mp4"
    _encode_color(path, codec)
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """MP4 H.264 (jeśli PyAV ma libx264) albo MPEG-4 Part 2."""
    av = pytest.importorskip("av")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    path = tmp_path_factory.mktemp("video") / f"synthetic_{codec}.mp4"
    _encode(path, codec)
    return path


@pytest.fixture(scope="session")
def real_video() -> Path:
    p = os.environ.get("SKYHUNT_TEST_VIDEO")
    if not p or not Path(p).is_file():
        pytest.skip("ustaw $SKYHUNT_TEST_VIDEO na prawdziwe nagranie")
    return Path(p)


@pytest.fixture
def cfg():
    from skyhunt.config import load_config

    return load_config(ROOT / "config.yaml")
