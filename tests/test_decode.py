"""Backendy dekodowania: poprawność Y (pozycja kropki), zakresy klatek, parytet między backendami."""
import shutil

import numpy as np
import pytest

from skyhunt.decode import BACKENDS, PyAVDecoder, cuda_available, make_decoder, open_decoder
from skyhunt.metadata import probe_video
from tests.conftest import SYN_N, dot_position

DCFG_CPU = {"backends": ["pyav", "ffmpeg"], "device": "cpu", "batch_frames": 16, "queue_depth": 2}


def decode_all(dec, bs=16, **kw) -> tuple[np.ndarray, list[int]]:
    ys, starts = [], []
    for b in dec.batches(bs, **kw):
        y = b.y.cpu().numpy() if hasattr(b.y, "cpu") else np.asarray(b.y)
        ys.append(y.copy())
        starts.append(b.start)
    return np.concatenate(ys), starts


@pytest.fixture(scope="module")
def pyav_frames(synthetic_video):
    meta = probe_video(synthetic_video)
    y, _ = decode_all(make_decoder("pyav", synthetic_video, meta, DCFG_CPU))
    return meta, y


def test_pyav_luma_correct(pyav_frames):
    meta, y = pyav_frames
    assert y.shape == (SYN_N, meta.height, meta.width) and y.dtype == np.uint8
    for i in range(SYN_N):
        yy, xx = np.unravel_index(np.argmax(y[i]), y[i].shape)
        x0, y0 = dot_position(i)
        assert abs(int(xx) - x0) <= 1 and abs(int(yy) - y0) <= 1, f"klatka {i}"
    assert abs(float(np.median(y)) - 40) <= 3   # tło


def test_range_selection(synthetic_video, pyav_frames):
    meta, full = pyav_frames
    dec = make_decoder("pyav", synthetic_video, meta, DCFG_CPU)
    y, starts = decode_all(dec, bs=8, start=5, stop=23)
    assert starts[0] == 5 and len(y) == 18
    np.testing.assert_array_equal(y, full[5:23])


def test_pict_types_from_pyav(synthetic_video):
    meta = probe_video(synthetic_video)
    dec = make_decoder("pyav", synthetic_video, meta, DCFG_CPU)
    picts = [p for b in dec.batches(16) for p in b.pict_types]
    assert len(picts) == SYN_N and picts[0] == "I"


def test_early_close_stops_producer(synthetic_video):
    meta = probe_video(synthetic_video)
    dec = make_decoder("pyav", synthetic_video, meta, {**DCFG_CPU, "queue_depth": 1})
    it = dec.batches(2)
    next(it)
    it.close()   # nie może zawisnąć


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="brak ffmpeg")
def test_ffmpeg_matches_pyav(synthetic_video, pyav_frames):
    meta, ref = pyav_frames
    y, _ = decode_all(make_decoder("ffmpeg", synthetic_video, meta, DCFG_CPU))
    np.testing.assert_array_equal(y, ref)


class _Broken(PyAVDecoder):
    name = "broken"

    def _check(self):
        raise RuntimeError("celowo niedostępny")


def test_open_decoder_fallback(synthetic_video, monkeypatch):
    monkeypatch.setitem(BACKENDS, "broken", _Broken)
    meta = probe_video(synthetic_video)
    dec = open_decoder(synthetic_video, meta, {**DCFG_CPU, "backends": ["torchcodec", "broken", "pyav"]})
    assert dec.name == "pyav"
    assert "pominięty" in dec.skipped["torchcodec"] and "celowo" in dec.skipped["broken"]


def test_open_decoder_all_fail(synthetic_video):
    meta = probe_video(synthetic_video)
    with pytest.raises(RuntimeError):
        open_decoder(synthetic_video, meta, {**DCFG_CPU, "backends": ["torchcodec"]})


def _gpu_backend_or_skip(name, video, meta):
    if not cuda_available():
        pytest.skip("brak CUDA")
    if meta.codec not in ("avc1", "avc3"):
        pytest.skip("test GPU tylko dla H.264")
    try:
        return make_decoder(name, video, meta, {"device": "cuda"})
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{name} niedostępny: {e}")


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["nvcodec", "torchaudio", "torchcodec"])
def test_gpu_backends_match_pyav(name, synthetic_video, pyav_frames):
    meta, ref = pyav_frames
    dec = _gpu_backend_or_skip(name, synthetic_video, meta)
    y, _ = decode_all(dec)
    assert y.shape == ref.shape
    tol = 0 if BACKENDS[name].exact_luma else 2
    assert int(np.abs(y.astype(np.int16) - ref.astype(np.int16)).max()) <= tol


@pytest.mark.gpu
def test_pyav_pinned_transfer_to_gpu(synthetic_video, pyav_frames):
    if not cuda_available():
        pytest.skip("brak CUDA")
    meta, ref = pyav_frames
    dec = make_decoder("pyav", synthetic_video, meta, {**DCFG_CPU, "device": "cuda"})
    assert dec.device.startswith("cuda")
    batches = list(dec.batches(16))
    assert all(b.y.is_cuda for b in batches)
    np.testing.assert_array_equal(np.concatenate([b.y.cpu().numpy() for b in batches]), ref)


@pytest.mark.video
@pytest.mark.parametrize("name", ["nvcodec", "torchaudio", "ffmpeg"])
def test_real_video_parity(name, real_video):
    """Pierwsze 48 klatek prawdziwego nagrania: backendy dokładne muszą być bit w bit jak PyAV."""
    pytest.importorskip("av")
    meta = probe_video(real_video)
    ref, _ = decode_all(make_decoder("pyav", real_video, meta, DCFG_CPU), stop=48)
    try:
        dec = make_decoder(name, real_video, meta, {**DCFG_CPU, "device": "cuda"})
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{name} niedostępny: {e}")
    y, _ = decode_all(dec, stop=48)
    np.testing.assert_array_equal(y, ref)
