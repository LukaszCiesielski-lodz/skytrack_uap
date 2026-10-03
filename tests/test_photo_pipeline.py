"""Sesje zdjęć RAW w pipeline: wejście (folder = sesja), odcisk manifestu, osobny rejestr etapów
(wideo bez zmian), nadpisania configu dla zdjęć, wyrównanie i stos z odrzucaniem kresek."""
import numpy as np
import pytest

from skyhunt.cli import iter_inputs
from skyhunt.config import config_for_file
from skyhunt.manifest import input_fingerprint, video_fingerprint
from skyhunt.pipeline import input_kind, input_outdir


def _session(tmp_path, name="deneb_0210", n=3):
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        (d / f"DSCF{4700 + i}.RAF").write_bytes(b"FUJIFILMCCD-RAW " + bytes([i]) * 100)
    return d


def test_iter_inputs_sessions_and_videos(tmp_path):
    (tmp_path / "a.MOV").write_bytes(b"x")
    s = _session(tmp_path)
    _session(tmp_path, "few", 2)                        # za mało zdjęć: nie sesja
    (tmp_path / "notes").mkdir()
    assert iter_inputs(tmp_path, [".mov"]) == [tmp_path / "a.MOV", s]
    assert iter_inputs(s, [".mov"]) == [s]              # folder sesji podany wprost
    assert input_kind(s) == "photos" and input_kind(tmp_path / "a.MOV") == "video"
    (tmp_path / "deneb.0210").mkdir()                   # kropka w nazwie folderu nie ucina nazwy
    assert input_outdir(tmp_path / "out", tmp_path / "deneb.0210") == tmp_path / "out" / "deneb.0210"
    assert input_outdir(tmp_path / "out", tmp_path / "a.MOV") == tmp_path / "out" / "a"


def test_fingerprint_folder_and_video(tmp_path):
    s = _session(tmp_path)
    a = input_fingerprint(s)
    assert a["n_files"] == 3 and a["name"] == "deneb_0210"
    assert input_fingerprint(s) == a
    (s / "DSCF4799.RAF").write_bytes(b"FUJIFILMCCD-RAW new")
    assert input_fingerprint(s) != a
    v = tmp_path / "v.MOV"
    v.write_bytes(b"\0" * 1000)
    assert input_fingerprint(v) == video_fingerprint(v)  # stare manifesty wideo zostają ważne


def test_photo_registry_separate_from_video():
    pytest.importorskip("scipy")
    from skyhunt import photo_stages  # noqa: F401
    from skyhunt.pipeline import PHOTO_PIPELINE, PIPELINE, pipeline_for

    assert list(PHOTO_PIPELINE.stages) == ["probe", "frames", "astrometry", "process", "report"]
    assert "frames" not in PIPELINE.stages and "process" not in PIPELINE.stages
    assert PIPELINE.stages["astrometry"].requires == ("detect",)
    assert PHOTO_PIPELINE.stages["astrometry"].requires == ("frames",)
    assert pipeline_for(__import__("pathlib").Path(__file__)) is PIPELINE


def test_photo_overrides_only_for_folders(tmp_path, cfg):
    s = _session(tmp_path)
    c = config_for_file(cfg, s)
    assert c["astrometry"]["downsample"] == 1 and c["color"]["transfer"] == "linear"
    assert config_for_file(cfg, tmp_path / "DSCF0001.MOV")["astrometry"]["downsample"] == cfg["astrometry"]["downsample"]
    dark = _session(tmp_path, "dark_0210")
    assert config_for_file(cfg, dark)["role"] == "dark"


def test_warp_and_stack_reject_streak():
    pytest.importorskip("scipy")
    from skyhunt.photo_stages import (StackAccumulator, coarse_grid, epoch_indices, upsample_map,
                                      warp_to_reference)

    rng = np.random.default_rng(0)
    h, w = 60, 90
    yy, xx = np.mgrid[0:h, 0:w]
    sky = 100 + 50 * np.exp(-((xx - 40.3) ** 2 + (yy - 30.6) ** 2) / 18.0)       # gwiazda w odniesieniu
    gx, gy = coarse_grid((h, w), (8, 6))
    acc = StackAccumulator((h, w), k=5.0, warmup=3, channels=0)
    for i in range(12):
        dx, dy = 0.4 * i, -0.2 * i                                            # dryf nieba
        img = 100 + 50 * np.exp(-((xx - 40.3 - dx) ** 2 + (yy - 30.6 - dy) ** 2) / 18.0)
        img = img + rng.normal(0, 1.0, (h, w))
        if i == 8:
            img[20, 10:80] += 200                                             # kreska satelity
        xmap = upsample_map(gx + dx, (h, w))
        ymap = upsample_map(gy + dy, (h, w))
        al = warp_to_reference(img.astype(np.float32), xmap, ymap)
        acc.add(al, 1.0)
    m = acc.mean()
    inner = (slice(5, 50), slice(5, 70))
    assert np.nanmax(np.abs(m[inner] - sky[inner])) < 3.0                     # gwiazda na miejscu, ostra
    assert np.nanmax(np.abs(m[20:24, 30] - 100)) < 2.0                         # kreska (wiersz 21–22) odrzucona
    tau = np.arange(30) * 1.0
    ev = ["ev0", "ev+1", "ev-1"] * 10
    idx = epoch_indices(tau, ev, every_s=12.0, min_epochs=3)
    assert idx[0] == 0 and idx[-1] == 27 and all(ev[i] == "ev0" for i in idx) and len(idx) >= 3


def test_probe_session_from_fake_rafs(tmp_path, cfg):
    pytest.importorskip("PIL")
    from skyhunt.photo_stages import probe_session
    from tests.test_raw import fake_raf

    d = tmp_path / "sess"
    d.mkdir()
    spec = [("21:00:01", (1, 2), 1), ("21:00:01", (1, 1), 2), ("21:00:02", (1, 4), 3),
            ("21:00:04", (1, 2), 1), ("21:00:04", (1, 1), 2), ("21:00:05", (1, 4), 3)]
    for i, (t, e, s) in enumerate(spec):
        fake_raf(d / f"DSCF{5000 + i}.RAF", when=f"2026:10:02 {t}", exposure=e, seq=s, count=20000 + i)
    res = probe_session(d, cfg)
    s, ph = res["session"], res["photos"]
    assert s["n_photos"] == 6 and s["n_sets"] == 2 and s["sequence_numbers"]
    assert s["classes"] == {"ev0": [0.5], "ev+1": [1.0], "ev-1": [0.25]}
    assert [p["ev"] for p in ph] == ["ev0", "ev+1", "ev-1"] * 2
    gap = cfg["photo"]["inter_frame_gap_s"]
    assert ph[0]["tau_open"] == 0.0 and ph[1]["tau_open"] == pytest.approx(0.5 + gap)
    assert ph[3]["tau_open"] == pytest.approx(s["cadence"]["period"])
    assert res["time_prior"]["start_utc"].startswith("2026-10-02T19:00:01")   # CEST → UTC
    assert not s["warnings"]


def test_pointing_model_interpolates_camera_creep():
    pytest.importorskip("scipy")
    from skyhunt.photo_stages import PointingModel, pointing_drift

    class Cam:                                         # kamera epoki: przesunięcie stałe + obrót nieba 0,5 px/s
        def __init__(self, off):
            self.off = off

        def pixel(self, ra, dec, tau):
            return np.asarray(ra, float) + 0.5 * tau + self.off, np.asarray(dec, float)

    ra, dec = np.array([10.0, 20.0]), np.array([5.0, 6.0])
    pm = PointingModel([Cam(2.0), Cam(0.0)], [10.0, 0.0], ra, dec)      # statyw „siadł” o 2 px w 10 s
    x, y = pm.coarse(5.0)
    assert np.allclose(x, ra + 2.5 + 1.0) and np.allclose(y, dec)
    assert np.allclose(pm.coarse(20.0)[0], ra + 10.0 + 2.0)          # poza zakresem: ostatnia epoka
    d = pointing_drift(Cam(0.0), [Cam(0.0), Cam(2.0)], [0.0, 10.0], 10.0, 5.0)
    assert [round(v["dx_px"], 6) for v in d] == [0.0, 2.0]


def test_torch_stack_matches_cpu_and_time_maps():
    pytest.importorskip("scipy")
    torch = pytest.importorskip("torch")
    from skyhunt.photo_stages import StackAccumulator, TimeMaps, TorchStackAccumulator, coarse_grid

    rng = np.random.default_rng(1)
    h, w = 40, 64
    gx, gy = coarse_grid((h, w), (8, 6))
    cpu = StackAccumulator((h, w), k=5.0, warmup=2, channels=3)
    gpu = TorchStackAccumulator((h, w), k=5.0, warmup=2, channels=3, device="cpu")
    for i in range(6):
        lum = (100 + rng.normal(0, 1, (h, w))).astype(np.float32)
        rgb = np.stack([lum / 3] * 3, axis=-1).astype(np.float32)
        xc, yc = gx + 0.3 * i, gy - 0.2 * i
        r1 = cpu.add_photo(lum, rgb, xc, yc, 1.0)
        r2 = gpu.add_photo(lum, rgb, xc, yc, 1.0)
        assert r1 == pytest.approx(r2, abs=0.02)
    m1, m2 = cpu.mean(), gpu.mean()
    inner = (slice(3, 35), slice(3, 55))
    assert np.allclose(m1[inner], m2[inner], atol=1e-3)
    assert np.allclose(cpu.mean_rgb()[inner], gpu.mean_rgb()[inner], atol=1e-3)
    st = gpu.state()
    again = TorchStackAccumulator((h, w), k=5.0, warmup=2, channels=3, device="cpu")
    again.load({f"ev0_{k}": v for k, v in st.items()}, "ev0_")
    assert np.allclose(again.mean()[inner], m2[inner]) and again.frames == 6

    class P:
        def coarse(self, tau):
            return np.array([1.0, 2.0]) + 0.5 * tau, np.array([3.0, 4.0]) - 0.1 * tau

    tm = TimeMaps(P(), 0.0, 100.0, 20.0)
    x, y = tm.at(37.0)
    assert np.allclose(x, [1.0 + 18.5, 2.0 + 18.5]) and np.allclose(y, [3.0 - 3.7, 4.0 - 3.7])


def test_flatten_removes_vignetting_keeps_stars():
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from skyhunt.photo_report import flatten

    h, w = 240, 360
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1000 * np.exp(-(((xx - w / 2) / 150) ** 2 + ((yy - h / 2) / 110) ** 2))
    star = 300 * np.exp(-((xx - 300) ** 2 + (yy - 40) ** 2) / 4.0)
    f = flatten(vign + star, box=24)
    assert abs(float(np.median(f))) < 20 and float(np.percentile(np.abs(f), 90)) < 40
    assert f[40, 300] > 250                                    # gwiazda w rogu zostaje
