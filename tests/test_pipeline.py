"""Manifest i wznawianie: cache, unieważnianie po zmianie configu, błędy, zmiana pliku."""
import pytest

from skyhunt.config import deep_merge
from skyhunt.io import read_json
from skyhunt.manifest import Manifest
from skyhunt.pipeline import Pipeline


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "raw" / "v.mov"
    p.parent.mkdir()
    p.write_bytes(b"abc" * 1000)
    return p


def make_pipeline(calls: list, fail: set | None = None) -> Pipeline:
    pl = Pipeline()
    fail = fail if fail is not None else set()

    @pl.stage("a", sections=("time",))
    def a(ctx):
        calls.append("a")
        (ctx.outdir / "a.txt").write_text("a")
        return {"outputs": ["a.txt"], "metrics": {"x": 1}}

    @pl.stage("b", sections=("decode.batch_frames",), requires=("a",))
    def b(ctx):
        calls.append("b")
        if "b" in fail:
            raise RuntimeError("awaria")
        (ctx.outdir / "b.txt").write_text("b")
        return {"outputs": ["b.txt"]}

    @pl.stage("c", requires=("a",))
    def c(ctx):
        calls.append("c")
        return {}

    return pl


def test_resume_and_cache(tmp_path, video, cfg):
    calls: list = []
    pl = make_pipeline(calls)
    out = tmp_path / "out"
    assert pl.run(video, cfg, out) == {"a": "done", "b": "done", "c": "done"}
    assert pl.run(video, cfg, out) == {"a": "cached", "b": "cached", "c": "cached"}
    assert calls == ["a", "b", "c"]
    man = read_json(out / "v" / "manifest.json")
    assert man["stages"]["a"]["metrics"] == {"x": 1}
    assert man["stages"]["a"]["version"]


def test_config_change_invalidates_downstream(tmp_path, video, cfg):
    calls: list = []
    pl = make_pipeline(calls)
    out = tmp_path / "out"
    pl.run(video, cfg, out)
    calls.clear()
    pl.run(video, deep_merge(cfg, {"decode": {"batch_frames": 3}}), out)
    assert calls == ["b"]
    calls.clear()
    pl.run(video, deep_merge(cfg, {"time": {"prior_sigma_s": 1}, "decode": {"batch_frames": 3}}), out)
    assert calls == ["a", "b", "c"]


def test_missing_output_triggers_rerun(tmp_path, video, cfg):
    calls: list = []
    pl = make_pipeline(calls)
    out = tmp_path / "out"
    pl.run(video, cfg, out)
    (out / "v" / "b.txt").unlink()
    calls.clear()
    pl.run(video, cfg, out)
    assert calls == ["b"]


def test_failure_is_recorded_and_resumed(tmp_path, video, cfg):
    calls: list = []
    fail = {"b"}
    pl = make_pipeline(calls, fail)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        pl.run(video, cfg, out)
    st = read_json(out / "v" / "manifest.json")["stages"]
    assert st["a"]["status"] == "done" and st["b"]["status"] == "failed" and "awaria" in st["b"]["error"]
    fail.clear()
    calls.clear()
    pl.run(video, cfg, out)
    assert calls == ["b", "c"]


def test_only_and_force(tmp_path, video, cfg):
    calls: list = []
    pl = make_pipeline(calls)
    out = tmp_path / "out"
    assert pl.run(video, cfg, out, only=["b"]) == {"a": "done", "b": "done"}
    calls.clear()
    pl.run(video, cfg, out, force=["a"])
    assert calls == ["a", "b", "c"]


def test_different_video_resets_manifest(tmp_path, video, cfg):
    calls: list = []
    pl = make_pipeline(calls)
    out = tmp_path / "out"
    pl.run(video, cfg, out)
    video.write_bytes(b"xyz" * 1000)   # ta sama nazwa, inna treść
    calls.clear()
    pl.run(video, cfg, out)
    assert calls == ["a", "b", "c"]


def test_corrupted_manifest(tmp_path, video):
    d = tmp_path / "out" / "v"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{ucięty")
    m = Manifest.load(d, video)
    assert m.data["stages"] == {} and m.reset_reason


def test_roles_gate_stages(tmp_path, video, cfg):
    calls: list = []
    pl = Pipeline()

    @pl.stage("a")
    def a(ctx):
        calls.append("a")
        return {}

    @pl.stage("sky_only", requires=("a",), roles=("sky",))
    def s(ctx):
        calls.append("sky_only")
        return {}

    @pl.stage("dark_only", requires=("a",), roles=("dark",))
    def d(ctx):
        calls.append("dark_only")
        return {}

    out = tmp_path / "out"
    assert pl.run(video, cfg, out) == {"a": "done", "sky_only": "done", "dark_only": "n/a"}
    calls.clear()
    st = pl.run(video, deep_merge(cfg, {"role": "dark"}), tmp_path / "out_dark")
    assert st == {"a": "done", "sky_only": "n/a", "dark_only": "done"}
    assert calls == ["a", "dark_only"]
    assert "sky_only" not in read_json(tmp_path / "out_dark" / "v" / "manifest.json")["stages"]


def test_registration_order_enforced():
    pl = Pipeline()
    with pytest.raises(ValueError):
        pl.stage("x", requires=("nie_ma",))(lambda ctx: None)
