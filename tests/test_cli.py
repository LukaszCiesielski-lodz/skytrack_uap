"""Całość na syntetycznym wideo (CPU): probe → stack, wznawianie, status, bench."""
import json

import numpy as np
import pytest

from skyhunt.cli import iter_inputs, main
from skyhunt.io import read_json
from tests.conftest import ROOT, SYN_N

CPU = ["--config", str(ROOT / "config.yaml"), "--set", "decode.device=cpu",
       "--set", "decode.backends=[pyav, ffmpeg]", "--set", "decode.batch_frames=8"]


def test_run_end_to_end(tmp_path, synthetic_video):
    out = tmp_path / "out"
    assert main(["run", str(synthetic_video), "--out", str(out), *CPU]) == 0
    d = out / synthetic_video.stem
    man = read_json(d / "manifest.json")
    assert man["stages"]["probe"]["status"] == "done"
    st = man["stages"]["stack"]
    assert st["status"] == "done" and st["metrics"]["frames_decoded"] == SYN_N
    mean = np.load(d / "stack_mean.npy")
    assert mean.shape == (64, 96)
    rows = (d / "frame_stats.csv").read_text().splitlines()
    assert len(rows) == SYN_N + 1 and rows[0].startswith("frame,t_s,is_key")
    meta = read_json(d / "meta.json")
    assert meta["time_prior"]["start_utc"]

    # drugie uruchomienie: wszystko z cache
    finished = st["finished_at"]
    assert main(["run", str(synthetic_video), "--out", str(out), *CPU]) == 0
    assert read_json(d / "manifest.json")["stages"]["stack"]["finished_at"] == finished


def test_status_and_probe(tmp_path, synthetic_video, capsys):
    out = tmp_path / "out"
    main(["run", str(synthetic_video), "--out", str(out), "--stages", "probe", *CPU])
    capsys.readouterr()
    main(["status", str(synthetic_video), "--out", str(out), *CPU])
    assert "probe=done" in capsys.readouterr().out
    main(["probe", str(synthetic_video), *CPU])
    assert json.loads(capsys.readouterr().out)["video"]["n_frames"] == SYN_N


def test_bench(tmp_path, synthetic_video):
    rc = main(["bench-decode", str(synthetic_video), "--backends", "pyav,torchcodec", "--frames", "24",
               "--out", str(tmp_path), *CPU])
    assert rc == 0
    res = read_json(tmp_path / "decode_bench.json")["results"]
    assert res[0]["ok"] and res[0]["frames_timed"] > 0
    assert "| pyav |" in (tmp_path / "decode_bench.md").read_text(encoding="utf-8")


def test_iter_inputs(tmp_path):
    for n in ("b.MOV", "a.mp4", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in iter_inputs(tmp_path, [".mp4", ".mov"])] == ["a.mp4", "b.MOV"]
    with pytest.raises(FileNotFoundError):
        iter_inputs(tmp_path / "nie_ma.mov", [".mov"])
