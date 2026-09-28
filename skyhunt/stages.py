"""Etapy pipeline'u. Import modułu rejestruje je w ``PIPELINE`` (kolejność = kolejność wykonania)."""
from __future__ import annotations

import time

import numpy as np

from .decode import open_decoder
from .metadata import VideoMeta, probe_video
from .pipeline import PIPELINE, StageContext
from .stack import StackAccumulator, keyframe_pulse, stretch_u8, write_frame_stats, write_png
from .timing import time_prior


@PIPELINE.stage("probe", sections=("time", "role"), rev=1)
def probe(ctx: StageContext) -> dict:
    """Metadane kontenera + wstępny czas startu (prior do synchronizacji po satelitach)."""
    meta = probe_video(ctx.video_path)
    try:
        prior = time_prior(meta, ctx.cfg["time"]).to_dict()
    except ValueError as e:   # brak znacznika czasu: czas wyłącznie z satelitów (M2)
        ctx.log.warning("[%s] %s", ctx.video_path.name, e)
        prior = None
    ctx.write_json("meta.json", {"video": meta.to_dict(), "time_prior": prior,
                                 "role": ctx.cfg.get("role", "sky")})
    ctx.log.info("[%s] %dx%d %s %.3f fps, %d klatek, GOP %s, klatki B: %s, start ≈ %s UTC",
                 ctx.video_path.name, meta.width, meta.height, meta.codec, meta.fps, meta.n_frames,
                 meta.gop_length, meta.has_bframes, prior and prior["start_utc"])
    return {"outputs": ["meta.json"],
            "metrics": {"n_frames": meta.n_frames, "fps": meta.fps, "gop_length": meta.gop_length,
                        "has_bframes": meta.has_bframes,
                        "start_utc_prior": prior and prior["start_utc"]}}


def load_meta(ctx: StageContext) -> VideoMeta:
    return VideoMeta.from_dict(ctx.read_json("meta.json")["video"])


@PIPELINE.stage("stack", sections=("decode.exact_luma_required", "decode.torchcodec_full_range", "stack"),
                requires=("probe",), rev=1)
def stack(ctx: StageContext) -> dict:
    """Jedno przejście przez całe nagranie: średnia, max, statystyki per klatka."""
    meta = load_meta(ctx)
    dcfg, scfg = ctx.cfg["decode"], ctx.cfg["stack"]
    dec = open_decoder(ctx.video_path, meta, dcfg)
    if dec.skipped:
        ctx.log.info("[%s] pominięte backendy: %s", ctx.video_path.name, dec.skipped)
    ctx.log.info("[%s] dekodowanie: %s → %s", ctx.video_path.name, dec.name, dec.device)

    acc = StackAccumulator(scfg["stats_subsample"], scfg["frame_quantiles"])
    t0 = time.perf_counter()
    for batch in dec.batches(dcfg["batch_frames"]):
        acc.add(batch)
    res = acc.result()   # kopiuje wyniki na CPU, więc czas obejmuje całą pracę GPU
    elapsed = time.perf_counter() - t0

    out = ctx.outdir
    np.save(out / "stack_mean.npy", res.mean)
    np.save(out / "stack_max.npy", res.max)
    lo, hi = scfg["preview_percentiles"]
    write_png(out / "stack_mean.png", stretch_u8(res.mean, lo, hi))
    write_png(out / "stack_max_minus_mean.png", stretch_u8(res.max.astype(np.float32) - res.mean, lo, hi))
    write_frame_stats(out / "frame_stats.csv", res, meta.fps, meta.keyframes)

    n = len(res.frames)
    pulse = keyframe_pulse(res.frame_means, meta.keyframes)
    if n != meta.n_frames:
        ctx.log.warning("[%s] zdekodowano %d klatek, kontener deklaruje %d",
                        ctx.video_path.name, n, meta.n_frames)
    ctx.log.info("[%s] stack: %d klatek w %.1f s (%.1f kl/s), pulsowanie I-klatek %.3f DN",
                 ctx.video_path.name, n, elapsed, n / elapsed, pulse)
    return {"outputs": ["stack_mean.npy", "stack_max.npy", "stack_mean.png",
                        "stack_max_minus_mean.png", "frame_stats.csv"],
            "metrics": {"backend": dec.name, "device": dec.device, "frames_decoded": n,
                        "frames_declared": meta.n_frames, "throughput_fps": n / elapsed,
                        "keyframe_pulse_dn": pulse, "mean_dn": float(res.frame_means.mean())}}
