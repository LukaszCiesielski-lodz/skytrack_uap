"""Etapy pipeline'u. Import modułu rejestruje je w ``PIPELINE`` (kolejność = kolejność wykonania)."""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path

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


# ---------------------------------------------------------------- pomocnicze

def prior_start(ctx: StageContext) -> datetime:
    prior = ctx.read_json("meta.json").get("time_prior")
    if not prior:
        raise RuntimeError("brak czasu startu w metadanych (time_prior); uzupełnij config/metadane")
    return datetime.fromisoformat(prior["start_utc"])


def _read_table(path) -> dict:
    import pandas as pd

    df = pd.read_parquet(path)
    return {c: df[c].to_numpy() for c in df.columns}


def _site(cfg: dict) -> tuple[float, float, float]:
    s = cfg["site"]
    return float(s["lat_deg"]), float(s["lon_deg"]), float(s["elevation_m"])


# ---------------------------------------------------------------- detekcja i tory

@PIPELINE.stage("detect", sections=("decode.exact_luma_required", "decode.torchcodec_full_range", "detect", "role"),
                requires=("probe",), rev=2)
def detect(ctx: StageContext) -> dict:
    """Tło, mapa SNR, komponenty per klatka; stacki epok (tylko nagrania nieba)."""
    import pandas as pd

    from .detect import run_detection, star_psf_sigma

    meta = load_meta(ctx)
    dcfg = ctx.cfg["detect"]
    res = run_detection(ctx.video_path, meta, ctx.cfg["decode"], dcfg,
                        save_epochs_to=ctx.outdir / "epochs" if ctx.role == "sky" else None, log=ctx.log)
    pd.DataFrame(res.detections).to_parquet(ctx.outdir / "detections.parquet", index=False)
    star = {"star_sigma_px": None, "n_stars": 0}
    if ctx.role == "sky" and res.middle_background is not None:
        star = star_psf_sigma(res.middle_background, res.middle_sigma, dcfg)
    per_frame = np.bincount(res.detections["frame"].astype(np.int64), minlength=max(res.n_frames, 1))
    stats = {"n_frames": res.n_frames, "detections": int(len(res.detections["frame"])),
             "per_frame_p50": float(np.percentile(per_frame, 50)), "per_frame_p99": float(np.percentile(per_frame, 99)),
             "flagged_frames": res.flagged_frames, "sigma_median_dn": res.sigma_median_dn,
             "throughput_fps": res.n_frames / max(res.elapsed_s, 1e-9), "backend": res.backend, "device": res.device,
             **star}
    ctx.write_json("detect_stats.json", stats)
    ctx.write_json("epochs.json", res.epochs)
    ctx.log.info("[%s] detekcja: %d detekcji (mediana %.0f/klatkę), %.1f kl/s, σ = %.2f DN, gwiazdy σ = %s px, "
                 "odrzucone klatki: %d", ctx.video_path.name, stats["detections"], stats["per_frame_p50"],
                 stats["throughput_fps"], res.sigma_median_dn, star["star_sigma_px"], len(res.flagged_frames))
    if res.flagged_frames and len(res.flagged_frames) > 0.1 * max(res.n_frames, 1):
        ctx.log.warning("[%s] %d/%d klatek odrzuconych (za dużo komponentów) — sprawdź detect.max_components_per_frame",
                        ctx.video_path.name, len(res.flagged_frames), res.n_frames)
    return {"outputs": ["detections.parquet", "detect_stats.json", "epochs.json"] + [e["file"] for e in res.epochs],
            "metrics": {k: stats[k] for k in ("detections", "per_frame_p50", "throughput_fps", "sigma_median_dn",
                                              "star_sigma_px")} | {"flagged_frames": len(res.flagged_frames)}}


@PIPELINE.stage("tracks", sections=("tracks", "camera.nominal_hfov_deg"), requires=("detect",), rev=1)
def tracks(ctx: StageContext) -> dict:
    import pandas as pd

    from .tracks import build_tracks

    meta = load_meta(ctx)
    det = _read_table(ctx.outdir / "detections.parquet")
    star = ctx.read_json("detect_stats.json").get("star_sigma_px")
    rows, points, stats = build_tracks(det, fps=meta.fps, width=meta.width, height=meta.height, star_sigma_px=star,
                                       nominal_hfov_deg=float(ctx.cfg["camera"]["nominal_hfov_deg"]),
                                       tcfg=ctx.cfg["tracks"])
    pd.DataFrame(rows).to_parquet(ctx.outdir / "tracks.parquet", index=False)
    pd.DataFrame(points).to_parquet(ctx.outdir / "track_points.parquet", index=False)
    ctx.log.info("[%s] tory: %d (z %d detekcji, %d statycznych)", ctx.video_path.name, stats["tracks"],
                 stats["detections"], stats["static"])
    return {"outputs": ["tracks.parquet", "track_points.parquet"], "metrics": stats}


@PIPELINE.stage("darkstats", sections=("darkstats", "classify"), requires=("tracks",), rev=1, roles=("dark",))
def darkstats(ctx: StageContext) -> dict:
    """Nagranie z zakrytym obiektywem: gorące piksele i liczba fałszywych torów na godzinę."""
    import pandas as pd

    from .tracks import classify_hint

    meta = load_meta(ctx)
    det = _read_table(ctx.outdir / "detections.parquet")
    n_frames = ctx.read_json("detect_stats.json")["n_frames"]
    xy = np.column_stack([np.round(det["x"]).astype(int), np.round(det["y"]).astype(int)]) if len(det["x"]) \
        else np.empty((0, 2), int)
    uniq, cnt = np.unique(xy, axis=0, return_counts=True) if len(xy) else (np.empty((0, 2), int), np.empty(0))
    hot = uniq[cnt >= float(ctx.cfg["darkstats"]["hot_min_frac"]) * n_frames]
    np.save(ctx.outdir / "hot_pixels.npy", hot)
    tr = pd.read_parquet(ctx.outdir / "tracks.parquet")
    duration = n_frames / meta.fps
    classes: dict[str, int] = {}
    for _, t in tr.iterrows():
        c, _ = classify_hint(t["deg_s_nominal"], t["dur_s"], t["curv_px"], t["cross_ratio"], t["f_peak_hz"],
                             t["f_power"], meta.fps, ctx.cfg["classify"])
        classes[c] = classes.get(c, 0) + 1
    snr = det["peak_snr"]
    out = {"file": ctx.video_path.name, "duration_s": duration, "tracks": int(len(tr)),
           "tracks_per_hour": len(tr) / duration * 3600 if duration else float("nan"), "by_class": classes,
           "detections_per_frame": len(det["x"]) / max(n_frames, 1), "hot_pixels": int(len(hot)),
           "peak_snr_p50_p99": [float(np.percentile(snr, 50)), float(np.percentile(snr, 99))] if len(snr) else None}
    ctx.write_json("darkstats.json", out)
    ctx.log.info("[%s] ciemne: %d fałszywych torów (%.1f/h), %d gorących pikseli", ctx.video_path.name,
                 out["tracks"], out["tracks_per_hour"], out["hot_pixels"])
    return {"outputs": ["darkstats.json", "hot_pixels.npy"], "metrics": out}


# ---------------------------------------------------------------- astrometria

@PIPELINE.stage("astrometry", sections=("astrometry", "site", "camera.nominal_hfov_deg", "camera.sensor_width_mm"),
                requires=("detect",), rev=1, roles=("sky",))
def astrometry(ctx: StageContext) -> dict:
    """Plate solve stacków epok, zgodność epok, rzeczywiste pole widzenia."""
    import shutil

    from .astrometry import (corr_residuals, crop_verdict, ensure_index, epoch_agreement, fov_from_wcs, load_wcs,
                             local_copy, solve_epoch, solver_available, write_solver_config)
    from .sky import FixedCamera, resolve_star

    acfg = ctx.cfg["astrometry"]
    meta = load_meta(ctx)
    if not solver_available(acfg):
        raise RuntimeError("brak solve-field: w notebooku uruchom `apt-get install astrometry.net`")
    adir = ctx.outdir / "astrometry"
    adir.mkdir(exist_ok=True)
    # indeksy i pliki robocze na dysku lokalnym (mmap nie działa niezawodnie na Google Drive)
    index_dir = ensure_index(acfg)
    local_index = Path(acfg.get("local_index_dir", "/tmp/skyhunt-astrometry-index"))
    local_copy(sorted(index_dir.glob("index-*.fits")), local_index)
    work = Path(acfg.get("work_dir", "/tmp/skyhunt-astrometry")) / ctx.video_path.stem
    work.mkdir(parents=True, exist_ok=True)
    solver_cfg = write_solver_config(local_index, work / "solver.cfg")
    hint = resolve_star(acfg["hint_star"]) if acfg.get("hint_star") else None
    results = []
    for e in ctx.read_json("epochs.json"):
        image = local_copy([ctx.outdir / e["file"]], work)[0]
        wcs_local = solve_epoch(acfg, image, work, solver_cfg, hint)
        entry = {**e, "solved": wcs_local is not None}
        if wcs_local is not None:
            for suffix in (".wcs", ".corr"):
                if wcs_local.with_suffix(suffix).exists():
                    shutil.copy2(wcs_local.with_suffix(suffix), adir / wcs_local.with_suffix(suffix).name)
            wcs_path = adir / wcs_local.name
            entry["wcs"] = f"astrometry/{wcs_path.name}"
            corr = wcs_path.with_suffix(".corr")
            if corr.exists():
                entry.update(corr_residuals(corr))
        results.append(entry)
    solved = [r for r in results if r["solved"]]
    if not solved:
        raise RuntimeError("plate solve nieudany dla wszystkich epok (sprawdź zakres skali i podpowiedź)")
    ref = max(solved, key=lambda r: r.get("n_matched", 0))
    t0, site = prior_start(ctx), _site(ctx.cfg)
    cams = [FixedCamera(load_wcs(ctx.outdir / r["wcs"]), r["tau_s"], t0, *site) for r in solved]
    rms = epoch_agreement(cams, (meta.height, meta.width), [r["tau_s"] for r in solved], solved.index(ref))
    fov = fov_from_wcs(cams[solved.index(ref)].wcs, (meta.height, meta.width))
    crop = crop_verdict(fov["fov_w_deg"], ctx.cfg["camera"])
    moved = rms > float(acfg["max_epoch_rms_px"])
    info = {"epochs": results, "n_epochs": len(results), "n_solved": len(solved), "reference": ref["file"],
            "reference_wcs": ref["wcs"], "reference_tau_s": ref["tau_s"], "epoch_rms_px": rms,
            "camera_moved": moved, "fov": fov, "crop": crop, "hint": acfg.get("hint_star")}
    ctx.write_json("wcs.json", info)
    level = ctx.log.warning if moved else ctx.log.info
    level("[%s] astrometria: %d/%d epok, pole %.2f°×%.2f°, %.2f″/px, %s, zgodność epok %.2f px",
          ctx.video_path.name, len(solved), len(results), fov["fov_w_deg"], fov["fov_h_deg"],
          fov["scale_arcsec_px"], crop["verdict"], rms)
    return {"outputs": ["wcs.json"] + [r["wcs"] for r in solved],
            "metrics": {"n_solved": len(solved), "n_epochs": len(results), "epoch_rms_px": rms, **fov,
                        "crop": crop["verdict"]}}


# ---------------------------------------------------------------- satelity

@PIPELINE.stage("tle", sections=("satellites",), requires=("probe",), rev=1, roles=("sky",))
def tle(ctx: StageContext) -> dict:
    """Zamrożony snapshot elementów orbit dla nagrania (CelesTrak ± Space-Track)."""
    from .satellites import fetch_spacetrack, load_catalog, select_celestrak, write_catalog_csv

    scfg = ctx.cfg["satellites"]
    t_rec = prior_start(ctx)
    snaps = select_celestrak(scfg, t_rec)
    st = fetch_spacetrack(scfg, t_rec)
    files = [(s["path"], s["source"]) for s in snaps] + ([(st["path"], "spacetrack")] if st else [])
    if not files:
        raise RuntimeError("brak elementów orbit: uruchom w notebooku komórkę „Snapshot elementów orbit” "
                           "albo ustaw dane Space-Track w Colab Secrets")
    cat = load_catalog(files, t_rec)
    write_catalog_csv(files, cat, ctx.outdir / "gp_elements.csv")
    ages = np.array([(t_rec - e).total_seconds() / 86400 for e in cat.epoch])
    info = {"n_objects": len(cat), "spacetrack": st is not None,
            "files": [{"name": Path(s["path"]).name, "source": s["source"], "group": s["group"],
                       "fetched": s["fetched"].isoformat()} for s in snaps + ([st] if st else [])],
            "age_days_median": float(np.median(np.abs(ages))) if len(ages) else None,
            "age_days_max": float(np.max(np.abs(ages))) if len(ages) else None}
    ctx.write_json("gp_source.json", info)
    ctx.log.info("[%s] katalog: %d obiektów, mediana wieku elementów %.2f d%s", ctx.video_path.name, len(cat),
                 info["age_days_median"] or float("nan"), " (+Space-Track)" if st else "")
    return {"outputs": ["gp_elements.csv", "gp_source.json"],
            "metrics": {k: info[k] for k in ("n_objects", "spacetrack", "age_days_median", "age_days_max")}}


@PIPELINE.stage("identify", sections=("identify", "classify", "site", "time", "camera", "satellites.ephemeris_dir",
                                      "report.identified_min_confidence"),
                requires=("tracks", "astrometry", "tle"), rev=1, roles=("sky",))
def identify(ctx: StageContext) -> dict:
    """Synchronizacja zegara po satelitach, identyfikacja NORAD, tabela końcowa torów."""
    import pandas as pd

    from .astrometry import load_wcs
    from .satellites import (SIDEREAL_DEG_S, Observer, identify_tracks, load_catalog, make_track_sky,
                             observer_offset, screen, sunlit_flags, synchronize)
    from .sky import FixedCamera, field_center, radec_to_vec
    from .timing import FrameClock
    from .tracks import classify_hint

    meta = load_meta(ctx)
    icfg, ccfg = ctx.cfg["identify"], ctx.cfg["classify"]
    t0, site = prior_start(ctx), _site(ctx.cfg)
    H, W = meta.height, meta.width
    wcsinfo = ctx.read_json("wcs.json")
    camera = FixedCamera(load_wcs(ctx.outdir / wcsinfo["reference_wcs"]), wcsinfo["reference_tau_s"], t0, *site)
    clock = FrameClock.from_config(t0, meta, ctx.cfg["camera"])
    tr = pd.read_parquet(ctx.outdir / "tracks.parquet")
    pts = pd.read_parquet(ctx.outdir / "track_points.parquet")

    # punkty torów na niebie (wektorowo dla wszystkich torów)
    if len(pts):
        pts = pts.sort_values(["track_id", "frame"]).reset_index(drop=True)
        tau = clock.tau(pts["frame"].to_numpy(), pts["y"].to_numpy())
        ra, dec = camera.icrs(pts["x"].to_numpy(), pts["y"].to_numpy(), tau)
        vec = camera.apparent_vectors(ra, dec, tau)
        pts = pts.assign(tau=tau, ra=ra, dec=dec)
    else:
        pts = pts.assign(tau=[], ra=[], dec=[])
        vec = np.empty((0, 3))
    skies = []
    for tid, idx in pts.groupby("track_id").indices.items():
        skies.append(make_track_sky(int(tid), pts["tau"].to_numpy()[idx], vec[idx], pts["ra"].to_numpy()[idx],
                                    pts["dec"].to_numpy()[idx]))
    by_id = {s.track_id: s for s in skies}

    catalog = load_catalog([(ctx.outdir / "gp_elements.csv", "snapshot")], t0)
    observer = Observer(*site, t0)
    _, radius = field_center(camera.wcs, (H, W))
    dur = meta.n_frames / meta.fps

    def grid_for(window_s: float):
        step = float(icfg["screen_step_s"])
        offsets = np.arange(-window_s, dur + window_s + step, step)
        ra_c, dec_c = camera.icrs(np.full(len(offsets), W / 2), np.full(len(offsets), H / 2), offsets)
        ctx.log.info("[%s] przesiew katalogu: ±%.0f s, %d chwil × %d obiektów", ctx.video_path.name, window_s,
                     len(offsets), len(catalog))
        return screen(observer, catalog, offsets, radec_to_vec(ra_c, dec_c), radius, icfg,
                      extra_margin_deg=window_s * SIDEREAL_DEG_S)

    sync, grid = synchronize(observer, catalog, skies, grid_for, float(ctx.cfg["time"]["prior_sigma_s"]),
                             float(ctx.cfg["time"]["sync_search_s"]), icfg)
    delta = sync.delta_s
    ctx.log.info("[%s] poprawka zegara Δ = %+.2f ± %.2f s (%s, %s)", ctx.video_path.name, delta, sync.sigma_s,
                 sync.confidence, sync.method)
    offset = None
    try:
        offset = observer_offset(observer, catalog, sync.members, by_id) if sync.synced else None
    except Exception as e:  # noqa: BLE001 — diagnostyka nie może zatrzymać identyfikacji
        ctx.log.warning("[%s] przesunięcie obserwatora: %s", ctx.video_path.name, e)
    if offset:
        bad = offset["horizontal_km"] > float(icfg["site_warn_km"])
        (ctx.log.warning if bad else ctx.log.info)(
            "[%s] paralaksa satelitów: obserwator przesunięty o %.2f km na północ, %.2f km na wschód "
            "(%d torów, odchylenie %.0f″ → %.0f″ po korekcie)%s", ctx.video_path.name, offset["north_km"],
            offset["east_km"], offset["n_tracks"], offset["rms_before_arcsec"], offset["rms_after_arcsec"],
            " — SPRAWDŹ WSPÓŁRZĘDNE (komórka „Miejsce obserwacji”)" if bad else "")
    matches = identify_tracks(observer, catalog, skies, grid, delta, icfg) if grid is not None else {}
    bests = [m[0] for m in matches.values() if m]
    for m in bests:   # predykcja toru satelity do rysowania
        s = by_id[m.track_id]
        sel = np.linspace(0, len(s.tau) - 1, min(len(s.tau), 60)).astype(int)
        u = observer.topocentric([catalog.satrecs[m.cat_index]], m.delta_s + s.tau[sel])["unit"][0]
        pra, pdec = camera.astrometric_radec(u, s.tau[sel], m.delta_s)
        m.pred_radec = np.column_stack([pra, pdec]).tolist()
    sunlit_flags(catalog, bests, observer, ctx.cfg["satellites"]["ephemeris_dir"],
                 {s.track_id: s.tau_mid for s in skies})

    # tabela końcowa
    from .report import CONF_RANK

    min_rank = CONF_RANK[str(ctx.cfg["report"]["identified_min_confidence"])]
    cam_sync = camera.with_reference(t0 + timedelta(seconds=delta))
    rows = []
    for _, t in tr.iterrows():
        tid = int(t["track_id"])
        s = by_id[tid]
        p = pts[pts["track_id"] == tid]
        az, alt = cam_sync.altaz(p["x"].to_numpy()[[0, -1]], p["y"].to_numpy()[[0, -1]])
        omega = s.gc.omega_deg_s
        curv_px = float(t["curv_px"])
        peak = float(p["peak_snr"].median()) if "peak_snr" in p else float("nan")
        hint, reason = classify_hint(omega, float(t["dur_s"]), curv_px, float(t["cross_ratio"]), float(t["f_peak_hz"]),
                                     float(t["f_power"]), meta.fps, ccfg, peak_snr=peak)
        best = (matches.get(tid) or [None])[0]
        kind = "sat" if best is not None and CONF_RANK.get(best.confidence, 0) >= min_rank else "unid"
        rows.append({
            **t.to_dict(), "kind": kind, "class_hint": hint, "class_reason": reason, "peak_snr_median": peak,
            "tau0": float(s.tau[0]), "tau1": float(s.tau[-1]), "tau_mid": s.tau_mid,
            "utc_start": (t0 + timedelta(seconds=float(s.tau[0]) + delta)).isoformat(),
            "utc_end": (t0 + timedelta(seconds=float(s.tau[-1]) + delta)).isoformat(),
            "omega_deg_s": omega, "curv_arcsec": s.gc.cross_rms_arcsec,
            "ra0": float(s.ra[0]), "dec0": float(s.dec[0]), "ra1": float(s.ra[-1]), "dec1": float(s.dec[-1]),
            "az0": float(az[0]), "alt0": float(alt[0]), "az1": float(az[1]), "alt1": float(alt[1]),
            "norad": best.norad if best else None, "sat_name": best.name if best else None,
            "confidence": best.confidence if best else None, "match_reason": best.reason if best else None,
            "sunlit": best.sunlit if best else None,
        })
    final = pd.DataFrame(rows)
    final.to_parquet(ctx.outdir / "tracks_final.parquet", index=False)
    final.to_csv(ctx.outdir / "tracks_final.csv", index=False)
    pts[["track_id", "frame", "x", "y", "tau", "ra", "dec"]].to_parquet(ctx.outdir / "track_sky.parquet", index=False)
    ctx.write_json("identifications.json", {str(k): [m.to_dict() | ({"pred_radec": getattr(m, "pred_radec", None)}
                                                                     if i == 0 else {}) for i, m in enumerate(v)]
                                            for k, v in matches.items()})

    # satelity przewidziane w kadrze
    pred_rows = []
    if grid is not None and len(grid.idx):
        step = float(icfg["sample_step_s"])
        taus = np.arange(0, dur + step, step)
        detected = {m.norad for m in bests if CONF_RANK.get(m.confidence, 0) >= min_rank}
        topo = observer.topocentric([catalog.satrecs[i] for i in grid.idx], delta + taus)
        S = len(grid.idx)
        flat = topo["unit"].reshape(-1, 3)
        tt = np.tile(taus, S)
        ok = np.isfinite(flat).all(axis=1)
        px = np.full(len(flat), np.nan)
        py = np.full(len(flat), np.nan)
        if ok.any():
            pra, pdec = camera.astrometric_radec(flat[ok], tt[ok], delta)
            px[ok], py[ok] = camera.pixel(pra, pdec, tt[ok])
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        inside = inside.reshape(S, len(taus))
        for j in np.flatnonzero(inside.any(axis=1)):
            k = np.flatnonzero(inside[j])
            i = int(grid.idx[j])
            pred_rows.append({"norad": int(catalog.norad[i]), "name": catalog.name[i],
                              "utc_in": (t0 + timedelta(seconds=delta + taus[k[0]])).isoformat(),
                              "utc_out": (t0 + timedelta(seconds=delta + taus[k[-1]])).isoformat(),
                              "range_km": float(topo["dist_km"][j, k[len(k) // 2]]),
                              "sunlit": None, "detected": int(catalog.norad[i]) in detected})
    pd.DataFrame(pred_rows, columns=["norad", "name", "utc_in", "utc_out", "range_km", "sunlit", "detected"]) \
        .to_csv(ctx.outdir / "fov_predicted.csv", index=False)

    gp = ctx.read_json("gp_source.json")
    age = gp.get("age_days_median")
    note = (f"{gp['n_objects']} obiektów ({', '.join(sorted({f['source'] for f in gp['files']}))}); mediana wieku "
            f"elementów {f'{age:.2f} d' if age is not None else '–'}"
            + ("" if gp["spacetrack"] else "; bez Space-Track: niepełne człony rakiet i śmieci"))
    ctx.write_json("time_sync.json", {**sync.to_dict(), "start_utc_prior": t0.isoformat(),
                                      "start_utc_synced": (t0 + timedelta(seconds=delta)).isoformat(),
                                      "catalog_note": note, "observer_offset": offset,
                                      "site": dict(zip(("lat_deg", "lon_deg", "elevation_m"), site))})
    n_sat = int((final["kind"] == "sat").sum()) if len(final) else 0
    ctx.log.info("[%s] zidentyfikowane satelity: %d / %d torów; przewidziane w kadrze: %d", ctx.video_path.name,
                 n_sat, len(final), len(pred_rows))
    return {"outputs": ["tracks_final.parquet", "tracks_final.csv", "track_sky.parquet", "identifications.json",
                        "fov_predicted.csv", "time_sync.json"],
            "metrics": {"site_offset_km": offset["horizontal_km"] if offset else None,
                        "delta_s": delta, "sigma_s": sync.sigma_s, "sync_confidence": sync.confidence,
                        "synced": sync.synced, "tracks": len(final), "satellites": n_sat,
                        "predicted_in_fov": len(pred_rows)}}


@PIPELINE.stage("report", sections=("report",), requires=("identify",), rev=1, roles=("sky",))
def report(ctx: StageContext) -> dict:
    from .report import build

    outputs = build(ctx.outdir, ctx.video_path, load_meta(ctx), ctx.cfg)
    pdfs = [o for o in outputs if o.endswith(".pdf")]
    ctx.log.info("[%s] raport: %d PDF, %d klipów → %s", ctx.video_path.name, len(pdfs), len(outputs) - len(pdfs),
                 ctx.outdir / "report")
    return {"outputs": outputs, "metrics": {"pdfs": len(pdfs), "clips": len(outputs) - len(pdfs)}}
