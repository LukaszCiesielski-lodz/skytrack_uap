"""Detektor per klatka (sekcja 5.4a handoffu): mapa SNR, matched filter, próg z histerezą,
komponenty spójne i ich momenty. Strumieniowo, na GPU (torch), z fallbackiem CPU.

Wynik: tabela detekcji (klatka, x, y, strumień, SNR, kowariancja kształtu) oraz stacki
„epok” (mediana ±1 s) do plate solve.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .background import FrameRing, median_mad, n_blocks, plan_block
from .decode import open_decoder
from .metadata import VideoMeta

DET_COLUMNS = ("frame", "x", "y", "flux", "sum_snr", "peak_snr", "npix", "cxx", "cxy", "cyy")


def gaussian_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def gaussian_filter(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separowalny filtr Gaussa dla [N, H, W]."""
    g = gaussian_kernel(sigma, img.device)
    r = (g.numel() - 1) // 2
    x = img[:, None]
    x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, r))
    x = F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))
    return x[:, 0]


def robust_std(x: torch.Tensor, subsample: int = 4) -> torch.Tensor:
    """Odporne σ per klatka [N] = 1.4826·MAD z podpróbki."""
    s = x[:, ::subsample, ::subsample].reshape(x.shape[0], -1)
    med = s.median(dim=1, keepdim=True).values
    return 1.4826 * (s - med).abs().median(dim=1).values


def snr_maps(frames: torch.Tensor, bg: torch.Tensor, sigma: torch.Tensor, dcfg: dict):
    """(różnica [DN], SNR, SNR po matched filter) dla klatek [N, H, W] float32."""
    diff = frames - bg
    if dcfg.get("subtract_frame_offset", True):
        k = int(dcfg.get("offset_subsample", 4))
        off = diff[:, ::k, ::k].reshape(diff.shape[0], -1).median(dim=1).values
        diff = diff - off[:, None, None]
    snr = diff / sigma
    psf = float(dcfg["psf_sigma_px"])
    filt = gaussian_filter(snr, psf)
    # Biały szum o σ=1 po filtrze Gaussa ma σ = 1/(2·√π·psf). Mapę dzielimy przez zmierzone
    # odporne σ (szum H.264 jest skorelowany, więc bywa większe), ale nie mniejsze niż teoria —
    # inaczej obraz bez szumu (np. syntetyczny) dałby ogromne „SNR” z artefaktów kompresji.
    white = 1.0 / (2.0 * math.sqrt(math.pi) * psf)
    if dcfg.get("renormalize_filtered", True):
        filt = filt / robust_std(filt).clamp(min=white)[:, None, None]
    else:
        filt = filt / white
    return diff, snr, filt


def _label_scipy(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    from scipy import ndimage

    lab, n = ndimage.label(mask.cpu().numpy())
    return torch.from_numpy(lab.astype(np.int64)).to(mask.device), int(n)


def _label_cupy(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    import cupy as cp
    from cupyx.scipy import ndimage as cnd

    lab, n = cnd.label(cp.from_dlpack(mask.to(torch.uint8).contiguous()))
    return torch.from_dlpack(lab).to(torch.int64), int(n)


def pick_labeler(name: str, device: torch.device):
    if name == "scipy" or (name == "auto" and device.type != "cuda"):
        return _label_scipy
    try:
        import cupy  # noqa: F401
        from cupyx.scipy import ndimage  # noqa: F401
        return _label_cupy
    except ImportError:
        if name == "cupy":
            raise
        return _label_scipy


def frame_components(filt: torch.Tensor, snr: torch.Tensor, diff: torch.Tensor, dcfg: dict,
                     labeler) -> tuple[np.ndarray, bool]:
    """Komponenty jednej klatki [H, W] → tablica [K, 9] (x, y, flux, sum_snr, peak, npix,
    cxx, cxy, cyy) oraz flaga „za dużo komponentów”. Maska: filt > snr_grow; komponent
    zostaje, gdy jego maksimum ≥ snr_seed. Wagi momentów: SNR bez filtra (≥0).

    Limit ``max_components_per_frame`` dotyczy komponentów z maksimum ≥ snr_seed; samych
    plamek szumu powyżej snr_grow jest na klatce 4K wiele tysięcy i nie świadczą o chmurach."""
    labels, n = labeler(filt > float(dcfg["snr_grow"]))
    if n == 0:
        return np.empty((0, 9)), False
    ys, xs = torch.nonzero(labels, as_tuple=True)
    lab = labels[ys, xs] - 1
    dev, dt = filt.device, torch.float64
    w = snr[ys, xs].clamp(min=0).to(dt)
    xf, yf = xs.to(dt), ys.to(dt)

    def acc(v):
        return torch.zeros(n, dtype=dt, device=dev).index_add_(0, lab, v)

    sw, swx, swy = acc(w), acc(w * xf), acc(w * yf)
    swxx, swxy, swyy = acc(w * xf * xf), acc(w * xf * yf), acc(w * yf * yf)
    flux = acc(diff[ys, xs].to(dt))
    npix = torch.bincount(lab, minlength=n).to(dt)
    peak = torch.full((n,), -math.inf, dtype=dt, device=dev).scatter_reduce_(
        0, lab, filt[ys, xs].to(dt), reduce="amax", include_self=True)
    keep = (peak >= float(dcfg["snr_seed"])) & (npix >= int(dcfg["min_pixels"])) & (sw > 0)
    if int(keep.sum()) > int(dcfg["max_components_per_frame"]):
        return np.empty((0, 9)), True
    sw_safe = sw.clamp(min=1e-12)
    x, y = swx / sw_safe, swy / sw_safe
    cxx, cxy, cyy = swxx / sw_safe - x * x, swxy / sw_safe - x * y, swyy / sw_safe - y * y
    out = torch.stack([x, y, flux, sw, peak, npix, cxx, cxy, cyy], dim=1)[keep]
    return out.cpu().numpy(), False


@dataclass
class DetectResult:
    detections: dict            # kolumna → np.ndarray
    epochs: list = field(default_factory=list)   # [{"block", "frame", "file"}]
    flagged_frames: list = field(default_factory=list)
    n_frames: int = 0
    sigma_median_dn: float = float("nan")
    middle_background: np.ndarray | None = None
    middle_sigma: np.ndarray | None = None
    elapsed_s: float = 0.0
    backend: str = ""
    device: str = ""


def epoch_blocks(nb: int, block_frames: int, fps: float, every_s: float) -> set[int]:
    """Bloki, których tło zapisujemy jako epokę do plate solve (co ``every_s``, min. 1)."""
    if nb <= 0:
        return set()
    step = max(1, int(round(every_s * fps / block_frames)))
    first = min(step // 2, nb // 2)
    return set(range(first, nb, step)) or {nb // 2}


def run_detection(video_path: Path, meta: VideoMeta, decode_cfg: dict, dcfg: dict, *,
                  save_epochs_to: Path | None = None, log: logging.Logger | None = None) -> DetectResult:
    """Jedno przejście przez nagranie: tło, SNR, komponenty. ``save_epochs_to``: katalog na
    FITS epok (None = bez epok, np. dla nagrań z zakrytym obiektywem)."""
    log = log or logging.getLogger("skyhunt")
    dec = open_decoder(video_path, meta, decode_cfg)
    device = torch.device(dec.device)
    H, W = meta.height, meta.width
    B, Wh, st = int(dcfg["bg_block_frames"]), int(dcfg["bg_half_window_frames"]), int(dcfg["bg_stride"])
    bs = int(decode_cfg["batch_frames"])
    # Zapas jednego bloku: gdy plik okaże się krótszy niż deklaruje kontener, środek
    # ostatniego bloku (a więc i okno tła) przesuwa się wstecz.
    ring = FrameRing(2 * Wh + 2 * B + bs + 2, H, W, device)
    labeler = pick_labeler(str(dcfg.get("labeler", "auto")), device)
    floor = float(dcfg["sigma_floor_dn"])
    step = max(1, int(dcfg.get("frames_per_step", 8)))

    n_plan = meta.n_frames
    nb_plan = n_blocks(n_plan, B)
    epochs_at = epoch_blocks(nb_plan, B, meta.fps, float(dcfg["epoch_every_s"])) if save_epochs_to else set()
    mid_block = nb_plan // 2
    rows: list[np.ndarray] = []
    res = DetectResult(detections={}, backend=dec.name, device=str(device))
    sigmas: list[float] = []

    def process(b: int, n_frames: int) -> None:
        blk = plan_block(b, n_frames, B, Wh, st)
        bg, mad = median_mad(ring.get(blk.window), int(dcfg.get("bg_tile_rows", 540)))
        sigma = (1.4826 * mad).clamp_(min=floor)
        sigmas.append(float(sigma.median()))
        if b == mid_block or (res.middle_background is None and b == n_blocks(n_frames, B) - 1):
            res.middle_background = bg.cpu().numpy()
            res.middle_sigma = sigma.cpu().numpy()
        if b in epochs_at:
            res.epochs.append(write_epoch(save_epochs_to, b, blk.center, bg.cpu().numpy(), meta))
        for f0 in range(blk.start, blk.stop, step):
            frames = list(range(f0, min(f0 + step, blk.stop)))
            diff, snr, filt = snr_maps(ring.get(frames).to(torch.float32), bg, sigma, dcfg)
            for i, f in enumerate(frames):
                comps, flagged = frame_components(filt[i], snr[i], diff[i], dcfg, labeler)
                if flagged:
                    res.flagged_frames.append(f)
                elif len(comps):
                    rows.append(np.column_stack([np.full(len(comps), f), comps]))
        if b + 1 < n_blocks(n_frames, B):
            ring.drop_before(plan_block(b + 1, n_frames, B, Wh, st).first_needed - B)

    t0 = time.perf_counter()
    next_b = 0
    extra = 0
    for batch in dec.batches(bs):
        # Klatki ponad liczbę z kontenera (stts) pomijamy — plan bloków opiera się na niej.
        keep = max(0, min(len(batch), n_plan - batch.start))
        extra += len(batch) - keep
        if keep == 0:
            continue
        y = batch.y if isinstance(batch.y, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(batch.y))
        ring.push(batch.start, y[:keep].to(device))
        while next_b < nb_plan and plan_block(next_b, n_plan, B, Wh, st).last_needed < ring.hi:
            process(next_b, n_plan)
            next_b += 1
            if next_b % max(1, nb_plan // 10) == 0:
                log.info("[%s] detekcja: blok %d/%d, %.1f kl/s", video_path.name, next_b, nb_plan,
                         ring.hi / (time.perf_counter() - t0))
    n_real = ring.hi
    if n_real != n_plan or extra:
        log.warning("[%s] zdekodowano %d klatek (+%d pominiętych), kontener deklaruje %d", video_path.name,
                    n_real, extra, n_plan)
    for b in range(next_b, n_blocks(n_real, B)):
        process(b, n_real)
    res.elapsed_s = time.perf_counter() - t0
    res.n_frames = n_real
    res.sigma_median_dn = float(np.median(sigmas)) if sigmas else float("nan")
    data = np.concatenate(rows) if rows else np.empty((0, len(DET_COLUMNS)))
    res.detections = {c: data[:, i] for i, c in enumerate(DET_COLUMNS)}
    res.detections["frame"] = res.detections["frame"].astype(np.int64)
    return res


def write_epoch(outdir: Path, block: int, frame: int, bg: np.ndarray, meta: VideoMeta) -> dict:
    """Stack epoki (mediana ±1 s) jako FITS float32 do plate solve."""
    from astropy.io import fits

    outdir.mkdir(parents=True, exist_ok=True)
    name = f"epoch_{frame:06d}.fits"
    hdr = fits.Header()
    hdr["FRAME"] = (int(frame), "klatka srodka epoki")
    hdr["TAU_S"] = (frame * meta.fps_den / meta.fps_num, "czas od startu nagrania [s]")
    fits.writeto(outdir / name, bg.astype(np.float32), hdr, overwrite=True)
    return {"block": int(block), "frame": int(frame), "tau_s": frame * meta.fps_den / meta.fps_num,
            "file": f"{outdir.name}/{name}"}


def star_psf_sigma(bg: np.ndarray, sigma: np.ndarray, dcfg: dict) -> dict:
    """Typowa szerokość gwiazd (σ osi mniejszej, px) na stacku epoki, tą samą metodą momentów
    co dla detekcji — do stosunku szerokość obiektu ÷ gwiazda (bliski = nieostry)."""
    t = torch.from_numpy(np.ascontiguousarray(bg, dtype=np.float32))[None]
    s = torch.from_numpy(np.ascontiguousarray(sigma, dtype=np.float32))[None]
    local = F.avg_pool2d(t[:, None], (1, 31), stride=1, padding=(0, 15), count_include_pad=False)
    local = F.avg_pool2d(local, (31, 1), stride=1, padding=(15, 0), count_include_pad=False)[:, 0]
    snr = (t - local) / s
    psf = float(dcfg["psf_sigma_px"])
    filt = gaussian_filter(snr, psf) * (2.0 * math.sqrt(math.pi) * psf)   # w jednostkach SNR (jak detekcja)
    cfg = {**dcfg, "snr_seed": float(dcfg["star_thr_sigma"]), "snr_grow": float(dcfg["star_thr_sigma"]) / 2,
           "max_components_per_frame": 10 ** 7}
    comps, _ = frame_components(filt[0], snr[0], t[0] - local[0], cfg, _label_scipy)
    npix = comps[:, 5] if len(comps) else np.empty(0)
    ok = (npix >= 4) & (npix <= 400)
    widths = []
    for cxx, cxy, cyy in comps[ok][:, 6:9]:
        ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
        widths.append(math.sqrt(max(ev[0], 0.0)))
    return {"star_sigma_px": float(np.median(widths)) if widths else None, "n_stars": len(widths)}
