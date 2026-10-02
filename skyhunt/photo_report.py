"""Raport sesji zdjęć RAW (F1): mapa nieba na głębokim stosie z konstelacjami, przebieg sesji
(rytm serii, tło, szum, dryf nieba, odrzucone piksele) i podgląd koloru ze stosu RGB.
Kreski satelitów, NORAD i fotometria dochodzą w F2–F3."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .io import read_json
from .report import A4, _mpl, draw_sky, fmt_utc, sky_stretch

C_EV = {"ev0": "#1f77b4", "ev+1": "#2ca02c", "ev-1": "#d62728"}


def rgb_preview(rgb: np.ndarray) -> np.ndarray:
    """Stos RGB → obraz 0…1: odjęcie tła (mediana) per kanał, wspólne rozciągnięcie asinh,
    balans tak, by tło było szare (kolor gwiazd zostaje względny)."""
    out = np.empty_like(rgb)
    s = rgb[::4, ::4]
    for c in range(3):
        ch = s[..., c][np.isfinite(s[..., c])]
        med = float(np.median(ch)) if len(ch) else 0.0
        sig = 1.4826 * float(np.median(np.abs(ch - med))) if len(ch) else 1.0
        out[..., c] = (rgb[..., c] - med) / max(sig, 1e-6)
    hi = float(np.nanpercentile(out[::4, ::4], 99.8))
    x = np.clip(np.nan_to_num(out) / max(hi, 1e-6), 0, None)
    return np.clip(np.arcsinh(8 * x) / np.arcsinh(8), 0, 1)


def build(outdir: Path, folder: Path, cfg: dict) -> list[str]:
    import pandas as pd
    from astropy.io import fits
    from matplotlib.backends.backend_pdf import PdfPages

    from .astrometry import load_wcs

    Figure = _mpl()
    rep = outdir / "report"
    rep.mkdir(exist_ok=True)
    meta = read_json(outdir / "meta.json")
    sess, prior = meta["session"], meta["time_prior"]
    proc = read_json(outdir / "process.json")
    wcsinfo = read_json(outdir / "wcs.json")
    wcs = load_wcs(outdir / wcsinfo["reference_wcs"])
    stats = pd.read_csv(outdir / "frame_stats.csv")
    classes = proc["classes"]
    deep = "ev+1" if "ev+1" in classes else next(iter(classes))
    img = fits.getdata(outdir / classes[deep]["file"]).astype(np.float32)
    rcfg = cfg["report"]
    hint = cfg.get("astrometry", {}).get("hint_star")
    always = list(rcfg.get("always_label") or []) + ([hint] if hint else [])
    start = datetime.fromisoformat(prior["start_utc"])
    cad = sess["cadence"]
    fov = wcsinfo.get("fov") or {}

    path = rep / "summary.pdf"
    with PdfPages(path) as pdf:
        fig = Figure(figsize=A4)
        fig.suptitle(f"skyhunt — {folder.name} — stos {classes[deep]['frames']} zdjęć "
                     f"({classes[deep]['exposure_s']:g} s, {deep})", x=0.02, ha="left", fontsize=13)
        ax = fig.add_axes([0.02, 0.1, 0.96, 0.82])
        st = sky_stretch(np.nan_to_num(img, nan=float(np.nanmedian(img))))
        draw_sky(ax, st, wcs, (0, st.shape[1], 0, st.shape[0]), rcfg, always, max_px=2100)
        fig.text(0.02, 0.03, "stos wyrównany modelem nieruchomej kamery (obrót nieba), odrzucone jasne piksele "
                             "przejściowe (kreski); niebieskie linie: konstelacje", fontsize=8)
        pdf.savefig(fig)

        fig = Figure(figsize=A4)
        fig.suptitle("Sesja zdjęć: czas, rytm, jakość", x=0.02, ha="left", fontsize=13)
        lines = [
            f"Folder: {folder.name}   zdjęć: {sess['n_photos']}   serii: {sess['n_sets']}   aparat: {sess.get('model') or '–'}",
            f"Klasy (czas naświetlania): " + ", ".join(f"{c} {v}" for c, v in sess["classes"].items()),
            f"ISO {sess['iso']}, f/{sess['fnumber']}",
            f"Start (EXIF + rytm, czas a priori): {fmt_utc(start)}   koniec ≈ "
            f"{fmt_utc(start + timedelta(seconds=float(sess['duration_s'])))}",
            f"Rytm serii: P = {cad['period']:.4f} s, start serii 0 ±{cad['t0_halfwidth']:.3f} s "
            f"({'regularny' if cad['regular'] else 'NIEREGULARNY'}; EXIF {'z' if sess['exif_subsec'] else 'bez'} "
            f"ułamków sekundy)",
            "Właściwy czas (poprawka zegara Δ z satelitów) — w F2.",
            "",
            f"Astrometria: {wcsinfo.get('n_solved', 0)}/{wcsinfo.get('n_epochs', 0)} zdjęć rozwiązanych, "
            f"zgodność {wcsinfo.get('epoch_rms_px', float('nan')):.2f} px (superpiksele)",
            f"Pole: {fov.get('fov_w_deg', float('nan')):.2f}° × {fov.get('fov_h_deg', float('nan')):.2f}°, "
            f"skala {fov.get('scale_arcsec_px', float('nan')):.1f}″/px (superpiksel 3×3)",
            "Stosy: " + ", ".join(f"{c}: {v['frames']} zdj." for c, v in classes.items()),
        ] + [f"UWAGA: {w}" for w in sess.get("warnings", [])]
        fig.text(0.02, 0.92, "\n".join(lines), family="monospace", fontsize=8, va="top")
        panels = [("bg_median", "tło (mediana) [DN·9]"), ("noise_mad", "szum σ [DN·9]"),
                  ("shift_px", "dryf nieba względem odniesienia [px]"), ("rejected_frac", "odrzucone piksele")]
        for k, (col, label) in enumerate(panels):
            a = fig.add_axes([0.07 + (k % 2) * 0.48, 0.37 - (k // 2) * 0.3, 0.4, 0.22])
            for c, g in stats.groupby("ev"):
                if col in g:
                    a.plot(g["tau_mid"], g[col], ".", ms=2, color=C_EV.get(c, "k"), label=c)
            a.set_xlabel("czas od początku sesji [s]", fontsize=7)
            a.set_title(label, fontsize=8)
            a.tick_params(labelsize=7)
            if k == 0:
                a.legend(fontsize=6)
        pdf.savefig(fig)

        rgb_file = outdir / "stack_rgb_ev0.npy"
        if rgb_file.exists():
            fig = Figure(figsize=A4)
            fig.suptitle("Kolor ze stosu RGB (ev0, superpiksele X-Trans, bez demozaikowania)", x=0.02, ha="left",
                         fontsize=13)
            ax = fig.add_axes([0.02, 0.06, 0.96, 0.86])
            ax.imshow(rgb_preview(np.load(rgb_file)), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            pdf.savefig(fig)
    return ["report/summary.pdf"]
