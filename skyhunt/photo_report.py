"""Raport sesji zdjęć RAW: mapa nieba na głębokim stosie z konstelacjami i kreskami obiektów,
kształt gwiazd w rogach, przebieg sesji (rytm serii, tło, szum, ruch statywu), podgląd koloru,
czas z satelitów (Δ, przerwa w serii, odczyt migawki), tabela obiektów i strona na obiekt.
Fotometria i błyski dochodzą w F3."""
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


def flatten(img: np.ndarray, box: int = 48) -> np.ndarray:
    """Odjęcie tła wielkoskalowego (winietowanie f/1.0 + łuna miasta): mediana w blokach ``box``
    px, wygładzona i powiększona do pełnego obrazu. Do pokazania gwiazd równo w całym kadrze."""
    from scipy.ndimage import median_filter, zoom

    a = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    h, w = a.shape
    hb, wb = h // box, w // box
    blocks = np.median(a[:hb * box, :wb * box].reshape(hb, box, wb, box), axis=(1, 3))
    blocks = median_filter(blocks, size=3, mode="nearest")
    bg = zoom(blocks, (h / hb, w / wb), order=1)[:h, :w]
    if bg.shape != a.shape:
        bg = np.pad(bg, ((0, h - bg.shape[0]), (0, w - bg.shape[1])), mode="edge")
    return a - bg


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
        flat = flatten(img)
        st = sky_stretch(flat)
        draw_sky(ax, st, wcs, (0, st.shape[1], 0, st.shape[0]), rcfg, always, max_px=2100)
        objs = objects(outdir, wcs)
        draw_objects(ax, objs, labels=True)
        fig.text(0.02, 0.03, "stos wyrównany do obrotu nieba i ruchu statywu, bez kresek (odrzucone jasne piksele "
                             "przejściowe); tło wielkoskalowe (winietowanie, łuna) odjęte; niebieskie linie: konstelacje; kreski: "
                             "zielone — satelity (NORAD), czerwone — niezidentyfikowane, pomarańczowe — samoloty (ADS-B)",
                 fontsize=8)
        pdf.savefig(fig)

        # wycinki w pełnej skali stosu: środek i rogi — ocena kształtu gwiazd (koma, ruch, ostrość)
        fig = Figure(figsize=A4)
        fig.suptitle(f"Kształt gwiazd w stosie ({deep}): środek i rogi, 160×160 px (superpiksele 3×3)",
                     x=0.02, ha="left", fontsize=12)
        h, w = flat.shape
        s = 160
        spots = {"lewy górny": (0, 0), "prawy górny": (0, w - s), "środek": (h // 2 - s // 2, w // 2 - s // 2),
                 "lewy dolny": (h - s, 0), "prawy dolny": (h - s, w - s)}
        pos = {"lewy górny": (0.04, 0.52), "prawy górny": (0.66, 0.52), "środek": (0.35, 0.30),
               "lewy dolny": (0.04, 0.06), "prawy dolny": (0.66, 0.06)}
        for name, (y0, x0) in spots.items():
            cut = flat[y0:y0 + s, x0:x0 + s]
            a = fig.add_axes([*pos[name], 0.3, 0.4])
            a.imshow(sky_stretch(cut), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a.set_title(name, fontsize=8)
            a.set_xticks([])
            a.set_yticks([])
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
            *(time_lines(read_json(outdir / "time_sync.json"))[:2] if (outdir / "time_sync.json").exists()
              else ["Właściwy czas (poprawka zegara Δ z satelitów): brak etapu identify"]),
            "",
            f"Astrometria: {wcsinfo.get('n_solved', 0)}/{wcsinfo.get('n_epochs', 0)} zdjęć rozwiązanych, "
            f"zgodność {wcsinfo.get('epoch_rms_px', float('nan')):.2f} px (superpiksele)",
            f"Pole: {fov.get('fov_w_deg', float('nan')):.2f}° × {fov.get('fov_h_deg', float('nan')):.2f}°, "
            f"skala {fov.get('scale_arcsec_px', float('nan')):.1f}″/px (superpiksel 3×3)",
            "Stosy: " + ", ".join(f"{c}: {v['frames']} zdj." for c, v in classes.items()),
        ] + [f"UWAGA: {w}" for w in sess.get("warnings", [])]
        fig.text(0.02, 0.92, "\n".join(lines), family="monospace", fontsize=8, va="top")
        panels = [("bg_median", "tło (mediana) [DN·9]"), ("noise_mad", "szum σ [DN·9]"),
                  ("drift", "ruch aparatu z plate solve (statyw) [px]"),
                  ("rejected_frac", "poza kadrem odniesienia + odrzucone kreski (ułamek pikseli)")]
        drift = proc.get("pointing_drift") or []
        for k, (col, label) in enumerate(panels):
            a = fig.add_axes([0.07 + (k % 2) * 0.48, 0.37 - (k // 2) * 0.3, 0.4, 0.22])
            if col == "drift":
                if drift:
                    t = [d["tau_s"] for d in drift]
                    a.plot(t, [d["dx_px"] for d in drift], "o-", ms=3, lw=0.8, label="x")
                    a.plot(t, [d["dy_px"] for d in drift], "s-", ms=3, lw=0.8, label="y")
                    a.axhline(0, color="k", lw=0.5)
                    a.legend(fontsize=6)
                    label += f" — max {proc.get('pointing_drift_max_px', 0):.2f} px, uwzględniony w stosie"
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
        if objs:
            satellite_pages(pdf, Figure, outdir, objs, flat, wcs, cfg, proc)
    return ["report/summary.pdf"]


# ---------------------------------------------------------------- obiekty (F2)

C_AIR = "#ff9f1a"


def objects(outdir: Path, wcs) -> list[dict]:
    """Obiekty z etapu identify (+ ADS-B): kreski w układzie stosu, predykcja z TLE, etykieta."""
    import pandas as pd

    from .report import C_SAT, C_UNID
    from .sky import _world2pix

    f = outdir / "tracks_final.parquet"
    if not f.exists():
        return []
    final = pd.read_parquet(f)
    if not len(final):
        return []
    st = pd.read_parquet(outdir / "streaks.parquet").set_index("streak_id")
    chn = pd.read_parquet(outdir / "chains.parquet")
    ids = read_json(outdir / "identifications.json") if (outdir / "identifications.json").exists() else {}
    air_f = outdir / "adsb_matches.csv"
    air = pd.read_csv(air_f) if air_f.exists() else pd.DataFrame(columns=["track_id"])
    air_by = {int(r["track_id"]): r for _, r in air.iterrows()}
    out = []
    for _, t in final.sort_values("tau0").iterrows():
        tid = int(t["track_id"])
        sids = chn.loc[chn["chain_id"] == tid - 1].sort_values("pos")["streak_id"].astype(int).tolist()
        segs = st.loc[sids].reset_index()
        a = air_by.get(tid)
        kind = "air" if (a is not None and t["kind"] != "sat") else str(t["kind"])
        pred = None
        best = (ids.get(str(tid)) or [{}])[0]
        if t["kind"] == "sat" and best.get("pred_radec"):
            pr = np.asarray(best["pred_radec"], float)
            px, py = _world2pix(wcs, pr[:, 0], pr[:, 1])
            pred = np.column_stack([px, py])
        if kind == "sat":
            label = f"#{tid} {t['sat_name']}"
        elif kind == "air":
            reg = a.get("reg")
            label = f"#{tid} {reg if isinstance(reg, str) else a.get('icao')}"
        else:
            label = f"#{tid} {t['class_hint']}"
        out.append({"track_id": tid, "row": t, "segs": segs, "kind": kind, "pred": pred, "label": label, "air": a,
                    "color": {"sat": C_SAT, "air": C_AIR}.get(kind, C_UNID)})
    return out


def draw_objects(ax, objs: list[dict], labels: bool = True, lw: float = 1.6) -> None:
    from .report import C_PRED

    for o in objs:
        for s in o["segs"].itertuples():
            ax.plot([s.xa, s.xb], [s.ya, s.yb], color=o["color"], lw=lw, solid_capstyle="butt")
        if o["pred"] is not None and len(o["pred"]) >= 2:
            ax.plot(o["pred"][:, 0], o["pred"][:, 1], "--", color=C_PRED, lw=0.7)
        if labels and len(o["segs"]):
            s = o["segs"].iloc[0]
            ax.text(s["xa"] + 8, s["ya"] - 8, o["label"], color=o["color"], fontsize=6, clip_on=True)


def _utc(s) -> str:
    from .report import fmt_utc

    return fmt_utc(datetime.fromisoformat(str(s)), 2) if isinstance(s, str) and s else "–"


def _f(v, spec: str, unit: str = "") -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    return f"{v:{spec}}{unit}" if np.isfinite(v) else "–"


def time_lines(sync: dict) -> list[str]:
    """Opis czasu sesji: Δ z satelitów, rytm, przerwa g, odczyt migawki, kontrola z satelitów."""
    pt = sync.get("photo_timing") or {}
    gap, chk, cad = pt.get("gap") or {}, pt.get("satellite_check") or {}, pt.get("cadence") or {}
    lines = []
    if sync.get("synced"):
        lines.append(f"Poprawka zegara Δ = {sync['delta_s']:+.3f} ± {sync['sigma_s']:.3f} s   pewność: "
                     f"{sync['confidence']}   metoda: {sync['method']}")
    else:
        lines.append("Czas NIEZSYNCHRONIZOWANY z satelitami — czas z EXIF (±0,5 s) i zegara aparatu")
    lines.append(f"Start sesji: EXIF {_utc(sync.get('start_utc_prior'))} → po synchronizacji "
                 f"{_utc(sync.get('start_utc_synced'))}")
    ref = sync.get("reference")
    if ref:
        lines.append(f"Satelita odniesienia: NORAD {ref['norad']} {ref['name']} (δ = {ref['delta_s']:+.3f} s, "
                     f"residuum {ref['rms_deg'] * 3600:.0f}″)")
    if cad:
        lines.append(f"Rytm serii z EXIF: P = {_f(cad.get('period'), '.4f', ' s')} (faza w obrębie sekundy → Δ)")
    if gap.get("fitted"):
        lines.append(f"Przerwa między zdjęciami serii (geometria {gap['n_chains']} łańcuchów): g = {gap['g_s']:.3f} ± "
                     f"{gap['sigma_s']:.3f} s; residua końców {_f(gap.get('rms_px'), '.2f', ' px')} = "
                     f"{_f(gap.get('rms_ms'), '.1f', ' ms')}; "
                     + ("użyta do czasu kresek" if pt.get("gap_from_geometry") else "NIEużyta (za duża niepewność)"))
    else:
        lines.append(f"Przerwa między zdjęciami serii: bez pomiaru ({gap.get('note', 'za mało łańcuchów')}), "
                     f"przyjęto {_f(pt.get('gap_used_s'), '.3f', ' s')}")
    lines.append(f"Odczyt migawki elektronicznej przyjęty: {_f(pt.get('rolling_shutter_s'), '.3f', ' s')} "
                 f"(streaks.link.rolling_shutter_s)")
    if chk.get("n_tracks"):
        lines.append(f"Kontrola z satelitów ({chk['n_tracks']} torów): residua czasu końców "
                     f"{_f(chk.get('rms_ms'), '.1f', ' ms')}; poprawka odczytu migawki "
                     f"{_f(chk.get('rolling_corr_s'), '+.3f')} ± {_f(chk.get('rolling_sigma_s'), '.3f', ' s')}; "
                     f"poprawka g {_f(chk.get('gap_corr_s'), '+.3f')} ± {_f(chk.get('gap_sigma_s'), '.3f', ' s')}")
    off = sync.get("observer_offset")
    if off:
        lines.append(f"Paralaksa satelitów: obserwator przesunięty o {off['north_km']:.2f} km N, "
                     f"{off['east_km']:.2f} km E ({off['n_tracks']} torów)")
    lines.append(f"Elementy orbit: {sync.get('catalog_note', '–')}")
    return lines


def _sunlit_text(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "oświetlenie ?"
    return "oświetlony" if bool(v) else "w cieniu Ziemi"


def satellite_pages(pdf, Figure, outdir: Path, objs: list[dict], flat: np.ndarray, wcs, cfg: dict,
                    proc: dict) -> None:
    """Strony F2: czas z satelitów, tabela obiektów, strona na obiekt (tor na stosie + wycinki
    z kolejnych zdjęć w ich własnych pikselach, z cache etapu process)."""
    from .report import _region, _table_pages, sky_stretch

    sync = read_json(outdir / "time_sync.json")
    fig = Figure(figsize=A4)
    fig.suptitle("Czas z satelitów i model czasu zdjęć", x=0.02, ha="left", fontsize=13)
    lines = time_lines(sync)
    members = sync.get("members") or []
    if members:
        lines += ["", "Satelity potwierdzające Δ:"] + [
            f"  NORAD {m['norad']:>6} {str(m['name'])[:28]:<28} δ = {m['delta_s']:+.3f} s  "
            f"residuum {m['rms_deg'] * 3600:5.0f}″  prędkość ×{m['speed_ratio']:.3f}" for m in members[:20]]
    fig.text(0.02, 0.92, "\n".join(lines), family="monospace", fontsize=7.5, va="top")
    pdf.savefig(fig)

    rows = []
    for o in objs:
        t, a = o["row"], o["air"]
        if o["kind"] == "sat":
            who, name, why = f"{int(t['norad'])}", str(t["sat_name"]), str(t["match_reason"])
        elif o["kind"] == "air":
            who = str(a.get("callsign") if isinstance(a.get("callsign"), str) else a.get("icao"))
            typ = a.get("type") if isinstance(a.get("type"), str) else ""
            name, why = f"{typ} {_f(a.get('alt_m'), '.0f', ' m')}", str(t["class_reason"])
        else:
            who, name, why = "–", "", str(t["class_reason"])
        rows.append([f"#{o['track_id']}", {"sat": "satelita", "air": "samolot (ADS-B)"}.get(o["kind"], t["class_hint"]),
                     who, name[:26], _utc(t["utc_start"]), _utc(t["utc_end"]), int(t["n_streaks"]),
                     _f(t["omega_deg_s"], ".2f"), str(t["confidence"] or "–"), why[:70]])
    _table_pages(pdf, Figure, "Obiekty — kreski na zdjęciach",
                 ["#", "klasa", "NORAD / lot", "nazwa", "UTC początek", "UTC koniec", "kresek", "°/s", "pewność",
                  "uzasadnienie"], rows)

    cache = Path(proc["cache_dir"]) if proc.get("cache_dir") else None
    st_full = sky_stretch(flat)
    for o in objs[: int(cfg["report"].get("photo_object_pages", 60))]:
        t = o["row"]
        fig = Figure(figsize=A4)
        fig.suptitle(f"#{o['track_id']} — " + o["label"].split(" ", 1)[-1], x=0.02, ha="left", fontsize=12)
        info = [f"UTC {_utc(t['utc_start'])} → {_utc(t['utc_end'])}   ({t['dur_s']:.1f} s, {int(t['n_streaks'])} kresek, "
                f"{t['omega_deg_s']:.2f}°/s)",
                f"Az/Alt {t['az0']:.1f}°/{t['alt0']:.1f}° → {t['az1']:.1f}°/{t['alt1']:.1f}°   RA/Dec {t['ra0']:.3f} "
                f"{t['dec0']:+.3f} → {t['ra1']:.3f} {t['dec1']:+.3f}",
                f"S/N kresek (mediana) {_f(t['peak_snr_median'], '.0f')}   {t['class_reason']}"]
        if o["kind"] == "sat":
            info.append(f"NORAD {int(t['norad'])} {t['sat_name']}: {t['match_reason']} (pewność {t['confidence']}, "
                        f"{_sunlit_text(t['sunlit'])})")
        if bool(t.get("dir_ambiguous", False)):
            info.append("Kierunek lotu nieznany (jedna kreska) — czasy końców mogą być zamienione")
        fig.text(0.02, 0.93, "\n".join(info), fontsize=7.5, va="top")
        segs = o["segs"]
        xy = np.vstack([segs[["xa", "ya"]].to_numpy(), segs[["xb", "yb"]].to_numpy()])
        ax = fig.add_axes([0.02, 0.05, 0.5, 0.72])
        x0, x1, y0, y1 = _region(xy, flat.shape, 40, 0.5 / 0.72 * A4[0] / A4[1])
        ax.imshow(st_full[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                  extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        draw_objects(ax, [o], labels=False, lw=1.0)
        ax.plot(segs["xa"].iloc[0], segs["ya"].iloc[0], "o", color=o["color"], ms=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("tor na stosie (układ nieba); przerywana: predykcja z elementów orbit", fontsize=7)
        pick = segs.iloc[np.unique(np.linspace(0, len(segs) - 1, min(len(segs), 6)).astype(int))]
        for k, (_, s) in enumerate(pick.iterrows()):
            f = cache / f"lum_{int(s['photo']):05d}.npy" if cache else None
            a = fig.add_axes([0.55 + (k % 2) * 0.22, 0.55 - (k // 2) * 0.25, 0.2, 0.2])
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{s['file']} ({s['ev']}, {s['exposure_s']:g} s)", fontsize=6)
            if f is None or not f.exists():
                a.text(0.5, 0.5, "brak cache\nzdjęcia", ha="center", va="center", fontsize=6, transform=a.transAxes)
                continue
            img = np.load(f).astype(np.float32)
            xs, ys = [s["xa_p"], s["xb_p"]], [s["ya_p"], s["yb_p"]]
            half = max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0])) / 2 + 12
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            bx0, bx1 = int(max(cx - half, 0)), int(min(cx + half, img.shape[1]))
            by0, by1 = int(max(cy - half, 0)), int(min(cy + half, img.shape[0]))
            cut = img[by0:by1, bx0:bx1]
            if cut.size == 0:
                continue
            a.imshow(sky_stretch(cut), cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                     extent=(bx0 - 0.5, bx1 - 0.5, by1 - 0.5, by0 - 0.5))
            a.plot(xs, ys, color=o["color"], lw=0.6, alpha=0.5)
            a.set_xlim(bx0, bx1)
            a.set_ylim(by1, by0)
        pdf.savefig(fig)
