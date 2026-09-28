"""Raport PDF: strona zbiorcza oraz PDF dla każdego obiektu.

- zidentyfikowany satelita: tło gwiazd z konstelacjami, tor (zielony), predykcja z elementów
  orbit (przerywana), dane NORAD, pasek klatek ≥ 1 s, klip MP4 obok i w załączniku PDF;
- obiekt niezidentyfikowany: tor na czerwono, prędkość kątowa, czas przelotu, początek
  i koniec w UTC, podpowiedź klasy, lista niesprawdzalnych hipotez.

Wejście: pliki z etapów ``detect``/``tracks``/``astrometry``/``identify`` w katalogu wyników.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .clips import extract_frames, filmstrip, plan_clip, positions_at, stretch_stack, write_annotated_mp4
from .io import read_json
from .metadata import VideoMeta
from .sky import _world2pix, constellation_overlay, named_stars_overlay, project_polyline, field_center

log = logging.getLogger("skyhunt")

C_SAT, C_UNID = "#2ecc40", "#ff2a2a"
C_PRED, C_CONST, C_STAR, C_GRID = "#00d5ff", "#7fb2ff", "#ffd84d", "#5a6a80"
A4 = (11.69, 8.27)
CONF_RANK = {"high": 3, "medium": 2, "low": 1, "": 0, None: 0}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    from matplotlib.figure import Figure

    return Figure


def fmt_utc(dt: datetime | None, digits: int = 2) -> str:
    if dt is None:
        return "–"
    dt = dt + timedelta(microseconds=5 * 10 ** (5 - digits))   # zaokrąglenie z przeniesieniem sekundy
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    frac = f".{dt.microsecond // 10 ** (6 - digits):0{digits}d}" if digits else ""
    return f"{s}{frac} UTC"


def sky_stretch(img: np.ndarray) -> np.ndarray:
    s = img[::4, ::4]
    med, hi = np.median(s), np.percentile(s, 99.7)
    x = np.clip((img - med) / max(hi - med, 1e-6), 0, None)
    return np.clip(np.arcsinh(10 * x) / np.arcsinh(10), 0, 1)


def grid_overlay(wcs, shape, step_deg: float) -> list[np.ndarray]:
    """Linie stałego RA i Dec co ``step_deg`` w pikselach."""
    center, radius = field_center(wcs, shape)
    from .sky import vec_to_radec

    ra_c, dec_c = vec_to_radec(center)
    lines = []
    lim = radius + 10
    decs = np.arange(math.floor((dec_c - lim) / step_deg) * step_deg, dec_c + lim + step_deg, step_deg)
    for d in decs:
        if abs(d) >= 89.9:
            continue
        ras = np.arange(0, 360.5, 1.0)
        lines += project_polyline(wcs, shape, np.column_stack([ras, np.full_like(ras, d)]), center=center,
                                  densify_deg=1.0, max_sep_deg=lim)
    for r in np.arange(0, 360, step_deg):
        ds = np.arange(-89, 89.5, 1.0)
        lines += project_polyline(wcs, shape, np.column_stack([np.full_like(ds, r), ds]), center=center,
                                  densify_deg=1.0, max_sep_deg=lim)
    return lines


def _region(xy: np.ndarray, shape, margin: float, aspect: float) -> tuple[int, int, int, int]:
    H, W = shape
    x0, x1 = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    y0, y1 = xy[:, 1].min() - margin, xy[:, 1].max() + margin
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        c, w = (x0 + x1) / 2, h * aspect
        x0, x1 = c - w / 2, c + w / 2
    else:
        c, h = (y0 + y1) / 2, w / aspect
        y0, y1 = c - h / 2, c + h / 2
    w, h = min(x1 - x0, W), min(y1 - y0, H)
    x0 = min(max(0, x0), W - w)
    y0 = min(max(0, y0), H - h)
    return int(x0), int(x0 + w), int(y0), int(y0 + h)


def draw_sky(ax, stretched: np.ndarray, wcs, region, rcfg: dict, always: list[str], *, labels=True,
             max_px: int = 1600, lw: float = 0.9, grid=True):
    x0, x1, y0, y1 = region
    ds = max(1, int(math.ceil((x1 - x0) / max_px)))
    ax.imshow(stretched[y0:y1:ds, x0:x1:ds], cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
    shape = stretched.shape
    if grid:
        for p in grid_overlay(wcs, shape, float(rcfg["grid_step_deg"])):
            ax.plot(p[:, 0], p[:, 1], color=C_GRID, lw=0.4, alpha=0.6)
    for c in constellation_overlay(wcs, shape, float(rcfg["densify_deg"])):
        for p in c["pieces"]:
            ax.plot(p[:, 0], p[:, 1], color=C_CONST, lw=lw, alpha=0.75)
        lx, ly = c["label_xy"]
        if labels and x0 <= lx < x1 and y0 <= ly < y1:
            ax.text(lx, ly, c["name"], color=C_CONST, fontsize=10, ha="center", alpha=0.95, clip_on=True)
    if labels:
        for s in named_stars_overlay(wcs, shape, float(rcfg["star_label_mag"]), always):
            if x0 <= s["x"] < x1 and y0 <= s["y"] < y1:
                ax.text(s["x"] + 12, s["y"] - 12, s["name"], color=C_STAR, fontsize=8, clip_on=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")


def _draw_path(ax, obj: dict, lw=2.0):
    xy = obj["path_xy"]
    ax.plot(xy[:, 0], xy[:, 1], color=obj["color"], lw=lw, solid_capstyle="round")
    ax.plot(*xy[0], "o", color=obj["color"], ms=5)
    if len(xy) >= 2:
        ax.annotate("", xy=xy[-1], xytext=xy[-2] + (xy[-2] - xy[-1]) * 0.0,
                    arrowprops={"arrowstyle": "-|>", "color": obj["color"], "lw": lw})
    if obj.get("ticks_xy") is not None and len(obj["ticks_xy"]):
        t = obj["ticks_xy"]
        ax.plot(t[:, 0], t[:, 1], "|", color="white", ms=9, mew=1.2)
    if obj.get("pred_xy") is not None and len(obj["pred_xy"]) >= 2:
        p = obj["pred_xy"]
        ax.plot(p[:, 0], p[:, 1], "--", color=C_PRED, lw=1.0)


def object_pdf(path: Path, obj: dict, rcfg: dict, always: list[str]) -> None:
    """PDF obiektu: strona z mapą i parametrami, strona z paskiem klatek."""
    Figure = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Circle

    shape = obj["stretched"].shape
    with PdfPages(path) as pdf:
        fig = Figure(figsize=A4)
        fig.suptitle(obj["title"], color="black", fontsize=13, x=0.02, ha="left", y=0.975)
        ax = fig.add_axes([0.02, 0.05, 0.62, 0.86])
        region = _region(obj["path_xy"], shape, float(rcfg["zoom_margin_px"]), (0.62 * A4[0]) / (0.86 * A4[1]))
        draw_sky(ax, obj["stretched"], obj["wcs"], region, rcfg, always)
        _draw_path(ax, obj)
        inset = fig.add_axes([0.66, 0.66, 0.32, 0.32 * A4[0] / A4[1] * shape[0] / shape[1]])
        draw_sky(inset, obj["stretched"], obj["wcs"], (0, shape[1], 0, shape[0]), rcfg, always, labels=False,
                 max_px=960, lw=0.6, grid=False)
        _draw_path(inset, obj, lw=1.2)
        x0, x1, y0, y1 = region
        inset.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="white", lw=0.6)
        text = "\n".join(f"{k:<26}{v}" for k, v in obj["params"])
        fig.text(0.66, 0.62, text, family="monospace", fontsize=7.6, va="top")
        if obj.get("notes"):
            fig.text(0.66, 0.16, "\n".join(obj["notes"]), fontsize=7, va="top", wrap=True)
        pdf.savefig(fig)

        panels = obj.get("panels") or []
        if panels:
            fig = Figure(figsize=A4)
            span = (panels[-1]["frame"] - panels[0]["frame"]) / obj["fps"]
            fig.suptitle(f"{obj['title']} — {len(panels)} klatek z {span:.2f} s nagrania "
                         f"(pełny klip: {obj.get('clip_name', '–')})", fontsize=11, x=0.02, ha="left")
            cols = 4
            rows = int(math.ceil(len(panels) / cols))
            stack = np.stack([p["img"] for p in panels]).astype(np.float32)
            lo, hi = np.percentile(stack, [1, 99.9])
            for i, p in enumerate(panels):
                a = fig.add_subplot(rows, cols, i + 1)
                a.imshow(p["img"], cmap="gray", vmin=lo, vmax=max(hi, lo + 1), interpolation="nearest")
                h, w = p["img"].shape
                if p["inside"]:
                    a.add_patch(Circle((w / 2 - 0.5, h / 2 - 0.5), radius=w / 8, fill=False,
                                       color=obj["color"], lw=1.0))
                a.set_title(f"kl. {p['frame']}  {p['label']}", fontsize=7)
                a.set_xticks([])
                a.set_yticks([])
            pdf.savefig(fig)


def attach_file(pdf_path: Path, file_path: Path) -> None:
    from pypdf import PdfWriter

    w = PdfWriter(clone_from=str(pdf_path))
    w.add_attachment(file_path.name, file_path.read_bytes())
    with open(pdf_path, "wb") as fh:
        w.write(fh)


def _table_pages(pdf, Figure, title: str, header: list[str], rows: list[list], per_page: int = 28) -> None:
    for p0 in range(0, max(len(rows), 1), per_page):
        fig = Figure(figsize=A4)
        fig.suptitle(title + (f" (str. {p0 // per_page + 1})" if len(rows) > per_page else ""), x=0.02,
                     ha="left", fontsize=12)
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.88])
        ax.axis("off")
        chunk = rows[p0:p0 + per_page] or [["–"] * len(header)]
        tbl = ax.table(cellText=chunk, colLabels=header, loc="upper center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        tbl.scale(1, 1.25)
        pdf.savefig(fig)


def summary_pdf(path: Path, s: dict, rcfg: dict, always: list[str]) -> None:
    Figure = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        fig = Figure(figsize=A4)
        fig.suptitle(s["title"], x=0.02, ha="left", fontsize=13)
        ax = fig.add_axes([0.02, 0.1, 0.96, 0.82])
        if s.get("stretched") is not None:
            shape = s["stretched"].shape
            draw_sky(ax, s["stretched"], s["wcs"], (0, shape[1], 0, shape[0]), rcfg, always, max_px=1920)
            for o in s["objects"]:
                _draw_path(ax, o, lw=1.4)
                ax.text(o["path_xy"][0, 0] + 15, o["path_xy"][0, 1] + 15, o["short"], color=o["color"], fontsize=7)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "brak rozwiązania astrometrycznego — mapa niedostępna", ha="center")
        fig.text(0.02, 0.03, s["legend"], fontsize=8)
        pdf.savefig(fig)

        fig = Figure(figsize=A4)
        fig.suptitle("Synchronizacja czasu, astrometria, katalog", x=0.02, ha="left", fontsize=13)
        fig.text(0.02, 0.92, "\n".join(s["info_lines"]), family="monospace", fontsize=8, va="top")
        members = s.get("sync_members") or []
        if members:
            ax = fig.add_axes([0.58, 0.45, 0.38, 0.4])
            d = np.array([m["delta_s"] for m in members]) - s["delta_s"]
            ax.errorbar(d, np.arange(len(d)), fmt="o", color=C_SAT)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks(np.arange(len(d)), [f"NORAD {m['norad']}" for m in members], fontsize=7)
            ax.set_xlabel("δ_j − Δ [s]")
            ax.set_title("zgodność poprawki czasu między satelitami", fontsize=9)
        pdf.savefig(fig)

        _table_pages(pdf, Figure, "Tory", s["table_header"], s["table_rows"])
        _table_pages(pdf, Figure, "Satelity przewidziane w kadrze (katalog), a niewykryte",
                     s["pred_header"], s["pred_rows"])
        fig = Figure(figsize=A4)
        fig.suptitle("Ograniczenia i kontrola fałszywych alarmów", x=0.02, ha="left", fontsize=13)
        fig.text(0.02, 0.92, "\n".join(s["notes"]), fontsize=9, va="top")
        pdf.savefig(fig)


# ---------------------------------------------------------------- składanie raportu

def _fmt(v, spec: str, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:{spec}}{unit}"


def build(outdir: Path, video: Path, meta: VideoMeta, cfg: dict) -> list[str]:
    """Wszystkie PDF-y i klipy dla pliku. Zwraca ścieżki wyników względem ``outdir``."""
    import pandas as pd
    from astropy.io import fits

    from .astrometry import load_wcs

    rcfg = cfg["report"]
    rep = outdir / "report"
    (rep / "objects").mkdir(parents=True, exist_ok=True)
    (rep / "clips").mkdir(parents=True, exist_ok=True)
    tracks = pd.read_parquet(outdir / "tracks_final.parquet")
    pts = pd.read_parquet(outdir / "track_sky.parquet")
    sync = read_json(outdir / "time_sync.json")
    ident = read_json(outdir / "identifications.json")
    wcsinfo = read_json(outdir / "wcs.json")
    pred = pd.read_csv(outdir / "fov_predicted.csv") if (outdir / "fov_predicted.csv").exists() else pd.DataFrame()
    hint = cfg.get("astrometry", {}).get("hint_star")
    always = list(rcfg.get("always_label") or []) + ([hint] if hint else [])
    start = datetime.fromisoformat(sync["start_utc_prior"])
    delta = float(sync["delta_s"])

    epochs = [e for e in wcsinfo.get("epochs", []) if e.get("solved")]
    loaded: dict[str, tuple] = {}

    def epoch_for(tau: float):
        if not epochs:
            return None
        e = min(epochs, key=lambda e: abs(e["tau_s"] - tau))
        if e["file"] not in loaded:
            img = fits.getdata(outdir / e["file"]).astype(np.float32)
            loaded.clear()   # trzymamy w pamięci jedną epokę 4K
            loaded[e["file"]] = (sky_stretch(img), load_wcs(outdir / e["wcs"]))
        return loaded[e["file"]]

    outputs, objects = [], []
    min_rank = CONF_RANK[str(rcfg["identified_min_confidence"])]
    for _, t in tracks.iterrows():
        tid = int(t["track_id"])
        tp = pts[pts["track_id"] == tid].sort_values("frame")
        ep = epoch_for(float(t["tau_mid"]))
        best = (ident.get(str(tid)) or [None])[0]
        is_sat = bool(best) and CONF_RANK.get(best.get("confidence"), 0) >= min_rank
        color = C_SAT if is_sat else C_UNID
        t0, t1 = start + timedelta(seconds=float(t["tau0"]) + delta), start + timedelta(seconds=float(t["tau1"]) + delta)
        obj = {"track_id": tid, "color": color, "fps": meta.fps,
               "short": f"{tid}" + (f" #{best['norad']}" if is_sat else "")}
        if ep is not None:
            stretched, wcs = ep
            px, py = _world2pix(wcs, tp["ra"].to_numpy(), tp["dec"].to_numpy())
            obj.update(stretched=stretched, wcs=wcs, path_xy=np.column_stack([px, py]))
            k = np.arange(math.ceil(tp["tau"].min()), math.floor(tp["tau"].max()) + 1)
            if len(k):
                tx, ty = np.interp(k, tp["tau"], px), np.interp(k, tp["tau"], py)
                obj["ticks_xy"] = np.column_stack([tx, ty])
            if is_sat and best.get("pred_radec"):
                pr = np.asarray(best["pred_radec"])
                qx, qy = _world2pix(wcs, pr[:, 0], pr[:, 1])
                obj["pred_xy"] = np.column_stack([qx, qy])
        sigma = f" ± {sync['sigma_s']:.2f} s" if sync.get("synced") else " (czas NIEZSYNCHRONIZOWANY)"
        params = [
            ("Tor", f"#{tid}  ({int(t['n'])} punktów)"),
            ("Początek", fmt_utc(t0) + sigma),
            ("Koniec", fmt_utc(t1)),
            ("Czas przelotu", _fmt(float(t["dur_s"]), ".2f", " s")),
            ("Prędkość kątowa", _fmt(float(t["omega_deg_s"]), ".3f", " °/s")),
            ("Start RA / Dec", f"{_fmt(t['ra0'], '.3f', '°')} / {_fmt(t['dec0'], '+.3f', '°')}"),
            ("Koniec RA / Dec", f"{_fmt(t['ra1'], '.3f', '°')} / {_fmt(t['dec1'], '+.3f', '°')}"),
            ("Start Az / Alt", f"{_fmt(t['az0'], '.2f', '°')} / {_fmt(t['alt0'], '.2f', '°')}"),
            ("Koniec Az / Alt", f"{_fmt(t['az1'], '.2f', '°')} / {_fmt(t['alt1'], '.2f', '°')}"),
            ("Krzywizna (od koła wielk.)", _fmt(float(t["curv_arcsec"]), ".0f", "″")),
            ("Pojawia się / znika", f"{'w kadrze' if t['starts_inside'] else 'na krawędzi'} / "
                                    f"{'w kadrze' if t['ends_inside'] else 'na krawędzi'}"),
        ]
        if is_sat:
            title = f"Satelita NORAD {best['norad']} — {best['name']}  (tor #{tid})"
            params += [
                ("NORAD / COSPAR", f"{best['norad']} / {best['object_id'] or '–'}"),
                ("Nazwa", best["name"]),
                ("Pewność", f"{best['confidence']}  ({best['source']})"),
                ("Epoka elementów", best["epoch"][:19] + " UTC"),
                ("Odległość / wysokość", f"{best['range_km']:.0f} km / {best['height_km']:.0f} km"),
                ("Oświetlony przez Słońce", {True: "tak", False: "NIE (w cieniu Ziemi)", None: "–"}[best.get("sunlit")]),
                ("Residuum poprzeczne", f"{best['cross_deg'] * 3600:.0f}″"),
                ("δ_j − Δ", f"{best['delta_s'] - delta:+.2f} s"),
                ("ω zmierz. / przewidz.", f"{t['omega_deg_s']:.3f} / {best['sat_omega_deg_s']:.3f} °/s"),
            ]
            notes = [best.get("reason", "")]
            name = f"sat_{best['norad']}_t{tid}"
        else:
            title = f"Obiekt niezidentyfikowany — tor #{tid}  ({t['class_hint']})"
            params += [("Podpowiedź klasy", str(t["class_hint"])),
                       ("Uzasadnienie", str(t["class_reason"])[:60]),
                       ("Szerokość ÷ gwiazda", _fmt(float(t["cross_ratio"]), ".2f")),
                       ("Modulacja jasności", _fmt(float(t["f_peak_hz"]), ".2f", " Hz")
                        + (f" (alias {t['f_alias_hz']:.2f} Hz)" if math.isfinite(float(t["f_alias_hz"])) else ""))]
            no_st = "bez Space-Track" in str(sync.get("catalog_note", ""))
            notes = [f"Katalog: {sync.get('catalog_note', '')}",
                     "Niesprawdzone: " + ("pełny katalog członów rakiet i śmieci (bez Space-Track), " if no_st else "")
                     + "ADS-B (samoloty), druga stacja (paralaksa).",
                     "Jedna kamera nie daje odległości: brak km/s dla obiektów ostrych."]
            if best:
                notes.insert(0, f"Najbliższy kandydat katalogowy: NORAD {best['norad']} {best['name']} "
                                f"(pewność {best.get('confidence') or 'poniżej progu'})")
            name = f"unid_t{tid}"
        obj.update(title=title, params=params, notes=notes)

        # klip i pasek klatek
        plan = plan_clip(tp["frame"].to_numpy(), tp["x"].to_numpy(), tp["y"].to_numpy(), meta, rcfg)
        clip_rel, u8 = None, None
        try:
            u8 = stretch_stack(extract_frames(video, meta, plan))
            if len(u8):
                fnum = np.arange(plan.start, plan.start + len(u8))
                fx, fy = positions_at(fnum.astype(float), tp["frame"].to_numpy(float), tp["x"].to_numpy(),
                                      tp["y"].to_numpy())
                cx, cy = plan.to_clip(fx, fy)
                upscale = max(1, int(float(rcfg["clip_min_size_px"]) // max(plan.out_w, plan.out_h)))
                inside = (fnum >= tp["frame"].min()) & (fnum <= tp["frame"].max())
                clip_path = rep / "clips" / f"t{tid}.mp4"
                write_annotated_mp4(u8, cx, cy, inside, (46, 204, 64) if is_sat else (255, 42, 42), clip_path,
                                    meta.fps, upscale)
                clip_rel = f"report/clips/t{tid}.mp4"
                outputs.append(clip_rel)
        except Exception as e:  # noqa: BLE001 — brak klipu nie blokuje PDF-u
            log.warning("tor %d: wycinek wideo nieudany: %s", tid, e)
        if u8 is not None and len(u8):
            panels = filmstrip(u8, plan, tp["frame"].to_numpy(), tp["x"].to_numpy(), tp["y"].to_numpy(), meta.fps, rcfg)
            for p in panels:
                p["label"] = fmt_utc(start + timedelta(seconds=p["frame"] / meta.fps + delta), 2)[11:]
            obj.update(panels=panels, clip_name=Path(clip_rel).name if clip_rel else "–")
        if "stretched" in obj:
            pdf_path = rep / "objects" / f"{name}.pdf"
            object_pdf(pdf_path, obj, rcfg, always)
            if clip_rel:
                attach_file(pdf_path, outdir / clip_rel)
            outputs.append(f"report/objects/{name}.pdf")
        objects.append(obj)

    # strona zbiorcza
    ref = epoch_for(float(wcsinfo.get("reference_tau_s", 0.0))) if epochs else None
    members = sync.get("members") or []
    fov = wcsinfo.get("fov") or {}
    info = [
        f"Plik: {video.name}   klatek: {meta.n_frames}   fps: {meta.fps:.3f}",
        f"Start (metadane, z poprawką zegara z configu): {fmt_utc(start, 0)}",
        f"Poprawka zegara Δ: {delta:+.2f} ± {sync['sigma_s']:.2f} s   pewność: {sync['confidence']}   "
        f"metoda: {sync['method']}",
        f"Start po synchronizacji: {fmt_utc(start + timedelta(seconds=delta))}",
    ]
    if sync.get("reference"):
        r = sync["reference"]
        info.append(f"Satelita referencyjny (pierwszy zidentyfikowany): NORAD {r['norad']} {r['name']}, tor #{r['track_id']}")
    info += ["", f"Astrometria: {wcsinfo.get('n_solved', 0)}/{wcsinfo.get('n_epochs', 0)} epok rozwiązanych, "
                 f"zgodność epok {wcsinfo.get('epoch_rms_px', float('nan')):.2f} px",
             f"Pole widzenia: {fov.get('fov_w_deg', float('nan')):.2f}° × {fov.get('fov_h_deg', float('nan')):.2f}°, "
             f"skala {fov.get('scale_arcsec_px', float('nan')):.2f}″/px, {wcsinfo.get('crop', {}).get('verdict', '–')}",
             "", f"Katalog: {sync.get('catalog_note', '–')}"]
    n_sat = sum(1 for o in objects if o["color"] == C_SAT)
    rows = []
    for _, t in tracks.iterrows():
        b = (ident.get(str(int(t["track_id"]))) or [None])[0]
        rows.append([int(t["track_id"]),
                     fmt_utc(start + timedelta(seconds=float(t["tau0"]) + delta))[11:],
                     f"{t['dur_s']:.2f}", f"{t['omega_deg_s']:.3f}",
                     f"{t['az0']:.1f}/{t['alt0']:.1f}", f"{t['az1']:.1f}/{t['alt1']:.1f}",
                     f"{b['norad']} {b['name'][:18]}" if b else "–",
                     (b or {}).get("confidence") or "–", t["class_hint"] if not b else ""])
    pred_rows = []
    if len(pred):
        for _, p in pred[~pred["detected"].astype(bool)].iterrows():
            pred_rows.append([int(p["norad"]), str(p["name"])[:24], str(p["utc_in"])[11:19], str(p["utc_out"])[11:19],
                              f"{p['range_km']:.0f}", {True: "tak", False: "nie"}.get(p.get("sunlit"), "–")])
    notes = [
        "• Detektor klasyczny per klatka (próg SNR w configu). Obiekty wolniejsze niż ~0,1 px/klatkę",
        "  (MEO/GEO) trafiają do modelu tła i nie są tu wykrywane — przyjdą z shift-and-stack.",
        "• Czas: poprawka zegara z pierwszego zidentyfikowanego satelity, potwierdzona zgodnością innych torów.",
        "• „Niezidentyfikowany” = brak dopasowania w użytym katalogu; to nie jest anomalia.",
        "  Kryteria anomalii (zamrożone w configu) dojdą w kolejnym etapie.",
        "• Jedna kamera nie daje odległości ani prędkości liniowej obiektów ostrych.",
    ]
    for p in sorted(outdir.parent.glob("*/darkstats.json")):
        d = read_json(p)
        notes.append(f"• Nagranie ciemne {d['file']}: {d['tracks_per_hour']:.1f} fałszywych torów/h "
                     f"({d['tracks']} torów w {d['duration_s']:.0f} s).")
    s = {"title": f"skyhunt — {video.name} — {n_sat} satelitów zidentyfikowanych, "
                  f"{len(objects) - n_sat} pozostałych torów",
         "stretched": ref[0] if ref else None, "wcs": ref[1] if ref else None, "objects": [o for o in objects if "path_xy" in o],
         "legend": "zielony: satelita (NORAD), czerwony: niezidentyfikowany; niebieskie linie: konstelacje; "
                   "białe kreski: co 1 s; przerywana: predykcja z elementów orbit",
         "info_lines": info, "sync_members": members, "delta_s": delta,
         "table_header": ["tor", "start UTC", "czas [s]", "°/s", "Az/Alt start", "Az/Alt koniec", "NORAD", "pewność",
                          "podpowiedź"],
         "table_rows": rows,
         "pred_header": ["NORAD", "nazwa", "wejście UTC", "wyjście UTC", "odl. [km]", "oświetl."],
         "pred_rows": pred_rows, "notes": notes}
    summary_pdf(rep / "summary.pdf", s, rcfg, always)
    outputs.insert(0, "report/summary.pdf")
    return outputs
