"""Konfiguracja: wczytanie config.yaml, nadpisania z CLI, walidacja, hash sekcji."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

PACKAGE_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
_MISSING = object()


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Kolejność: argument → $SKYHUNT_CONFIG → ./config.yaml → config.yaml obok pakietu."""
    for cand in (path, os.environ.get("SKYHUNT_CONFIG"), "config.yaml", PACKAGE_CONFIG):
        if cand and Path(cand).is_file():
            return Path(cand)
    raise FileNotFoundError("nie znaleziono config.yaml (podaj --config)")


def load_config(path: str | Path | None = None, overrides: dict | None = None,
                sites: str | Path | None = None) -> dict:
    """``sites``: plik YAML {nazwa pliku: {lat_deg, lon_deg, elevation_m}} z miejscem każdej
    obserwacji (domyślnie ``$SKYHUNT_SITES``). Trzymany poza repo (Drive), nakładany jako
    ``files.<plik>.site``; pliki bez wpisu używają sekcji ``site``."""
    p = resolve_config_path(path)
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    sites = sites or os.environ.get("SKYHUNT_SITES")
    if sites and Path(sites).is_file():
        with open(sites, encoding="utf-8") as fh:
            per_site = yaml.safe_load(fh) or {}
        files: dict = {}
        for f, s in per_site.items():
            s = dict(s or {})
            over = {"role": s.pop("role")} if "role" in s else {}
            if s:
                over["site"] = s
            files[str(f)] = over
        cfg = deep_merge(cfg, {"files": files})
        cfg["_sites"] = str(sites)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate(cfg)
    cfg["_source"] = str(p)
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_for_file(cfg: dict, video_path: str | Path) -> dict:
    """Config z nałożonymi nadpisaniami z sekcji ``files`` dla danej nazwy pliku
    (np. inna poprawka zegara aparatu dla starszego nagrania)."""
    name = Path(video_path).name
    per_file = dict((cfg.get("files") or {}).get(name) or {})
    if "role" not in per_file and "dark" in name.lower():   # np. dark_0929.MOV: zakryty obiektyw
        per_file["role"] = "dark"
    if per_file.get("role") == "dark" and cfg.get("dark_overrides"):
        per_file = deep_merge(cfg["dark_overrides"], per_file)   # wspólne ustawienia nagrań ciemnych
    return deep_merge(cfg, per_file) if per_file else cfg


def cfg_get(cfg: dict, dotted: str, default: Any = _MISSING) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"brak klucza konfiguracji: {dotted}")
            return default
        node = node[part]
    return node


def parse_set(items: Iterable[str]) -> dict:
    """``["decode.batch_frames=16", "time.camera_clock_ahead_s=0"]`` → zagnieżdżony dict.
    Wartość jest parsowana jako YAML (liczby, listy, true/false)."""
    out: dict = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"oczekiwano KLUCZ=WARTOŚĆ, jest: {item!r}")
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def config_hash(cfg: dict, sections: Iterable[str]) -> str:
    """Hash wybranych sekcji configu; zmiana dowolnej z nich zmienia hash."""
    payload = {s: cfg_get(cfg, s, None) for s in sorted(sections)}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def validate(cfg: dict) -> None:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    from .decode import BACKENDS

    errors = []
    lat, lon = cfg_get(cfg, "site.lat_deg", None), cfg_get(cfg, "site.lon_deg", None)
    if not isinstance(lat, (int, float)) or not -90 <= lat <= 90:
        errors.append("site.lat_deg musi być liczbą w [-90, 90]")
    if not isinstance(lon, (int, float)) or not -180 <= lon <= 180:
        errors.append("site.lon_deg musi być liczbą w [-180, 180]")
    if not isinstance(cfg_get(cfg, "site.elevation_m", None), (int, float)):
        errors.append("site.elevation_m musi być liczbą")
    tz = cfg_get(cfg, "time.camera_tz", None)
    try:
        ZoneInfo(str(tz))
    except (ZoneInfoNotFoundError, ValueError):
        errors.append(f"time.camera_tz: nieznana strefa {tz!r}")
    if cfg_get(cfg, "time.start_source", None) not in ("maker_datetime", "mvhd_creation",
                                                        "mvhd_creation_minus_duration"):
        errors.append("time.start_source: maker_datetime | mvhd_creation | mvhd_creation_minus_duration")
    backends = cfg_get(cfg, "decode.backends", None)
    if not isinstance(backends, list) or not backends:
        errors.append("decode.backends: niepusta lista")
    else:
        unknown = [b for b in backends if b not in BACKENDS]
        if unknown:
            errors.append(f"decode.backends: nieznane {unknown}, dostępne {list(BACKENDS)}")
    if not isinstance(cfg_get(cfg, "decode.batch_frames", None), int) or cfg["decode"]["batch_frames"] < 1:
        errors.append("decode.batch_frames: liczba całkowita >= 1")
    det = cfg.get("detect") or {}
    if det:
        if not 0 < float(det.get("snr_grow", 0)) <= float(det.get("snr_seed", 0)):
            errors.append("detect: wymagane 0 < snr_grow ≤ snr_seed")
        if det.get("labeler") not in ("auto", "cupy", "scipy"):
            errors.append("detect.labeler: auto | cupy | scipy")
        if int(det.get("bg_stride", 0)) < 1 or int(det.get("bg_half_window_frames", 0)) < int(det.get("bg_stride", 1)):
            errors.append("detect: bg_stride ≥ 1 i bg_half_window_frames ≥ bg_stride")
    ast = cfg.get("astrometry") or {}
    if "scale_low_deg" in ast and not 0 < float(ast["scale_low_deg"]) < float(ast["scale_high_deg"]):
        errors.append("astrometry: wymagane 0 < scale_low_deg < scale_high_deg")
    if any(not 4107 <= int(n) <= 4119 for n in ast.get("index_series", [])):
        errors.append("astrometry.index_series: indeksy 4107–4119 (seria 4100)")
    if ast.get("hint_star"):
        from .sky import resolve_star
        try:
            resolve_star(str(ast["hint_star"]))
        except KeyError as e:
            errors.append(f"astrometry.hint_star: {e}")
    idf = cfg.get("identify") or {}
    if idf and idf.get("reference") not in ("first", "best", "median"):
        errors.append("identify.reference: first | best | median")
    sat = cfg.get("satellites") or {}
    if sat and not sat.get("celestrak_groups"):
        errors.append("satellites.celestrak_groups: niepusta lista")
    rep = cfg.get("report") or {}
    if rep and rep.get("identified_min_confidence") not in ("high", "medium", "low"):
        errors.append("report.identified_min_confidence: high | medium | low")
    col = cfg.get("color") or {}
    if col:
        if str(col.get("enabled", "auto")) not in ("auto", "off"):
            errors.append("color.enabled: auto | off")
        tr = col.get("transfer", "bt709")
        if tr not in ("bt709", "linear") and not (isinstance(tr, (int, float)) and 1.0 <= tr <= 3.0):
            errors.append("color.transfer: bt709 | linear | gamma z [1, 3]")
        ann = col.get("annulus_px")
        if not (isinstance(ann, list) and len(ann) == 2 and col.get("aperture_px", 0) < ann[0] < ann[1]):
            errors.append("color: wymagane aperture_px < annulus_px[0] < annulus_px[1]")
        if not 0 < float(col.get("sat_level", 0)) <= 1:
            errors.append("color.sat_level: (0, 1]")
    for fname, over in (cfg.get("files") or {}).items():
        if not isinstance(over, dict):
            errors.append(f"files.{fname}: oczekiwano słownika nadpisań")
        elif over.get("role", "sky") not in ("sky", "dark"):
            errors.append(f"files.{fname}.role: sky | dark")
        elif (over.get("astrometry") or {}).get("hint_star"):
            from .sky import resolve_star
            try:
                resolve_star(str(over["astrometry"]["hint_star"]))
            except KeyError as e:
                errors.append(f"files.{fname}.astrometry.hint_star: {e}")
        site = over.get("site") if isinstance(over, dict) else None
        if site is not None:
            ok = isinstance(site, dict) and all(isinstance(site.get(k), (int, float)) for k in ("lat_deg", "lon_deg"))
            if not ok or not -90 <= site["lat_deg"] <= 90 or not -180 <= site["lon_deg"] <= 180:
                errors.append(f"files.{fname}.site: wymagane lat_deg ∈ [-90, 90] i lon_deg ∈ [-180, 180]")
    if errors:
        raise ValueError("Błędy w config.yaml:\n  - " + "\n  - ".join(errors))
