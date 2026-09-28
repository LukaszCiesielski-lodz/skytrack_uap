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


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    p = resolve_config_path(path)
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
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
    per_file = (cfg.get("files") or {}).get(Path(video_path).name)
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
    if errors:
        raise ValueError("Błędy w config.yaml:\n  - " + "\n  - ".join(errors))
