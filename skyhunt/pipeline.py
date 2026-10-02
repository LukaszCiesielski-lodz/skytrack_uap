"""Rejestr etapów pipeline'u i uruchamianie per plik ze wznawianiem.

Hash etapu = f(nazwa, ``rev`` etapu, sekcje configu, od których zależy, hashe etapów
wymaganych). Zmiana configu unieważnia więc etap i wszystko, co od niego zależy;
zmiana logiki etapu wymaga podbicia ``rev``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import config_for_file, config_hash
from .io import read_json, write_json
from .manifest import Manifest

ALL_ROLES = ("sky", "dark")


def input_kind(path: Path) -> str:
    """Rodzaj wejścia: ``photos`` (folder sesji zdjęć RAW) albo ``video`` (plik)."""
    return "photos" if Path(path).is_dir() else "video"


def input_outdir(out_root: Path, path: Path) -> Path:
    """Katalog wyników: nazwa pliku bez rozszerzenia albo pełna nazwa folderu sesji."""
    path = Path(path)
    return Path(out_root) / (path.name if path.is_dir() else path.stem)


def file_role(cfg: dict) -> str:
    """Rola pliku z configu (sekcja ``files``): ``sky`` (domyślnie) albo ``dark``."""
    return str(cfg.get("role") or "sky")


@dataclass(frozen=True)
class Stage:
    name: str
    func: Callable[["StageContext"], dict | None]
    sections: tuple[str, ...]
    requires: tuple[str, ...]
    rev: int
    roles: tuple[str, ...] = ALL_ROLES


@dataclass
class StageContext:
    video_path: Path
    outdir: Path
    cfg: dict
    manifest: Manifest
    log: logging.Logger

    @property
    def role(self) -> str:
        return file_role(self.cfg)

    @property
    def kind(self) -> str:
        return input_kind(self.video_path)

    @property
    def input_path(self) -> Path:
        """Plik wideo albo folder sesji zdjęć (``video_path`` zostaje dla zgodności)."""
        return self.video_path

    @property
    def out_root(self) -> Path:
        return self.outdir.parent

    def read_json(self, name: str) -> Any:
        return read_json(self.outdir / name)

    def write_json(self, name: str, obj: Any) -> None:
        write_json(self.outdir / name, obj)


class Pipeline:
    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}

    def stage(self, name: str, sections: Iterable[str] = (), requires: Iterable[str] = (), rev: int = 1,
              roles: Iterable[str] = ALL_ROLES):
        """Dekorator rejestrujący etap. Wymagane etapy muszą być zarejestrowane wcześniej,
        więc kolejność rejestracji jest zarazem kolejnością wykonania. ``roles``: dla jakich
        plików etap ma sens (np. astrometria tylko dla ``sky``, statystyki ciemne dla ``dark``)."""
        requires, roles = tuple(requires), tuple(roles)

        def deco(func):
            if name in self.stages:
                raise ValueError(f"etap {name!r} już zarejestrowany")
            missing = [r for r in requires if r not in self.stages]
            if missing:
                raise ValueError(f"etap {name!r} wymaga niezarejestrowanych: {missing}")
            self.stages[name] = Stage(name, func, tuple(sections), requires, rev, roles)
            return func

        return deco

    def stage_hash(self, name: str, cfg: dict, memo: dict[str, str] | None = None) -> str:
        memo = {} if memo is None else memo
        if name not in memo:
            st = self.stages[name]
            payload = {"stage": name, "rev": st.rev, "config": config_hash(cfg, st.sections),
                       "upstream": {r: self.stage_hash(r, cfg, memo) for r in st.requires}}
            blob = json.dumps(payload, sort_keys=True).encode()
            memo[name] = hashlib.sha256(blob).hexdigest()[:16]
        return memo[name]

    def closure(self, names: Iterable[str]) -> list[str]:
        """Etapy ``names`` z zależnościami, w kolejności wykonania."""
        need: set[str] = set()

        def add(n: str) -> None:
            if n not in self.stages:
                raise KeyError(f"nieznany etap {n!r}; dostępne: {list(self.stages)}")
            if n not in need:
                need.add(n)
                for r in self.stages[n].requires:
                    add(r)

        for n in names:
            add(n)
        return [n for n in self.stages if n in need]

    def dependents(self, names: Iterable[str]) -> set[str]:
        """Etapy ``names`` i wszystkie zależne od nich (``all`` = wszystkie)."""
        names = set(names)
        if "all" in names:
            return set(self.stages)
        out: set[str] = set()
        for n, st in self.stages.items():
            if n in names or any(r in out for r in st.requires):
                out.add(n)
        return out

    def run(self, video_path: Path, cfg: dict, out_root: Path, only: Iterable[str] | None = None,
            force: Iterable[str] = (), log: logging.Logger | None = None) -> dict[str, str]:
        """Uruchamia etapy dla jednego pliku. Zwraca {etap: 'cached' | 'done' | 'n/a'}; błąd
        etapu jest zapisywany w manifeście i przerywa dalsze etapy tego pliku."""
        video_path = Path(video_path)
        log = log or logging.getLogger("skyhunt")
        cfg = config_for_file(cfg, video_path)
        outdir = input_outdir(out_root, video_path)
        outdir.mkdir(parents=True, exist_ok=True)
        man = Manifest.load(outdir, video_path)
        if man.reset_reason:
            log.warning("[%s] manifest od nowa: %s", video_path.name, man.reset_reason)
        names = self.closure(only) if only else list(self.stages)
        forced = self.dependents(force) if force else set()
        ctx = StageContext(video_path, outdir, cfg, man, log)
        memo: dict[str, str] = {}
        status: dict[str, str] = {}
        role = file_role(cfg)
        for name in names:
            if role not in self.stages[name].roles:
                status[name] = "n/a"
                continue
            h = self.stage_hash(name, cfg, memo)
            if name not in forced and man.is_done(name, h):
                status[name] = "cached"
                log.info("[%s] %s: gotowe wcześniej (hash %s)", video_path.name, name, h)
                continue
            log.info("[%s] %s: start", video_path.name, name)
            man.mark_running(name, h)
            man.save()
            t0 = time.perf_counter()
            try:
                result = self.stages[name].func(ctx) or {}
            except Exception as e:
                man.mark_failed(name, h, f"{type(e).__name__}: {e}")
                man.save()
                raise
            man.mark_done(name, h, result.get("outputs", []), result.get("metrics", {}),
                          time.perf_counter() - t0)
            man.save()
            status[name] = "done"
            log.info("[%s] %s: gotowe w %.1f s", video_path.name, name, time.perf_counter() - t0)
        return status


PIPELINE = Pipeline()          # wideo (pliki MOV/MP4)
PHOTO_PIPELINE = Pipeline()    # sesje zdjęć RAW (folder z plikami RAF); etapy w photo_stages.py


def pipeline_for(path: Path) -> Pipeline:
    return PHOTO_PIPELINE if input_kind(path) == "photos" else PIPELINE
