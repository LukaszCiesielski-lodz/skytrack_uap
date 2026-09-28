"""Zapis/odczyt JSON odporny na przerwania (Colab, Google Drive)."""
from __future__ import annotations

import functools
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from . import __version__


def to_jsonable(obj: Any) -> Any:
    """Zamienia typy numpy/Path/NaN na typy JSON (NaN i inf → None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", 1) == 0:
        obj = obj.item()  # skalar numpy/torch
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    """Zapis atomowy: plik tymczasowy + os.replace, więc przerwany zapis nie zostawia śmieci."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def git_rev() -> str | None:
    """Skrócony hash commita, z którego uruchomiono kod (None poza repozytorium)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def code_version() -> dict:
    return {"version": __version__, "git_rev": git_rev()}
