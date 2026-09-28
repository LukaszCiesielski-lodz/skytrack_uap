"""``manifest.json`` per plik: które etapy są gotowe, z jakim hashem configu i wersją kodu.

Po rozłączeniu Colaba przetwarzanie wznawia się od pierwszego etapu, który nie jest
``done`` z aktualnym hashem albo którego pliki wynikowe zniknęły.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .io import code_version, read_json, write_json

SCHEMA = 1


def video_fingerprint(path: Path) -> dict:
    """Nazwa + rozmiar + hash pierwszego 1 MiB (w nim moov z czasem nagrania)."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = hashlib.sha256(fh.read(1 << 20)).hexdigest()[:16]
    return {"name": path.name, "size_bytes": path.stat().st_size, "head_sha256": head}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    FILENAME = "manifest.json"

    def __init__(self, path: Path, data: dict):
        self.path = Path(path)
        self.data = data
        self.reset_reason: str | None = None

    @classmethod
    def load(cls, outdir: Path, video_path: Path) -> "Manifest":
        path = Path(outdir) / cls.FILENAME
        fp = video_fingerprint(video_path)
        data, reason = None, None
        if path.exists():
            try:
                data = read_json(path)
            except (ValueError, OSError):
                reason = "uszkodzony manifest"
        if isinstance(data, dict) and data.get("schema") == SCHEMA and data.get("video") == fp:
            return cls(path, data)
        if isinstance(data, dict):
            reason = "inny plik wideo lub schemat manifestu"
        m = cls(path, {"schema": SCHEMA, "video": fp, "stages": {}})
        m.reset_reason = reason
        return m

    def stage(self, name: str) -> dict | None:
        return self.data["stages"].get(name)

    def is_done(self, name: str, stage_hash: str) -> bool:
        st = self.stage(name)
        return bool(st and st.get("status") == "done" and st.get("hash") == stage_hash
                    and all((self.path.parent / o).exists() for o in st.get("outputs", [])))

    def mark_running(self, name: str, stage_hash: str) -> None:
        self.data["stages"][name] = {"status": "running", "hash": stage_hash, "started_at": _now(),
                                     **code_version()}

    def mark_done(self, name: str, stage_hash: str, outputs: list[str], metrics: dict,
                  elapsed_s: float) -> None:
        st = self.data["stages"].setdefault(name, {"hash": stage_hash, **code_version()})
        st.update(status="done", hash=stage_hash, finished_at=_now(), elapsed_s=round(elapsed_s, 3),
                  outputs=list(outputs), metrics=metrics)
        st.pop("error", None)

    def mark_failed(self, name: str, stage_hash: str, error: str) -> None:
        st = self.data["stages"].setdefault(name, {"hash": stage_hash, **code_version()})
        st.update(status="failed", hash=stage_hash, finished_at=_now(), error=error)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.data)
