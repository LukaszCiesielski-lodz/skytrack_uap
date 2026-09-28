"""Metadane wideo z kontenera MP4/MOV — własny parser atomów, bez ffprobe.

Czyta tylko nagłówek ``moov`` (dla Fuji ~1 MB na początku pliku), nie dotyka ``mdat``,
więc działa w milisekundach także dla plików wielogigabajtowych na Google Drive.

Pozycje klatek kluczowych (GOP) pochodzą z ``stss``; obecność niezerowych przesunięć
w ``ctts`` oznacza klatki B (zmieniona kolejność dekodowania). Oba są potrzebne do
maskowania artefaktów H.264 (sekcja 5.3 handoffu).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Iterator

_MAC_EPOCH = datetime(1904, 1, 1)
# Data w formacie EXIF zapisana przez aparat w udta (Fuji: moment startu nagrania).
_EXIF_DATE = re.compile(rb"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_MAKES = re.compile(rb"FUJIFILM|Canon|NIKON|SONY|Panasonic|OLYMPUS|OM Digital|Apple|GoPro")
# Heurystyka modelu: Fuji (X-E3, X-T4, X100V, GFX 100S); inne aparaty → None.
_MODEL = re.compile(rb"\b(X-[A-Z][A-Z0-9]{0,3}|X[0-9]{3}[A-Z]{0,2}|GFX ?[0-9]{2,3}[A-Z]{0,2})\b")


@dataclass
class VideoMeta:
    path: str
    size_bytes: int
    codec: str               # fourcc z stsd, np. "avc1"
    width: int
    height: int
    fps_num: int
    fps_den: int
    n_frames: int
    duration_s: float
    keyframes: list[int] = field(default_factory=list)   # indeksy klatek (od 0)
    has_bframes: bool = False
    mvhd_creation: str | None = None   # naiwny ISO; Fuji: zegar aparatu opisany jako UTC
    maker_datetime: str | None = None  # naiwny ISO z daty EXIF w udta
    make: str | None = None
    model: str | None = None

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def gop_length(self) -> int | None:
        """Najczęstszy odstęp między klatkami kluczowymi."""
        if len(self.keyframes) < 2:
            return None
        gaps = [b - a for a, b in zip(self.keyframes, self.keyframes[1:])]
        return max(set(gaps), key=gaps.count)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = self.fps
        d["gop_length"] = self.gop_length
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VideoMeta":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _u(buf: bytes, off: int, n: int) -> int:
    return int.from_bytes(buf[off:off + n], "big")


def _boxes(buf: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Atomy w ``buf[start:end]`` jako (typ, początek_payloadu, koniec)."""
    pos = start
    while pos + 8 <= end:
        size, typ, hdr = _u(buf, pos, 4), bytes(buf[pos + 4:pos + 8]), 8
        if size == 1:
            if pos + 16 > end:
                return
            size, hdr = _u(buf, pos + 8, 8), 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            return
        yield typ, pos + hdr, pos + size
        pos += size


def _find(buf: bytes, start: int, end: int, path: list[bytes]) -> tuple[int, int] | None:
    for typ, s, e in _boxes(buf, start, end):
        if typ == path[0]:
            if len(path) == 1:
                return s, e
            hit = _find(buf, s, e, path[1:])
            if hit:
                return hit
    return None


def read_moov(path: Path) -> bytes:
    """Payload atomu moov (szuka go na najwyższym poziomie, także za mdat)."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        pos = 0
        while pos + 8 <= size:
            fh.seek(pos)
            h = fh.read(16)
            bsize, typ, hdr = int.from_bytes(h[:4], "big"), h[4:8], 8
            if bsize == 1:
                bsize, hdr = int.from_bytes(h[8:16], "big"), 16
            elif bsize == 0:
                bsize = size - pos
            if bsize < hdr:
                break
            if typ == b"moov":
                fh.seek(pos + hdr)
                return fh.read(bsize - hdr)
            pos += bsize
    raise ValueError(f"{path}: brak atomu moov (to nie MP4/MOV albo plik jest ucięty)")


def _mvhd_like(buf: bytes, s: int) -> tuple[int, int, int]:
    """(creation, timescale, duration) z mvhd/mdhd, wersja 0 lub 1."""
    if buf[s] == 1:
        return _u(buf, s + 4, 8), _u(buf, s + 20, 4), _u(buf, s + 24, 8)
    return _u(buf, s + 4, 4), _u(buf, s + 12, 4), _u(buf, s + 16, 4)


def probe_video(path: str | Path) -> VideoMeta:
    path = Path(path)
    moov = read_moov(path)
    end = len(moov)

    mvhd = _find(moov, 0, end, [b"mvhd"])
    if not mvhd:
        raise ValueError(f"{path}: brak mvhd")
    m_ct, m_ts, m_dur = _mvhd_like(moov, mvhd[0])

    for typ, ts, te in _boxes(moov, 0, end):
        if typ != b"trak":
            continue
        hdlr = _find(moov, ts, te, [b"mdia", b"hdlr"])
        if hdlr and moov[hdlr[0] + 8:hdlr[0] + 12] == b"vide":
            break
    else:
        raise ValueError(f"{path}: brak ścieżki wideo")

    mdhd = _find(moov, ts, te, [b"mdia", b"mdhd"])
    stbl = _find(moov, ts, te, [b"mdia", b"minf", b"stbl"])
    if not mdhd or not stbl:
        raise ValueError(f"{path}: niekompletna ścieżka wideo (mdhd/stbl)")
    _, timescale, t_dur = _mvhd_like(moov, mdhd[0])

    stsd = _find(moov, stbl[0], stbl[1], [b"stsd"])
    stts = _find(moov, stbl[0], stbl[1], [b"stts"])
    if not stsd or not stts:
        raise ValueError(f"{path}: brak stsd/stts w ścieżce wideo")
    e0 = stsd[0] + 8  # pierwszy wpis opisu próbki
    codec = moov[e0 + 4:e0 + 8].decode("latin-1")
    width, height = _u(moov, e0 + 32, 2), _u(moov, e0 + 34, 2)

    entries =[(_u(moov, stts[0] + 8 + 8 * i, 4), _u(moov, stts[0] + 12 + 8 * i, 4))
               for i in range(_u(moov, stts[0] + 4, 4))]
    n_frames = sum(c for c, _ in entries)
    delta = max(entries, key=lambda e: e[0])[1] if entries else 1
    fps = Fraction(timescale, delta or 1)

    stss = _find(moov, stbl[0], stbl[1], [b"stss"])
    if stss:
        keyframes = [_u(moov, stss[0] + 8 + 4 * i, 4) - 1 for i in range(_u(moov, stss[0] + 4, 4))]
    else:  # brak stss = każda próbka jest synchronizująca (np. intra-only)
        keyframes = list(range(n_frames))

    ctts = _find(moov, stbl[0], stbl[1], [b"ctts"])
    has_b = bool(ctts) and any(_u(moov, ctts[0] + 12 + 8 * i, 4)
                               for i in range(_u(moov, ctts[0] + 4, 4)))

    # Napisy producenta szukamy tylko w udta: regex na ~1 MB binarnych tabel próbek
    # łapałby przypadkowe bajty.
    udta = _find(moov, 0, end, [b"udta"])
    maker = moov[udta[0]:udta[1]] if udta else b""
    date = _EXIF_DATE.search(maker)
    make = _MAKES.search(maker)
    model = _MODEL.search(maker)
    duration = t_dur / timescale if timescale else (m_dur / m_ts if m_ts else 0.0)
    return VideoMeta(
        path=str(path),
        size_bytes=path.stat().st_size,
        codec=codec,
        width=width,
        height=height,
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        n_frames=n_frames,
        duration_s=duration,
        keyframes=keyframes,
        has_bframes=has_b,
        mvhd_creation=(_MAC_EPOCH + timedelta(seconds=m_ct)).isoformat() if m_ct else None,
        maker_datetime=("{}-{}-{}T{}:{}:{}".format(*(g.decode() for g in date.groups()))
                        if date else None),
        make=make.group().decode() if make else None,
        model=model.group(1).decode() if model else None,
    )
