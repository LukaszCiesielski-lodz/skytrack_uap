"""Zdjęcia RAW Fujifilm (RAF, matryca X-Trans): metadane i dane z matrycy.

Metadane czytamy bez zewnętrznych narzędzi: RAF zawiera JPEG podglądu z pełnym EXIF
(offset i długość w nagłówku, bajty 84–92, big-endian), a w nim MakerNote Fuji z numerem
zdjęcia w serii bracketingu. Sprawdzone na X-E3 (diagnostyka F0, 2.10.2026): czas EXIF ma
rozdzielczość 1 s (bez SubSec), każde zdjęcie serii ma własny czas, SequenceNumber 1-2-3,
dane 14 bit także przy migawce elektronicznej, czerń 1019, biel 16383.

Dane z matrycy (``rawpy``/LibRaw, bez demozaikowania) sumujemy w superpiksele 3×3: w każdym
bloku 3×3 wzoru X-Trans jest 5 G, 2 R i 2 B, więc suma bloku to luminancja (9× więcej fotonów
niż piksel), a średnie per kolor dają RGB bez interpolacji. Skala 3× większa niż piksel
matrycy (~48″/px przy 50 mm) — kreska satelity w 1/2 s to wciąż kilkadziesiąt px.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAF_MAGIC = b"FUJIFILMCCD-RAW"
# MakerNote Fuji (tagi jak w exiftool, FujiFilm.pm)
FUJI_TAGS = {0x1032: "exposure_count", 0x1050: "shutter_type", 0x1100: "auto_bracketing",
             0x1101: "sequence_number", 0x1438: "image_count"}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8}


def raf_jpeg(path: Path) -> bytes:
    """JPEG podglądu z pliku RAF (z nim EXIF)."""
    with open(path, "rb") as fh:
        head = fh.read(100)
        if not head.startswith(RAF_MAGIC):
            raise ValueError(f"{Path(path).name}: to nie jest plik RAF (brak {RAF_MAGIC!r})")
        off, ln = struct.unpack(">II", head[84:92])
        fh.seek(off)
        return fh.read(ln)


def parse_fuji_makernote(mn: bytes) -> dict:
    """MakerNote Fuji: ``FUJIFILM`` + przesunięcie IFD (LE), IFD little-endian z przesunięciami
    liczonymi od początku MakerNote. Zwraca wybrane tagi (``FUJI_TAGS``) jako liczby."""
    out: dict = {}
    if not mn or not mn.startswith(b"FUJIFILM") or len(mn) < 14:
        return out
    ifd = struct.unpack_from("<I", mn, 8)[0]
    if ifd + 2 > len(mn):
        return out
    n = struct.unpack_from("<H", mn, ifd)[0]
    for i in range(n):
        p = ifd + 2 + 12 * i
        if p + 12 > len(mn):
            break
        tag, typ, cnt = struct.unpack_from("<HHI", mn, p)
        if tag not in FUJI_TAGS or typ not in (1, 3, 4, 8, 9):
            continue
        size = _TYPE_SIZE[typ] * cnt
        pos = p + 8 if size <= 4 else struct.unpack_from("<I", mn, p + 8)[0]
        fmt = {1: "<B", 3: "<H", 4: "<I", 8: "<h", 9: "<i"}[typ]
        if pos + struct.calcsize(fmt) <= len(mn):
            out[FUJI_TAGS[tag]] = int(struct.unpack_from(fmt, mn, pos)[0])
    return out


def _num(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        v = v[0] if v else None
        if v is None:
            return None
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_raf_exif(path: Path) -> dict:
    """Metadane jednego zdjęcia: czas (napis EXIF), ułamek sekundy (jeśli jest), czas
    naświetlania, ISO, przysłona, aparat oraz pola MakerNote (numer w serii, licznik)."""
    from PIL import Image

    exif = Image.open(io.BytesIO(raf_jpeg(path))).getexif()
    sub = exif.get_ifd(0x8769)

    def tag(t):
        v = sub.get(t)
        return exif.get(t) if v is None else v

    mn = tag(0x927C)
    out = {"file": Path(path).name,
           "datetime": str(tag(0x9003) or tag(0x0132) or ""),
           "subsec": (str(tag(0x9291)).strip() or None) if tag(0x9291) is not None else None,
           "exposure_s": _num(tag(0x829A)), "iso": _num(tag(0x8827)), "fnumber": _num(tag(0x829D)),
           "make": str(exif.get(0x010F) or "").strip(), "model": str(exif.get(0x0110) or "").strip()}
    out.update({k: None for k in FUJI_TAGS.values()})
    out.update(parse_fuji_makernote(mn if isinstance(mn, bytes) else b""))
    return out


# ---------------------------------------------------------------- dane z matrycy

@dataclass
class RawFrame:
    cfa: np.ndarray          # [H, W] uint16, obszar widoczny
    colors: np.ndarray       # [H, W] uint8: 0 = R, 1 = G, 2 = B
    black: np.ndarray        # poziom czerni per kolor [3]
    white: float


def read_raw(path: Path) -> RawFrame:
    """Dane z matrycy bez demozaikowania (rawpy/LibRaw obsługuje X-Trans)."""
    import rawpy

    with rawpy.imread(str(path)) as r:
        cfa = r.raw_image_visible.copy()
        colors = r.raw_colors_visible.astype(np.uint8)
        bl = np.asarray(r.black_level_per_channel, float)
        white = float(r.white_level)
    colors = np.where(colors == 3, 1, colors).astype(np.uint8)          # drugi G (RGBG) → G
    black = np.array([bl[0], bl[1], bl[2]], float)
    return RawFrame(cfa, colors, black, white)


def superpixels(frame: RawFrame, *, sat_frac: float = 0.98) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bloki 3×3 → (luminancja [h, w] = suma 9 pikseli po odjęciu czerni, RGB [h, w, 3] = średnie
    per kolor, maska nasycenia [h, w] = choć jeden piksel bloku ≥ ``sat_frac``·biel).
    Kadr przycinany do wielokrotności 3 (od góry-lewej, więc wzór X-Trans zostaje wyrównany)."""
    H, W = frame.cfa.shape
    h, w = H // 3, W // 3
    cfa = frame.cfa[:3 * h, :3 * w]
    col = frame.colors[:3 * h, :3 * w]
    sat = (cfa >= sat_frac * frame.white).reshape(h, 3, w, 3).any(axis=(1, 3))
    lin = cfa.astype(np.float32) - frame.black[col].astype(np.float32)
    blocks = lin.reshape(h, 3, w, 3)
    cblocks = col.reshape(h, 3, w, 3)
    lum = blocks.sum(axis=(1, 3))
    rgb = np.empty((h, w, 3), np.float32)
    for c in range(3):
        m = cblocks == c
        cnt = m.sum(axis=(1, 3))
        rgb[..., c] = np.where(cnt > 0, (blocks * m).sum(axis=(1, 3)) / np.maximum(cnt, 1), 0.0)
    return lum.astype(np.float32), rgb, sat


def xtrans_pattern() -> np.ndarray:
    """Wzór X-Trans 6×6 z X-E3 (0 = R, 1 = G, 2 = B) — do testów i kontroli."""
    return np.array([[0, 2, 1, 2, 0, 1],
                     [1, 1, 0, 1, 1, 2],
                     [1, 1, 2, 1, 1, 0],
                     [2, 0, 1, 0, 2, 1],
                     [1, 1, 2, 1, 1, 0],
                     [1, 1, 0, 1, 1, 2]], np.uint8)


def list_photos(folder: Path, extensions=(".raf",)) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts)
