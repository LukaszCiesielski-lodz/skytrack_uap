"""Czas nagrania: wstępne oszacowanie z metadanych i poprawki zegara aparatu.

Zegar aparatu nie jest wiarygodny (Fuji zapisuje czas lokalny oznaczony jako UTC,
a zegar może się spieszyć). Tu liczymy tylko *prior* z niepewnością; właściwy czas
wyznacza synchronizacja po przelotach satelitów (M2, sekcja 5.6).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .metadata import VideoMeta

START_SOURCES = ("maker_datetime", "mvhd_creation", "mvhd_creation_minus_duration")


def camera_to_utc(camera_time: datetime, tz: str, clock_ahead_s: float) -> datetime:
    """Czas ścienny aparatu (strefa ``tz``, zegar spieszy się o ``clock_ahead_s``) → UTC."""
    local = (camera_time.replace(tzinfo=None) - timedelta(seconds=clock_ahead_s))
    return local.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def camera_start(meta: VideoMeta, source: str) -> datetime | None:
    """Czas startu nagrania wg zegara aparatu (naiwny) z wybranego źródła."""
    if source == "maker_datetime" and meta.maker_datetime:
        return datetime.fromisoformat(meta.maker_datetime)
    if source == "mvhd_creation" and meta.mvhd_creation:
        return datetime.fromisoformat(meta.mvhd_creation)
    if source == "mvhd_creation_minus_duration" and meta.mvhd_creation:
        return datetime.fromisoformat(meta.mvhd_creation) - timedelta(seconds=meta.duration_s)
    return None


@dataclass
class TimePrior:
    start_utc: str            # ISO 8601 z +00:00
    sigma_s: float            # 1σ niepewności
    source: str
    camera_time: str          # odczyt zegara aparatu (naiwny)
    candidates_utc: dict[str, str] = field(default_factory=dict)  # wszystkie źródła, do porównania

    def to_dict(self) -> dict:
        return asdict(self)


def time_prior(meta: VideoMeta, tcfg: dict) -> TimePrior:
    tz, ahead = tcfg["camera_tz"], float(tcfg["camera_clock_ahead_s"])
    cands = {s: camera_start(meta, s) for s in START_SOURCES}
    utc = {s: camera_to_utc(t, tz, ahead).isoformat() for s, t in cands.items() if t}
    order = [tcfg["start_source"]] + [s for s in START_SOURCES if s != tcfg["start_source"]]
    for rank, src in enumerate(order):
        if cands[src] is None:
            continue
        sigma = float(tcfg["prior_sigma_s"]) * (3 if rank else 1)  # zapasowe źródło: 3× σ
        return TimePrior(utc[src], sigma, src, cands[src].isoformat(), utc)
    raise ValueError(f"{meta.path}: brak znacznika czasu w metadanych")
