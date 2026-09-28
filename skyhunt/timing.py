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


@dataclass
class FrameClock:
    """Czas klatki: τ (od startu nagrania, środek ekspozycji, opcjonalnie rolling shutter)
    oraz UTC = start_prior + Δ + τ, gdzie Δ to poprawka zegara z synchronizacji."""
    start_utc: datetime
    fps: float
    height: int
    exposure_offset_s: float
    rolling_shutter_s: float = 0.0
    delta_s: float = 0.0

    @classmethod
    def from_config(cls, start_utc: datetime, meta: VideoMeta, camera_cfg: dict, delta_s: float = 0.0):
        off = camera_cfg.get("exposure_offset_s")
        if off is None:
            off = float(camera_cfg["shutter_s"]) / 2
        return cls(start_utc, meta.fps, meta.height, float(off), float(camera_cfg.get("rolling_shutter_s") or 0.0),
                   delta_s)

    def tau(self, frame, y=None):
        import numpy as np

        t = np.asarray(frame, float) / self.fps + self.exposure_offset_s
        if y is not None and self.rolling_shutter_s:
            t = t + self.rolling_shutter_s * np.asarray(y, float) / self.height
        return t

    def utc(self, tau: float) -> datetime:
        return self.start_utc + timedelta(seconds=float(tau) + self.delta_s)


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
