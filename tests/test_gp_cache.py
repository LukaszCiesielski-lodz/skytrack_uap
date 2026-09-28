"""Wybór i pobieranie snapshotów elementów orbit: bez sieci (atrapa urlopen)."""
import logging
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from skyhunt.satellites import FetchError, fetch_spacetrack, fetch_url, list_snapshots, select_celestrak

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
T_REC = datetime(2026, 9, 27, 18, 26, 8, tzinfo=timezone.utc)


class Resp:
    def __init__(self, data=b"OBJECT_NAME,NORAD_CAT_ID\nX,1\n", status=200):
        self.data, self.status = data, status

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Opener:
    def __init__(self, fail_code=None):
        self.urls, self.fail_code = [], fail_code

    def __call__(self, url, data=None, timeout=None):
        self.urls.append(url)
        if self.fail_code:
            raise urllib.error.HTTPError(url, self.fail_code, "blocked", {}, None)
        return Resp()


def scfg(tmp_path, groups=("active", "visual")):
    return {"cache_dir": str(tmp_path), "celestrak_url": "https://celestrak.org/NORAD/elements/gp.php",
            "celestrak_groups": list(groups), "auto_fetch": True, "max_elements_age_days": 3, "min_refetch_h": 2,
            "spacetrack": "auto", "history_days_before": 3, "history_days_after": 1}


def touch(tmp_path, name):
    (tmp_path / name).write_text("OBJECT_NAME,NORAD_CAT_ID\n")


def test_uses_earliest_snapshot_after_recording(tmp_path):
    touch(tmp_path, "celestrak_active_20260927T1500Z.csv")   # przed nagraniem
    touch(tmp_path, "celestrak_active_20260927T2000Z.csv")   # najwcześniejszy po
    touch(tmp_path, "celestrak_active_20260928T0900Z.csv")
    op = Opener()
    got = select_celestrak(scfg(tmp_path, ["active"]), T_REC, now=NOW, opener=op)
    assert [g["path"].name for g in got] == ["celestrak_active_20260927T2000Z.csv"]
    assert op.urls == []


def test_fetches_missing_group(tmp_path):
    touch(tmp_path, "celestrak_active_20260927T2000Z.csv")
    op = Opener()
    got = select_celestrak(scfg(tmp_path), T_REC, now=NOW, opener=op)
    assert len(op.urls) == 1 and "GROUP=visual" in op.urls[0] and "FORMAT=csv" in op.urls[0]
    assert {g["group"] for g in got} == {"active", "visual"}
    assert any(s["group"] == "visual" for s in list_snapshots(tmp_path))


def test_http_403_stops_fetching_without_retry(tmp_path, caplog):
    op = Opener(fail_code=403)
    with caplog.at_level(logging.WARNING):
        got = select_celestrak(scfg(tmp_path), T_REC, now=NOW, opener=op)
    assert len(op.urls) == 1          # druga grupa już nie jest pobierana
    assert got == []
    assert "403" in caplog.text


def test_no_refetch_within_min_interval(tmp_path):
    t_rec = NOW - timedelta(hours=1)
    touch(tmp_path, "celestrak_active_20260928T1030Z.csv")   # 90 min temu, przed „nagraniem”
    op = Opener()
    got = select_celestrak(scfg(tmp_path, ["active"]), t_rec, now=NOW, opener=op)
    assert op.urls == [] and got[0]["path"].name == "celestrak_active_20260928T1030Z.csv"


def test_old_recording_not_fetched(tmp_path):
    op = Opener()
    got = select_celestrak(scfg(tmp_path, ["active"]), NOW - timedelta(days=10), now=NOW, opener=op)
    assert op.urls == [] and got == []


def test_fetch_url_non_200():
    with pytest.raises(FetchError):
        fetch_url("https://x/y", lambda url, timeout=None: Resp(status=301))


def test_spacetrack_credentials_never_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SPACETRACK_USER", "tajny_login")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "tajne_haslo")
    calls = []

    class STOpener:
        def open(self, url, data=None, timeout=None):
            calls.append((url, data))
            if "query" in url:
                raise urllib.error.HTTPError(url + "?tajne_haslo", 500, "blad tajny_login", {}, None)
            return Resp(b"")

    with caplog.at_level(logging.WARNING):
        assert fetch_spacetrack(scfg(tmp_path), T_REC, now=NOW, opener_factory=STOpener) is None
    assert "tajne_haslo" not in caplog.text and "tajny_login" not in caplog.text
    assert b"tajne_haslo" in calls[0][1]   # hasło idzie tylko w treści POST logowania

    class OK(STOpener):
        def open(self, url, data=None, timeout=None):
            calls.append((url, data))
            return Resp(b"NORAD_CAT_ID,EPOCH\n25544,2026-09-27T12:00:00\n")

    got = fetch_spacetrack(scfg(tmp_path), T_REC, now=NOW, opener_factory=OK)
    assert got["path"].exists() and got["group"] == "gp-history-2026-09-24--2026-09-29"
    n = len(calls)
    assert fetch_spacetrack(scfg(tmp_path), T_REC, now=NOW, opener_factory=OK)["path"] == got["path"]
    assert len(calls) == n            # drugi raz z cache


def test_spacetrack_disabled_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    assert fetch_spacetrack(scfg(tmp_path), T_REC, now=NOW) is None
