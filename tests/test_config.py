import copy

import pytest

from skyhunt.config import cfg_get, config_for_file, config_hash, deep_merge, parse_set, validate


def test_repo_config_is_valid(cfg):
    assert cfg["site"]["lat_deg"] == pytest.approx(51.718042)
    assert cfg["site"]["elevation_m"] == 210
    assert cfg["decode"]["backends"][0] == "nvcodec"


def test_hash_stable_and_section_sensitive(cfg):
    h = config_hash(cfg, ["time"])
    assert h == config_hash(copy.deepcopy(cfg), ["time"])
    changed = deep_merge(cfg, {"time": {"camera_clock_ahead_s": 1}})
    assert config_hash(changed, ["time"]) != h
    assert config_hash(changed, ["decode"]) == config_hash(cfg, ["decode"])


def test_hash_of_dotted_subkey(cfg):
    a = config_hash(cfg, ["decode.exact_luma_required"])
    b = config_hash(deep_merge(cfg, {"decode": {"batch_frames": 7}}), ["decode.exact_luma_required"])
    assert a == b


def test_per_file_overrides(cfg):
    old = config_for_file(cfg, "/drive/raw/DSCF4641.MOV")
    assert old["time"]["camera_clock_ahead_s"] == 600
    assert old["astrometry"]["hint_star"] == "Deneb"
    new = config_for_file(cfg, "/drive/raw/DSCF9999.MOV")
    assert new["time"]["camera_clock_ahead_s"] == 0
    assert config_hash(old, ["time"]) != config_hash(new, ["time"])


def test_sites_file_overrides_site_per_file(tmp_path):
    from skyhunt.config import PACKAGE_CONFIG, load_config

    sites = tmp_path / "sites.yaml"
    sites.write_text("DSCF0001.MOV: {lat_deg: 50.5, lon_deg: 20.25, elevation_m: 300}\n", encoding="utf-8")
    cfg = load_config(PACKAGE_CONFIG, sites=sites)
    a = config_for_file(cfg, "/raw/DSCF0001.MOV")
    assert (a["site"]["lat_deg"], a["site"]["lon_deg"], a["site"]["elevation_m"]) == (50.5, 20.25, 300)
    assert config_for_file(cfg, "/raw/DSCF0002.MOV")["site"] == cfg["site"]
    assert config_for_file(cfg, "/raw/DSCF4641.MOV")["time"]["camera_clock_ahead_s"] == 600  # repo nadal działa
    sites.write_text("DSCF0001.MOV: {lat_deg: 99, lon_deg: 20}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(PACKAGE_CONFIG, sites=sites)


def test_sites_file_hint_star_per_file(tmp_path):
    from skyhunt.config import PACKAGE_CONFIG, load_config

    sites = tmp_path / "sites.yaml"
    sites.write_text("DSCF0001.MOV: {lat_deg: 50.5, lon_deg: 20.25, elevation_m: 300, hint_star: Altair}\n"
                     "DSCF0002.MOV: {lat_deg: 50.5, lon_deg: 20.25, elevation_m: 300, hint_star: null}\n",
                     encoding="utf-8")
    cfg = load_config(PACKAGE_CONFIG, sites=sites)
    a = config_for_file(cfg, "/raw/DSCF0001.MOV")
    assert a["astrometry"]["hint_star"] == "Altair" and "hint_star" not in a["site"]
    assert config_for_file(cfg, "/raw/DSCF0002.MOV")["astrometry"]["hint_star"] == "Deneb"   # domyślna
    sites.write_text("DSCF0001.MOV: {lat_deg: 50.5, lon_deg: 20.25, elevation_m: 300, hint_star: Nieznana}\n",
                     encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(PACKAGE_CONFIG, sites=sites)


def test_dark_role_from_sites_file_or_name(tmp_path):
    from skyhunt.config import PACKAGE_CONFIG, load_config

    sites = tmp_path / "sites.yaml"
    sites.write_text("DSCF0100.MOV: {role: dark}\n", encoding="utf-8")
    cfg = load_config(PACKAGE_CONFIG, sites=sites)
    assert config_for_file(cfg, "/raw/DSCF0100.MOV")["role"] == "dark"
    assert "site" not in (cfg["files"]["DSCF0100.MOV"])
    assert config_for_file(cfg, "/raw/Dark_0929.MOV")["role"] == "dark"      # z nazwy pliku
    for name in ("DSCF0100.MOV", "Dark_0929.MOV", "dark_frames.MOV"):        # wspólne ustawienia ciemnych
        assert config_for_file(cfg, f"/raw/{name}")["detect"]["sigma_floor_dn"] == 1.5
    assert config_for_file(cfg, "/raw/DSCF0101.MOV")["detect"]["sigma_floor_dn"] == 0.5
    assert config_for_file(cfg, "/raw/DSCF0101.MOV").get("role", "sky") == "sky"


def test_parse_set():
    o = parse_set(["decode.batch_frames=16", "decode.backends=[pyav]", "time.camera_tz=UTC"])
    assert o == {"decode": {"batch_frames": 16, "backends": ["pyav"]}, "time": {"camera_tz": "UTC"}}
    with pytest.raises(ValueError):
        parse_set(["bez_znaku_rownosci"])


def test_cfg_get():
    assert cfg_get({"a": {"b": 1}}, "a.b") == 1
    assert cfg_get({"a": {}}, "a.x", None) is None
    with pytest.raises(KeyError):
        cfg_get({}, "a.b")


@pytest.mark.parametrize("patch", [
    {"site": {"lat_deg": 123}},
    {"time": {"camera_tz": "Mars/Olympus"}},
    {"time": {"start_source": "zgadnij"}},
    {"decode": {"backends": ["vhs"]}},
    {"decode": {"batch_frames": 0}},
])
def test_validate_rejects(cfg, patch):
    with pytest.raises(ValueError):
        validate(deep_merge(cfg, patch))
