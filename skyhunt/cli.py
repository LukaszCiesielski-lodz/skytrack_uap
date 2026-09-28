"""Interfejs wiersza poleceń: ``skyhunt probe | bench-decode | run | status``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import cfg_get, config_for_file, load_config, parse_set
from .io import to_jsonable, write_json

log = logging.getLogger("skyhunt")


def iter_inputs(path: Path, extensions: list[str]) -> list[Path]:
    """Plik → [plik]; katalog → pliki wideo (bez podkatalogów), posortowane."""
    path = Path(path)
    if path.is_dir():
        exts = {e.lower() for e in extensions}
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None, help="ścieżka do config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KLUCZ=WARTOŚĆ",
                   help="nadpisanie configu, np. --set decode.batch_frames=16 (można powtarzać)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    ap = argparse.ArgumentParser(prog="skyhunt", description="Detekcja obiektów ruchomych na wideo nieba.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", parents=[common], help="metadane i wstępny czas nagrania (bez zapisu)")
    p.add_argument("input", type=Path, help="plik wideo albo katalog")

    b = sub.add_parser("bench-decode", parents=[common], help="przepustowość backendów dekodowania")
    b.add_argument("video", type=Path)
    b.add_argument("--backends", default=None, help="lista po przecinku (domyślnie wszystkie)")
    b.add_argument("--frames", type=int, default=None, help="liczba klatek (domyślnie bench.frames)")
    b.add_argument("--batch", type=int, default=None)
    b.add_argument("--out", type=Path, default=None, help="katalog na decode_bench.json/.md")

    r = sub.add_parser("run", parents=[common], help="uruchom pipeline (ze wznawianiem)")
    r.add_argument("input", type=Path, nargs="?", default=None, help="plik lub katalog (domyślnie paths.raw_dir)")
    r.add_argument("--out", type=Path, default=None, help="katalog wyników (domyślnie paths.out_dir)")
    r.add_argument("--stages", default=None, help="etapy po przecinku (z zależnościami); domyślnie wszystkie")
    r.add_argument("--force", default="", help="etapy do przeliczenia (z zależnymi), 'all' = wszystko")

    s = sub.add_parser("status", parents=[common], help="stan etapów z manifestów")
    s.add_argument("input", type=Path, nargs="?", default=None)
    s.add_argument("--out", type=Path, default=None)
    return ap


def _split(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def cmd_probe(args, cfg) -> int:
    from .metadata import probe_video
    from .timing import time_prior

    for p in iter_inputs(args.input, cfg["input"]["extensions"]):
        meta = probe_video(p)
        try:
            prior = time_prior(meta, config_for_file(cfg, p)["time"]).to_dict()
        except ValueError as e:
            prior = {"error": str(e)}
        d = meta.to_dict()
        d["keyframes"] = f"{len(meta.keyframes)} klatek kluczowych"   # pełna lista w meta.json
        print(json.dumps(to_jsonable({"video": d, "time_prior": prior}), indent=2, ensure_ascii=False))
    return 0


def cmd_bench(args, cfg) -> int:
    from .bench import benchmark, environment_info, to_markdown
    from .metadata import probe_video

    meta = probe_video(args.video)
    n = args.frames or cfg_get(cfg, "bench.frames", 480)
    env = environment_info()
    results = benchmark(args.video, meta, cfg["decode"], _split(args.backends) or None, n, args.batch)
    md = to_markdown(results, meta, env, n)
    print(md)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "decode_bench.json", {"env": env, "frames": n, "results": results})
        (args.out / "decode_bench.md").write_text(md, encoding="utf-8")
        log.info("zapisano %s", args.out / "decode_bench.md")
    return 0 if any(r["ok"] for r in results) else 1


def cmd_run(args, cfg) -> int:
    from . import stages  # noqa: F401 — rejestracja etapów
    from .pipeline import PIPELINE

    src = args.input or Path(cfg["paths"]["raw_dir"])
    out = args.out or Path(cfg["paths"]["out_dir"])
    files = iter_inputs(src, cfg["input"]["extensions"])
    if not files:
        log.error("brak plików wideo w %s", src)
        return 1
    failed = []
    for f in files:
        try:
            st = PIPELINE.run(f, cfg, out, only=_split(args.stages) or None, force=_split(args.force), log=log)
            log.info("[%s] %s", f.name, st)
        except Exception:  # noqa: BLE001 — błąd jednego pliku nie zatrzymuje pozostałych
            log.exception("[%s] błąd", f.name)
            failed.append(f.name)
    if failed:
        log.error("nieudane pliki: %s", failed)
    return 1 if failed else 0


def cmd_status(args, cfg) -> int:
    from .io import read_json

    src = args.input or Path(cfg["paths"]["raw_dir"])
    out = args.out or Path(cfg["paths"]["out_dir"])
    for f in iter_inputs(src, cfg["input"]["extensions"]):
        mpath = out / f.stem / "manifest.json"
        if not mpath.exists():
            print(f"{f.name}: brak manifestu")
            continue
        stages = read_json(mpath).get("stages", {})
        summary = ", ".join(f"{k}={v.get('status')}" for k, v in stages.items()) or "brak etapów"
        print(f"{f.name}: {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, parse_set(args.set))
    log.debug("config: %s", cfg["_source"])
    handler = {"probe": cmd_probe, "bench-decode": cmd_bench, "run": cmd_run, "status": cmd_status}[args.cmd]
    return handler(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
