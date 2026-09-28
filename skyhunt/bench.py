"""Pomiar przepustowości backendów dekodowania i zgodności ich wyników (parytet Y)."""
from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .decode import BACKENDS, make_decoder
from .metadata import VideoMeta

# Kolejność wyboru referencji do porównania (backendy dające oryginalne Y).
_REFERENCE_ORDER = ("pyav", "ffmpeg", "nvcodec", "torchaudio")


def _sync(y) -> None:
    if type(y).__module__.startswith("torch") and y.is_cuda:
        import torch

        torch.cuda.synchronize(y.device)


def _to_numpy(y) -> np.ndarray:
    return y.detach().cpu().numpy() if type(y).__module__.startswith("torch") else np.array(y)


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "av", "PyNvVideoCodec", "torchcodec", "torchaudio", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def benchmark(path: Path, meta: VideoMeta, dcfg: dict, backends: Iterable[str] | None = None,
              n_frames: int = 480, batch_size: int | None = None) -> list[dict]:
    """Dla każdego backendu: czas inicjalizacji (1. paczka), przepustowość na kolejnych
    paczkach (z synchronizacją GPU) i różnica Y względem backendu referencyjnego."""
    bs = int(batch_size or dcfg.get("batch_frames", 32))
    results, samples = [], {}
    for name in backends or BACKENDS:
        r: dict = {"backend": name, "ok": False}
        it = None
        try:
            dec = make_decoder(name, path, meta, dcfg)
            r.update(device=dec.device, exact_luma=dec.exact_luma)
            t_open = time.perf_counter()
            it = dec.batches(bs, stop=n_frames)
            first = next(it)
            _sync(first.y)
            r["init_s"] = time.perf_counter() - t_open
            samples[name] = _to_numpy(first.y[: min(4, len(first))])
            n, last, t0 = 0, None, time.perf_counter()
            for b in it:
                n += len(b)
                last = b.y
            if last is not None:
                _sync(last)
            dt = time.perf_counter() - t0
            r["frames_timed"] = n
            if n and dt > 0:
                r.update(fps=n / dt, mpix_per_s=n * meta.width * meta.height / dt / 1e6,
                         realtime_x=n / dt / meta.fps)
            r["ok"] = True
        except Exception as e:  # noqa: BLE001 — raportujemy i idziemy dalej
            r["error"] = f"{type(e).__name__}: {e}"
        finally:
            if it is not None:
                it.close()
        results.append(r)

    ref = next((n for n in _REFERENCE_ORDER if n in samples), None)
    for r in results:
        if not r["ok"] or ref is None or r["backend"] == ref:
            continue
        a, b = samples[r["backend"]], samples[ref]
        r["parity_ref"] = ref
        if a.shape != b.shape:
            r["parity_error"] = f"kształt {a.shape} vs {b.shape}"
            continue
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        r.update(max_abs_diff=int(d.max()), mean_abs_diff=float(d.mean()))
    return results


def to_markdown(results: list[dict], meta: VideoMeta, env: dict, n_frames: int) -> str:
    lines = [
        f"Plik: `{Path(meta.path).name}` ({meta.width}×{meta.height} {meta.codec}, {meta.fps:.3f} fps), "
        f"{n_frames} klatek; GPU: {env.get('gpu', 'brak')}; torch {env.get('torch')}, "
        f"PyAV {env.get('av')}, PyNvVideoCodec {env.get('PyNvVideoCodec')}, "
        f"torchcodec {env.get('torchcodec')}, torchaudio {env.get('torchaudio')}",
        "",
        "| backend | urządzenie | init [s] | kl/s | × czas rzecz. | max |ΔY| vs ref | uwagi |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if not r["ok"]:
            lines.append(f"| {r['backend']} | – | – | – | – | – | {r.get('error', '')[:120]} |")
            continue
        diff = r.get("max_abs_diff", "ref" if r.get("parity_ref") is None else "–")
        note = "" if r["exact_luma"] else "Y z RGB"
        lines.append(f"| {r['backend']} | {r['device']} | {r['init_s']:.2f} | {r.get('fps', 0):.1f} | "
                     f"{r.get('realtime_x', 0):.2f} | {diff} | {note} |")
    return "\n".join(lines) + "\n"
