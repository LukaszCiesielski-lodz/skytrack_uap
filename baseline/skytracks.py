#!/usr/bin/env python3
"""
skytracks.py - detekcja i klasyfikacja obiektow ruchomych na nagraniach nieba
(statyw, ostrosc na nieskonczonosc). Czyta wideo strumieniowo przez ffmpeg,
wiec dziala na plikach wielo-GB bez ladowania calosci do RAM.

Uzycie:
  python skytracks.py nagranie.mp4 --hfov 26.4
  python skytracks.py katalog_z_mp4/ --hfov 26.4 --scale 1920

Wymaga: ffmpeg/ffprobe w PATH, pip install numpy scipy pillow

--hfov  poziome pole widzenia w stopniach. Dla 50 mm na APS-C (23.5 mm) = 26.4.
        Jesli aparat przycina 4K (crop), podaj mniejsze pole - najlepiej
        skalibruj: wrzuc klatke do nova.astrometry.net i wez pole z wyniku.

Wyjscie (obok pliku wideo, katalog <nazwa>_tracks/):
  tracks.csv  - jeden wiersz na tor z metrykami i klasa
  stack.png   - max-minus-tlo z ponumerowanymi torami
"""
import argparse, json, subprocess, sys, csv
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    n = int(s.get("nb_frames") or float(j["format"]["duration"]) * fps)
    return int(s["width"]), int(s["height"]), fps, n


def frames(path, w, h):
    """Generator klatek w skali szarosci (uint8), przeskalowanych do w x h."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    size = w * h
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w)
    p.wait()


def background(path, w, h, n, samples=60):
    """Mediana z rownomiernie probkowanych klatek (gwiazdy na statywie sa praktycznie stale)."""
    step = max(1, n // samples)
    acc = [f.copy() for i, f in enumerate(frames(path, w, h)) if i % step == 0]
    return np.median(np.stack(acc), 0).astype(np.float32)


def star_width(bg, thr):
    """Typowa szerokosc (sigma, px) gwiazd liczona TA SAMA metoda co dla obiektow ruchomych."""
    d = ndimage.uniform_filter(bg - ndimage.median_filter(bg, 15), 3)
    lab, n = ndimage.label(d > thr)
    sig = []
    for k, s in enumerate(ndimage.find_objects(lab)):
        m = lab[s] == k + 1
        if not 4 <= m.sum() <= 60:
            continue
        yy, xx = np.nonzero(m)
        cov = np.cov(np.vstack([xx, yy]), aweights=np.clip(d[s][m], 1e-3, None))
        sig.append(np.sqrt(max(np.linalg.eigvalsh(cov)[0], 0)))
    return float(np.median(sig)) if sig else 1.0


def detect(f, bg, thr):
    d = ndimage.uniform_filter(f.astype(np.float32) - bg, 3)
    lab, n = ndimage.label(d > thr)
    out = []
    for k, s in enumerate(ndimage.find_objects(lab)):
        m = lab[s] == k + 1
        if m.sum() < 4:
            continue
        yy, xx = np.nonzero(m)
        wts = np.clip(d[s][m], 1e-3, None)
        cy, cx = np.average(yy, weights=wts), np.average(xx, weights=wts)
        cov = np.cov(np.vstack([xx, yy]), aweights=wts) if m.sum() > 2 else np.eye(2)
        out.append(dict(x=cx + s[1].start, y=cy + s[0].start, flux=float(wts.sum()), cov=cov))
    return out


def link(dets_by_frame, static_min=8, gap=2):
    # usun obiekty statyczne (migoczace gwiazdy, hot pixele)
    from collections import Counter
    cnt = Counter((int(d["x"] / 4), int(d["y"] / 4)) for fr in dets_by_frame for d in fr)
    tracks = []
    for i, fr in enumerate(dets_by_frame):
        for d in fr:
            if cnt[(int(d["x"] / 4), int(d["y"] / 4))] >= static_min:
                continue
            best, bd = None, 1e9
            for t in tracks:
                li, ld = t[-1]
                if not 1 <= i - li <= gap:
                    continue
                if len(t) >= 2:
                    pi, pd = t[-2]
                    vx, vy = (ld["x"] - pd["x"]) / (li - pi), (ld["y"] - pd["y"]) / (li - pi)
                    px, py = ld["x"] + vx * (i - li), ld["y"] + vy * (i - li)
                    lim = 6 + 0.1 * np.hypot(vx, vy)
                else:
                    px, py, lim = ld["x"], ld["y"], 40
                dist = np.hypot(d["x"] - px, d["y"] - py)
                if dist < lim and dist < bd:
                    best, bd = t, dist
            (best.append((i, d)) if best is not None else tracks.append([(i, d)]))
    return tracks


def analyze(t, fps, degpx, star_sig):
    fr = np.array([i for i, _ in t], float)
    x = np.array([d["x"] for _, d in t]); y = np.array([d["y"] for _, d in t])
    fl = np.array([d["flux"] for _, d in t])
    px_, py_ = np.polyfit(fr, x, 1), np.polyfit(fr, y, 1)
    v = np.hypot(px_[0], py_[0])                      # px / klatke
    res = np.sqrt(np.mean((np.polyval(px_, fr) - x) ** 2 + (np.polyval(py_, fr) - y) ** 2))
    # szerokosc poprzeczna do kierunku ruchu
    nvec = np.array([-py_[0], px_[0]]) / (v + 1e-9)
    cross = np.median([np.sqrt(max(nvec @ d["cov"] @ nvec, 0)) for _, d in t])
    # okresowosc jasnosci
    f_peak, p_ratio = np.nan, 0.0
    if len(fl) >= 24:
        k = 9
        det = fl - np.convolve(np.pad(fl, k // 2, mode="edge"), np.ones(k) / k, "valid")
        F = np.abs(np.fft.rfft(det * np.hanning(len(det))))
        fq = np.fft.rfftfreq(len(det), 1 / fps)
        F[fq < 0.5] = 0
        j = int(np.argmax(F))
        f_peak, p_ratio = float(fq[j]), float(F[j] / (np.median(F[F > 0]) + 1e-9))
    return dict(start_s=fr[0] / fps, dur_s=(fr[-1] - fr[0]) / fps, n=len(t),
                x0=x[0], y0=y[0], x1=x[-1], y1=y[-1],
                deg_s=v * fps * degpx, curv_px=res, cross_ratio=cross / star_sig,
                flux_cv=float(fl.std() / fl.mean()), f_peak_hz=f_peak, f_power=p_ratio)


def classify(m, fps):
    periodic = m["f_power"] > 6
    straight = m["curv_px"] < 2.0
    near = m["cross_ratio"] > 1.8
    if m["deg_s"] > 1.8 and m["dur_s"] < 2.0 and straight:
        return "meteor?"
    if periodic and 0.4 <= m["f_peak_hz"] <= 2.0 and m["deg_s"] < 1.8:
        return "samolot? (swiatla stroboskopowe)"
    if near or (periodic and m["f_peak_hz"] > 2.0) or not straight:
        alias = fps - m["f_peak_hz"] if periodic else None
        tag = f" trzepot {m['f_peak_hz']:.1f} lub {alias:.1f} Hz" if periodic else ""
        return "BLISKI: ptak/nietoperz/owad" + tag
    if straight and 0.05 <= m["deg_s"] <= 1.8:
        return "satelita"
    return "niesklasyfikowany"


def run(path, hfov, scale, thr, min_len):
    W, H, fps, n = probe(path)
    w, h = scale, int(round(scale * H / W / 2)) * 2
    degpx = hfov / w
    print(f"[{path.name}] {W}x{H} {fps:.3f} fps ~{n} klatek -> analiza w {w}x{h}, {degpx*3600:.0f}\"/px")
    bg = background(path, w, h, n)
    ssig = star_width(bg, thr)
    dets, mx = [], np.zeros_like(bg)
    for f in frames(path, w, h):
        dets.append(detect(f, bg, thr))
        np.maximum(mx, f, out=mx)
    tracks = [t for t in link(dets) if len(t) >= min_len
              and np.hypot(t[-1][1]["x"] - t[0][1]["x"], t[-1][1]["y"] - t[0][1]["y"]) > 10]
    outdir = path.with_name(path.stem + "_tracks"); outdir.mkdir(exist_ok=True)
    rows = []
    for k, t in enumerate(tracks, 1):
        m = analyze(t, fps, degpx, ssig)
        m["id"], m["klasa"] = k, classify(m, fps)
        rows.append(m)
    keys = ["id", "klasa", "start_s", "dur_s", "n", "deg_s", "curv_px", "cross_ratio",
            "flux_cv", "f_peak_hz", "f_power", "x0", "y0", "x1", "y1"]
    with open(outdir / "tracks.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys, extrasaction="ignore"); wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    img = Image.fromarray(np.clip((mx - bg) * 4, 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    for r in rows:
        col = (255, 80, 80) if r["klasa"].startswith("BLISKI") else (80, 200, 255) if r["klasa"] == "satelita" else (255, 220, 60)
        dr.text((r["x0"] + 5, r["y0"] + 5), str(r["id"]), fill=col)
    img.save(outdir / "stack.png")
    for r in rows:
        print(f"  #{r['id']:>2} t={r['start_s']:6.1f}s {r['deg_s']:5.2f}°/s dur={r['dur_s']:4.1f}s "
              f"krzyw={r['curv_px']:4.1f}px szer/gw={r['cross_ratio']:.2f} "
              f"f={r['f_peak_hz']:.1f}Hz(x{r['f_power']:.0f})  -> {r['klasa']}")
    print(f"  wyniki: {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="plik wideo albo katalog")
    ap.add_argument("--hfov", type=float, default=26.4, help="poziome pole widzenia [deg]")
    ap.add_argument("--scale", type=int, default=1920, help="szerokosc analizy [px]")
    ap.add_argument("--thr", type=float, default=12, help="prog detekcji ponad tlem [DN]")
    ap.add_argument("--min-len", type=int, default=6, help="min. liczba klatek toru")
    a = ap.parse_args()
    files = sorted(p for p in a.input.iterdir() if p.suffix.lower() in (".mp4", ".mov")) if a.input.is_dir() else [a.input]
    for p in files:
        try:
            run(p, a.hfov, a.scale, a.thr, a.min_len)
        except Exception as e:
            print(f"[{p.name}] blad: {e}", file=sys.stderr)
