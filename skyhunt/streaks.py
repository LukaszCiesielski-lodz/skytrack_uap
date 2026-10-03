"""Kreski na zdjęciach RAW (F2): wykrywanie, końce, łańcuchy jednego obiektu, przerwa w serii.

Wszystko dzieje się w układzie zdjęcia odniesienia (stos): piksel = stały kierunek ICRS, więc
gwiazdy stoją, a tor satelity jest prostą (rzut gnomoniczny). Różnica D = zdjęcie − mediana
sąsiednich zdjęć tej samej klasy jasności usuwa gwiazdy i tło; kreska z sąsiada nie przechodzi
przez medianę. Jasne gwiazdy (resztki po wyrównaniu) są maskowane.

Wykrywanie: bank filtrów odcinkowych na D/σ (odpowiedź w jednostkach σ, normalizowana
empirycznie, bo interpolacja przy wyrównaniu koreluje szum), progi z histerezą, składowe
spójne, PCA → wydłużone obiekty, sklejanie współliniowych kawałków (kreska przecięta maską
gwiazdy, przerywane światła samolotu). Końce: dopasowanie profilu wzdłuż kreski modelem
„prostokąt ⊗ PSF” (dwie funkcje erf), środek poprzeczny z centroidów.

Łańcuchy: kreski jednego obiektu w kolejnych zdjęciach (ruch x(t), y(t) wielomianem); kierunek
lotu wynika z kolejności. Przerwa g między zdjęciami serii z geometrii łańcuchów (bez TLE).
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- filtr odcinkowy


def line_kernels(length: int, n_angles: int) -> np.ndarray:
    """Jądra odcinków długości ``length`` (nieparzysta) w ``n_angles`` kierunkach 0…180°, rysowane
    wagami dwuliniowymi, Σw² = 1 (odpowiedź na biały szum o σ = 1 ma σ = 1)."""
    k = int(length) | 1
    c = k // 2
    out = np.zeros((int(n_angles), k, k), np.float64)
    s = np.linspace(-c, c, 8 * k)
    for a in range(int(n_angles)):
        th = math.pi * a / n_angles
        xs, ys = c + s * math.cos(th), c + s * math.sin(th)
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                xi, yi = x0 + dx, y0 + dy
                ok = (xi >= 0) & (xi < k) & (yi >= 0) & (yi < k)
                np.add.at(out[a], (yi[ok], xi[ok]), (wx * wy)[ok])
        out[a] /= np.sqrt((out[a] ** 2).sum())
    return out.astype(np.float32)


def line_response(z, kernels: np.ndarray, device: str = "cpu"):
    """Maksimum po kierunkach odpowiedzi filtrów odcinkowych na obraz S/N ``z`` [h, w]
    (tensor albo numpy). Każdy kierunek normalizowany medianą i MAD (szum skorelowany).
    Zwraca tensor [h, w] na ``device``."""
    import torch

    F = torch.nn.functional
    zt = z if torch.is_tensor(z) else torch.from_numpy(np.ascontiguousarray(z, np.float32))
    zt = zt.to(device).float()[None, None]
    kt = torch.from_numpy(kernels)[:, None].to(zt.device)
    r = F.conv2d(zt, kt, padding=kernels.shape[-1] // 2)[0]
    sub = r[:, ::5, ::5].reshape(r.shape[0], -1)
    med = sub.median(dim=1).values
    mad = 1.4826 * (sub - med[:, None]).abs().median(dim=1).values
    r = (r - med[:, None, None]) / mad.clamp(min=1e-6)[:, None, None]
    return r.max(dim=0).values


def neighbor_median(stack):
    """Mediana (z NaN) po osi 0 tensora [n, h, w]; dla parzystej liczby wartości średnia dwóch
    środkowych. Zwraca (mediana, liczba wartości)."""
    import torch

    finite = torch.isfinite(stack)
    n = finite.sum(dim=0)
    v = torch.where(finite, stack, torch.full_like(stack, float("inf"))).sort(dim=0).values
    lo = ((n - 1).clamp(min=0) // 2)[None].long()
    hi = (n // 2).clamp(max=stack.shape[0] - 1)[None].long()
    med = 0.5 * (v.gather(0, lo) + v.gather(0, hi))[0]
    return torch.where(n > 0, med, torch.full_like(med, float("nan"))), n


def block_sigma(img: np.ndarray, box: int = 64) -> np.ndarray:
    """Mapa szumu: MAD w blokach ``box`` px, powiększona do pełnego obrazu (stała w bloku)."""
    h, w = img.shape
    hb, wb = max(h // box, 1), max(w // box, 1)
    a = img[:hb * box, :wb * box].reshape(hb, box, wb, box).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    med = np.nanmedian(a, axis=2)
    sig = 1.4826 * np.nanmedian(np.abs(a - med[..., None]), axis=2)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig))
    ys = np.minimum(np.arange(h) // box, hb - 1)
    xs = np.minimum(np.arange(w) // box, wb - 1)
    return sig[np.ix_(ys, xs)].astype(np.float32)


# ---------------------------------------------------------------- składowe → odcinki


def components(rmax: np.ndarray, dcfg: dict) -> list[dict]:
    """Składowe ``rmax > grow_sigma`` z maksimum ≥ ``det_sigma`` → wydłużone odcinki (PCA).
    Punkt (gwiazda, gorący piksel, promień kosmiczny) daje w maksimum po kierunkach okrągłą plamę."""
    from scipy import ndimage as ndi

    det, grow = float(dcfg["det_sigma"]), float(dcfg["grow_sigma"])
    lab, n = ndi.label(rmax > grow, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    peak = np.asarray(ndi.maximum(rmax, lab, index=np.arange(1, n + 1)))
    keep = np.flatnonzero(peak >= det) + 1
    keep = keep[np.argsort(-peak[keep - 1])][: int(dcfg["max_per_photo"]) * 4]
    objs = ndi.find_objects(lab)
    half = (int(dcfg["line_len_px"]) | 1) / 2.0
    out = []
    for k in keep:
        sl = objs[k - 1]
        m = lab[sl] == k
        ys, xs = np.nonzero(m)
        if len(ys) < int(dcfg["min_pix"]):
            continue
        w = rmax[sl][m] - grow + 1e-3
        xs = xs + sl[1].start
        ys = ys + sl[0].start
        cx, cy = np.average(xs, weights=w), np.average(ys, weights=w)
        dx, dy = xs - cx, ys - cy
        cov = np.array([[np.average(dx * dx, weights=w), np.average(dx * dy, weights=w)],
                        [np.average(dx * dy, weights=w), np.average(dy * dy, weights=w)]])
        ev, evec = np.linalg.eigh(cov)
        if ev[1] <= 0 or math.sqrt(ev[1] / max(ev[0], 1e-6)) < float(dcfg["min_elong"]):
            continue
        u = evec[:, 1]
        proj = dx * u[0] + dy * u[1]
        lo, hi = float(proj.min()), float(proj.max())
        if hi - lo - 2 * half < 0.5 * float(dcfg["min_len_px"]):   # filtr wydłuża o ~pół jądra z każdej strony
            continue
        shrink = min(half, 0.25 * (hi - lo))
        p0 = np.array([cx, cy]) + u * (lo + shrink)
        p1 = np.array([cx, cy]) + u * (hi - shrink)
        out.append({"p0": p0, "p1": p1, "peak": float(peak[k - 1]), "pieces": [(0.0, float(np.hypot(*(p1 - p0))))]})
    return out


def _line_dist(p, a, u) -> float:
    d = np.asarray(p, float) - a
    return abs(d[0] * u[1] - d[1] * u[0])


def merge_collinear(segs: list[dict], dcfg: dict) -> list[dict]:
    """Skleja współliniowe odcinki jednego zdjęcia (przerwa ≤ ``merge_gap_px``). Zapamiętuje
    kawałki (pozycje wzdłuż osi), żeby rozpoznać przerywaną kreskę samolotu."""
    segs = [dict(s) for s in segs]
    tol_a = math.radians(float(dcfg["merge_angle_deg"]))
    tol_p, gap = float(dcfg["merge_perp_px"]), float(dcfg["merge_gap_px"])
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                ua = (a["p1"] - a["p0"]) / max(np.hypot(*(a["p1"] - a["p0"])), 1e-9)
                ub = (b["p1"] - b["p0"]) / max(np.hypot(*(b["p1"] - b["p0"])), 1e-9)
                if math.acos(min(1.0, abs(float(ua @ ub)))) > tol_a:
                    continue
                sa = sorted(((a["p0"] - a["p0"]) @ ua, (a["p1"] - a["p0"]) @ ua))
                sb = sorted(((b["p0"] - a["p0"]) @ ua, (b["p1"] - a["p0"]) @ ua))
                g = max(sb[0] - sa[1], sa[0] - sb[1])
                if g > gap:
                    continue
                if max(_line_dist(b["p0"], a["p0"], ua), _line_dist(b["p1"], a["p0"], ua)) > tol_p + 0.01 * max(g, 0):
                    continue
                pts = [a["p0"], a["p1"], b["p0"], b["p1"]]
                s = [float((p - a["p0"]) @ ua) for p in pts]
                lo, hi = int(np.argmin(s)), int(np.argmax(s))
                base = min(s)
                ob = s[2] - base                      # kawałki b liczone od b.p0 wzdłuż ub (może być ub = −ua)
                pieces = ([(p0 - base, p1 - base) for p0, p1 in a["pieces"]]
                          + [((ob + p0, ob + p1) if ua @ ub > 0 else (ob - p1, ob - p0)) for p0, p1 in b["pieces"]])
                segs[i] = {"p0": pts[lo], "p1": pts[hi], "peak": max(a["peak"], b["peak"]),
                           "pieces": sorted(pieces)}
                segs.pop(j)
                changed = True
                break
            if changed:
                break
    return segs


def is_dashed(pieces: list[tuple[float, float]], min_pieces: int, max_cv: float) -> bool:
    """Przerywana kreska o równych odstępach (światła pozycyjne samolotu)."""
    if len(pieces) < int(min_pieces):
        return False
    c = np.array([0.5 * (a + b) for a, b in sorted(pieces)])
    d = np.diff(c)
    return bool(d.mean() > 0 and d.std() / d.mean() <= float(max_cv))


# ---------------------------------------------------------------- końce kreski


def _box_model(p, s):
    from scipy.special import erf

    b, amp, s0, s1, w = p
    k = 1.0 / (math.sqrt(2.0) * w)
    return b + 0.5 * amp * (erf((s - s0) * k) - erf((s - s1) * k))


def refine_streak(diff: np.ndarray, p0: np.ndarray, p1: np.ndarray, dcfg: dict) -> dict | None:
    """Doprecyzowanie odcinka na obrazie różnicowym (NaN = maska): oś z centroidów poprzecznych,
    końce i amplituda z dopasowania profilu wzdłuż osi (prostokąt ⊗ PSF). Zwraca końce a, b,
    ich niepewności [px] wzdłuż osi, amplitudę (suma w poprzek na px długości), S/N i profil."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares

    p0, p1 = np.asarray(p0, float).copy(), np.asarray(p1, float).copy()
    half = int(dcfg["profile_half_px"])
    pad = float(dcfg["profile_pad_px"])
    for _ in range(2):
        L = float(np.hypot(*(p1 - p0)))
        if L < 2:
            return None
        u = (p1 - p0) / L
        n = np.array([-u[1], u[0]])
        ss = np.arange(0.0, L + 1e-6, 2.0)
        offs = np.arange(-4.0, 4.01, 0.5)
        X = p0[0] + ss[:, None] * u[0] + offs[None] * n[0]
        Y = p0[1] + ss[:, None] * u[1] + offs[None] * n[1]
        prof = map_coordinates(diff, [Y, X], order=1, mode="constant", cval=np.nan)
        pos = np.clip(np.nan_to_num(prof, nan=0.0), 0, None)
        wsum = pos.sum(axis=1)
        ok = wsum > np.nanmedian(wsum) * 0.5
        if ok.sum() < 3:
            break
        cen = (pos * offs[None]).sum(axis=1)[ok] / wsum[ok]
        coef = np.polyfit(ss[ok], cen, 1)
        p0 = p0 + n * coef[1]
        p1 = p1 + n * (coef[1] + coef[0] * L)
    L = float(np.hypot(*(p1 - p0)))
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    ss = np.arange(-pad, L + pad + 1e-6, 0.5)
    offs = np.arange(-half, half + 1, 1.0)
    X = p0[0] + ss[:, None] * u[0] + offs[None] * n[0]
    Y = p0[1] + ss[:, None] * u[1] + offs[None] * n[1]
    prof = map_coordinates(diff, [Y, X], order=1, mode="constant", cval=np.nan)
    cnt = np.isfinite(prof).sum(axis=1)
    f = np.where(cnt >= max(1, len(offs) // 2), np.nansum(prof, axis=1) / np.maximum(cnt, 1) * len(offs), np.nan)
    good = np.isfinite(f)
    if good.sum() < 8:
        return None
    inside = good & (ss >= 0) & (ss <= L)
    outside = good & ~((ss >= 0) & (ss <= L))
    b0 = float(np.median(f[outside])) if outside.sum() >= 3 else 0.0
    a0 = float(np.median(f[inside])) - b0 if inside.any() else float(np.nanmax(f))
    x0 = [b0, max(a0, 1e-3), 0.0, L, 1.0]
    lo = [-np.inf, 0.0, -pad, 0.0, 0.3]
    hi = [np.inf, np.inf, L + pad, L + pad, 4.0]
    try:
        res = least_squares(lambda p: _box_model(p, ss[good]) - f[good], x0, bounds=(lo, hi))
    except ValueError:
        return None
    b, amp, s0, s1, w = res.x
    if s1 < s0:
        s0, s1 = s1, s0
    r = res.fun
    dof = max(len(r) - 5, 1)
    sig_f = float(np.sqrt(np.sum(r ** 2) / dof))
    try:
        J = res.jac
        cov = np.linalg.pinv(J.T @ J) * sig_f ** 2 * 2.0     # próbki co 0,5 px są skorelowane
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    length = float(s1 - s0)
    snr = float(amp * math.sqrt(max(length, 1.0)) / max(sig_f * math.sqrt(0.5), 1e-9))
    a, bb = p0 + u * s0, p0 + u * s1
    sel = (ss >= s0 - 3) & (ss <= s1 + 3) & good
    return {"a": a, "b": bb, "length": length, "amp": float(amp), "bg": float(b), "psf_sigma": float(w),
            "err_a": float(err[2]), "err_b": float(err[3]), "snr": snr, "noise": sig_f,
            "profile_s": (ss[sel] - s0).astype(np.float32), "profile_f": (f[sel] - b).astype(np.float32)}


def end_clipped(point: np.ndarray, outward: np.ndarray, blocked: np.ndarray, edge_px: int) -> bool:
    """Koniec kreski obcięty: za nim (do ``edge_px``) brzeg kadru albo piksel zablokowany
    (poza zdjęciem, maska gwiazdy) — widoczny koniec nie jest wtedy chwilą otwarcia/zamknięcia."""
    h, w = blocked.shape
    for k in range(1, int(edge_px) + 2):
        x, y = point + outward * k
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h) or blocked[yi, xi]:
            return True
    return False


def sample_grid(coarse: np.ndarray, shape: tuple[int, int], x, y) -> np.ndarray:
    """Wartość mapy zgrubnej [gy, gx] (węzły od brzegu do brzegu obrazu ``shape``) w punktach (x, y)."""
    from scipy.ndimage import map_coordinates

    gy, gx = coarse.shape
    h, w = shape
    return map_coordinates(coarse, [np.asarray(y, float) * (gy - 1) / max(h - 1, 1),
                                    np.asarray(x, float) * (gx - 1) / max(w - 1, 1)], order=1, mode="nearest")


# ---------------------------------------------------------------- łańcuchy


def endpoint_rows(row, orient: int, rolling_s: float, height: int, dg: float = 0.0) -> list[tuple]:
    """Końce kreski w kolejności lotu: (x, y, t, czysty). ``orient`` +1: a → b, −1: b → a.
    t = otwarcie + m·dg + r·wiersz/H (+ czas naświetlania dla końca wyjścia)."""
    t_open = float(row["tau_open"]) + int(row["m"]) * dg
    A = (float(row["xa"]), float(row["ya"]), float(row["ya_s"]), not bool(row["clip_a"]))
    B = (float(row["xb"]), float(row["yb"]), float(row["yb_s"]), not bool(row["clip_b"]))
    ent, ex = (A, B) if orient > 0 else (B, A)
    T = float(row["exposure_s"])
    return [(ent[0], ent[1], t_open + rolling_s * ent[2] / height, ent[3]),
            (ex[0], ex[1], t_open + T + rolling_s * ex[2] / height, ex[3])]


def _fit_motion(P: np.ndarray, deg: int):
    """x(t), y(t) wielomianami stopnia ``deg`` z czystych końców; None, gdy za mało punktów."""
    ok = P[:, 3] > 0
    if ok.sum() < deg + 2:
        return None
    t = P[ok, 2]
    tm = float(t.mean())
    cx = np.polyfit(t - tm, P[ok, 0], deg)
    cy = np.polyfit(t - tm, P[ok, 1], deg)
    return tm, cx, cy


def _predict(model, t):
    tm, cx, cy = model
    return np.polyval(cx, np.asarray(t) - tm), np.polyval(cy, np.asarray(t) - tm)


def _speed(model, t) -> float:
    tm, cx, cy = model
    return float(np.hypot(np.polyval(np.polyder(cx), t - tm), np.polyval(np.polyder(cy), t - tm)))


def _residual(P: np.ndarray, model) -> float:
    ok = P[:, 3] > 0
    if not ok.any():
        return float("inf")
    x, y = _predict(model, P[ok, 2])
    return float(np.max(np.hypot(P[ok, 0] - x, P[ok, 1] - y)))


def _perp_ok(P: np.ndarray, model, tol: float) -> bool:
    """Obcięte końce: tylko odległość od toru (kierunek ruchu w chwili końca)."""
    for x, y, t, ok in P:
        if ok:
            continue
        px, py = _predict(model, [t - 0.05, t + 0.05])
        u = np.array([px[1] - px[0], py[1] - py[0]])
        nu = np.hypot(*u)
        if nu > 0 and _line_dist((x, y), np.array([px[0], py[0]]), u / nu) > tol:
            return False
    return True


def link_chains(st, lcfg: dict, rolling_s: float, height: int) -> list[dict]:
    """Łańcuchy kresek jednego obiektu. ``st``: tabela kresek (photo, tau_open, exposure_s, m,
    xa, ya, xb, yb, ya_s, yb_s, clip_a, clip_b, snr). Zwraca [{"items": [(indeks, orient)], …}]
    w kolejności czasu; pojedyncze kreski mają orient 0 (kierunek nieznany)."""
    rows = st.to_dict("records")
    order = sorted(range(len(rows)), key=lambda k: (rows[k]["photo"], -float(rows[k]["snr"])))
    max_gap = float(lcfg["max_gap_s"])
    pos_tol, t_tol = float(lcfg["pos_tol_px"]), float(lcfg["timing_tol_s"])
    perp_tol = float(lcfg["perp_tol_px"])
    chains: list[dict] = []

    def last_exit(ch):
        k, o = ch["items"][-1]
        q = rows[k]
        return float(q["tau_open"]) + float(q["exposure_s"])

    for k in order:
        q = rows[k]
        best, best_score = None, 1.0
        for ch in chains:
            if ch["photo"] >= q["photo"]:
                continue
            gap = float(q["tau_open"]) - last_exit(ch)
            if gap > max_gap or gap < -0.01:
                continue
            if ch["items"][0][1] == 0:     # pojedyncza kreska: oba kierunki obu kresek
                k0 = ch["items"][0][0]
                L0 = math.hypot(rows[k0]["xb"] - rows[k0]["xa"], rows[k0]["yb"] - rows[k0]["ya"])
                L1 = math.hypot(q["xb"] - q["xa"], q["yb"] - q["ya"])
                v0, v1 = L0 / float(rows[k0]["exposure_s"]), L1 / float(q["exposure_s"])
                clean = not (rows[k0]["clip_a"] or rows[k0]["clip_b"] or q["clip_a"] or q["clip_b"])
                if clean and abs(v1 / max(v0, 1e-9) - 1) > float(lcfg["speed_tol"]):
                    continue
                for oc in (1, -1):
                    for ok_ in (1, -1):
                        P = np.array(endpoint_rows(rows[k0], oc, rolling_s, height)
                                     + endpoint_rows(q, ok_, rolling_s, height), float)
                        model = _fit_motion(P, 1)
                        if model is None:
                            continue
                        tol = pos_tol + t_tol * _speed(model, P[:, 2].mean())
                        if not _perp_ok(P, model, perp_tol):
                            continue
                        score = _residual(P, model) / tol
                        if score < best_score:
                            best, best_score = (ch, [(k0, oc), (k, ok_)]), score
            else:
                Pc = np.array([e for kk, o in ch["items"] for e in endpoint_rows(rows[kk], o, rolling_s, height)], float)
                span = Pc[:, 2].max() - Pc[:, 2].min()
                deg = 2 if (len(ch["items"]) >= 4 and span >= 10) else 1
                model = _fit_motion(Pc, deg)
                if model is None:
                    continue
                for ok_ in (1, -1):
                    Pq = np.array(endpoint_rows(q, ok_, rolling_s, height), float)
                    if not Pq[:, 3].any():
                        continue
                    tol = pos_tol + t_tol * _speed(model, Pq[0, 2]) + 0.02 * _speed(model, Pq[0, 2]) * gap
                    if not _perp_ok(Pq, model, perp_tol + 0.01 * _speed(model, Pq[0, 2]) * gap):
                        continue
                    score = _residual(Pq, model) / tol
                    if score < best_score:
                        best, best_score = (ch, ch["items"] + [(k, ok_)]), score
        if best is None:
            chains.append({"items": [(k, 0)], "photo": q["photo"]})
        else:
            ch, items = best
            ch["items"], ch["photo"] = items, q["photo"]
    chains = merge_chains(chains, rows, lcfg, rolling_s, height)
    chains.sort(key=lambda c: (rows[c["items"][0][0]]["tau_open"]))
    return chains


def _chain_points(rows: list[dict], items, rolling_s: float, height: int) -> np.ndarray:
    return np.array([e for k, o in items for e in endpoint_rows(rows[k], o, rolling_s, height)], float)


def merge_chains(chains: list[dict], rows: list[dict], lcfg: dict, rolling_s: float, height: int) -> list[dict]:
    """Sklejanie porwanych łańcuchów jednego obiektu (brak kilku kresek z rzędu: słaba kreska −1 EV,
    maska jasnej gwiazdy). Ruch z końcówki łańcucha A (ostatnie ``merge_fit_s`` s, liniowo)
    przedłużony na łańcuch B zaczynający się ≤ ``merge_max_gap_s`` s później; wszystkie czyste
    końce B muszą leżeć w tolerancji rosnącej z przerwą (prędkość kątowa satelity się zmienia)."""
    max_gap = float(lcfg.get("merge_max_gap_s", 30.0))
    fit_s = float(lcfg.get("merge_fit_s", 12.0))
    pos_tol, t_tol = float(lcfg["pos_tol_px"]), float(lcfg["timing_tol_s"])
    single = [c for c in chains if len(c["items"]) < 2]
    multi = sorted((c for c in chains if len(c["items"]) >= 2), key=lambda c: rows[c["items"][0][0]]["tau_open"])
    i = 0
    while i < len(multi):
        A = multi[i]
        PA = _chain_points(rows, A["items"], rolling_s, height)
        t_end = float(PA[:, 2].max())
        model = _fit_motion(PA[PA[:, 2] >= t_end - fit_s], 1) or _fit_motion(PA, 1)
        best, best_score = None, 1.0
        PA_last = _chain_points(rows, A["items"][-1:], rolling_s, height)
        if model is not None:
            for j in range(i + 1, len(multi)):
                PB = _chain_points(rows, multi[j]["items"], rolling_s, height)
                t_start = float(PB[:, 2].min())
                gap = t_start - t_end
                if gap <= 0 or gap > max_gap:
                    continue
                v = _speed(model, t_start)
                tol = pos_tol + t_tol * v + 0.02 * v * gap
                # tylko przez przerwę: A → pierwsza kreska B i B → ostatnia kreska A (ruch się zmienia)
                score = _residual(_chain_points(rows, multi[j]["items"][:1], rolling_s, height), model) / tol
                model_b = _fit_motion(PB[PB[:, 2] <= t_start + fit_s], 1)
                if model_b is not None:
                    score = max(score, _residual(PA_last, model_b) / tol)
                if score < best_score:
                    best, best_score = j, score
        if best is None:
            i += 1
            continue
        B = multi.pop(best)
        A["items"], A["photo"] = A["items"] + B["items"], B["photo"]
    return single + multi


# ---------------------------------------------------------------- przerwa w serii


def _chain_chi2(P: np.ndarray, w: np.ndarray, deg: int) -> float:
    ok = P[:, 3] > 0
    t = P[ok, 2]
    tm = t.mean()
    A = np.vander(t - tm, deg + 1)
    Aw = A * w[ok, None]
    chi = 0.0
    for col in (0, 1):
        coef, *_ = np.linalg.lstsq(Aw, P[ok, col] * w[ok], rcond=None)
        chi += float(np.sum((Aw @ coef - P[ok, col] * w[ok]) ** 2))
    return chi


def fit_gap(st, chains: list[dict], g0: float, rolling_s: float, height: int, lcfg: dict) -> dict:
    """Przerwa g między zdjęciami serii z łańcuchów ≥ ``min_chain_timing`` kresek: dla każdego
    dg minimalizujemy (liniowo) ruch x(t), y(t) każdego łańcucha, χ²(dg) → minimum i σ z krzywizny
    (skalowanej χ² na stopień swobody). Zwraca też residua końców [px] i [ms]."""
    rows = st.to_dict("records")
    use = [c for c in chains if len(c["items"]) >= int(lcfg["min_chain_timing"])
           and len({rows[k]["m"] for k, _ in c["items"]}) >= 2]
    out = {"fitted": False, "g0_s": g0, "dg_s": 0.0, "g_s": g0, "sigma_s": float("nan"), "n_chains": len(use),
           "n_points": 0, "rms_px": float("nan"), "rms_ms": float("nan")}
    if not use:
        out["note"] = "brak łańcuchów z kreskami z różnych pozycji w serii"
        return out
    floor = float(lcfg["endpoint_err_floor_px"])

    def setup(dg):
        data = []
        for c in use:
            P, err = [], []
            for k, o in c["items"]:
                P += endpoint_rows(rows[k], o, rolling_s, height, dg)
                ea, eb = float(rows[k].get("err_a", floor)), float(rows[k].get("err_b", floor))
                e = (ea, eb) if o > 0 else (eb, ea)
                err += [max(x, floor) if np.isfinite(x) else 1.0 for x in e]
            P = np.array(P, float)
            span = P[:, 2].max() - P[:, 2].min()
            data.append((P, 1.0 / np.array(err), 2 if (len(c["items"]) >= 4 and span >= 10) else 1))
        return data

    def chi2(dg):
        return sum(_chain_chi2(P, w, d) for P, w, d in setup(dg) if (P[:, 3] > 0).sum() >= d + 2)

    grid = np.arange(-g0, float(lcfg.get("max_gap_fit_s", 0.5)), 0.002)
    vals = np.array([chi2(d) for d in grid])
    k = int(np.argmin(vals))
    fine = np.arange(grid[k] - 0.004, grid[k] + 0.004, 0.0002)
    fv = np.array([chi2(d) for d in fine])
    j = int(np.argmin(fv))
    dg = float(fine[j])
    data = setup(dg)
    npts = int(sum((P[:, 3] > 0).sum() for P, _, _ in data))
    npar = int(sum(2 * (d + 1) for P, _, d in data)) + 1
    dof = max(2 * npts - npar, 1)
    red = float(fv[j]) / dof
    sel = slice(max(j - 10, 0), j + 11)
    curv = np.polyfit(fine[sel] - dg, fv[sel], 2)[0] if len(fv[sel]) >= 3 else 0.0
    sigma = math.sqrt(max(red, 1.0) / curv) if curv > 0 else float("nan")
    res_px, res_ms = [], []
    per_set: dict[int, dict[int, list[float]]] = {}      # seria → łańcuch → residua czasu [s]
    for ci, (c, (P, w, d)) in enumerate(zip(use, data)):
        model = _fit_motion(P, d)
        if model is None:
            continue
        ok = P[:, 3] > 0
        x, y = _predict(model, P[ok, 2])
        r = np.hypot(P[ok, 0] - x, P[ok, 1] - y)
        v = np.array([_speed(model, t) for t in P[ok, 2]])
        res_px += list(r)
        res_ms += list(1000 * r / np.maximum(v, 1e-6))
        tm, cx, cy = model
        vx = np.polyval(np.polyder(cx), P[ok, 2] - tm)
        vy = np.polyval(np.polyder(cy), P[ok, 2] - tm)
        dt = ((P[ok, 0] - x) * vx + (P[ok, 1] - y) * vy) / np.maximum(vx ** 2 + vy ** 2, 1e-9)
        sets = np.array([rows[kk]["set"] for kk, _ in c["items"] for _ in (0, 1)])[ok]
        for s, e in zip(sets, dt):
            per_set.setdefault(int(s), {}).setdefault(ci, []).append(float(e))
    # wspólne przesunięcie całej serii w kilku łańcuchach naraz = niestały start serii (interwałometr)
    prods = []
    for chains_in_set in per_set.values():
        means = [float(np.mean(v)) for v in chains_in_set.values()]
        prods += [means[a] * means[b] for a in range(len(means)) for b in range(a + 1, len(means))]
    cov = float(np.mean(prods)) if prods else float("nan")
    out.update({"fitted": bool(np.isfinite(sigma)), "dg_s": dg, "g_s": g0 + dg, "sigma_s": sigma, "n_points": npts,
                "chi2_dof": red, "rms_px": float(np.sqrt(np.mean(np.square(res_px)))) if res_px else float("nan"),
                "rms_ms": float(np.sqrt(np.mean(np.square(res_ms)))) if res_ms else float("nan"),
                "set_jitter_ms": 1000 * math.sqrt(max(cov, 0.0)) if np.isfinite(cov) else float("nan"),
                "set_jitter_pairs": len(prods), "at_grid_edge": bool(k in (0, len(grid) - 1))})
    return out
