"""Zdjęcie astronomiczne z sesji RAW: HDR z bracketingu, odrzucanie ruchomych obiektów, tło,
kolor, rozciągnięcie, wyrównanie w pełnej rozdzielczości — na danych syntetycznych."""
import numpy as np
import pytest

pytest.importorskip("scipy")


def test_hdr_uses_short_exposure_where_long_is_saturated():
    from skyhunt.astrophoto import hdr_merge

    long = np.full((4, 5, 3), 100.0, np.float32)
    long[0, 0] = 1000.0                                  # jądro galaktyki: plateau nasycenia
    short = np.full((4, 5, 3), 25.0, np.float32)
    short[0, 0] = 500.0                                  # 2000 DN/s naprawdę
    n = np.full((4, 5), 10, np.int16)
    out = hdr_merge({"ev+1": long, "ev-1": short}, {"ev+1": n, "ev-1": n}, {"ev+1": 1.0, "ev-1": 0.25})
    assert out[0, 0] == pytest.approx([2000.0] * 3)
    assert np.allclose(out[1:, 1:], 100.0)


def test_clip_stack_removes_satellite():
    torch = pytest.importorskip("torch")
    from skyhunt.astrophoto import ClipStack

    rng = np.random.default_rng(0)
    frames = [torch.from_numpy((10 + rng.normal(0, 1, (3, 8, 9))).astype(np.float32)) for _ in range(12)]
    frames[5][:, 4, 2:7] = 1000.0                        # kreska satelity w jednym zdjęciu
    frames[7][:, 0, 0] = float("nan")                    # poza kadrem
    st = ClipStack(3.0)
    for f in frames:
        st.add1(f)
    st.finish1()
    rej = [st.add2(f) for f in frames]
    m, n = st.result()
    assert abs(float(m[0, 4, 4]) - 10) < 1.5 and int(n[4, 4]) == 11 and int(n[0, 0]) <= 11
    assert max(rej) > 0


def test_background_map_follows_gradient_not_object():
    from skyhunt.astrophoto import background_map

    H, W = 256, 384
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sky = 50 + 0.2 * xx + 0.1 * yy                       # łuna z jednej strony
    img = np.repeat(sky[..., None], 3, axis=-1)
    obj = ((xx - 190) / 60) ** 2 + ((yy - 128) / 30) ** 2 < 1
    img[obj] += 400.0                                    # galaktyka: nie może trafić do tła
    bg = background_map(img, obj, box=32, smooth_blocks=0.5)
    err = np.abs(bg[..., 1] - sky)
    assert np.median(err) < 2.0 and err[128, 190] < 15.0


def test_star_color_factors_whiten_stars():
    from skyhunt.astrophoto import star_color_factors

    img = np.zeros((60, 80, 3), np.float32)
    mask = np.zeros((60, 80), bool)
    for k in range(15):
        y, x = 5 + (k // 5) * 20, 6 + (k % 5) * 15
        img[y - 1:y + 2, x - 1:x + 2] = np.array([20.0, 10.0, 5.0]) * (1 + k)   # R/G = 2, B/G = 0,5
        mask[y - 2:y + 3, x - 2:x + 3] = True
    f, n = star_color_factors(img, mask)
    assert n >= 10 and f == pytest.approx([0.5, 1.0, 2.0], rel=1e-3)


def test_asinh_stretch_background_level_and_range():
    from skyhunt.astrophoto import asinh_stretch

    rng = np.random.default_rng(2)
    img = rng.normal(0, 1, (100, 120, 3)).astype(np.float32)
    img[50, 60] = [400, 380, 300]
    out = asinh_stretch(img, np.ones((100, 120), bool), 0.1, 1.3, 0.0)
    assert out.min() >= 0 and out.max() <= 1
    assert 0.03 < float(np.median(out)) < 0.2 and out[50, 60].mean() > 0.8


def test_full_res_warper_identity():
    torch = pytest.importorskip("torch")
    from skyhunt.astrophoto import FullResWarper
    from skyhunt.photo_stages import coarse_grid

    h, w = 10, 12
    gx, gy = coarse_grid((h, w), (6, 5))                 # mapa tożsamościowa w superpikselach
    rgb = np.random.default_rng(3).uniform(0, 100, (3 * h, 3 * w, 3)).astype(np.float32)
    out = FullResWarper((h, w), (3 * h, 3 * w), "cpu")(rgb, gx, gy)
    got = out.permute(1, 2, 0).numpy()
    assert np.allclose(got[1:3 * h - 2, 1:3 * w - 2], rgb[1:3 * h - 2, 1:3 * w - 2], atol=1e-3)


def _stars(shape, n, sigma, rng, shift=None, amp=(20, 100)):
    """Pole gwiazd (Gauss) w 3 kanałach; ``shift(c, x, y)`` → przesunięcie gwiazdy w kanale c."""
    H, W = shape
    img = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[-6:7, -6:7]
    for _ in range(n):
        x, y, a = rng.uniform(12, W - 12), rng.uniform(12, H - 12), rng.uniform(*amp)
        for c in range(3):
            dx, dy = shift(c, x, y) if shift else (0.0, 0.0)
            xi, yi = int(x), int(y)
            g = a * np.exp(-((xx - (x + dx - xi)) ** 2 + (yy - (y + dy - yi)) ** 2) / (2 * sigma ** 2))
            img[yi - 6:yi + 7, xi - 6:xi + 7, c] += g
    return img


def test_lateral_ca_fit_and_correction():
    from scipy.ndimage import binary_dilation

    from skyhunt.astrophoto import apply_lateral_ca, fit_lateral_ca

    H, W, s = 400, 600, 0.003
    cx, cy = (W - 1) / 2, (H - 1) / 2

    def shift(c, x, y):                                  # R powiększone, B pomniejszone względem G
        k = {0: s, 1: 0.0, 2: -s}[c]
        return k * (x - cx), k * (y - cy)

    img = _stars((H, W), 160, 1.2, np.random.default_rng(4), shift)
    smask = binary_dilation(img.mean(axis=-1) > 3, iterations=2)
    info = fit_lateral_ca(img, smask)
    assert info["R"]["scale"] == pytest.approx(s, abs=5e-4) and info["B"]["scale"] == pytest.approx(-s, abs=5e-4)
    again = fit_lateral_ca(apply_lateral_ca(img, info), smask)
    assert abs(again["R"]["scale"]) < 5e-4 and abs(again["B"]["scale"]) < 5e-4


def test_valid_rect_inside_rotated_coverage():
    from skyhunt.astrophoto import valid_rect

    H, W = 300, 450
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.radians(3)                                   # pokrycie: prostokąt obrócony o 3°
    u = (xx - W / 2) * np.cos(th) + (yy - H / 2) * np.sin(th)
    v = -(xx - W / 2) * np.sin(th) + (yy - H / 2) * np.cos(th)
    valid = (np.abs(u) < 210) & (np.abs(v) < 130)
    y0, y1, x0, x1 = valid_rect(valid, frac=1.0, step=2)
    assert valid[y0:y1, x0:x1].all() and (y1 - y0) * (x1 - x0) > 0.7 * valid.sum()


def test_n2n_tiling_is_seamless():
    torch = pytest.importorskip("torch")
    from skyhunt.n2n import _unet, apply

    net = _unet()
    with torch.no_grad():                                # poprawka = 0 → sieć tożsamościowa
        net.out.weight.zero_()
        net.out.bias.zero_()
    img = np.random.default_rng(5).normal(0, 1, (300, 700, 3)).astype(np.float32)
    out = apply(net.eval(), img, "cpu", tile=128, overlap=16)
    assert np.allclose(out, img, atol=1e-5)


def test_psf_measurement_and_deconvolution_sharpen_stars(cfg):
    pytest.importorskip("torch")
    from scipy.ndimage import binary_dilation

    from skyhunt.deconv import deconvolve, measure_psfs

    rng = np.random.default_rng(6)
    img = _stars((300, 450), 120, 1.5, rng, amp=(5, 400)) + rng.normal(0, 0.2, (300, 450, 3)).astype(np.float32)
    L = img.mean(axis=-1)
    smask = binary_dilation(L > 3, iterations=3)
    psf = measure_psfs(L, smask, (1, 1))[0, 0]
    sig = np.sqrt((psf * (np.arange(15) - 7)[:, None] ** 2).sum())
    assert 1.2 < sig < 2.2                               # σ = 1,5 px (+ rozmycie centrowania)
    acfg = dict(cfg["astrophoto"]) | {"deconv_grid": [1, 1], "deconv_strength": 1.0}
    out, info = deconvolve(img, smask, ~smask, acfg, "cpu")
    assert info["applied"]
    assert out.mean(axis=-1)[smask].max() > 1.3 * L[smask].max()     # gwiazdy ostrzejsze (wyższe szczyty)
