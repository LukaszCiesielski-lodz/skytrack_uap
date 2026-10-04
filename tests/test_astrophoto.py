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
