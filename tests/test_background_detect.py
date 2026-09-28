"""Tło (mediana/MAD), bufor klatek, mapa SNR i komponenty na syntetycznej kostce z gwiazdami,
migotaniem, gorącym pikselem i poruszającą się kropką."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from skyhunt.background import FrameRing, median_mad, n_blocks, plan_block  # noqa: E402
from skyhunt.detect import (_label_scipy, epoch_blocks, frame_components, gaussian_filter,  # noqa: E402
                            snr_maps, star_psf_sigma)


@pytest.fixture
def dcfg(cfg):
    return cfg["detect"]


def test_median_mad_matches_numpy():
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 255, size=(25, 20, 30), dtype=np.uint8)
    med, mad = median_mad(torch.from_numpy(stack), tile_rows=7)
    ref = np.median(stack, axis=0)
    np.testing.assert_allclose(med.numpy(), ref)
    np.testing.assert_allclose(mad.numpy(), np.median(np.abs(stack - ref), axis=0))


def test_plan_block_and_edges():
    assert n_blocks(100, 24) == 5
    b0 = plan_block(0, 100, 24, 24, 2)
    assert (b0.start, b0.stop, b0.center) == (0, 24, 12) and b0.window[0] == 0 and 12 in b0.window
    last = plan_block(4, 100, 24, 24, 2)
    assert (last.start, last.stop, last.center) == (96, 100, 98) and max(last.window) <= 99
    mid = plan_block(2, 100, 24, 24, 2)
    assert len(mid.window) == 25 and mid.window[0] == mid.center - 24


def test_frame_ring():
    ring = FrameRing(4, 2, 3, "cpu")
    ring.push(0, torch.arange(3, dtype=torch.uint8)[:, None, None].expand(3, 2, 3))
    assert ring.get([1, 2])[:, 0, 0].tolist() == [1, 2]
    with pytest.raises(RuntimeError):
        ring.push(3, torch.zeros((2, 2, 3), dtype=torch.uint8))   # 5 klatek > 4
    ring.drop_before(2)
    ring.push(3, torch.full((2, 2, 3), 7, dtype=torch.uint8))
    assert ring.get([4])[0, 0, 0] == 7
    with pytest.raises(IndexError):
        ring.get([0])


def test_epoch_blocks():
    assert epoch_blocks(320, 24, 23.976, 60) == set(range(30, 320, 60))
    assert epoch_blocks(2, 24, 24.0, 60) == {1}


def synthetic_cube(n=72, h=128, w=160, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    cube = 20 + rng.normal(0, 2, size=(n, h, w))
    stars = [(rng.uniform(10, w - 10), rng.uniform(10, h - 10), rng.uniform(40, 120)) for _ in range(15)]
    pos = []
    for i in range(n):
        for k, (sx, sy, a) in enumerate(stars):
            amp = a * (rng.uniform(0.3, 1.7) if k < 5 else 1.0)   # 5 gwiazd migocze
            cube[i] += amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.3 ** 2))
        x, y = 20 + 1.4 * i, 30 + 0.5 * i
        pos.append((x, y))
        cube[i] += 16 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2 ** 2))   # szczyt ~8σ
        cube[i, 100, 140] = 250   # gorący piksel
    return np.clip(cube, 0, 255).astype(np.uint8), pos


def test_moving_dot_detected(dcfg):
    cube, pos = synthetic_cube()
    t = torch.from_numpy(cube)
    hits, junk = 0, 0
    for c in range(12, 60, 24):
        blk = plan_block(c // 24, len(cube), 24, 24, 2)
        bg, mad = median_mad(t[list(blk.window)])
        sigma = (1.4826 * mad).clamp(min=float(dcfg["sigma_floor_dn"]))
        frames = list(range(blk.start, blk.stop))
        diff, snr, filt = snr_maps(t[frames].float(), bg, sigma, dcfg)
        for k, f in enumerate(frames):
            comps, flagged = frame_components(filt[k], snr[k], diff[k], dcfg, _label_scipy)
            assert not flagged
            d = np.hypot(comps[:, 0] - pos[f][0], comps[:, 1] - pos[f][1]) if len(comps) else np.array([np.inf])
            hits += d.min() <= 0.7
            junk += int((d > 5).sum())
    assert hits >= 0.9 * 48
    # poza kropką zostają głównie jaśniejsze chwile 5 migoczących gwiazd (usuwa je dopiero
    # filtr statyczny w torach); stały gorący piksel znika z tłem
    assert junk / 48 < 6


def test_gaussian_filter_normalized():
    img = torch.zeros((1, 21, 21))
    img[0, 10, 10] = 1
    out = gaussian_filter(img, 1.5)
    assert float(out.sum()) == pytest.approx(1.0, rel=1e-5)
    assert out[0].argmax() == 10 * 21 + 10


def test_star_psf_sigma(dcfg):
    rng = np.random.default_rng(3)
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w]
    img = 30 + rng.normal(0, 0.5, size=(h, w))
    for _ in range(30):
        sx, sy = rng.uniform(15, w - 15), rng.uniform(15, h - 15)
        img += rng.uniform(50, 150) * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.6 ** 2))
    res = star_psf_sigma(img.astype(np.float32), np.full((h, w), 2.0, np.float32), dcfg)
    assert res["n_stars"] >= 20
    assert 1.0 < res["star_sigma_px"] < 2.2
