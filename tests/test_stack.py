import numpy as np
import pytest

from skyhunt.decode import FrameBatch, cuda_available
from skyhunt.stack import StackAccumulator, keyframe_pulse, stretch_u8


def frames(n=10, h=16, w=24, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, h, w), dtype=np.uint8)


def run_acc(y, bs, to):
    acc = StackAccumulator(subsample=2, quantiles=(0.5, 0.999))
    for s in range(0, len(y), bs):
        acc.add(FrameBatch(s, to(y[s:s + bs])))
    return acc.result()


def check(res, y):
    np.testing.assert_allclose(res.mean, y.mean(0), rtol=1e-6)
    np.testing.assert_array_equal(res.max, y.max(0))
    np.testing.assert_array_equal(res.frames, np.arange(len(y)))
    np.testing.assert_allclose(res.frame_means, y.reshape(len(y), -1).mean(1))
    np.testing.assert_allclose(res.frame_quantiles[:, 0],
                               np.quantile(y[:, ::2, ::2].reshape(len(y), -1).astype(np.float32), 0.5, axis=1),
                               atol=1e-3)


def test_numpy_accumulator():
    y = frames()
    check(run_acc(y, 3, lambda a: a), y)


def test_torch_accumulator_matches_numpy():
    torch = pytest.importorskip("torch")
    y = frames()
    dev = "cuda" if cuda_available() else "cpu"
    check(run_acc(y, 4, lambda a: torch.from_numpy(a.copy()).to(dev)), y)


def test_empty_raises():
    with pytest.raises(ValueError):
        StackAccumulator().result()


def test_keyframe_pulse():
    means = np.full(48, 10.0)
    means[[0, 24]] += 0.5            # I-klatki jaśniejsze o 0.5 DN
    assert keyframe_pulse(means, [0, 24]) == pytest.approx(0.5)
    assert keyframe_pulse(np.full(48, 10.0), [0, 24]) == pytest.approx(0.0)
    assert np.isnan(keyframe_pulse(means, []))


def test_stretch_u8():
    img = np.linspace(0, 100, 10_000, dtype=np.float32).reshape(100, 100)
    out = stretch_u8(img, 0, 100)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert stretch_u8(np.zeros((4, 4))).max() == 0   # obraz stały nie dzieli przez zero
