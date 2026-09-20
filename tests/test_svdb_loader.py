"""Tests for SVDB resample + fill helpers."""

import numpy as np


def test_resample_scales_peaks():
    from src.data.svdb_loader import resample_signal_and_peaks

    sig = np.sin(np.linspace(0, 20, 1280))
    peaks = np.array([100, 200, 300])
    y, p2 = resample_signal_and_peaks(sig, peaks, fs_in=128.0, fs_out=360.0)
    assert len(y) == int(round(1280 * 360 / 128))
    assert np.allclose(p2 / 360.0, peaks / 128.0, atol=1e-2)


def test_fill_classes_from_pool():
    from src.data.svdb_loader import fill_classes_from_pool

    rng = np.random.default_rng(0)
    tb = np.zeros((10, 5, 187), np.float32)
    ty = np.array([0] * 8 + [1, 1], np.int64)
    tr = np.zeros((10, 5, 4), np.float32)
    pb = np.ones((20, 5, 187), np.float32)
    py = np.array([1] * 10 + [2] * 10, np.int64)
    pr = np.ones((20, 5, 4), np.float32)
    b, y, r = fill_classes_from_pool(tb, ty, tr, pb, py, pr, [1, 2], target_per_class=5, rng=rng)
    assert int(np.sum(y == 1)) == 5
    assert int(np.sum(y == 2)) == 5
    assert b.shape[0] == y.shape[0] == r.shape[0]
