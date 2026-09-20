"""Tests for 5-beat context window construction (method A)."""

import numpy as np
import pytest


def test_make_context_windows_shapes_and_center_label():
    from src.data.context import make_context_windows

    n, L = 10, 187
    beats = np.arange(n * L, dtype=np.float32).reshape(n, L)
    labels = np.arange(n, dtype=np.int64) % 5
    rr = np.arange(n * 4, dtype=np.float32).reshape(n, 4)

    b5, y, rr5 = make_context_windows(beats, labels, rr, half_window=2)

    # indices 2..7 inclusive → 6 windows
    assert b5.shape == (6, 5, 187)
    assert y.shape == (6,)
    assert rr5.shape == (6, 5, 4)
    # center of first window is original index 2
    assert np.allclose(b5[0, 2], beats[2])
    assert y[0] == labels[2]
    assert np.allclose(rr5[0, 2], rr[2])
    # neighbors
    assert np.allclose(b5[0, 0], beats[0])
    assert np.allclose(b5[0, 4], beats[4])


def test_make_context_windows_too_short():
    from src.data.context import make_context_windows

    beats = np.zeros((4, 187), dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)
    rr = np.zeros((4, 4), dtype=np.float32)
    b5, y, rr5 = make_context_windows(beats, labels, rr, half_window=2)
    assert b5.shape[0] == 0
    assert y.shape[0] == 0
    assert rr5.shape[0] == 0
