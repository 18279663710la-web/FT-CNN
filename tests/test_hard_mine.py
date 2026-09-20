"""Tests for train-only S↔N hard mining (no test leakage helpers)."""

import numpy as np


def test_select_sn_hard_indices():
    from src.data.hard_mine import select_sn_hard_indices

    #        0    1    2    3    4    5
    y_true = np.array([0, 0, 1, 1, 1, 2])
    y_pred = np.array([0, 1, 0, 1, 2, 2])
    # N→S: idx1; S→N: idx2; S→V idx4 ignored for S→N
    idx = select_sn_hard_indices(y_true, y_pred)
    assert set(idx.tolist()) == {1, 2}


def test_oversample_by_indices():
    from src.data.hard_mine import oversample_by_indices

    beats = np.arange(12, dtype=np.float32).reshape(4, 3)
    labels = np.array([0, 1, 0, 1])
    rr = np.arange(8, dtype=np.float32).reshape(4, 2)
    out_b, out_y, out_rr = oversample_by_indices(beats, labels, rr, np.array([1, 3]), repeat=2)
    assert len(out_y) == 4 + 4  # two indices * repeat 2
    assert out_y[-4:].tolist() == [1, 1, 1, 1]
