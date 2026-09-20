"""Tests for S RR prematurity aug and center_prem_seq branch."""

import numpy as np


def test_shorten_center_rr_updates_ratio():
    from src.data.rr_augment import shorten_center_rr_prev

    rr = np.zeros((1, 5, 4), dtype=np.float32)
    rr[0, 2] = [1.0, 1.0, 1.0, 1.0]
    out = shorten_center_rr_prev(rr, center=2, scale=0.9)
    assert abs(float(out[0, 2, 0]) - 0.9) < 1e-5
    assert abs(float(out[0, 2, 3]) - 0.9) < 1e-5
    # original unchanged
    assert float(rr[0, 2, 0]) == 1.0


def test_augment_s_rr_appends_only_s():
    from src.data.rr_augment import augment_s_rr_prematurity

    beats = np.zeros((4, 5, 187, 1), dtype=np.float32)
    labels = np.array([0, 1, 1, 2], dtype=np.int64)
    rr = np.ones((4, 5, 4), dtype=np.float32)
    b2, y2, r2 = augment_s_rr_prematurity(
        beats, labels, rr, copies=1, scale_min=0.9, scale_max=0.9, rng=np.random.default_rng(0)
    )
    assert len(y2) == 4 + 2  # two S clones
    assert int(np.sum(y2 == 1)) == 4
    # cloned S have shortened prev
    assert float(r2[-1, 2, 0]) < 1.0


def test_center_prem_seq_model_builds():
    import tensorflow as tf
    from src.models.cnn_lstm import build_cnn_lstm

    m = build_cnn_lstm(
        rr_branch=True,
        keep_step_rr=True,
        rr_branch_mode="center_prem_seq",
        rr_branch_units=64,
        use_attention=True,
        center_skip=True,
    )
    names = [l.name for l in m.layers]
    assert "rr_prem_feats" in names
    assert "rr_seq_flat" in names
    assert "rr_prem_seq" in names
    out = m(
        [tf.random.normal((2, 5, 187, 1)), tf.random.normal((2, 5, 4))],
        training=False,
    )
    assert tuple(out.shape) == (2, 5)
