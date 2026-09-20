"""Tests for explicit RR prematurity features (plan B / v3.2)."""

import numpy as np
import pytest


def test_center_prem_feature_math():
    """1-ratio and local-prev are computed as intended."""
    import tensorflow as tf

    from src.models.cnn_lstm import rr_center_prematurity_features

    # T=5, center=2; feats = [prev, post, local, ratio]
    rr = np.zeros((1, 5, 4), dtype=np.float32)
    rr[0, 2] = [0.6, 0.9, 0.8, 0.75]  # center
    rr[0, 1, 3] = 0.95  # left ratio
    rr[0, 3, 3] = 1.05  # right ratio
    out = rr_center_prematurity_features(tf.constant(rr), center=2, context_len=5).numpy()[0]
    # center 4 + prem(1-0.75=0.25) + (local-prev=0.2) + left_ratio + right_ratio
    np.testing.assert_allclose(out[:4], [0.6, 0.9, 0.8, 0.75], atol=1e-5)
    np.testing.assert_allclose(out[4], 0.25, atol=1e-5)
    np.testing.assert_allclose(out[5], 0.2, atol=1e-5)
    np.testing.assert_allclose(out[6], 0.95, atol=1e-5)
    np.testing.assert_allclose(out[7], 1.05, atol=1e-5)


def test_v32_model_builds_with_center_prem():
    import tensorflow as tf

    from src.models.cnn_lstm import build_cnn_lstm

    m = build_cnn_lstm(
        use_attention=True,
        center_skip=True,
        rr_branch=True,
        keep_step_rr=True,
        rr_branch_mode="center_prem",
    )
    names = [l.name for l in m.layers]
    assert "rr_prem_feats" in names or any("prem" in n for n in names)
    assert "fuse_step" in names
    assert any("rr_branch" in n for n in names)
    out = m(
        [tf.random.normal((2, 5, 187, 1)), tf.random.normal((2, 5, 4))],
        training=False,
    )
    assert tuple(out.shape) == (2, 5)
    assert int(m.get_layer("lstm").input.shape[-1]) == 132
