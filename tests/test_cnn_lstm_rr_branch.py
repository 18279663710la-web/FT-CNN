"""Tests for dedicated RR/prematurity branch (design B)."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def model_rr_branch():
    from src.models.cnn_lstm import build_cnn_lstm

    return build_cnn_lstm(
        use_attention=True,
        center_skip=True,
        rr_branch=True,
        rr_branch_units=32,
        keep_step_rr=False,
    )


def test_rr_branch_output_shape(model_rr_branch):
    import tensorflow as tf

    b = 4
    out = model_rr_branch(
        [tf.random.normal((b, 5, 187, 1)), tf.random.normal((b, 5, 4))],
        training=False,
    )
    assert tuple(out.shape) == (b, 5)


def test_rr_branch_layers_present(model_rr_branch):
    names = [l.name for l in model_rr_branch.layers]
    assert any("rr_branch" in n for n in names)
    assert any("fuse_rr" in n or "fuse_head" in n for n in names)


def test_rr_branch_no_step_rr_concat(model_rr_branch):
    """When keep_step_rr=False, LSTM sees morph-only features (no fuse_step)."""
    names = [l.name for l in model_rr_branch.layers]
    assert "fuse_step" not in names
    assert any(n.startswith("rr_branch") for n in names)
    feat_dim = int(model_rr_branch.get_layer("lstm").input.shape[-1])
    assert feat_dim == 128


def test_rr_branch_off_keeps_legacy_fuse():
    from src.models.cnn_lstm import build_cnn_lstm
    import tensorflow as tf

    m = build_cnn_lstm(rr_branch=False, keep_step_rr=True)
    assert "fuse_step" in [l.name for l in m.layers]
    feat_dim = int(m.get_layer("lstm").input.shape[-1])
    assert feat_dim == 132  # 128 morph + 4 RR
    out = m(
        [tf.random.normal((2, 5, 187, 1)), tf.random.normal((2, 5, 4))],
        training=False,
    )
    assert tuple(out.shape) == (2, 5)
