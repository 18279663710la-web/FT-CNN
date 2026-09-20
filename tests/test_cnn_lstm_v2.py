"""Tests for CNN+LSTM v2: attention + center skip."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def model():
    from src.models.cnn_lstm import build_cnn_lstm

    return build_cnn_lstm(use_attention=True, center_skip=True)


def test_v2_output_shape(model):
    import tensorflow as tf

    b = 4
    out = model(
        [tf.random.normal((b, 5, 187, 1)), tf.random.normal((b, 5, 4))],
        training=False,
    )
    assert tuple(out.shape) == (b, 5)


def test_v2_has_attention_and_center_fuse(model):
    names = [l.name for l in model.layers]
    assert any("attn" in n for n in names)
    assert any("center" in n or "fuse_head" in n for n in names)
