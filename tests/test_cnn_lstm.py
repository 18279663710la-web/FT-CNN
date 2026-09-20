"""Tests for CNN+LSTM context model (method A)."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def model():
    from src.models.cnn_lstm import build_cnn_lstm

    return build_cnn_lstm()


def test_cnn_lstm_output_shape(model):
    import tensorflow as tf

    b = 4
    beats = tf.random.normal((b, 5, 187, 1))
    rr = tf.random.normal((b, 5, 4))
    out = model([beats, rr], training=False)
    assert tuple(out.shape) == (b, 5)


def test_cnn_lstm_softmax(model):
    import tensorflow as tf

    beats = tf.zeros((2, 5, 187, 1))
    rr = tf.zeros((2, 5, 4))
    out = model([beats, rr], training=False).numpy()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(out >= 0)
