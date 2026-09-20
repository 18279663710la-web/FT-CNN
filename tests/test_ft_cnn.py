import numpy as np
import pytest


@pytest.fixture(scope="module")
def model():
    from src.models.ft_cnn import build_ft_cnn

    return build_ft_cnn()


def test_ft_cnn_output_shape(model):
    import tensorflow as tf

    b = 4
    beats = tf.random.normal((b, 187, 1))
    rr = tf.random.normal((b, 4))
    out = model([beats, rr], training=False)
    assert tuple(out.shape) == (b, 5)


def test_ft_cnn_has_expected_layers(model):
    names = [l.name for l in model.layers]
    assert "spatial_dropout2" in names
    assert "prelu2" in names
    assert "leaky3" in names
    assert "gap" in names
    assert model.count_params() > 0


def test_ft_cnn_softmax_probs(model):
    import tensorflow as tf

    beats = tf.zeros((2, 187, 1))
    rr = tf.zeros((2, 4))
    out = model([beats, rr], training=False).numpy()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(out >= 0)
