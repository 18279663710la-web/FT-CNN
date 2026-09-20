import numpy as np
import pytest


def test_focal_loss_scalar_and_hard_examples_higher():
    import tensorflow as tf

    from src.models.losses import categorical_focal_loss

    loss_fn = categorical_focal_loss(gamma=2.0, class_weight=None)
    y_true = tf.constant([[1, 0, 0, 0, 0]], dtype=tf.float32)

    # Easy correct prediction
    y_easy = tf.constant([[0.95, 0.0125, 0.0125, 0.0125, 0.0125]], dtype=tf.float32)
    # Hard / wrong prediction
    y_hard = tf.constant([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=tf.float32)

    easy = float(loss_fn(y_true, y_easy).numpy())
    hard = float(loss_fn(y_true, y_hard).numpy())
    assert np.isscalar(easy) or np.ndim(easy) == 0
    assert hard > easy


def test_class_weight_increases_minority_loss():
    import tensorflow as tf

    from src.models.losses import categorical_focal_loss

    y_true = tf.constant([[0, 1, 0, 0, 0]], dtype=tf.float32)  # class S
    y_pred = tf.constant([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=tf.float32)
    w = np.array([1.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32)

    unweighted = categorical_focal_loss(gamma=2.0, class_weight=None)
    weighted = categorical_focal_loss(gamma=2.0, class_weight=w)
    assert float(weighted(y_true, y_pred).numpy()) > float(unweighted(y_true, y_pred).numpy())
