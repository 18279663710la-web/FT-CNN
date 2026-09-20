"""Weighted categorical CE with focal modulation (paper γ=2.0)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import tensorflow as tf


def categorical_focal_loss(
    gamma: float = 2.0,
    class_weight: Optional[Union[Sequence[float], np.ndarray]] = None,
    from_logits: bool = False,
    epsilon: float = 1e-7,
):
    """
    FL = α_t * (1 - p_t)^γ * CE

    where CE is categorical cross-entropy and α_t is the class weight for the true class.
    """
    weights = None
    if class_weight is not None:
        weights = tf.constant(np.asarray(class_weight, dtype=np.float32), dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # CE per sample
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating = tf.pow(1.0 - p_t, gamma)
        fl = modulating * ce

        if weights is not None:
            alpha_t = tf.reduce_sum(y_true * weights, axis=-1)
            fl = alpha_t * fl
        return tf.reduce_mean(fl)

    loss.__name__ = "categorical_focal_loss"
    return loss
