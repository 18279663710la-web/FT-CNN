"""Fine-Tuned CNN (FT-CNN) dual-branch architecture from the paper."""

from __future__ import annotations

from typing import Any, Optional

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def build_ft_cnn(
    beat_len: int = 187,
    rr_dim: int = 4,
    num_classes: int = 5,
    conv_l2: float = 1e-4,
    dense_l2: float = 1e-4,
    spatial_dropout: float = 0.25,
    dense_dropout: float = 0.5,
    leaky_relu_alpha: float = 0.01,
    temperature: float = 1.0,
    name: str = "ft_cnn",
) -> Model:
    """
    Morphology branch (187,) + RR branch (4,) → 5-class softmax.

    Block1: Conv1D(32,k=5) → BN → ReLU → MaxPool(2)
    Block2: Conv1D(64,k=3) → SpatialDropout → PReLU → AvgPool(2)
    Block3: Conv1D(128,k=3) → BN → LeakyReLU → GAP
    Head: Dense(256)+BN → Dense(128)+Dropout → Dense(64)+L2 → Softmax
    """
    l2 = regularizers.L2(conv_l2)
    d_l2 = regularizers.L2(dense_l2)

    beat_in = layers.Input(shape=(beat_len, 1), name="beat")
    rr_in = layers.Input(shape=(rr_dim,), name="rr")

    x = layers.Conv1D(
        32,
        5,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
        name="conv1",
    )(beat_in)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name="maxpool1")(x)

    x = layers.Conv1D(
        64,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
        name="conv2",
    )(x)
    x = layers.SpatialDropout1D(spatial_dropout, name="spatial_dropout2")(x)
    x = layers.PReLU(name="prelu2")(x)
    x = layers.AveragePooling1D(pool_size=2, strides=2, name="avgpool2")(x)

    x = layers.Conv1D(
        128,
        3,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2,
        name="conv3",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    # TF/Keras 2.x uses `alpha`; Keras 3 uses `negative_slope`
    try:
        x = layers.LeakyReLU(alpha=leaky_relu_alpha, name="leaky3")(x)
    except TypeError:
        x = layers.LeakyReLU(negative_slope=leaky_relu_alpha, name="leaky3")(x)
    f_cnn = layers.GlobalAveragePooling1D(name="gap")(x)

    fused = layers.Concatenate(name="fuse")([f_cnn, rr_in])

    h = layers.Dense(
        256,
        kernel_initializer="he_normal",
        kernel_regularizer=d_l2,
        name="fc256",
    )(fused)
    h = layers.BatchNormalization(name="bn_fc256")(h)
    h = layers.ReLU(name="relu_fc256")(h)

    h = layers.Dense(
        128,
        kernel_initializer="he_normal",
        kernel_regularizer=d_l2,
        name="fc128",
    )(h)
    h = layers.Dropout(dense_dropout, name="dropout_fc128")(h)
    h = layers.ReLU(name="relu_fc128")(h)

    h = layers.Dense(
        64,
        kernel_initializer="he_normal",
        kernel_regularizer=d_l2,
        name="fc64",
    )(h)
    h = layers.ReLU(name="relu_fc64")(h)

    logits = layers.Dense(
        num_classes,
        kernel_initializer="glorot_uniform",
        name="logits",
    )(h)

    if temperature != 1.0:
        logits = layers.Lambda(lambda z: z / temperature, name="temperature")(logits)

    probs = layers.Softmax(name="softmax")(logits)
    return Model(inputs=[beat_in, rr_in], outputs=probs, name=name)


def build_ft_cnn_from_config(cfg: Optional[dict[str, Any]] = None) -> Model:
    m = (cfg or {}).get("model", {})
    return build_ft_cnn(
        conv_l2=float(m.get("conv_l2", 1e-4)),
        dense_l2=float(m.get("dense_l2", 1e-4)),
        spatial_dropout=float(m.get("spatial_dropout", 0.25)),
        dense_dropout=float(m.get("dense_dropout", 0.5)),
        leaky_relu_alpha=float(m.get("leaky_relu_alpha", 0.01)),
        temperature=float(m.get("temperature", 1.0)),
        num_classes=int(m.get("num_classes", 5)),
    )
