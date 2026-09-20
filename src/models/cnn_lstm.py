"""Shared 1D-CNN morphology encoder + LSTM/Attention over 5-beat context."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def build_morphology_encoder(
    beat_len: int = 187,
    conv_l2: float = 1e-4,
    spatial_dropout: float = 0.25,
    leaky_relu_alpha: float = 0.01,
    name: str = "morph_encoder",
) -> Model:
    """FT-CNN morphology tower through GAP → (128,)."""
    l2 = regularizers.L2(conv_l2)
    beat_in = layers.Input(shape=(beat_len, 1), name="beat")

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
    try:
        x = layers.LeakyReLU(alpha=leaky_relu_alpha, name="leaky3")(x)
    except TypeError:
        x = layers.LeakyReLU(negative_slope=leaky_relu_alpha, name="leaky3")(x)
    f_cnn = layers.GlobalAveragePooling1D(name="gap")(x)
    return Model(inputs=beat_in, outputs=f_cnn, name=name)


def _center_attention_prior_np(context_len: int, center_boost: float = 1.5) -> np.ndarray:
    """Fixed prior logits favoring the center step, shape (1, T, 1)."""
    prior = np.zeros((context_len,), dtype=np.float32)
    c = context_len // 2
    for i in range(context_len):
        dist = abs(i - c)
        prior[i] = center_boost - 0.5 * float(dist)
    return prior.reshape(1, context_len, 1)


class CenterPriorAdd(layers.Layer):
    """Add a fixed (non-trainable) center-favoring prior to attention scores."""

    def __init__(self, context_len: int, center_boost: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.context_len = int(context_len)
        self.center_boost = float(center_boost)

    def build(self, input_shape):
        prior = _center_attention_prior_np(self.context_len, self.center_boost)
        self.prior = self.add_weight(
            name="prior",
            shape=prior.shape,
            initializer=tf.constant_initializer(prior),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, score):
        return score + self.prior

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"context_len": self.context_len, "center_boost": self.center_boost}
        )
        return cfg


class AttnPool(layers.Layer):
    """Weighted sum over time: sum_t attn_t * seq_t."""

    def call(self, inputs):
        attn, seq = inputs
        return tf.reduce_sum(attn * seq, axis=1)

    def compute_output_shape(self, input_shape):
        # inputs: [attn (B,T,1), seq (B,T,U)] -> (B,U)
        seq_shape = input_shape[1]
        return (seq_shape[0], seq_shape[2])


class CenterSlice(layers.Layer):
    def __init__(self, center: int, **kwargs):
        super().__init__(**kwargs)
        self.center = int(center)

    def call(self, x):
        return x[:, self.center, :]

    def compute_output_shape(self, input_shape):
        # (B, T, F) -> (B, F)
        return (input_shape[0], input_shape[2])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"center": self.center})
        return cfg


def rr_center_prematurity_features(
    rr_seq: tf.Tensor,
    center: int,
    context_len: int,
) -> tf.Tensor:
    """
    Build explicit prematurity features from RR sequence.

    RR layout per beat: [RR_prev, RR_post, RR_local, RR_ratio].
    Output (B, 8): center(4) + (1-ratio) + (local-prev) + left_ratio + right_ratio.
    """
    c = int(center)
    left_i = max(c - 1, 0)
    right_i = min(c + 1, int(context_len) - 1)
    center_rr = rr_seq[:, c, :]
    prem = 1.0 - center_rr[:, 3:4]
    shortfall = center_rr[:, 2:3] - center_rr[:, 0:1]
    left_ratio = rr_seq[:, left_i, 3:4]
    right_ratio = rr_seq[:, right_i, 3:4]
    return tf.concat([center_rr, prem, shortfall, left_ratio, right_ratio], axis=-1)


def build_cnn_lstm(
    beat_len: int = 187,
    context_len: int = 5,
    rr_dim: int = 4,
    num_classes: int = 5,
    conv_l2: float = 1e-4,
    dense_l2: float = 1e-4,
    spatial_dropout: float = 0.25,
    dense_dropout: float = 0.5,
    leaky_relu_alpha: float = 0.01,
    lstm_units: int = 64,
    lstm_dropout: float = 0.25,
    use_attention: bool = True,
    center_skip: bool = True,
    center_attn_boost: float = 1.5,
    prior_as_weight: bool = False,
    rr_branch: bool = False,
    rr_branch_units: int = 32,
    rr_branch_dropout: float = 0.25,
    rr_branch_mode: str = "center",
    keep_step_rr: bool = True,
    name: str = "cnn_lstm",
) -> Model:
    """
    Shared CNN per beat → (optional step RR) → LSTM → attention pool
    → optional center CNN skip → optional dedicated RR-MLP → softmax.

    rr_branch_mode:
      - "center": center-beat RR(4) → MLP
      - "center_prem": center RR + explicit prematurity feats(8) → MLP
      - "seq": flatten all context RR → MLP
      - "center_prem_seq": concat(center_prem, flattened seq) → MLP
    """
    d_l2 = regularizers.L2(dense_l2)
    center = context_len // 2

    beats_in = layers.Input(shape=(context_len, beat_len, 1), name="beats")
    rr_in = layers.Input(shape=(context_len, rr_dim), name="rr_seq")

    encoder = build_morphology_encoder(
        beat_len=beat_len,
        conv_l2=conv_l2,
        spatial_dropout=spatial_dropout,
        leaky_relu_alpha=leaky_relu_alpha,
        name="morph_encoder",
    )
    f_cnn = layers.TimeDistributed(encoder, name="td_morph")(beats_in)
    if keep_step_rr:
        step = layers.Concatenate(name="fuse_step")([f_cnn, rr_in])
    else:
        step = f_cnn

    if use_attention:
        seq = layers.LSTM(
            lstm_units,
            dropout=lstm_dropout,
            recurrent_dropout=0.0,
            return_sequences=True,
            name="lstm",
        )(step)
        score_h = layers.Dense(lstm_units, activation="tanh", name="attn_hidden")(seq)
        score = layers.Dense(1, name="attn_score")(score_h)
        if prior_as_weight:
            score = CenterPriorAdd(
                context_len=context_len,
                center_boost=center_attn_boost,
                name="attn_score_prior",
            )(score)
        else:
            prior = _center_attention_prior_np(context_len, center_attn_boost)
            score = layers.Lambda(
                lambda s, p=prior: s + tf.cast(p, s.dtype),
                name="attn_score_prior",
            )(score)
        attn = layers.Softmax(axis=1, name="attn_weights")(score)
        temporal = AttnPool(name="attn_pool")([attn, seq])
    else:
        temporal = layers.LSTM(
            lstm_units,
            dropout=lstm_dropout,
            recurrent_dropout=0.0,
            name="lstm",
        )(step)

    parts = [temporal]
    if center_skip:
        parts.append(CenterSlice(center, name="center_cnn")(f_cnn))
    if rr_branch:
        mode = str(rr_branch_mode).lower()
        if mode == "center_prem":
            rr_feat = layers.Lambda(
                lambda x, c=center, t=context_len: rr_center_prematurity_features(x, c, t),
                name="rr_prem_feats",
            )(rr_in)
        elif mode == "seq":
            rr_feat = layers.Flatten(name="rr_seq_flat")(rr_in)
        elif mode == "center_prem_seq":
            prem = layers.Lambda(
                lambda x, c=center, t=context_len: rr_center_prematurity_features(x, c, t),
                name="rr_prem_feats",
            )(rr_in)
            flat = layers.Flatten(name="rr_seq_flat")(rr_in)
            rr_feat = layers.Concatenate(name="rr_prem_seq")([prem, flat])
        else:
            rr_feat = CenterSlice(center, name="center_rr")(rr_in)
        rr_h = layers.Dense(
            rr_branch_units,
            kernel_initializer="he_normal",
            kernel_regularizer=d_l2,
            name="rr_branch_fc1",
        )(rr_feat)
        rr_h = layers.BatchNormalization(name="rr_branch_bn1")(rr_h)
        rr_h = layers.ReLU(name="rr_branch_relu1")(rr_h)
        rr_h = layers.Dropout(rr_branch_dropout, name="rr_branch_drop1")(rr_h)
        rr_h = layers.Dense(
            max(rr_branch_units // 2, 8),
            kernel_initializer="he_normal",
            kernel_regularizer=d_l2,
            name="rr_branch_fc2",
        )(rr_h)
        rr_h = layers.ReLU(name="rr_branch_relu2")(rr_h)
        parts.append(rr_h)

    if len(parts) == 1:
        h = parts[0]
    elif rr_branch:
        h = layers.Concatenate(name="fuse_rr")(parts)
    else:
        h = layers.Concatenate(name="fuse_head")(parts)

    h = layers.Dense(
        128,
        kernel_initializer="he_normal",
        kernel_regularizer=d_l2,
        name="fc128",
    )(h)
    h = layers.BatchNormalization(name="bn_fc128")(h)
    h = layers.ReLU(name="relu_fc128")(h)
    h = layers.Dropout(dense_dropout, name="dropout_fc128")(h)

    logits = layers.Dense(
        num_classes,
        kernel_initializer="glorot_uniform",
        name="logits",
    )(h)
    probs = layers.Softmax(name="softmax")(logits)
    return Model(inputs=[beats_in, rr_in], outputs=probs, name=name)


def build_cnn_lstm_from_config(cfg: Optional[dict[str, Any]] = None) -> Model:
    m = (cfg or {}).get("model", {})
    ctx = (cfg or {}).get("context", {})
    half = int(ctx.get("half_window", 2))
    rr_branch = bool(m.get("rr_branch", False))
    # When RR branch is on, default to morph-only LSTM steps unless overridden.
    keep_step_rr = bool(m.get("keep_step_rr", not rr_branch))
    return build_cnn_lstm(
        context_len=2 * half + 1,
        conv_l2=float(m.get("conv_l2", 1e-4)),
        dense_l2=float(m.get("dense_l2", 1e-4)),
        spatial_dropout=float(m.get("spatial_dropout", 0.25)),
        dense_dropout=float(m.get("dense_dropout", 0.5)),
        leaky_relu_alpha=float(m.get("leaky_relu_alpha", 0.01)),
        lstm_units=int(m.get("lstm_units", 64)),
        lstm_dropout=float(m.get("lstm_dropout", 0.25)),
        use_attention=bool(m.get("use_attention", True)),
        center_skip=bool(m.get("center_skip", True)),
        center_attn_boost=float(m.get("center_attn_boost", 1.5)),
        # Default False so existing AutoDL checkpoints (Add prior, 0 weights) load.
        prior_as_weight=bool(m.get("prior_as_weight", False)),
        rr_branch=rr_branch,
        rr_branch_units=int(m.get("rr_branch_units", 32)),
        rr_branch_dropout=float(m.get("rr_branch_dropout", 0.25)),
        rr_branch_mode=str(m.get("rr_branch_mode", "center")),
        keep_step_rr=keep_step_rr,
        num_classes=int(m.get("num_classes", 5)),
    )
