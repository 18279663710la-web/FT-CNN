"""Forward-pass smoke test for FT-CNN."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ft_cnn import build_ft_cnn
from src.utils.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    model = build_ft_cnn(
        conv_l2=float(cfg["model"]["conv_l2"]),
        dense_l2=float(cfg["model"]["dense_l2"]),
        spatial_dropout=float(cfg["model"]["spatial_dropout"]),
        dense_dropout=float(cfg["model"]["dense_dropout"]),
        leaky_relu_alpha=float(cfg["model"]["leaky_relu_alpha"]),
        temperature=float(cfg["model"]["temperature"]),
    )
    beats = tf.random.normal((8, 187, 1))
    rr = tf.random.normal((8, 4))
    out = model([beats, rr], training=False)
    print(model.summary())
    print("output shape:", tuple(out.shape))
    print("probs sum:", np.round(out.numpy().sum(axis=1), 4))
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
