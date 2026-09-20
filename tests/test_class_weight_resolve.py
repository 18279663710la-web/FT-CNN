"""Tests for mild/manual class-weight resolution (S-focused)."""

import numpy as np


def test_manual_class_weight_from_config():
    from src.train import resolve_class_weights

    labels = np.array([0, 0, 1, 2], dtype=np.int64)
    cfg = {
        "train": {
            "use_class_weight": True,
            "class_weight": [1.0, 2.5, 1.0, 1.0, 1.0],
        }
    }
    w = resolve_class_weights(cfg, labels)
    assert w is not None
    np.testing.assert_allclose(w, [1.0, 2.5, 1.0, 1.0, 1.0])


def test_class_weight_off_returns_none():
    from src.train import resolve_class_weights

    labels = np.array([0, 1], dtype=np.int64)
    cfg = {"train": {"use_class_weight": False, "class_weight": [1, 2, 1, 1, 1]}}
    assert resolve_class_weights(cfg, labels) is None


def test_inv_freq_when_enabled_without_manual():
    from src.train import resolve_class_weights

    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    cfg = {"train": {"use_class_weight": True}}
    w = resolve_class_weights(cfg, labels, num_classes=5)
    assert w is not None
    assert w[1] > w[0]  # rarer S heavier than N
