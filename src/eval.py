"""Evaluate a trained FT-CNN / CNN-LSTM checkpoint on the test cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.aami_map import CLASS_NAMES
from src.utils.config import load_config


def load_npz(path: Path):
    data = np.load(path)
    beats = data["beats"].astype(np.float32)
    if beats.ndim == 2:
        beats = beats[..., None]
    elif beats.ndim == 3:
        beats = beats[..., None]
    labels = data["labels"].astype(np.int64)
    rr = data["rr"].astype(np.float32)
    return beats, labels, rr


def _build_model(cfg: dict):
    from src.models.cnn_lstm import build_cnn_lstm_from_config
    from src.models.ft_cnn import build_ft_cnn_from_config

    arch = str(cfg.get("model", {}).get("architecture", "ft_cnn"))
    if arch == "cnn_lstm":
        return build_cnn_lstm_from_config(cfg)
    return build_ft_cnn_from_config(cfg)


def load_trained_model(model_path: Path, cfg: dict | None = None):
    """Prefer rebuild+load_weights (robust); fall back to load_model."""
    from src.models.cnn_lstm import AttnPool, CenterPriorAdd, CenterSlice

    custom = {
        "AttnPool": AttnPool,
        "CenterSlice": CenterSlice,
        "CenterPriorAdd": CenterPriorAdd,
    }
    path = str(model_path)
    if cfg is not None:
        try:
            model = _build_model(cfg)
            model.load_weights(path, by_name=True, skip_mismatch=True)
            print("Loaded weights into rebuilt model:", path)
            return model
        except Exception as e:
            print("rebuild+load_weights failed:", repr(e), "trying load_model...")
    model = tf.keras.models.load_model(path, compile=False, custom_objects=custom)
    print("Loaded full model:", path)
    return model


def evaluate(
    model_path: Path,
    cache_dir: Path,
    artifact_dir: Path,
    cfg: dict | None = None,
) -> dict:
    beats, labels, rr = load_npz(cache_dir / "test.npz")
    model = load_trained_model(model_path, cfg=cfg)
    probs = model.predict([beats, rr], batch_size=64, verbose=0)
    preds = probs.argmax(axis=1)

    report = classification_report(
        labels,
        preds,
        labels=list(range(5)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    acc = float((preds == labels).mean())
    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(artifact_dir / "confusion_matrix.npy", cm)
    print(f"Accuracy: {acc:.4f}")
    print(
        classification_report(
            labels, preds, labels=list(range(5)), target_names=CLASS_NAMES, zero_division=0
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = root / args.config
    cfg = load_config(cfg_path)

    cache_dir = Path(args.cache_dir) if args.cache_dir else root / cfg["paths"]["cache_dir"]
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else root / cfg["paths"]["artifact_dir"]
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir

    evaluate(Path(args.checkpoint), cache_dir, artifact_dir, cfg=cfg)


if __name__ == "__main__":
    main()
