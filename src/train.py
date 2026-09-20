"""Train FT-CNN on cached MIT-BIH arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from src.models.cnn_lstm import build_cnn_lstm_from_config
from src.models.ft_cnn import build_ft_cnn_from_config
from src.models.losses import categorical_focal_loss
from src.utils.config import load_config
from src.utils.seed import set_seed


def configure_gpu_memory() -> None:
    """Avoid TF grabbing all VRAM (fixes CUBLAS_STATUS_NOT_INITIALIZED on shared GPUs)."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


configure_gpu_memory()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    beats = data["beats"].astype(np.float32)
    if beats.ndim == 2:
        beats = beats[..., None]
    elif beats.ndim == 3:
        # (N, T, L) context windows
        beats = beats[..., None]
    labels = data["labels"].astype(np.int64)
    rr = data["rr"].astype(np.float32)
    return beats, labels, rr


def class_weights_from_labels(labels: np.ndarray, num_classes: int = 5) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv * (num_classes / inv.sum())
    return w.astype(np.float32)


def resolve_class_weights(
    cfg: dict,
    labels: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray | None:
    """
    Resolve per-class loss weights.

    - use_class_weight false → None
    - train.class_weight list provided → use as-is (mild S boost etc.)
    - else → inverse-frequency from labels
    """
    tcfg = cfg.get("train", {})
    if not bool(tcfg.get("use_class_weight", False)):
        return None
    manual = tcfg.get("class_weight")
    if manual is not None:
        w = np.asarray(manual, dtype=np.float32).reshape(-1)
        if w.shape[0] != num_classes:
            raise ValueError(
                f"train.class_weight length {w.shape[0]} != num_classes {num_classes}"
            )
        return w
    return class_weights_from_labels(labels, num_classes=num_classes)


def make_datasets(
    beats: np.ndarray,
    labels: np.ndarray,
    rr: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    y = tf.keras.utils.to_categorical(labels, num_classes=5)
    ds = tf.data.Dataset.from_tensor_slices(((beats, rr), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(labels), 10000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(cfg: dict):
    arch = str(cfg.get("model", {}).get("architecture", "ft_cnn"))
    if arch == "cnn_lstm":
        return build_cnn_lstm_from_config(cfg)
    return build_ft_cnn_from_config(cfg)


def train(cfg: dict, cache_dir: Path, artifact_dir: Path) -> Path:
    set_seed(int(cfg.get("seed", 42)))
    tcfg = cfg["train"]
    hm = tcfg.get("hard_mine") or {}
    hard_enabled = bool(hm.get("enabled", False))

    train_b, train_y, train_rr = load_npz(cache_dir / "train_fit.npz")
    val_b, val_y, val_rr = load_npz(cache_dir / "val.npz")
    print("beats_shape", train_b.shape, "rr_shape", train_rr.shape)

    rr_aug = tcfg.get("rr_augment_s") or {}
    if bool(rr_aug.get("enabled", False)):
        from src.data.rr_augment import augment_s_rr_prematurity

        half = int(cfg.get("context", {}).get("half_window", 2))
        train_b, train_y, train_rr = augment_s_rr_prematurity(
            train_b,
            train_y,
            train_rr,
            center=half,
            copies=int(rr_aug.get("copies", 1)),
            scale_min=float(rr_aug.get("scale_min", 0.85)),
            scale_max=float(rr_aug.get("scale_max", 0.95)),
            rng=np.random.default_rng(int(cfg.get("seed", 42))),
        )
        print(
            "RR_AUGMENT_S",
            "copies=",
            rr_aug.get("copies", 1),
            "scale=",
            [rr_aug.get("scale_min", 0.85), rr_aug.get("scale_max", 0.95)],
            "new_shape=",
            train_b.shape,
            "class_counts=",
            np.bincount(train_y, minlength=5).tolist(),
        )

    # Paper: after balancing augmentation, do not apply compensatory class weights.
    # Optional mild manual weights (e.g. boost S) via train.class_weight.
    weights = resolve_class_weights(cfg, train_y)
    print(
        "class_counts_train=",
        np.bincount(train_y, minlength=5).tolist(),
        "use_class_weight=",
        bool(tcfg.get("use_class_weight", False)),
        "class_weight=",
        None if weights is None else weights.tolist(),
    )

    model = build_model(cfg)
    init_ckpt = hm.get("init_checkpoint")
    if init_ckpt:
        ckpt_path = Path(str(init_ckpt))
        if not ckpt_path.is_file():
            # allow relative to project root (parent of src)
            root = Path(__file__).resolve().parents[1]
            ckpt_path = root / init_ckpt
        model.load_weights(str(ckpt_path), by_name=True, skip_mismatch=True)
        print("Loaded init checkpoint:", ckpt_path)

    if hard_enabled:
        from src.data.hard_mine import oversample_by_indices, select_sn_hard_indices

        # Mine ONLY on train_fit (never test) to avoid leakage.
        probs = model.predict([train_b, train_rr], batch_size=64, verbose=0)
        preds = probs.argmax(axis=1)
        hard_idx = select_sn_hard_indices(
            train_y,
            preds,
            include_s_as_n=bool(hm.get("include_s_as_n", True)),
            include_n_as_s=bool(hm.get("include_n_as_s", True)),
        )
        n_sn = int(np.sum((train_y == 1) & (preds == 0)))
        n_ns = int(np.sum((train_y == 0) & (preds == 1)))
        repeat = int(hm.get("repeat", 3))
        print(
            "HARD_MINE train_only",
            "S→N=",
            n_sn,
            "N→S=",
            n_ns,
            "hard_idx=",
            len(hard_idx),
            "repeat=",
            repeat,
        )
        train_b, train_y, train_rr = oversample_by_indices(
            train_b, train_y, train_rr, hard_idx, repeat=repeat
        )
        print(
            "after_oversample",
            train_b.shape,
            "class_counts=",
            np.bincount(train_y, minlength=5).tolist(),
        )

    loss_fn = categorical_focal_loss(gamma=float(tcfg["focal_gamma"]), class_weight=weights)

    # Cosine annealing with warm restarts: period = cosine_restart_period epochs
    steps_per_epoch = int(np.ceil(len(train_y) / tcfg["batch_size"]))
    first_decay_steps = max(1, int(tcfg["cosine_restart_period"]) * steps_per_epoch)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=float(tcfg["initial_lr"]),
        first_decay_steps=first_decay_steps,
        t_mul=1.0,
        m_mul=1.0,
        alpha=float(tcfg["min_lr"]) / float(tcfg["initial_lr"]),
    )
    opt = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=float(tcfg["adam_beta_1"]),
        beta_2=float(tcfg["adam_beta_2"]),
        epsilon=float(tcfg["adam_epsilon"]),
    )
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.h5"

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(tcfg["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.CSVLogger(str(artifact_dir / "train_log.csv")),
    ]

    train_ds = make_datasets(train_b, train_y, train_rr, int(tcfg["batch_size"]), shuffle=True)
    val_ds = make_datasets(val_b, val_y, val_rr, int(tcfg["batch_size"]), shuffle=False)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(tcfg["max_epochs"]),
        callbacks=cbs,
    )
    model.save(best_path)

    hist_path = artifact_dir / "history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override config train.max_epochs")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Override artifact output dir")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache dir")
    parser.add_argument("--focal-gamma", type=float, default=None, help="0 = plain CE")
    parser.add_argument("--use-class-weight", type=str, default=None, choices=["true", "false"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = root / args.config
    cfg = load_config(cfg_path)
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = int(args.max_epochs)
    if args.focal_gamma is not None:
        cfg["train"]["focal_gamma"] = float(args.focal_gamma)
    if args.use_class_weight is not None:
        cfg["train"]["use_class_weight"] = args.use_class_weight == "true"

    cache_dir = Path(args.cache_dir) if args.cache_dir else root / cfg["paths"]["cache_dir"]
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else root / cfg["paths"]["artifact_dir"]
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(
        "TRAIN_CFG",
        "focal_gamma=",
        cfg["train"]["focal_gamma"],
        "use_class_weight=",
        cfg["train"].get("use_class_weight", False),
        "cache=",
        cache_dir,
        "artifacts=",
        artifact_dir,
    )
    best = train(cfg, cache_dir, artifact_dir)
    print(f"Saved best model to {best}")


if __name__ == "__main__":
    main()
