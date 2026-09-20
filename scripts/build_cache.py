"""Build preprocessed train/val/test caches from MIT-BIH WFDB files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.augment import maybe_augment_beat
from src.data.splits import get_splits
from src.data.svdb_loader import build_svdb_context_pool, fill_classes_from_pool
from src.data.wfdb_loader import build_split_arrays
from src.utils.config import load_config
from src.utils.seed import set_seed


def _zscore_rr(rr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # rr: (N,4) or (N,T,4)
    if rr.ndim == 3:
        flat = rr.reshape(-1, rr.shape[-1])
        out = (flat - mean) / (std + 1e-8)
        return out.reshape(rr.shape)
    return (rr - mean) / (std + 1e-8)


def _augment_one_beat(beat: np.ndarray, aug_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    out, ok = maybe_augment_beat(
        beat,
        amp_range=(aug_cfg["amp_scale_min"], aug_cfg["amp_scale_max"]),
        warp_range=(aug_cfg["warp_min"], aug_cfg["warp_max"]),
        max_r_shift=int(aug_cfg["max_r_shift"]),
        max_qrs_amp_change=float(aug_cfg["max_qrs_amp_change"]),
        rng=rng,
    )
    tries = 0
    while not ok and tries < 5:
        out, ok = maybe_augment_beat(
            beat,
            amp_range=(aug_cfg["amp_scale_min"], aug_cfg["amp_scale_max"]),
            warp_range=(aug_cfg["warp_min"], aug_cfg["warp_max"]),
            max_r_shift=int(aug_cfg["max_r_shift"]),
            max_qrs_amp_change=float(aug_cfg["max_qrs_amp_change"]),
            rng=rng,
        )
        tries += 1
    return out if ok else beat.copy()


def balance_with_augmentation(
    beats: np.ndarray,
    labels: np.ndarray,
    rr: np.ndarray,
    target_per_class: int,
    aug_cfg: dict,
    rng: np.random.Generator,
    aug_classes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supports single-beat (N,L) or context (N,T,L); augments center beat only."""
    out_b, out_y, out_rr = [beats.copy()], [labels.copy()], [rr.copy()]
    classes = list(range(5)) if aug_classes is None else list(aug_classes)
    ctx = beats.ndim == 3
    center = beats.shape[1] // 2 if ctx else None
    for c in classes:
        idx = np.where(labels == c)[0]
        need = max(0, target_per_class - len(idx))
        if need == 0 or len(idx) == 0:
            continue
        for _ in tqdm(range(need), desc=f"aug class {c}", leave=False):
            i = int(rng.choice(idx))
            if ctx:
                window = beats[i].copy()
                window[center] = _augment_one_beat(window[center], aug_cfg, rng)
                out_b.append(window[None, ...])
            else:
                beat = _augment_one_beat(beats[i], aug_cfg, rng)
                out_b.append(beat[None, ...])
            out_y.append(np.array([c], dtype=labels.dtype))
            out_rr.append(rr[i][None, ...])
    return (
        np.concatenate(out_b, axis=0),
        np.concatenate(out_y, axis=0),
        np.concatenate(out_rr, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--target-per-class", type=int, default=None)
    parser.add_argument(
        "--aug-classes",
        type=str,
        default=None,
        help="Comma-separated class ids to augment, e.g. 1,2,3,4 (skip N)",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--skip-svdb-fill",
        action="store_true",
        help="Do not fill S/V from SVDB even if configured",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    data_root = Path(cfg["paths"]["data_root"])
    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / cfg["paths"]["cache_dir"]
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    arch = str(cfg.get("model", {}).get("architecture", "ft_cnn"))
    half = None
    if arch == "cnn_lstm":
        half = int(cfg.get("context", {}).get("half_window", 2))
        print(f"Architecture=cnn_lstm context_half_window={half}")

    splits = get_splits(cfg)
    arrays = {}
    for name, recs in splits.items():
        print(f"Building {name} from {len(recs)} records...")
        arrays[name] = build_split_arrays(
            data_root, recs, cfg, context_half_window=half
        )

    svdb_cfg = cfg.get("svdb", {}) or {}
    svdb_fill_meta = None
    do_svdb = bool(svdb_cfg.get("enabled", False)) and not args.skip_svdb_fill
    if do_svdb:
        svdb_root = Path(svdb_cfg["data_root"])
        target = int(svdb_cfg.get("target_per_class", cfg["augment"]["target_per_class"]))
        fill_classes = [int(x) for x in svdb_cfg.get("fill_classes", [1, 2])]
        print(f"SVDB fill from {svdb_root} classes={fill_classes} target={target}")
        pool = build_svdb_context_pool(
            svdb_root,
            cfg,
            half_window=int(half or cfg.get("context", {}).get("half_window", 2)),
            class_ids=fill_classes,
        )
        print(
            "SVDB pool size",
            len(pool["labels"]),
            "counts",
            np.bincount(pool["labels"], minlength=5).tolist(),
        )
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        b, y, rr = fill_classes_from_pool(
            arrays["train_fit"]["beats"],
            arrays["train_fit"]["labels"],
            arrays["train_fit"]["rr"],
            pool["beats"],
            pool["labels"],
            pool["rr"],
            class_ids=fill_classes,
            target_per_class=target,
            rng=rng,
        )
        arrays["train_fit"]["beats"] = b.astype(np.float32)
        arrays["train_fit"]["labels"] = y
        arrays["train_fit"]["rr"] = rr.astype(np.float32)
        rid = arrays["train_fit"]["record_ids"]
        if len(rid) < len(y):
            pad = np.full(len(y) - len(rid), -2, dtype=np.int32)  # -2 = SVDB
            arrays["train_fit"]["record_ids"] = np.concatenate([rid, pad])
        svdb_fill_meta = {
            "enabled": True,
            "data_root": str(svdb_root),
            "fill_classes": fill_classes,
            "target_per_class": target,
            "pool_counts": np.bincount(pool["labels"], minlength=5).tolist(),
        }

    # RR z-score from train_fit only (after optional SVDB fill)
    train_rr = arrays["train_fit"]["rr"]
    if train_rr.ndim == 3:
        flat = train_rr.reshape(-1, train_rr.shape[-1])
        rr_mean = flat.mean(axis=0)
        rr_std = flat.std(axis=0)
    else:
        rr_mean = train_rr.mean(axis=0)
        rr_std = train_rr.std(axis=0)
    for name in arrays:
        arrays[name]["rr"] = _zscore_rr(arrays[name]["rr"], rr_mean, rr_std).astype(np.float32)

    # Morph aug only if not skipped and not replaced by SVDB-only balancing
    use_morph_aug = (not args.skip_augment) and bool(cfg.get("augment", {}).get("enabled", True))
    if do_svdb and not bool(svdb_cfg.get("also_morph_aug", False)):
        use_morph_aug = False
        print("Skip morph augmentation (SVDB fill mode)")

    if use_morph_aug:
        rng = np.random.default_rng(int(cfg.get("seed", 42)) + 1)
        target = int(args.target_per_class or cfg["augment"]["target_per_class"])
        aug_classes = None
        if args.aug_classes:
            aug_classes = [int(x) for x in args.aug_classes.split(",") if x.strip() != ""]
        print("AUGMENT target_per_class=", target, "aug_classes=", aug_classes)
        b, y, rr = balance_with_augmentation(
            arrays["train_fit"]["beats"],
            arrays["train_fit"]["labels"],
            arrays["train_fit"]["rr"],
            target_per_class=target,
            aug_cfg=cfg["augment"],
            rng=rng,
            aug_classes=aug_classes,
        )
        rid = arrays["train_fit"]["record_ids"]
        if len(rid) < len(y):
            pad = np.full(len(y) - len(rid), -1, dtype=np.int32)
            rid = np.concatenate([rid, pad])
        arrays["train_fit"]["beats"] = b.astype(np.float32)
        arrays["train_fit"]["labels"] = y
        arrays["train_fit"]["rr"] = rr.astype(np.float32)
        arrays["train_fit"]["record_ids"] = rid

    meta = {
        "rr_mean": rr_mean.tolist(),
        "rr_std": rr_std.tolist(),
        "counts": {k: int(v["labels"].shape[0]) for k, v in arrays.items()},
        "class_counts_train": np.bincount(arrays["train_fit"]["labels"], minlength=5).tolist(),
        "target_per_class": int(args.target_per_class or cfg["augment"]["target_per_class"]),
        "aug_classes": args.aug_classes,
        "architecture": arch,
        "context_half_window": half,
        "beats_shape": list(arrays["train_fit"]["beats"].shape),
        "rr_shape": list(arrays["train_fit"]["rr"].shape),
        "svdb_fill": svdb_fill_meta,
        "morph_aug": use_morph_aug,
    }
    for name, data in arrays.items():
        out = cache_dir / f"{name}.npz"
        np.savez_compressed(
            out,
            beats=data["beats"],
            labels=data["labels"],
            rr=data["rr"],
            record_ids=data.get("record_ids", np.array([])),
        )
        print(f"Wrote {out}  n={data['labels'].shape[0]}")

    with (cache_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Done.", meta["counts"], "class_counts_train", meta["class_counts_train"])


if __name__ == "__main__":
    main()
