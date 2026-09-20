"""Load MIT-BIH SVDB records: resample 128Hz → target_fs, extract AAMI beats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy import signal as sps

from src.data.aami_map import symbol_to_aami
from src.data.context import make_context_windows
from src.data.wfdb_loader import beats_from_arrays, select_lead_index


def resample_signal_and_peaks(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs_in: float,
    fs_out: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample 1D signal and scale peak indices."""
    x = np.asarray(signal, dtype=np.float64)
    n_out = int(round(len(x) * fs_out / fs_in))
    y = sps.resample(x, n_out)
    scale = fs_out / fs_in
    peaks_out = np.clip(np.round(np.asarray(peaks, dtype=np.float64) * scale).astype(int), 0, n_out - 1)
    return y, peaks_out


def load_svdb_record_beats(
    data_root: Path | str,
    record_name: str,
    lead_preference: Sequence[str],
    preprocess_cfg: dict[str, Any],
    target_fs: float = 360.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import wfdb

    root = Path(data_root)
    path = root / str(record_name)
    record = wfdb.rdrecord(str(path))
    ann = wfdb.rdann(str(path), "atr")
    fs_in = float(record.fs)

    lead_idx = select_lead_index(list(record.sig_name), lead_preference)
    raw = record.p_signal[:, lead_idx]

    peaks, symbols = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol_to_aami(symbol) is None:
            continue
        peaks.append(int(sample))
        symbols.append(symbol)
    if not peaks:
        L = int(preprocess_cfg.get("samples_before_r", 93)) + int(
            preprocess_cfg.get("samples_after_r", 94)
        )
        return (
            np.zeros((0, L), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
        )

    sig360, peaks360 = resample_signal_and_peaks(raw, np.asarray(peaks), fs_in, target_fs)
    return beats_from_arrays(
        sig360,
        peaks360,
        symbols,
        fs=target_fs,
        before=int(preprocess_cfg.get("samples_before_r", 93)),
        after=int(preprocess_cfg.get("samples_after_r", 94)),
        local_n=int(preprocess_cfg.get("rr_local_n", 10)),
        bandpass=True,
        low_hz=float(preprocess_cfg.get("bandpass_low_hz", 0.5)),
        high_hz=float(preprocess_cfg.get("bandpass_high_hz", 45.0)),
        order=int(preprocess_cfg.get("bandpass_order", 3)),
    )


def list_svdb_records(data_root: Path | str) -> list[str]:
    root = Path(data_root)
    rec_file = root / "RECORDS"
    if rec_file.is_file():
        return [ln.strip() for ln in rec_file.read_text().splitlines() if ln.strip()]
    return sorted({p.stem for p in root.glob("*.dat")})


def build_svdb_context_pool(
    data_root: Path | str,
    cfg: dict[str, Any],
    half_window: int = 2,
    class_ids: Optional[Sequence[int]] = None,
) -> dict[str, np.ndarray]:
    """Build context windows from all SVDB records; optionally keep only given classes."""
    lead_pref = cfg.get("svdb", {}).get(
        "lead_preference",
        cfg["preprocess"].get("lead_preference", ["ECG1", "ECG2", "II", "MLII"]),
    )
    target_fs = float(cfg.get("svdb", {}).get("target_fs", cfg["preprocess"].get("fs", 360)))
    keep = None if class_ids is None else set(int(c) for c in class_ids)

    beats_l, labels_l, rr_l = [], [], []
    for name in list_svdb_records(data_root):
        b, y, rr = load_svdb_record_beats(
            data_root, name, lead_pref, cfg["preprocess"], target_fs=target_fs
        )
        if len(y) == 0:
            continue
        b, y, rr = make_context_windows(b, y, rr, half_window=half_window)
        if len(y) == 0:
            continue
        if keep is not None:
            mask = np.isin(y, list(keep))
            if not np.any(mask):
                continue
            b, y, rr = b[mask], y[mask], rr[mask]
        beats_l.append(b)
        labels_l.append(y)
        rr_l.append(rr)

    if not beats_l:
        raise RuntimeError(f"No SVDB context beats from {data_root}")
    return {
        "beats": np.concatenate(beats_l, axis=0).astype(np.float32),
        "labels": np.concatenate(labels_l, axis=0).astype(np.int64),
        "rr": np.concatenate(rr_l, axis=0).astype(np.float32),
    }


def fill_classes_from_pool(
    train_beats: np.ndarray,
    train_labels: np.ndarray,
    train_rr: np.ndarray,
    pool_beats: np.ndarray,
    pool_labels: np.ndarray,
    pool_rr: np.ndarray,
    class_ids: Sequence[int],
    target_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append samples from pool until each class_id reaches target_per_class."""
    out_b = [train_beats]
    out_y = [train_labels]
    out_rr = [train_rr]
    for c in class_ids:
        have = int(np.sum(train_labels == c))
        need = max(0, int(target_per_class) - have)
        if need == 0:
            continue
        idx = np.where(pool_labels == c)[0]
        if len(idx) == 0:
            print(f"WARN: SVDB pool has 0 samples for class {c}")
            continue
        choose = rng.choice(idx, size=need, replace=(len(idx) < need))
        out_b.append(pool_beats[choose])
        out_y.append(pool_labels[choose])
        out_rr.append(pool_rr[choose])
        print(f"SVDB fill class {c}: have={have} need={need} pool={len(idx)}")
    return (
        np.concatenate(out_b, axis=0),
        np.concatenate(out_y, axis=0),
        np.concatenate(out_rr, axis=0),
    )
