"""Load MIT-BIH WFDB records and extract AAMI beats + RR features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from src.data.aami_map import symbol_to_aami
from src.data.context import make_context_windows
from src.data.preprocess import bandpass_filter, compute_rr_features, extract_beat, zscore


def select_lead_index(sig_names: Sequence[str], preference: Sequence[str]) -> int:
    upper = [str(n).upper() for n in sig_names]
    pref = [str(p).upper() for p in preference]
    for p in pref:
        if p in upper:
            return upper.index(p)
    return 0


def beats_from_arrays(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    symbols: Sequence[str],
    fs: float = 360.0,
    before: int = 93,
    after: int = 94,
    local_n: int = 10,
    bandpass: bool = True,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract labeled beats from in-memory signal + annotations."""
    x = np.asarray(signal, dtype=np.float64)
    if bandpass:
        x = bandpass_filter(x, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    x = zscore(x)

    peaks = np.asarray(r_peaks, dtype=int)
    rr_all = compute_rr_features(peaks, local_n=local_n)

    beats, labels, rr = [], [], []
    for i, (r, sym) in enumerate(zip(peaks, symbols)):
        cls = symbol_to_aami(sym)
        if cls is None:
            continue
        beat = extract_beat(x, int(r), before=before, after=after)
        if beat is None:
            continue
        beats.append(beat)
        labels.append(cls)
        rr.append(rr_all[i])

    if not beats:
        return (
            np.zeros((0, before + after), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
        )
    return (
        np.asarray(beats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(rr, dtype=np.float32),
    )


def load_record_beats(
    data_root: Path | str,
    record_id: int,
    lead_preference: Sequence[str],
    preprocess_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import wfdb

    root = Path(data_root)
    rec_name = f"{int(record_id):03d}" if int(record_id) < 1000 else str(int(record_id))
    # MIT-BIH uses 3-digit names like 100, 101
    rec_name = str(int(record_id))
    path = root / rec_name

    record = wfdb.rdrecord(str(path))
    ann = wfdb.rdann(str(path), "atr")

    lead_idx = select_lead_index(list(record.sig_name), lead_preference)
    signal = record.p_signal[:, lead_idx]

    # Keep beat annotations only
    peaks, symbols = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol_to_aami(symbol) is None and symbol not in {"N", "L", "R", "A", "a", "J", "S", "V", "E", "F", "/", "f", "Q", "e", "j", "!", "x"}:
            continue
        if symbol_to_aami(symbol) is None:
            continue
        peaks.append(int(sample))
        symbols.append(symbol)

    return beats_from_arrays(
        signal,
        np.asarray(peaks, dtype=int),
        symbols,
        fs=float(preprocess_cfg.get("fs", 360)),
        before=int(preprocess_cfg.get("samples_before_r", 93)),
        after=int(preprocess_cfg.get("samples_after_r", 94)),
        local_n=int(preprocess_cfg.get("rr_local_n", 10)),
        bandpass=True,
        low_hz=float(preprocess_cfg.get("bandpass_low_hz", 0.5)),
        high_hz=float(preprocess_cfg.get("bandpass_high_hz", 45.0)),
        order=int(preprocess_cfg.get("bandpass_order", 3)),
    )


def build_split_arrays(
    data_root: Path | str,
    record_ids: Iterable[int],
    cfg: dict[str, Any],
    context_half_window: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract beats per record. If context_half_window is set (e.g. 2),
    return 5-beat stacks: beats (N,T,L), rr (N,T,4), labels = center.
    """
    lead_pref = cfg["preprocess"]["lead_preference"]
    half = context_half_window
    if half is None and cfg.get("model", {}).get("architecture") == "cnn_lstm":
        half = int(cfg.get("context", {}).get("half_window", 2))

    beats_l, labels_l, rr_l, rec_l = [], [], [], []
    for rid in record_ids:
        b, y, rr = load_record_beats(data_root, int(rid), lead_pref, cfg["preprocess"])
        if len(y) == 0:
            continue
        if half is not None:
            b, y, rr = make_context_windows(b, y, rr, half_window=int(half))
            if len(y) == 0:
                continue
        beats_l.append(b)
        labels_l.append(y)
        rr_l.append(rr)
        rec_l.append(np.full(len(y), int(rid), dtype=np.int32))
    if not beats_l:
        raise RuntimeError(f"No beats extracted for records: {list(record_ids)}")
    return {
        "beats": np.concatenate(beats_l, axis=0),
        "labels": np.concatenate(labels_l, axis=0),
        "rr": np.concatenate(rr_l, axis=0),
        "record_ids": np.concatenate(rec_l, axis=0),
    }
