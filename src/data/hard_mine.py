"""Hard-example selection for S↔N (train-only; never use test)."""

from __future__ import annotations

import numpy as np


def select_sn_hard_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_s_as_n: bool = True,
    include_n_as_s: bool = True,
) -> np.ndarray:
    """
    Return indices of S→N misses and/or N→S false alarms.

    Classes: N=0, S=1.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    masks = []
    if include_s_as_n:
        masks.append((y_true == 1) & (y_pred == 0))
    if include_n_as_s:
        masks.append((y_true == 0) & (y_pred == 1))
    if not masks:
        return np.array([], dtype=np.int64)
    hard = np.zeros(len(y_true), dtype=bool)
    for m in masks:
        hard |= m
    return np.flatnonzero(hard).astype(np.int64)


def oversample_by_indices(
    beats: np.ndarray,
    labels: np.ndarray,
    rr: np.ndarray,
    indices: np.ndarray,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append `indices` repeated `repeat` times to the training arrays."""
    if repeat <= 0 or len(indices) == 0:
        return beats, labels, rr
    extra_idx = np.tile(np.asarray(indices, dtype=np.int64), int(repeat))
    return (
        np.concatenate([beats, beats[extra_idx]], axis=0),
        np.concatenate([labels, labels[extra_idx]], axis=0),
        np.concatenate([rr, rr[extra_idx]], axis=0),
    )
