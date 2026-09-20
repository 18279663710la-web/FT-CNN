"""Build multi-beat context windows for CNN+LSTM (method A)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_context_windows(
    beats: NDArray[np.floating],
    labels: NDArray[np.integer],
    rr: NDArray[np.floating],
    half_window: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack consecutive beats: for center i, use [i-h, ..., i, ..., i+h].

    Returns
    -------
    beats_ctx : (N, 2*h+1, L)
    labels_center : (N,)
    rr_seq : (N, 2*h+1, R)
    """
    beats = np.asarray(beats)
    labels = np.asarray(labels)
    rr = np.asarray(rr)
    n = len(labels)
    h = int(half_window)
    width = 2 * h + 1
    if n < width:
        L = beats.shape[1] if beats.ndim == 2 else 0
        R = rr.shape[1] if rr.ndim == 2 else 0
        return (
            np.zeros((0, width, L), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, width, R), dtype=np.float32),
        )

    centers = range(h, n - h)
    b_list, y_list, r_list = [], [], []
    for i in centers:
        b_list.append(beats[i - h : i + h + 1])
        y_list.append(labels[i])
        r_list.append(rr[i - h : i + h + 1])

    return (
        np.asarray(b_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(r_list, dtype=np.float32),
    )
