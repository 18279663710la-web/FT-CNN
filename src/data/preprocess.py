"""ECG filtering, beat windowing, and RR feature extraction."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 45.0,
    order: int = 3,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + eps)


def extract_beat(
    signal: np.ndarray,
    r_idx: int,
    before: int = 93,
    after: int = 94,
) -> Optional[np.ndarray]:
    """Extract fixed-length window centered on R; None if out of bounds."""
    x = np.asarray(signal)
    start = int(r_idx) - before
    end = int(r_idx) + after  # exclusive end → length = before + after
    if start < 0 or end > len(x):
        return None
    return x[start:end].astype(np.float64, copy=True)


def compute_rr_features(r_peaks: np.ndarray, local_n: int = 10) -> np.ndarray:
    """
    For each beat i return [RR_prev, RR_post, RR_local, RR_ratio].

    Edge beats use available neighbors; RR_local is mean of up to local_n
    neighboring RR intervals around beat i.
    """
    peaks = np.asarray(r_peaks, dtype=np.float64)
    n = len(peaks)
    feats = np.zeros((n, 4), dtype=np.float64)
    if n == 0:
        return feats

    rr = np.diff(peaks)  # length n-1; rr[i] = peaks[i+1]-peaks[i]

    for i in range(n):
        prev = rr[i - 1] if i - 1 >= 0 else (rr[0] if len(rr) else 0.0)
        post = rr[i] if i < len(rr) else (rr[-1] if len(rr) else 0.0)

        # Local mean over RR intervals whose right peak is within a window of beats
        lo = max(0, i - local_n // 2)
        hi = min(len(rr), i + local_n // 2)
        if hi <= lo:
            local = prev if prev != 0 else 1.0
        else:
            local = float(np.mean(rr[lo:hi]))
        if local == 0:
            local = 1.0
        ratio = float(prev / local)
        feats[i] = [prev, post, local, ratio]
    return feats
