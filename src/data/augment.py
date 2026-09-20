"""Training-only ECG beat augmentation with morphology reject rules."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def amplitude_scale(
    beat: NDArray[np.floating],
    low: float,
    high: float,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.floating], float]:
    scale = float(rng.uniform(low, high))
    return beat * scale, scale


def temporal_warp(
    beat: NDArray[np.floating],
    low: float,
    high: float,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.floating], float]:
    """Stretch/compress time axis then resample back to original length."""
    alpha = float(rng.uniform(low, high))
    n = len(beat)
    src = np.linspace(0.0, 1.0, n)
    # Warp: new duration = n * alpha, then crop/pad via interpolation onto [0,1]
    warped_len = max(2, int(round(n * alpha)))
    warped_t = np.linspace(0.0, 1.0, warped_len)
    warped = np.interp(warped_t, src, beat)
    out = np.interp(src, np.linspace(0.0, 1.0, warped_len), warped)
    return out.astype(np.float64, copy=False), alpha


def _qrs_amplitude(beat: NDArray[np.floating], r_pos: int = 93, half_w: int = 20) -> float:
    lo = max(0, r_pos - half_w)
    hi = min(len(beat), r_pos + half_w + 1)
    seg = beat[lo:hi]
    return float(np.max(np.abs(seg))) if len(seg) else 0.0


def maybe_augment_beat(
    beat: NDArray[np.floating],
    amp_range: Tuple[float, float],
    warp_range: Tuple[float, float],
    max_r_shift: int = 5,
    max_qrs_amp_change: float = 0.2,
    rng: np.random.Generator | None = None,
    r_pos: int = 93,
) -> Tuple[NDArray[np.floating], bool]:
    """
    Apply amplitude scale + temporal warp.
    Reject if R-peak shifts > max_r_shift or QRS amp changes > max_qrs_amp_change.
    On reject, return the original beat and accepted=False.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(beat, dtype=np.float64)
    orig_amp = _qrs_amplitude(x, r_pos=r_pos)

    y, _ = amplitude_scale(x, amp_range[0], amp_range[1], rng)
    y, _ = temporal_warp(y, warp_range[0], warp_range[1], rng)

    # Approximate R location as argmax |y| near original r_pos
    search_lo = max(0, r_pos - max_r_shift * 3)
    search_hi = min(len(y), r_pos + max_r_shift * 3 + 1)
    local = y[search_lo:search_hi]
    new_r = int(search_lo + np.argmax(np.abs(local))) if len(local) else r_pos
    r_shift = abs(new_r - r_pos)

    new_amp = _qrs_amplitude(y, r_pos=new_r)
    amp_change = abs(new_amp - orig_amp) / (orig_amp + 1e-8)

    if r_shift > max_r_shift or amp_change > max_qrs_amp_change:
        return x.copy(), False
    return y, True
