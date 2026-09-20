"""Mild RR prematurity augmentation for S beats (train-only)."""

from __future__ import annotations

import numpy as np


def shorten_center_rr_prev(
    rr: np.ndarray,
    center: int,
    scale: float | np.ndarray,
) -> np.ndarray:
    """
    Copy RR windows and shorten center RR_prev by `scale`, then refresh RR_ratio.

    RR layout: [prev, post, local, ratio].
    """
    out = np.array(rr, dtype=np.float32, copy=True)
    scale_arr = np.asarray(scale, dtype=np.float32)
    out[..., center, 0] = out[..., center, 0] * scale_arr
    local = np.maximum(out[..., center, 2], 1e-6)
    out[..., center, 3] = out[..., center, 0] / local
    return out


def augment_s_rr_prematurity(
    beats: np.ndarray,
    labels: np.ndarray,
    rr: np.ndarray,
    *,
    center: int = 2,
    copies: int = 1,
    scale_min: float = 0.85,
    scale_max: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Append `copies` RR-perturbed clones of each S sample (label==1).
    Morphology (beats) is copied unchanged; only center prematurity RR is nudged.
    """
    if copies <= 0:
        return beats, labels, rr
    rng = rng or np.random.default_rng(42)
    s_idx = np.flatnonzero(labels == 1)
    if len(s_idx) == 0:
        return beats, labels, rr

    extra_b, extra_y, extra_rr = [], [], []
    for _ in range(int(copies)):
        scales = rng.uniform(scale_min, scale_max, size=len(s_idx)).astype(np.float32)
        rr_aug = shorten_center_rr_prev(rr[s_idx], center=center, scale=scales)
        extra_b.append(beats[s_idx])
        extra_y.append(labels[s_idx])
        extra_rr.append(rr_aug)

    return (
        np.concatenate([beats, *extra_b], axis=0),
        np.concatenate([labels, *extra_y], axis=0),
        np.concatenate([rr, *extra_rr], axis=0),
    )
