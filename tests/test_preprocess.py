import numpy as np
import pytest


def test_extract_beat_length_and_bounds():
    from src.data.preprocess import extract_beat

    sig = np.arange(1000, dtype=np.float64)
    beat = extract_beat(sig, r_idx=200, before=93, after=94)
    assert beat is not None
    assert beat.shape == (187,)
    assert beat[93] == 200

    assert extract_beat(sig, r_idx=10, before=93, after=94) is None
    assert extract_beat(sig, r_idx=990, before=93, after=94) is None


def test_bandpass_attenuates_dc():
    from src.data.preprocess import bandpass_filter

    fs = 360.0
    t = np.arange(0, 2.0, 1 / fs)
    # strong DC + mid-band sinusoid
    x = 5.0 + 0.5 * np.sin(2 * np.pi * 5.0 * t)
    y = bandpass_filter(x, fs=fs, low_hz=0.5, high_hz=45.0, order=3)
    assert abs(np.mean(y)) < abs(np.mean(x)) * 0.2


def test_rr_features_shapes_and_ratio():
    from src.data.preprocess import compute_rr_features

    # peaks at 0,100,200,300,400 samples → RR=100
    peaks = np.array([0, 100, 200, 300, 400], dtype=int)
    feats = compute_rr_features(peaks, local_n=3)
    assert feats.shape == (5, 4)
    # middle beat index 2: prev=100, post=100, local~100, ratio~1
    assert feats[2, 0] == pytest.approx(100.0)
    assert feats[2, 1] == pytest.approx(100.0)
    assert feats[2, 3] == pytest.approx(feats[2, 0] / feats[2, 2])
