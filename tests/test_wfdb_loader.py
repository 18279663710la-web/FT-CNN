import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_select_lead_prefers_mlii():
    from src.data.wfdb_loader import select_lead_index

    names = ["V1", "MLII"]
    assert select_lead_index(names, ["MLII", "II", "V5"]) == 1


def test_process_synthetic_record_shapes(tmp_path):
    """Build a tiny synthetic signal+annotations path without real WFDB files."""
    from src.data.wfdb_loader import beats_from_arrays

    fs = 360
    t = np.arange(0, 10 * fs) / fs
    # synthetic ECG-like pulses every ~1s
    sig = np.sin(2 * np.pi * 1.0 * t) * 0.1
    peaks = np.arange(fs, 9 * fs, fs)
    for p in peaks:
        sig[p] = 1.5
    symbols = ["N"] * len(peaks)
    beats, labels, rr = beats_from_arrays(
        sig,
        peaks,
        symbols,
        fs=fs,
        before=93,
        after=94,
        local_n=3,
        bandpass=True,
    )
    assert beats.ndim == 2 and beats.shape[1] == 187
    assert labels.shape[0] == beats.shape[0]
    assert rr.shape == (beats.shape[0], 4)
    assert set(labels.tolist()).issubset({0, 1, 2, 3, 4})
