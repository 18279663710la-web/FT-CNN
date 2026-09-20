import numpy as np
import pytest


def test_amplitude_scale_changes_signal():
    from src.data.augment import amplitude_scale

    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 187)
    y, scale = amplitude_scale(x, 0.9, 1.1, rng)
    assert scale != 1.0 or np.allclose(y, x)
    assert y.shape == x.shape


def test_reject_extreme_scale():
    from src.data.augment import maybe_augment_beat

    rng = np.random.default_rng(1)
    x = np.zeros(187)
    x[93] = 1.0  # R peak
    # Force extreme scale beyond 20% QRS amp change → reject → return original
    out, accepted = maybe_augment_beat(
        x,
        amp_range=(3.0, 3.0),
        warp_range=(1.0, 1.0),
        max_r_shift=5,
        max_qrs_amp_change=0.2,
        rng=rng,
    )
    assert accepted is False
    assert np.allclose(out, x)


def test_accept_mild_scale():
    from src.data.augment import maybe_augment_beat

    rng = np.random.default_rng(2)
    x = np.zeros(187)
    x[93] = 1.0
    out, accepted = maybe_augment_beat(
        x,
        amp_range=(1.05, 1.05),
        warp_range=(1.0, 1.0),
        max_r_shift=5,
        max_qrs_amp_change=0.2,
        rng=rng,
    )
    assert accepted is True
    assert out[93] == pytest.approx(1.05)
