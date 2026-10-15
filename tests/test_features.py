import numpy as np

from src.features import window_features
from src.load import FS
from src.preprocess import WINDOW_SAMPLES, keep_mask, make_windows


def sine_window(freq_hz, n_channels=19):
    t = np.arange(WINDOW_SAMPLES) / FS
    wave = np.sin(2 * np.pi * freq_hz * t)
    return np.tile(wave[:, None], (1, n_channels))


def test_6hz_sine_is_theta():
    feats = window_features(sine_window(6))
    assert feats["theta"][0] > 10 * feats["delta"][0]
    assert feats["theta"][0] > 10 * feats["alpha"][0]
    assert feats["theta"][0] > 10 * feats["beta"][0]
    assert feats["rel_theta"][0] > 0.9


def test_20hz_sine_is_beta():
    feats = window_features(sine_window(20))
    assert feats["rel_beta"][0] > 0.9


def test_tbr_is_theta_over_beta():
    # 6 Hz + a weaker 20 Hz wave, so both bands are non-zero
    window = sine_window(6) + 0.5 * sine_window(20)
    feats = window_features(window)
    assert np.allclose(feats["tbr"], feats["theta"] / feats["beta"])
    assert feats["tbr"][0] > 1


def test_relative_powers_sum_to_one():
    window = np.random.default_rng(0).standard_normal((WINDOW_SAMPLES, 19))
    feats = window_features(window)
    total = sum(feats["rel_" + b] for b in ["delta", "theta", "alpha", "beta"])
    assert np.allclose(total, 1.0)


def test_make_windows_drops_leftover():
    data = np.zeros((WINDOW_SAMPLES * 3 + 100, 19))
    assert make_windows(data).shape == (3, WINDOW_SAMPLES, 19)


def test_keep_mask_drops_big_artifact():
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((10, WINDOW_SAMPLES, 19))
    windows[3] *= 100  # one huge window
    mask = keep_mask(windows)
    assert not mask[3]
    assert mask.sum() == 9
