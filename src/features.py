"""Welch PSD -> band power per channel for one 4 s window."""
import numpy as np
from scipy.signal import welch

from src.load import FS

# (low, high) in Hz; each band includes low and excludes high, so they tile 1-30 Hz
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def psd(window, fs=FS):
    """Welch PSD of a samples x channels window. Returns (freqs, psd) with psd as freqs x channels."""
    return welch(window, fs=fs, nperseg=256, axis=0)


def band_power(freqs, power, low, high):
    """Absolute power in [low, high): PSD summed over the bins, times the bin width."""
    in_band = (freqs >= low) & (freqs < high)
    bin_width = freqs[1] - freqs[0]
    return power[in_band].sum(axis=0) * bin_width


def window_features(window, fs=FS):
    """Return a dict of arrays (one value per channel):
    delta/theta/alpha/beta, rel_delta/.., and tbr = theta / beta."""
    freqs, power = psd(window, fs)
    out = {name: band_power(freqs, power, lo, hi) for name, (lo, hi) in BANDS.items()}
    total = band_power(freqs, power, 1, 30)
    for name in BANDS:
        out["rel_" + name] = out[name] / total
    out["tbr"] = out["theta"] / out["beta"]
    return out
