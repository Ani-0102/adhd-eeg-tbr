"""Bandpass filter a recording and cut it into non-overlapping 4 s windows."""
import numpy as np
from scipy.signal import butter, filtfilt

from src.load import FS

WINDOW_S = 4
WINDOW_SAMPLES = WINDOW_S * FS  # 512
# Units in the .mat files are not documented and amplitudes are large (median window
# peak-to-peak ~1600), so a fixed microvolt threshold would drop everything.
# Instead: drop a window if its worst channel is more than 3x this subject's median.
ARTIFACT_FACTOR = 3.0


def bandpass(data, low=1.0, high=40.0, fs=FS, order=4):
    """Zero-phase Butterworth bandpass along time (axis 0)."""
    nyquist = fs / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, data, axis=0)


def make_windows(data):
    """Split samples x channels into an array of shape (n_windows, 512, channels).
    The leftover samples at the end (less than 4 s) are dropped."""
    n_windows = data.shape[0] // WINDOW_SAMPLES
    trimmed = data[: n_windows * WINDOW_SAMPLES]
    return trimmed.reshape(n_windows, WINDOW_SAMPLES, data.shape[1])


def peak_to_peak(windows):
    """Per window and channel: max minus min. Shape (n_windows, channels)."""
    return windows.max(axis=1) - windows.min(axis=1)


def keep_mask(windows, factor=ARTIFACT_FACTOR):
    """True for windows whose worst-channel peak-to-peak is within factor x the
    subject's median worst-channel peak-to-peak. Uses no group information."""
    worst = peak_to_peak(windows).max(axis=1)
    return worst <= factor * np.median(worst)
