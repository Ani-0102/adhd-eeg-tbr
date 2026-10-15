"""Save a sanity-check PSD plot: one ADHD and one Control subject, channel Cz."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features import BANDS, psd
from src.load import CHANNELS, load_all
from src.preprocess import bandpass, keep_mask, make_windows


def mean_psd(data, channel_index):
    """Average the PSD of one channel over a subject's kept windows."""
    windows = make_windows(bandpass(data))
    windows = windows[keep_mask(windows)]
    spectra = []
    for window in windows:
        freqs, power = psd(window)
        spectra.append(power[:, channel_index])
    return freqs, np.mean(spectra, axis=0)


if __name__ == "__main__":
    table, signals = load_all()
    cz = CHANNELS.index("Cz")
    fig, ax = plt.subplots(figsize=(7, 4))
    for group in ["ADHD", "Control"]:
        subject_id = table[table.group == group].subject_id.iloc[0]
        freqs, power = mean_psd(signals[subject_id], cz)
        ax.semilogy(freqs, power, label=f"{group} ({subject_id})")
    ax.set_xlim(1, 40)
    ax.set_ylim(0.5, 1e4)  # below 1 Hz the bandpass leaves almost nothing
    for name, (low, high) in BANDS.items():
        ax.axvline(low, color="grey", linewidth=0.5)
        ax.text(low + 0.3, ax.get_ylim()[1] * 0.5, name, fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (raw units^2 / Hz)")
    ax.set_title("Mean Welch PSD at Cz, one subject per group")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/psd_example.png", dpi=120)
    print("saved figs/psd_example.png")
