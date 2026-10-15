"""Load every subject's .mat file into memory plus a one-row-per-subject table."""
import glob
import os

import pandas as pd
import scipy.io

RAW_DIR = "data/raw"
FS = 128  # Hz, from the dataset page
# Order from Channel_Labels.docx (the .mat files have no labels)
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]


def group_from_path(path):
    # folder names look like ADHD_part1 / Control_part2
    return "ADHD" if "ADHD" in path else "Control"


def load_recording(path):
    """Return the samples x 19 channels matrix, as floats."""
    mat = scipy.io.loadmat(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return mat[name].astype(float)


def load_all(raw_dir=RAW_DIR):
    """Return (table, signals). signals maps subject_id -> samples x channels array."""
    rows = []
    signals = {}
    for path in sorted(glob.glob(f"{raw_dir}/*/*/*.mat")):
        subject_id = os.path.splitext(os.path.basename(path))[0]
        data = load_recording(path)
        assert data.shape[1] == len(CHANNELS), f"{subject_id}: {data.shape[1]} channels"
        signals[subject_id] = data
        rows.append({
            "subject_id": subject_id,
            "group": group_from_path(path),
            "n_samples": data.shape[0],
            "duration_s": data.shape[0] / FS,
            "source_file": path,
        })
    return pd.DataFrame(rows), signals


if __name__ == "__main__":
    table, _ = load_all()
    print(table.groupby("group")["duration_s"].agg(["count", "mean", "min", "max"]).round(1))
    print("Channel order:", CHANNELS)
