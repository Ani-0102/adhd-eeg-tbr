"""Run load -> preprocess -> features for all subjects and write data/eeg.duckdb."""
import duckdb
import pandas as pd

from src.features import window_features
from src.load import CHANNELS, FS, load_all
from src.preprocess import bandpass, keep_mask, make_windows
from src.preprocess import WINDOW_S

DB_PATH = "data/eeg.duckdb"
FEATURE_NAMES = ["delta", "theta", "alpha", "beta",
                 "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "tbr"]


def process_subject(subject_id, data):
    """Return (windows_rows, feature_rows) for one subject."""
    windows = make_windows(bandpass(data))
    kept = keep_mask(windows)
    window_rows = []
    feature_rows = []
    for i, window in enumerate(windows):
        window_id = f"{subject_id}_{i}"
        window_rows.append({"window_id": window_id, "subject_id": subject_id,
                            "start_s": i * WINDOW_S, "kept": bool(kept[i])})
        if not kept[i]:
            continue
        feats = window_features(window)
        for c, channel in enumerate(CHANNELS):
            row = {"window_id": window_id, "subject_id": subject_id, "channel": channel}
            for name in FEATURE_NAMES:
                row[name] = float(feats[name][c])
            feature_rows.append(row)
    return window_rows, feature_rows


def build(db_path=DB_PATH, raw_dir="data/raw"):
    table, signals = load_all(raw_dir)
    window_rows = []
    feature_rows = []
    for subject_id, data in signals.items():
        w, f = process_subject(subject_id, data)
        window_rows += w
        feature_rows += f
    windows = pd.DataFrame(window_rows)
    features = pd.DataFrame(feature_rows)

    kept_counts = windows[windows.kept].groupby("subject_id").size()
    subjects = table[["subject_id", "group", "duration_s"]].copy()
    subjects["label"] = (subjects["group"] == "ADHD").astype(int)
    subjects["n_windows_kept"] = subjects["subject_id"].map(kept_counts).fillna(0).astype(int)
    subjects = subjects[["subject_id", "group", "label", "duration_s", "n_windows_kept"]]

    recordings = pd.DataFrame([
        {"subject_id": row.subject_id, "channel": ch, "n_samples": row.n_samples, "fs": FS}
        for row in table.itertuples() for ch in CHANNELS
    ])

    con = duckdb.connect(db_path)
    for name, df in [("subjects", subjects), ("recordings", recordings),
                     ("windows", windows), ("features", features)]:
        con.register("df_in", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df_in")
        con.unregister("df_in")
    con.close()
    return subjects, windows, features


if __name__ == "__main__":
    subjects, windows, features = build()
    print("subjects:", len(subjects), "| windows:", len(windows),
          "| kept:", int(windows.kept.sum()), "| feature rows:", len(features))
    print(subjects.groupby("group").agg(n=("subject_id", "count"),
          mean_dur_s=("duration_s", "mean"), kept=("n_windows_kept", "sum")).round(1))
