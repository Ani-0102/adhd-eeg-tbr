"""Stretch: a small 1D CNN on raw 4 s windows, scored per subject with the SQL folds.
Hyperparameters are fixed up front (no tuning on test folds)."""
import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.load import load_all
from src.models import bootstrap_ci, fold_split
from src.preprocess import bandpass, keep_mask, make_windows

DB_PATH = "data/eeg.duckdb"
EPOCHS = 8
BATCH = 64
LEARNING_RATE = 1e-3
N_SHUFFLES = 3
SEED = 0


class SmallCNN(nn.Module):
    """Input (batch, 19 channels, 512 samples) -> one logit per window."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(19, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


def load_windows():
    """All kept windows as (n_windows, 19, 512), each channel z-scored within its own window,
    plus the subject id of every window."""
    table, signals = load_all()
    arrays, owners = [], []
    for subject_id in table.subject_id:
        windows = make_windows(bandpass(signals[subject_id]))
        windows = windows[keep_mask(windows)]
        arrays.append(windows.transpose(0, 2, 1))  # -> (n, channels, samples)
        owners += [subject_id] * len(windows)
    x = np.concatenate(arrays)
    x = (x - x.mean(axis=2, keepdims=True)) / x.std(axis=2, keepdims=True)
    return x.astype(np.float32), np.array(owners)


def train(x, y):
    """Train a fresh CNN for a fixed number of epochs."""
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    x, y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), BATCH):
            pick = order[start:start + BATCH]
            optimizer.zero_grad()
            loss_fn(model(x[pick]), y[pick]).backward()
            optimizer.step()
    return model


def window_probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def average_per_subject(probs, owners, subject_ids):
    """Mean window probability for each subject, in the order of subject_ids."""
    frame = pd.DataFrame({"subject": owners, "p": probs})
    return frame.groupby("subject").p.mean().loc[subject_ids].to_numpy()


def cross_validate(x, owners, subject_ids, y, folds):
    """Train on windows of train subjects only; score test subjects by averaging their windows.
    Returns (AUROC per fold, out-of-fold subject scores)."""
    label_of = dict(zip(subject_ids, y))
    window_y = np.array([label_of[s] for s in owners])
    oof = np.zeros(len(subject_ids))
    aucs = []
    for k in sorted(set(folds)):
        train_pos, test_pos = fold_split(folds, k)
        train_windows = np.isin(owners, subject_ids[train_pos])
        test_windows = np.isin(owners, subject_ids[test_pos])
        torch.manual_seed(SEED + k)
        model = train(x[train_windows], window_y[train_windows])
        probs = window_probabilities(model, x[test_windows])
        oof[test_pos] = average_per_subject(probs, owners[test_windows], subject_ids[test_pos])
        aucs.append(roc_auc_score(y[test_pos], oof[test_pos]))
    return np.array(aucs), oof


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    meta = (con.sql("SELECT s.subject_id, s.label, sp.fold FROM subjects s "
                    "JOIN splits sp USING (subject_id) ORDER BY s.subject_id").df())
    subject_ids = meta.subject_id.to_numpy()
    y, folds = meta.label.to_numpy(), meta.fold.to_numpy()

    x, owners = load_windows()
    print("windows:", x.shape)
    rng = np.random.default_rng(SEED)

    aucs, oof = cross_validate(x, owners, subject_ids, y, folds)
    low, high = bootstrap_ci(y, oof, rng)
    print(f"CNN AUROC {aucs.mean():.3f} +- {aucs.std():.3f}, folds {aucs.round(2)}, CI [{low:.3f}, {high:.3f}]")

    shuffled = []
    for _ in range(N_SHUFFLES):
        a, _ = cross_validate(x, owners, subject_ids, rng.permutation(y), folds)
        shuffled.append(a.mean())
    print("shuffled-label AUROC:", np.round(shuffled, 3))

    row = pd.DataFrame([{"model": "C 1D CNN on raw windows", "n_features": 19 * 512,
                         "auroc_mean": aucs.mean(), "auroc_std": aucs.std(),
                         "auroc_pooled": roc_auc_score(y, oof), "ci_low": low, "ci_high": high,
                         "shuffled_auroc_mean": np.mean(shuffled),
                         "shuffled_auroc_std": np.std(shuffled)}])
    con.register("cnn_in", row)
    con.execute("CREATE OR REPLACE TABLE cnn_results AS SELECT * FROM cnn_in")
    con.close()
