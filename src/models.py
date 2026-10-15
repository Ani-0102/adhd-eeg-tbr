"""Subject-level logistic regression, cross-validated with the SQL `splits` folds."""
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = "data/eeg.duckdb"
N_BOOTSTRAP = 2000
N_SHUFFLES = 20
SEED = 0
POWER_COLUMNS = ["delta", "theta", "alpha", "beta",
                 "rel_delta", "rel_theta", "rel_alpha", "rel_beta"]


def make_model():
    """Scaler and classifier in one Pipeline, so scaling is fit on the training folds only."""
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000))])


def wide(subject_channel, column):
    """One row per subject, one column per channel, log10 of the subject's mean value."""
    table = subject_channel.pivot(index="subject_id", columns="channel", values=column)
    table.columns = [f"{column}_{c}" for c in table.columns]
    return np.log10(table)


def feature_sets(subject_channel):
    """Return {model name: DataFrame indexed by subject_id}."""
    tbr = wide(subject_channel, "tbr")
    all_bands = pd.concat([wide(subject_channel, c) for c in POWER_COLUMNS], axis=1)
    return {
        "A1 TBR, mean of channels": tbr.mean(axis=1).to_frame("tbr_mean"),
        "A2 TBR, Cz only": tbr[["tbr_Cz"]],
        "A3 TBR, all 19 channels": tbr,
        "B all band powers (log abs + rel)": all_bands,
    }


def fold_split(folds, k):
    """Positions of train and test subjects for fold k. Fold comes from the SQL splits table."""
    test = np.where(folds == k)[0]
    train = np.where(folds != k)[0]
    return train, test


def cross_validate(X, y, folds):
    """Return (AUROC per fold, out-of-fold predicted probability for every subject)."""
    oof = np.zeros(len(y))
    aucs = []
    for k in sorted(set(folds)):
        train, test = fold_split(folds, k)
        model = make_model()
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], oof[test]))
    return np.array(aucs), oof


def bootstrap_ci(y, scores, rng):
    """95% CI of AUROC, resampling subjects with replacement."""
    aucs = []
    while len(aucs) < N_BOOTSTRAP:
        pick = rng.integers(0, len(y), len(y))
        if len(set(y[pick])) < 2:  # need both classes to score
            continue
        aucs.append(roc_auc_score(y[pick], scores[pick]))
    return np.percentile(aucs, [2.5, 97.5])


def shuffled_auc(X, y, folds, rng):
    """Mean fold AUROC over several runs with the labels randomly shuffled (should be ~0.5)."""
    means = []
    for _ in range(N_SHUFFLES):
        aucs, _ = cross_validate(X, rng.permutation(y), folds)
        means.append(aucs.mean())
    return np.mean(means), np.std(means)


def run_all(subject_channel, subjects, splits):
    meta = subjects.merge(splits, on="subject_id").sort_values("subject_id").set_index("subject_id")
    y = meta["label"].to_numpy()
    folds = meta["fold"].to_numpy()
    rng = np.random.default_rng(SEED)

    rows = []
    curves = {}
    for name, features in feature_sets(subject_channel).items():
        X = features.loc[meta.index].to_numpy()
        aucs, oof = cross_validate(X, y, folds)
        low, high = bootstrap_ci(y, oof, rng)
        shuf_mean, shuf_std = shuffled_auc(X, y, folds, rng)
        rows.append({"model": name, "n_features": X.shape[1],
                     "auroc_mean": aucs.mean(), "auroc_std": aucs.std(),
                     "auroc_pooled": roc_auc_score(y, oof),
                     "ci_low": low, "ci_high": high,
                     "shuffled_auroc_mean": shuf_mean, "shuffled_auroc_std": shuf_std})
        curves[name] = roc_curve(y, oof)[:2]
    return pd.DataFrame(rows), curves


def plot_roc(curves, results, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (fpr, tpr) in curves.items():
        auc = results.loc[results.model == name, "auroc_pooled"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC, subject-level 5-fold CV")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    results, curves = run_all(con.sql("SELECT * FROM subject_channel").df(),
                              con.sql("SELECT * FROM subjects").df(),
                              con.sql("SELECT * FROM splits").df())
    con.register("results_in", results)
    con.execute("CREATE OR REPLACE TABLE model_results AS SELECT * FROM results_in")
    con.close()
    pd.set_option("display.width", 200)
    print(results.drop(columns="n_features").round(3).to_string(index=False))
    plot_roc(curves, results, "figs/roc.png")
