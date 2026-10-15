import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import cross_validate, fold_split, make_model, shuffled_auc
from src.stats import run_sql_file
from tests.test_sql import fake_db


def sql_folds():
    """Subject ids and folds produced by the real splits.sql on fake subjects."""
    con, _ = fake_db()
    run_sql_file(con, "sql/splits.sql")
    splits = con.sql("SELECT * FROM splits ORDER BY subject_id").df()
    return splits.subject_id.to_numpy(), splits.fold.to_numpy()


def test_no_subject_in_both_train_and_test():
    subject_ids, folds = sql_folds()
    for k in range(5):
        train, test = fold_split(folds, k)
        assert set(subject_ids[train]).isdisjoint(subject_ids[test])
        assert len(train) + len(test) == len(subject_ids)


def test_every_subject_is_tested_exactly_once():
    subject_ids, folds = sql_folds()
    tested = np.concatenate([fold_split(folds, k)[1] for k in range(5)])
    assert sorted(tested) == list(range(len(subject_ids)))


def test_model_is_a_pipeline_with_a_scaler():
    model = make_model()
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scale"], StandardScaler)


def test_scaler_is_fit_on_train_rows_only():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    X[80:] += 50  # test rows look very different from train rows
    y = rng.integers(0, 2, 100)
    train = np.arange(80)
    model = make_model().fit(X[train], y[train])
    scaler_mean = model.named_steps["scale"].mean_
    assert np.allclose(scaler_mean, X[train].mean(axis=0))
    assert not np.allclose(scaler_mean, X.mean(axis=0))


def test_shuffled_labels_give_chance_auroc():
    rng = np.random.default_rng(1)
    _, folds = sql_folds()
    y = np.array([1] * 61 + [0] * 60)
    X = rng.standard_normal((121, 10))  # pure noise
    mean_auc, _ = shuffled_auc(X, y, folds, rng)
    assert 0.35 < mean_auc < 0.65  # loose bound around 0.5


def test_model_finds_a_planted_signal():
    rng = np.random.default_rng(2)
    _, folds = sql_folds()
    y = np.array([1] * 61 + [0] * 60)
    X = rng.standard_normal((121, 5))
    X[:, 0] += 2 * y  # feature 0 carries the label
    aucs, _ = cross_validate(X, y, folds)
    assert aucs.mean() > 0.85
