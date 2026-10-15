import duckdb
import numpy as np
import pandas as pd

from src.load import CHANNELS
from src.stats import channel_tests, run_sql_file


def fake_db(n_adhd=61, n_control=60):
    """In-memory DB with the same tables the SQL expects, filled with random numbers."""
    rng = np.random.default_rng(0)
    subjects = pd.DataFrame({
        "subject_id": [f"a{i}" for i in range(n_adhd)] + [f"c{i}" for i in range(n_control)],
        "group": ["ADHD"] * n_adhd + ["Control"] * n_control,
    })
    rows = []
    for subject_id in subjects.subject_id:
        for window in range(5):
            for channel in ["Cz", "Pz"]:
                theta, beta = rng.uniform(1, 5, 2)
                rows.append({"window_id": f"{subject_id}_{window}", "subject_id": subject_id,
                             "channel": channel, "delta": 1.0, "theta": theta, "alpha": 1.0,
                             "beta": beta, "rel_delta": 0.1, "rel_theta": 0.2,
                             "rel_alpha": 0.3, "rel_beta": 0.4, "tbr": theta / beta})
    features = pd.DataFrame(rows)
    con = duckdb.connect()
    con.register("subjects_in", subjects)
    con.register("features_in", features)
    con.execute("CREATE TABLE subjects AS SELECT * FROM subjects_in")
    con.execute("CREATE TABLE features AS SELECT * FROM features_in")
    return con, features


def test_group_summary_matches_pandas():
    con, features = fake_db()
    run_sql_file(con, "sql/summaries.sql")
    sql = con.sql("SELECT * FROM group_channel_summary").df()

    group = features.subject_id.str[0].map({"a": "ADHD", "c": "Control"})
    expected = features.assign(group=group).groupby(["group", "channel"]).tbr.agg(
        ["count", "mean", "median"]).reset_index()
    merged = sql.merge(expected, on=["group", "channel"])
    assert len(merged) == 4
    assert (merged.n_windows == merged["count"]).all()
    assert np.allclose(merged.mean_tbr, merged["mean"])
    assert np.allclose(merged.median_tbr, merged["median"])


def test_subject_view_has_one_row_per_subject_and_channel():
    con, _ = fake_db()
    run_sql_file(con, "sql/summaries.sql")
    view = con.sql("SELECT * FROM subject_channel").df()
    assert len(view) == 121 * 2
    assert not view.duplicated(["subject_id", "channel"]).any()
    assert (view.n_windows == 5).all()


def test_every_subject_in_exactly_one_fold():
    con, _ = fake_db()
    run_sql_file(con, "sql/splits.sql")
    splits = con.sql("SELECT * FROM splits").df()
    assert len(splits) == 121
    assert splits.subject_id.is_unique  # no subject can be in both train and test
    assert set(splits.fold) == {0, 1, 2, 3, 4}


def test_folds_are_stratified_by_group():
    con, _ = fake_db()
    run_sql_file(con, "sql/splits.sql")
    counts = con.sql("""SELECT sp.fold, s."group", count(*) AS n
                        FROM splits sp JOIN subjects s USING (subject_id)
                        GROUP BY 1, 2""").df()
    assert counts.n.between(12, 13).all()


def test_splits_are_repeatable():
    con1, _ = fake_db()
    con2, _ = fake_db()
    run_sql_file(con1, "sql/splits.sql")
    run_sql_file(con2, "sql/splits.sql")
    a = con1.sql("SELECT * FROM splits ORDER BY subject_id").df()
    b = con2.sql("SELECT * FROM splits ORDER BY subject_id").df()
    assert a.equals(b)


def test_channel_tests_finds_a_planted_difference():
    # every channel gets the same planted shift: ADHD TBR about 2 higher than Control
    rng = np.random.default_rng(1)
    rows = []
    for channel in CHANNELS:
        for i in range(60):
            for group, shift in [("ADHD", 2.0), ("Control", 0.0)]:
                rows.append({"subject_id": f"{group}{i}", "group": group,
                             "channel": channel, "tbr": rng.normal(3 + shift, 0.5)})
    result = channel_tests(pd.DataFrame(rows))
    assert len(result) == 19
    assert (result.effect_r > 0.9).all()  # ADHD clearly higher
    assert result.significant.all()
