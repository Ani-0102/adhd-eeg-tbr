import pandas as pd
from streamlit.testing.v1 import AppTest


def test_exported_features_are_subject_level_only():
    features = pd.read_parquet("data/app_features.parquet")
    assert features.subject_id.nunique() == 121
    assert len(features) == 121 * 19
    assert not features.duplicated(["subject_id", "channel"]).any()
    assert "window_id" not in features.columns  # no window-level data in the app file


def test_app_runs_on_every_page():
    app = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not app.exception
    for page in ["Model comparison", "Literature check", "About the data", "TBR by group"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
