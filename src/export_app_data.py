"""Export the small derived files the Streamlit app reads. No raw EEG leaves data/raw/."""
import duckdb
import pandas as pd

DB_PATH = "data/eeg.duckdb"

if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)
    # subject x channel features (mean over each subject's kept windows) + recording length
    features = con.sql("""
        SELECT sc.*, s.duration_s
        FROM subject_channel sc JOIN subjects s USING (subject_id)
        ORDER BY sc.subject_id, sc.channel""").df()
    features.to_parquet("data/app_features.parquet", index=False)
    con.sql("SELECT * FROM group_stats").df().to_csv("data/app_group_stats.csv", index=False)
    models = con.sql("SELECT * FROM model_results").df()
    tables = [t[0] for t in con.sql("SHOW TABLES").fetchall()]
    if "cnn_results" in tables:  # stretch model, present only if src/cnn.py was run
        models = pd.concat([models, con.sql("SELECT * FROM cnn_results").df()])
    models.to_csv("data/app_model_results.csv", index=False)
    con.close()
    print("subjects:", features.subject_id.nunique(), "| rows:", len(features))
