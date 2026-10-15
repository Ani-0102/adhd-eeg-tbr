# NOTES — progress log (Claude Code updates this)

## Checkpoints
- [x] 1. Load + features
- [x] 2. SQL + stats
- [x] 3. Models
- [x] 4. App + README

## Setup status
- [x] Dataset downloaded into data/raw/ (IEEE DataPort, DOI 10.21227/rzfh-zn36; 121 .mat files: ADHD 61, Control 60)
- [x] License checked: CC BY (attribution), per IEEE DataPort FAQ "Who Owns the Datasets on IEEE DataPort?" (uploads are made available under CC-BY by the Terms of Service). The dataset page itself shows no licence line and the version (likely 4.0) is unconfirmed. Derived features can be committed: y, with attribution/citation (in README).

## Dataset facts (verified from files)
- Subjects: ADHD 61 / Control 60 (121 .mat files; matches dataset page)
- Channels / order: 19, from Channel_Labels.docx: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz (differs from the order listed on the web page; .mat files have no labels)
- Sampling rate: 128 Hz (from the dataset page; .mat files do not store it)
- Recording length per group (mean s): ADHD 154.6 (72-338), Control 124.9 (62-203). LENGTH CONFOUND: ADHD ~24% longer. Never a feature.
- Age/sex available? NO: .mat files hold only the signal matrix. Cannot control for age (Limitations). Also: ADHD kids had taken Ritalin up to 6 months (page).
- File quirks: one variable per .mat named after the file; dtype int16 for 89 files, float64 for 32; no NaNs; units undocumented (std ~190, so not clean microvolts).

## Decisions
- Window 4 s non-overlapping; bandpass 1-40 Hz; artifact rule: drop window if worst-channel peak-to-peak > 3x that subject's median (scale-free; a fixed µV cut dropped 100% of windows because amplitudes are large and units unknown). Uses no group info.

## Results (fill in as they arrive)
- Windows kept per group: ADHD 2274/2330 (97.6%), Control 1800/1843 (97.7%); per subject 15-83 kept. 77,406 feature rows (4074 windows x 19 ch).
- Unadjusted pooled TBR (sanity only, NOT a result): mean ADHD 2.99 vs Control 3.68 (ADHD lower, opposite of textbook). Needs subject-level stats in checkpoint 2.
- FDR-significant TBR channels: 3 of 19 (BH q<0.05): T7 (r=-0.35, q=0.009), P7 (r=-0.35, q=0.009), Pz (r=-0.33, q=0.010). Subject-level Mann-Whitney, n=61 vs 60. ADHD TBR is LOWER than Control in all 19 channels (r from -0.03 to -0.35), opposite of the classic 'high TBR in ADHD' claim. 7 more channels have raw p<0.05 but q 0.086-0.089 (F3 F4 P3 F7 T8 P8 Fz). Possible explanations NOT tested: Ritalin use in the ADHD group, length/task confound, undocumented units/filtering. Do not claim a cause.
- AUROC A (TBR only), mean fold AUROC ± std over 5 folds, [95% bootstrap CI over subjects of pooled out-of-fold AUROC]: A1 mean-of-channels 0.634 ± 0.108 [0.520, 0.719]; A2 Cz only 0.585 ± 0.085 [0.465, 0.681]; A3 all 19 ch 0.553 ± 0.091 [0.446, 0.650]. Only A1 has a CI just above 0.5; A2/A3 CIs include 0.5.
- AUROC B (all bands, 152 features): 0.676 ± 0.135 [0.575, 0.767]. Best model but weak/moderate; fold std is large (n≈24 per test fold). Default L2 logistic regression, nothing tuned.
- Shuffled-label AUROC (mean of 20 shuffles): A1 0.483, A2 0.508, A3 0.504, B 0.440 ± 0.063. Near chance; B's 0.44 is slightly below 0.5, a known small-n CV effect, not signal. Real B (0.676) sits well above it.

## Log
<!-- date — checkpoint — what changed — open issues -->
2026-10-14 — setup — downloaded 4 zips + Channel_Labels via Chrome, unzipped to data/raw/<zip>/<zip>/*.mat; added .gitignore. Licence not stated on page: still to check.
2026-10-14 — checkpoint 1 — built src/{load,preprocess,features,build_db,make_figs}.py, tests/test_features.py (6 pass), data/eeg.duckdb (subjects, recordings, windows, features), figs/psd_example.png, venv + requirements.txt, pytest.ini. Open: licence still unchecked; DuckDB column `group` is a reserved word, quote it as "group" in SQL.
2026-10-14 — checkpoint 2 — sql/splits.sql (5 folds, stratified, hash seed '42': folds 12-13 ADHD / 12 Control each), sql/summaries.sql (views subject_channel, group_channel_summary), src/stats.py (MWU + rank-biserial r + BH FDR; writes group_stats table), figs/tbr_effect_size.png, figs/tbr_subject_box.png, tests/test_sql.py. 12 tests pass. Open: licence unchecked; fold = hash order within group, so folds are random not balanced on recording length.
2026-10-14 — checkpoint 3 — src/models.py (subject-level logistic regression in Pipeline(StandardScaler, LogReg), folds from SQL splits, bootstrap CI, shuffled-label check, model_results table, figs/roc.png), tests/test_models.py. 18 tests pass. Open: licence unchecked; no feature selection or tuning done (by design); length/Ritalin confounds can still inflate or distort AUROC.
2026-10-14 — checkpoint 4 — src/export_app_data.py -> data/app_features.parquet (2299 rows = 121 subjects x 19 ch, subject-level only), data/app_group_stats.csv, data/app_model_results.csv; app.py (3 pages), requirements.txt (app, pinned) + requirements-pipeline.txt, README.md with deploy checklist, tests/test_app.py (AppTest on every page). 20 tests pass; headless server health check ok. Open: licence still unchecked (blocks committing derived files); no git repo; live URL TBD.

## Resume bullet (real numbers)
"Tested whether the EEG theta/beta ratio separates 121 children (61 ADHD, 60 control) on a public dataset; stored band-power features in DuckDB and built subject-level cross-validation splits in SQL to prevent leakage; TBR alone reached 0.55-0.63 AUROC vs 0.68 for all band powers (95% CI 0.58-0.77), and TBR was lower, not higher, in ADHD (3 of 19 channels FDR-significant)."
2026-10-14 — stretch (Ani said go) — src/cnn.py: 1D CNN (2 conv layers, fixed 8 epochs, Adam 1e-3, no tuning), per-window z-score, subject-level SQL folds, window probs averaged per subject. AUROC 0.792 ± 0.102, CI [0.713, 0.865] (seed 0); seeds 1/2 gave 0.767/0.785; shuffled labels 0.528/0.448/0.340 (mean 0.440). Stronger than band-power B (0.676) but UNEXPLAINED; do not call it a biomarker (Ritalin, length, task confounds untested). src/pubmed_check.py (NCBI E-utilities via certifi SSL context; 32 papers -> data/literature_check.csv; list only, no LLM). Lit context: Arns 2012 (higher TBR but heterogeneous, not reliable diagnostic), Boxum 2025 (no diagnostic value). App now has 4 pages. 23 tests pass.
2026-10-14 — licence — CC BY basis recorded (see Setup status). Residual: dataset page states no licence/version; README cites dataset. Optionally ask the authors (nasrabadi@shahed.ac.ir) to confirm.
