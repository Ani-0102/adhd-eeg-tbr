"""Streamlit demo: ADHD vs control EEG theta/beta ratio. Research demo, not a diagnostic tool."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ADHD EEG TBR", layout="wide")


@st.cache_data
def load_data():
    features = pd.read_parquet("data/app_features.parquet")
    stats = pd.read_csv("data/app_group_stats.csv")
    models = pd.read_csv("data/app_model_results.csv")
    return features, stats, models


def page_tbr(features, stats):
    st.header("Theta/beta ratio (TBR) by group")
    st.write("One dot = one child (mean TBR over their clean 4 s windows).")
    channel = st.selectbox("Channel", list(stats.channel), index=list(stats.channel).index("Pz"))
    one = features[features.channel == channel]
    row = stats[stats.channel == channel].iloc[0]

    fig, ax = plt.subplots(figsize=(4, 4))
    groups = ["ADHD", "Control"]
    ax.boxplot([one[one.group == g].tbr for g in groups], showfliers=False)
    for i, g in enumerate(groups):
        values = one[one.group == g].tbr
        ax.scatter([i + 1] * len(values), values, s=8, alpha=0.4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Subject mean TBR")
    ax.set_title(channel)
    left, right = st.columns(2)
    left.pyplot(fig)
    right.metric("Effect size r (negative = ADHD lower)", f"{row.effect_r:.2f}")
    right.metric("p (uncorrected)", f"{row.p:.3f}")
    right.metric("p after FDR (19 channels)", f"{row.p_fdr:.3f}")

    st.subheader("All channels")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    colors = ["tab:red" if s else "tab:grey" for s in stats.significant]
    ax2.bar(stats.channel, stats.effect_r, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Rank-biserial r")
    ax2.set_title("Red = FDR < 0.05")
    st.pyplot(fig2)
    st.caption("ADHD TBR is lower than control in every channel, the opposite of the "
               "often-quoted 'high TBR in ADHD'. See the limitations page.")


def page_models(models):
    st.header("Can EEG band power classify ADHD?")
    table = models[["model", "n_features", "auroc_mean", "auroc_std", "ci_low", "ci_high",
                    "shuffled_auroc_mean"]].copy()
    table.columns = ["Model", "Features", "AUROC (mean of folds)", "Std over folds",
                     "95% CI low", "95% CI high", "AUROC with shuffled labels"]
    st.dataframe(table.round(3), hide_index=True)
    st.write("Logistic regression, 5-fold cross-validation with whole children held out "
             "(split by subject). Scores near 0.5 are chance. The CI comes from resampling "
             "children; a CI that includes 0.5 means the model is not clearly better than chance.")
    st.caption("Models A and B use band-power features; model C (if shown) is a small 1D CNN on raw "
               "4 s windows whose window predictions are averaged per child. The ROC curves below "
               "cover A and B only.")
    st.image("figs/roc.png", width=450)


def page_literature():
    st.header("Literature check (PubMed)")
    st.markdown("""
Two meta-analyses found by the search below (via [PubMed](https://pubmed.ncbi.nlm.nih.gov/)):
- Arns, Conners & Kraemer, *J Atten Disord* 2012: [doi](https://doi.org/10.1177/1087054712460087).
  Nine studies (1253 children with ADHD, 517 without) reported higher TBR in ADHD at Cz with eyes open
  (grand-mean effect size 0.62 to 0.75), but heterogeneity was significant, so these were treated as
  overestimates. The authors concluded excess TBR is not a reliable diagnostic measure.
- Boxum et al., *Appl Psychophysiol Biofeedback* 2025: [doi](https://doi.org/10.1007/s10484-024-09649-y).
  A meta-analytic approach on 417 patients found no link between TBR-based EEG subtypes and behavior
  and concluded, in line with earlier work, that TBR has no diagnostic value for ADHD.

**How this project compares:** we found *lower* TBR in ADHD, not higher, in a task (not eyes-open
resting) recording of medicated children. So this is not a replication of the high-TBR finding. It is
consistent with the literature's caution that TBR is an unreliable diagnostic marker.
""")
    papers = pd.read_csv("data/literature_check.csv")
    st.caption(f"{len(papers)} papers returned by three PubMed searches, newest first. This is a "
               "list of search results, not a systematic review.")
    st.dataframe(papers[["year", "title", "journal", "types", "url"]], hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("PubMed")})


def page_about(features):
    st.header("About the data and limitations")
    per_subject = features.drop_duplicates("subject_id")
    counts = per_subject.group.value_counts()
    length = per_subject.groupby("group").duration_s.mean()
    st.write(
        f"Data: 'EEG data for ADHD / Control children' (Nasrabadi et al., IEEE DataPort, "
        f"doi 10.21227/rzfh-zn36). {counts['ADHD']} ADHD and {counts['Control']} control "
        f"children, 19 channels, 128 Hz, recorded during a visual attention task.")
    st.markdown(f"""
- **Task EEG, not resting EEG.** Children counted cartoon characters, so results may not carry over to rest.
- **Recording length differs by group:** ADHD {length['ADHD']:.0f} s vs control {length['Control']:.0f} s on average, because the task ended when the child finished. Length was never used as a feature.
- **Medication:** the ADHD children had taken Ritalin for up to 6 months, which can change EEG. It may explain the lower TBR, but this was not tested.
- **No age or sex** in the files, so neither could be controlled for.
- **Small, single dataset.** Fold-to-fold AUROC varies a lot with about 24 children per test fold.
- **Units are undocumented** in the files, so absolute power values are only comparable within this dataset.
""")
    st.warning("Research demo. This is not a diagnostic tool and must not be used to assess any child.")


features, stats, models = load_data()
page = st.sidebar.radio("Page", ["TBR by group", "Model comparison", "Literature check", "About the data"])
st.sidebar.caption("Research demo, not a diagnostic tool.")
if page == "TBR by group":
    page_tbr(features, stats)
elif page == "Model comparison":
    page_models(models)
elif page == "Literature check":
    page_literature()
else:
    page_about(features)
