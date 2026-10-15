"""Build splits + SQL views, then test subject-level TBR differences per channel."""
import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

from src.load import CHANNELS

DB_PATH = "data/eeg.duckdb"


def run_sql_file(con, path):
    with open(path) as f:
        con.execute(f.read())


def channel_tests(subject_channel):
    """One Mann-Whitney U test per channel on subject-level mean TBR.
    effect_r is the rank-biserial correlation: > 0 means ADHD higher, range -1..1."""
    rows = []
    for channel in CHANNELS:
        one = subject_channel[subject_channel.channel == channel]
        adhd = one[one.group == "ADHD"].tbr
        control = one[one.group == "Control"].tbr
        u = mannwhitneyu(adhd, control, alternative="two-sided")
        rows.append({
            "channel": channel,
            "n_adhd": len(adhd),
            "n_control": len(control),
            "median_adhd": adhd.median(),
            "median_control": control.median(),
            "u": u.statistic,
            "p": u.pvalue,
            "effect_r": 2 * u.statistic / (len(adhd) * len(control)) - 1,
        })
    result = pd.DataFrame(rows)
    # Benjamini-Hochberg across the 19 channel tests
    result["p_fdr"] = false_discovery_control(result.p, method="bh")
    result["significant"] = result.p_fdr < 0.05
    return result


def plot_effect_sizes(result, path):
    colors = ["tab:red" if s else "tab:grey" for s in result.significant]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(result.channel, result.effect_r, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Rank-biserial r (>0: ADHD higher)")
    ax.set_title("Subject-level TBR, ADHD vs Control (red = FDR < 0.05)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_subject_boxes(subject_channel, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, channel in enumerate(CHANNELS):
        one = subject_channel[subject_channel.channel == channel]
        ax.boxplot(one[one.group == "ADHD"].tbr, positions=[i * 3], widths=0.8,
                   boxprops={"color": "tab:blue"}, showfliers=False)
        ax.boxplot(one[one.group == "Control"].tbr, positions=[i * 3 + 1], widths=0.8,
                   boxprops={"color": "tab:orange"}, showfliers=False)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(CHANNELS))])
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("Subject mean TBR")
    ax.set_title("TBR per subject by channel (blue = ADHD, orange = Control)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH)
    run_sql_file(con, "sql/splits.sql")
    run_sql_file(con, "sql/summaries.sql")

    folds = con.sql("""SELECT sp.fold, s."group", count(*) AS n
                       FROM splits sp JOIN subjects s USING (subject_id)
                       GROUP BY 1, 2 ORDER BY 1, 2""").df()
    print(folds.pivot(index="fold", columns="group", values="n"))

    subject_channel = con.sql("SELECT * FROM subject_channel").df()
    result = channel_tests(subject_channel)
    con.register("result_in", result)
    con.execute("CREATE OR REPLACE TABLE group_stats AS SELECT * FROM result_in")

    print(result[["channel", "median_adhd", "median_control", "effect_r", "p", "p_fdr"]]
          .round(3).to_string(index=False))
    print("FDR-significant channels:", list(result[result.significant].channel))
    plot_effect_sizes(result, "figs/tbr_effect_size.png")
    plot_subject_boxes(subject_channel, "figs/tbr_subject_box.png")
    con.close()
