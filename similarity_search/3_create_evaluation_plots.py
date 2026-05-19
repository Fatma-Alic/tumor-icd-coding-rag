# Plot weighted F1 scores and accuracy values
# for exact and partial predictions of each embedding model.

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


exact_csv = Path("/home/alic/PredictICD10/FINAL/evaluation/exakt_match_matrics.csv")
partial_csv = Path("/home/alic/PredictICD10/FINAL/evaluation/partial_match_metrics.csv")
out_dir = Path("/home/alic/PredictICD10/FINAL/evaluation/per_model_plots")
out_png = Path(
    "/home/alic/PredictICD10/FINAL/evaluation/"
    "Accuracy_F1score_exact_vs_partial_one_plot_icd10.png"
)

# Column names
f1_col = "WeightedF1_Top1 (%)"  # Change to "MacroF1_Top1 (%)" if needed
acc_col = "Top1 (%)"  # Accuracy in percent

# Load CSV files
df_exact = pd.read_csv(exact_csv)
df_partial = pd.read_csv(partial_csv)

# Check required columns
req_cols = {"Model", f1_col, acc_col}
miss_e = req_cols - set(df_exact.columns)
miss_p = req_cols - set(df_partial.columns)

if miss_e:
    raise ValueError(f"Exact CSV is missing columns: {miss_e}")
if miss_p:
    raise ValueError(f"Partial CSV is missing columns: {miss_p}")

# Merge values per model
df = pd.merge(
    df_exact[["Model", f1_col, acc_col]].rename(
        columns={f1_col: "F1_exact", acc_col: "Acc_exact"}
    ),
    df_partial[["Model", f1_col, acc_col]].rename(
        columns={f1_col: "F1_partial", acc_col: "Acc_partial"}
    ),
    on="Model",
    how="inner",
).copy()

# Sort values by descending exact accuracy
df = df.sort_values("Acc_exact", ascending=False).reset_index(drop=True)


def safe_name(name: str) -> str:
    """
    Create a file-safe version of a model name.

    Parameters:
        name (str): The original model name.

    Returns:
        str: A cleaned file name with only safe characters.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150]



# Plot per model
for _, row in df.iterrows():
    model = str(row["Model"])
    values = [
        float(row["F1_exact"]),
        float(row["F1_partial"]),
        float(row["Acc_exact"]),
        float(row["Acc_partial"]),
    ]
    labels = [
        "F1 (Exact)",
        "F1 (Partial)",
        "Accuracy (Exact)",
        "Accuracy (Partial)",
    ]

    x = np.arange(len(labels), dtype=float)
    bar_w = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, width=bar_w)

    # Labels and title
    ax.set_title(f"{model}\nWeighted F1 & Accuracy - Exact vs. Partial (Top-1)")
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")

    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add value labels above bars
    for xi, value in zip(x, values):
        ax.text(
            xi,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=True,
        )

    fig.tight_layout()

    out_path = out_dir / f"metrics_{safe_name(model)}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved file: {out_path}")

# Combined plot for accuracy and F1 score
labels_col = "Model"

# Order by descending exact accuracy
models = df[labels_col].tolist()
acc_exact = pd.to_numeric(df["Acc_exact"], errors="coerce").to_numpy()
acc_partial = pd.to_numeric(df["Acc_partial"], errors="coerce").to_numpy()
f1_exact = pd.to_numeric(df["F1_exact"], errors="coerce").to_numpy()
f1_partial = pd.to_numeric(df["F1_partial"], errors="coerce").to_numpy()
n_models = len(models)

# Colors: F1 vs accuracy
c_f1 = "#5572a8"
c_acc = "#4ec47b"

# Plot
x = np.arange(n_models, dtype=float)
fig_h = 6
fig, ax = plt.subplots(figsize=(max(12, n_models * 0.9), fig_h))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.set_axisbelow(True)

max_val = np.nanmax(
    [f1_exact.max(), f1_partial.max(), acc_exact.max(), acc_partial.max()]
)
ax.set_ylim(0, min(110, max(100, max_val + 20)))
ax.margins(y=0.02)

w = 0.21  # Wider bars
x = np.arange(n_models, dtype=float)  # X coordinates for each model group

# Order per model (left -> right):
# Accuracy exact, Accuracy partial, F1 exact, F1 partial
bars_acc_exact = ax.bar(
    x - 1.5 * w,
    acc_exact,
    w,
    label="Accuracy (exact match)",
    color=c_acc,
    edgecolor="black",
    linewidth=0.4,
    zorder=3,
)
bars_acc_partial = ax.bar(
    x - 0.5 * w,
    acc_partial,
    w,
    label="Accuracy (partial match)",
    color=c_acc,
    edgecolor="black",
    linewidth=0.4,
    hatch="//",
    zorder=3,
)
bars_f1_exact = ax.bar(
    x + 0.5 * w,
    f1_exact,
    w,
    label="F1 (exact match)",
    color=c_f1,
    edgecolor="black",
    linewidth=0.4,
    zorder=3,
)
bars_f1_partial = ax.bar(
    x + 1.5 * w,
    f1_partial,
    w,
    label="F1 (partial match)",
    color=c_f1,
    edgecolor="black",
    linewidth=0.4,
    hatch="//",
    zorder=3,
)

# X-axis: model names
ax.set_xticks(x, models, rotation=45, ha="center", fontsize=8)
ax.tick_params(axis="x", which="both", pad=6, length=0)

# Y-axis and title
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_title(
    "Performance of classifier head in ICD-10 coding",
    fontsize=12,
    fontweight="bold",
)

# Add values above bars
for bar_container in (
    bars_acc_exact,
    bars_acc_partial,
    bars_f1_exact,
    bars_f1_partial,
):
    ax.bar_label(bar_container, fmt="%.1f", padding=2, fontsize=7)

# Legend
ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=1.0,
    edgecolor="black",
    fontsize=8,
    title="Metric",
    title_fontsize=9,
)

fig.tight_layout()
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved file: {out_png}")