#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =========================
# Paths
# =========================
IN_RESULTS_CSV = Path("/home/alic/LLIT-RAG/ICDO/Qwen_results_icdo.csv")
OUT_PNG = Path(
    "/home/alic/LLIT-RAG/"
    "Accuracy_exact_vs_partial_grouped_methods_Chroma_Classifier_icdo_2.png"
)

FORCED_EMBEDDING = "Qwen3-Embedding-8B"  # kept for compatibility (not used)


# =========================
# CSV reader (robust separator detection)
# =========================
def read_csv_flexible(
    path: Path,
    required_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Read a CSV file with automatic separator testing.

    Parameters:
        path (Path): Path to the CSV file.
        required_cols (Optional[Iterable[str]]): Columns that should exist.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    required = set(required_cols) if required_cols is not None else None

    seps = [";", ",", "\t", "|"]
    candidates: List[Tuple[int, int, str, pd.DataFrame]] = []

    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            ncols = len(df.columns)
            has_req = (
                1
                if required is not None
                and required.issubset(set(df.columns))
                else 0
            )
            candidates.append((has_req, ncols, sep, df))
        except Exception:
            continue

    try:
        df_sniff = pd.read_csv(path, sep=None, engine="python")
        ncols = len(df_sniff.columns)
        has_req = (
            1
            if required is not None
            and required.issubset(set(df_sniff.columns))
            else 0
        )
        candidates.append((has_req, ncols, "sniff", df_sniff))
    except Exception:
        pass

    if not candidates:
        raise ValueError(f"Could not read CSV with common separators: {path}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_has_req, best_ncols, best_sep, best_df = candidates[0]

    if required is not None and best_has_req == 0:
        info = [(sep, ncols) for (_, ncols, sep, _) in candidates]
        raise ValueError(
            f"{path}: could not find required cols {sorted(required)}. "
            f"Parsed candidates (sep,ncols)={info}. "
            f"Best sep was '{best_sep}' with {best_ncols} cols: "
            f"{list(best_df.columns)}"
        )

    return best_df


# =========================
# Robust matching helpers (kept for compatibility)
# =========================
def normalize_for_match(s: str) -> str:
    """
    Normalize a string for matching.

    Parameters:
        s (str): Input text.

    Returns:
        str: Normalized text.
    """
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


# =========================
# Percent conversion
# =========================
def to_percent_series(s: pd.Series) -> pd.Series:
    """
    Convert a series to percent values.

    Parameters:
        s (pd.Series): Input series.

    Returns:
        pd.Series: Series with percent values.
    """

    def _conv(v):
        if pd.isna(v):
            return np.nan

        if isinstance(v, str):
            vv = v.strip()
            if vv.endswith("%"):
                try:
                    return float(vv[:-1])
                except ValueError:
                    return np.nan
            try:
                fv = float(vv)
            except ValueError:
                return np.nan
        else:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return np.nan

        return fv * 100.0 if fv <= 1.0 else fv

    return s.map(_conv)


# =========================
# Recompute Acc from counts (kept for compatibility, not used for table input)
# =========================
COUNT_SYNONYMS = {
    "n": [
        "Num Questions",
        "Num_Questions",
        "N",
        "n",
        "Total",
        "Total Questions",
        "Total_Questions",
    ],
    "invalid": [
        "Invalid Answer",
        "Invalid Answers",
        "Invalid",
        "Invalid_Answer",
        "Invalid_Count",
    ],
    "exact": [
        "Exact Match",
        "Exact",
        "Exact_Match",
        "ExactMatch",
        "Exact_Count",
    ],
    "partial": [
        "Partial Match Code",
        "Partial Match",
        "Partial",
        "Partial_Match",
        "Partial_Count",
    ],
    "false_code": [
        "False Code",
        "False",
        "False_Code",
        "FalseCount",
        "False_Count",
    ],
}


def _first_existing_col(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Return the first column name that exists in the dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        candidates (List[str]): Possible column names.

    Returns:
        Optional[str]: First matching column name or None.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def recompute_acc_invalid_as_false(
    df: pd.DataFrame,
    strict_total_denominator: bool = True,
    fold_invalid_into_false_code: bool = True,
    set_invalid_to_zero: bool = False,
    verbose_label: str = "",
) -> pd.DataFrame:
    """
    Recompute exact and partial accuracy from counts.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        strict_total_denominator (bool): If True, use total count as denominator.
        fold_invalid_into_false_code (bool): If True, add invalid to false code.
        set_invalid_to_zero (bool): If True, reset invalid count to zero.
        verbose_label (str): Text for the info print.

    Returns:
        pd.DataFrame: Dataframe with recomputed accuracy columns.
    """
    out = df.copy()

    col_n = _first_existing_col(out, COUNT_SYNONYMS["n"])
    col_invalid = _first_existing_col(out, COUNT_SYNONYMS["invalid"])
    col_exact = _first_existing_col(out, COUNT_SYNONYMS["exact"])
    col_partial = _first_existing_col(out, COUNT_SYNONYMS["partial"])
    col_false = _first_existing_col(out, COUNT_SYNONYMS["false_code"])

    have_counts = all([col_n, col_exact, col_partial])
    if not have_counts:
        return out

    n = pd.to_numeric(out[col_n], errors="coerce")
    exact = pd.to_numeric(out[col_exact], errors="coerce").fillna(0)
    partial = pd.to_numeric(out[col_partial], errors="coerce").fillna(0)

    if col_invalid is not None:
        invalid = pd.to_numeric(out[col_invalid], errors="coerce").fillna(0)
    else:
        invalid = pd.Series(0, index=out.index)

    denom = n if strict_total_denominator else (n - invalid)
    denom = denom.replace(0, np.nan)

    out["Acc_exact"] = (exact / denom) * 100.0
    out["Acc_partial"] = ((exact + partial) / denom) * 100.0

    if fold_invalid_into_false_code and col_false is not None and col_invalid is not None:
        false_code = pd.to_numeric(out[col_false], errors="coerce").fillna(0)
        out[col_false] = false_code + invalid
        if set_invalid_to_zero:
            out[col_invalid] = 0

    if verbose_label:
        print(
            "[INFO] Recomputed accuracy from counts "
            f"(invalid treated as false): {verbose_label}"
        )

    return out


# =========================
# Normalize metrics (kept for compatibility)
# =========================
def normalize_partial_only(
    df: pd.DataFrame,
    include_invalid: bool,
    verbose_label: str = "",
) -> pd.DataFrame:
    """
    Normalize only the partial accuracy.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        include_invalid (bool): If True, recompute using invalid counts.
        verbose_label (str): Text for the info print.

    Returns:
        pd.DataFrame: Dataframe with normalized partial accuracy.
    """
    out = df.copy()

    if include_invalid:
        out2 = recompute_acc_invalid_as_false(
            out,
            strict_total_denominator=True,
            fold_invalid_into_false_code=True,
            set_invalid_to_zero=False,
            verbose_label=verbose_label,
        )
        if "Acc_partial" in out2.columns and out2["Acc_partial"].notna().any():
            return out2

    if "Partial Accuracy" not in out.columns:
        raise ValueError(
            "Partial CSV missing 'Partial Accuracy' "
            f"(and no counts recompute). Columns: {list(out.columns)}"
        )

    out["Acc_partial"] = to_percent_series(out["Partial Accuracy"])
    return out


def normalize_exact_only(
    df: pd.DataFrame,
    include_invalid: bool,
    verbose_label: str = "",
) -> pd.DataFrame:
    """
    Normalize only the exact accuracy.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        include_invalid (bool): If True, recompute using invalid counts.
        verbose_label (str): Text for the info print.

    Returns:
        pd.DataFrame: Dataframe with normalized exact accuracy.
    """
    out = df.copy()

    if include_invalid:
        out2 = recompute_acc_invalid_as_false(
            out,
            strict_total_denominator=True,
            fold_invalid_into_false_code=True,
            set_invalid_to_zero=False,
            verbose_label=verbose_label,
        )
        if "Acc_exact" in out2.columns and out2["Acc_exact"].notna().any():
            return out2

    if "Accuracy" not in out.columns:
        raise ValueError(
            "Exact CSV missing 'Accuracy' "
            f"(and no counts recompute). Columns: {list(out.columns)}"
        )

    out["Acc_exact"] = to_percent_series(out["Accuracy"])
    return out


def normalize_full(
    df: pd.DataFrame,
    include_invalid: bool,
    verbose_label: str = "",
) -> pd.DataFrame:
    """
    Normalize exact and partial accuracy.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        include_invalid (bool): If True, recompute using invalid counts.
        verbose_label (str): Text for the info print.

    Returns:
        pd.DataFrame: Dataframe with normalized exact and partial accuracy.
    """
    out = df.copy()

    if include_invalid:
        out2 = recompute_acc_invalid_as_false(
            out,
            strict_total_denominator=True,
            fold_invalid_into_false_code=True,
            set_invalid_to_zero=False,
            verbose_label=verbose_label,
        )
        if (
            "Acc_exact" in out2.columns
            and out2["Acc_exact"].notna().any()
            and "Acc_partial" in out2.columns
            and out2["Acc_partial"].notna().any()
        ):
            return out2

    if "Accuracy" not in out.columns or "Partial Accuracy" not in out.columns:
        raise ValueError(
            "Full metrics CSV missing Accuracy/Partial Accuracy "
            f"(and no counts recompute). Columns: {list(out.columns)}"
        )

    out["Acc_exact"] = to_percent_series(out["Accuracy"])
    out["Acc_partial"] = to_percent_series(out["Partial Accuracy"])
    return out


def pick_best_row(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Return the row with the highest value in one column.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        col (str): Column used for selection.

    Returns:
        pd.Series: Best row.
    """
    tmp = df.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    if tmp[col].notna().sum() == 0:
        raise ValueError(f"No numeric values in '{col}' after conversion.")

    return tmp.loc[tmp[col].idxmax()]


# =========================
# Load values from consolidated table (NEW)
# =========================
def _llm_to_key(llm: str) -> str:
    """
    Map an LLM label to a short key.

    Parameters:
        llm (str): LLM label.

    Returns:
        str: Short LLM key.
    """
    llm = str(llm).strip()
    mapping = {
        "Llama 3.1 base": "L31_base",
        "Llama 3.1 PEFT": "L31_peft",
        "Llama 3.3 base": "L33_base",
        "Llama 3.3 PEFT": "L33_peft",
        "—": "L31_base",
        "-": "L31_base",
    }
    return mapping.get(llm, "L31_base")


def _method_to_group(method: str) -> str:
    """
    Map a method label to the group label used in the plot.

    Parameters:
        method (str): Method label.

    Returns:
        str: Group label.
    """
    method = str(method).strip()
    mapping = {
        "Retrieval-baseline: similarity search": "Similarity search",
        "Retrieval-baseline: classifier head": "Classifier head",
        "Zero-shot-prompting": "Zero-shot",
        "Three-shot-prompting": "Three-shot",
        "RAG-prompting (similarity search)": "RAG (similarity search)",
        "RAG-prompting (classifier head)": "RAG (classifier head)",
    }
    return mapping.get(method, method)


def load_values_from_results_table(csv_path: Path):
    """
    Load exact and partial accuracy values from the results table.

    Parameters:
        csv_path (Path): Path to the results CSV file.

    Returns:
        tuple: Exact values, partial values, and source paths.
    """
    df = read_csv_flexible(
        csv_path,
        required_cols={"Method", "LLM", "Accuracy", "Partial Accuracy"},
    )
    df.columns = [c.strip() for c in df.columns]

    df = df[df["Method"].notna()].copy()
    df["Method"] = df["Method"].astype(str).str.strip()
    df = df[df["Method"] != ""].copy()

    df["LLM"] = df["LLM"].astype(str).str.strip()
    df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    df["Partial Accuracy"] = pd.to_numeric(df["Partial Accuracy"], errors="coerce")

    values_exact: Dict[Tuple[str, str], float] = {}
    values_partial: Dict[Tuple[str, str], float] = {}
    source_paths: Dict[Tuple[str, str], str] = {}

    for _, row in df.iterrows():
        group = _method_to_group(row["Method"])
        model_key = _llm_to_key(row["LLM"])

        acc_exact = row["Accuracy"]
        acc_partial = row["Partial Accuracy"]

        if pd.isna(acc_exact) and pd.isna(acc_partial):
            continue

        values_exact[(group, model_key)] = (
            float(acc_exact) if not pd.isna(acc_exact) else np.nan
        )
        values_partial[(group, model_key)] = (
            float(acc_partial) if not pd.isna(acc_partial) else np.nan
        )
        source_paths[(group, model_key)] = str(csv_path)

    return values_exact, values_partial, source_paths


# =========================
# Plot
# =========================
def build_plot_grouped(
    methods: List[str],
    values_exact: Dict[Tuple[str, str], float],
    values_partial: Dict[Tuple[str, str], float],
    out_png: Path,
    source_paths: Dict[Tuple[str, str], str],
) -> None:
    """
    Build and save the grouped bar plot.

    Parameters:
        methods (List[str]): Ordered list of method groups.
        values_exact (Dict[Tuple[str, str], float]): Exact accuracy values.
        values_partial (Dict[Tuple[str, str], float]): Partial accuracy values.
        out_png (Path): Output PNG path.
        source_paths (Dict[Tuple[str, str], str]): Source path per value.

    Returns:
        None
    """
    model_colors = {
        "L31_base": "#0072B2",
        "L31_peft": "#56B4E9",
        "L33_base": "#D55E00",
        "L33_peft": "#E69F00",
    }
    model_labels = {
        "L31_base": "Llama 3.1 8B (base)",
        "L31_peft": "Llama 3.1 8B (PEFT)",
        "L33_base": "Llama 3.3 70B (base)",
        "L33_peft": "Llama 3.3 70B (PEFT)",
    }
    similarity_color = "#7A7A7A"
    gtds_color = "#4ec47b"
    desired_model_order = ["L31_base", "L31_peft", "L33_base", "L33_peft"]

    bar_w = 0.30
    pair_gap = 0.0
    model_gap = 0.16
    group_gap = 0.55
    extra_gap_after = {"Similarity search": 0.95}

    model_span = 1.6 * bar_w + pair_gap
    slot = model_span + model_gap

    method_to_models = {m: [] for m in methods}
    for meth, mk in set(list(values_exact.keys()) + list(values_partial.keys())):
        if meth in method_to_models:
            method_to_models[meth].append(mk)

    for method in methods:
        method_to_models[method] = [
            key
            for key in desired_model_order
            if key in method_to_models[method]
        ]

    centers = []
    exact_x, partial_x = [], []
    exact_y, partial_y = [], []
    exact_c, partial_c = [], []

    x0 = 0.0
    for meth in methods:
        models_here = method_to_models[meth]
        if not models_here:
            continue

        group_start = x0
        group_end = x0 + (len(models_here) - 1) * slot + model_span
        centers.append(((group_start + group_end) / 2.0, meth))

        for i, mk in enumerate(models_here):
            model_left = x0 + i * slot
            x_exact = model_left
            x_partial = model_left + bar_w + pair_gap

            exact_x.append(x_exact)
            partial_x.append(x_partial)

            exact_y.append(values_exact.get((meth, mk), np.nan))
            partial_y.append(values_partial.get((meth, mk), np.nan))

            if meth in ["Similarity search", "Classifier head"]:
                exact_c.append(similarity_color)
                partial_c.append(similarity_color)
            else:
                exact_c.append(model_colors[mk])
                partial_c.append(model_colors[mk])

        x0 = group_end + group_gap + extra_gap_after.get(meth, 0.0)

    exact_x = np.array(exact_x, dtype=float)
    partial_x = np.array(partial_x, dtype=float)
    exact_y = np.array(exact_y, dtype=float)
    partial_y = np.array(partial_y, dtype=float)

    fig_h = 9.5
    fig_w = max(16.0, len(centers) * 2.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.30, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 125)

    ax.axhspan(
        84,
        96,
        facecolor=gtds_color,
        alpha=0.15,
        hatch="//",
        edgecolor=(0, 0, 0, 0),
        linewidth=0,
        zorder=1,
    )
    ax.axhspan(
        55,
        75,
        facecolor=gtds_color,
        alpha=0.12,
        edgecolor=(0, 0, 0, 0),
        linewidth=0,
        zorder=1,
    )

    bars_exact = ax.bar(
        exact_x,
        exact_y,
        width=bar_w,
        align="edge",
        color=exact_c,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    bars_part = ax.bar(
        partial_x,
        partial_y,
        width=bar_w,
        align="edge",
        color=partial_c,
        edgecolor="black",
        linewidth=0.5,
        hatch="//",
        zorder=3,
    )

    def label_above(container, dy=1.2):
        for rect in container:
            h = rect.get_height()
            if np.isnan(h):
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            ax.text(
                x,
                h + dy,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                clip_on=True,
            )

    label_above(bars_exact)
    label_above(bars_part)

    ax.set_xticks([c for c, _ in centers], [lab for _, lab in centers], fontsize=14)
    ax.tick_params(axis="x", pad=6, length=0)

    ax.set_ylabel("Score (%)", fontsize=14)
    ax.set_title(
        "\n\nComparison of methods (similarity search, classifier, zero-shot, "
        "three-shot, RAG) for ICD-O coding\n",
        fontsize=17,
        fontweight="bold",
        pad=10,
    )

    metric_handles = [
        Patch(facecolor="white", edgecolor="black", label="Exact match"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="//",
            label="Partial match",
        ),
    ]
    model_handles = [
        Patch(facecolor=model_colors[k], edgecolor="black", label=model_labels[k])
        for k in ["L31_base", "L31_peft", "L33_base", "L33_peft"]
    ]
    header_metric = Line2D([], [], linestyle="none", label=r"$\bf{Metric}$")
    header_model = Line2D([], [], linestyle="none", label=r"$\bf{Model}$")
    blank = Line2D([], [], linestyle="none", label="")
    handles = (
        [header_metric]
        + metric_handles
        + [blank]
        + [blank]
        + [header_model]
        + model_handles
    )

    fig.tight_layout(rect=[0, 0.12, 0.78, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved to: {out_png}")


# =========================
# Main
# =========================
def main() -> None:
    """
    Run the full plot generation.

    Parameters:
        None

    Returns:
        None
    """
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    methods = [
        "Similarity search",
        "Classifier head",
        "Zero-shot",
        "Three-shot",
        "RAG (similarity search)",
        "RAG (classifier head)",
    ]

    values_exact, values_partial, source_paths = load_values_from_results_table(
        IN_RESULTS_CSV
    )

    build_plot_grouped(
        methods,
        values_exact,
        values_partial,
        OUT_PNG,
        source_paths,
    )


if __name__ == "__main__":
    main()