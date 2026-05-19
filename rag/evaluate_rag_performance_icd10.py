#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, Dict, Tuple, List
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# =========================
# Paths
# =========================
IN_RESULTS_CSV = Path("/home/alic/LLIT-RAG/ICD10/Qwen_results_icd10.csv")
OUT_PNG = Path("/home/alic/LLIT-RAG/Accuracy_exact_vs_partial_grouped_methods_Chroma_Classifier_icd10_noinvalids.png")


# =========================
# CSV reader (robust separator detection)
# =========================
def read_csv_flexible(path: Path, required_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Robust CSV reader:
    - tries common separators ; , \\t |
    - also tries sep=None sniffing
    - chooses parse that best matches required_cols (if given), otherwise most columns
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    required = set(required_cols) if required_cols is not None else None
    seps = [";", ",", "\t", "|"]

    candidates: List[Tuple[int, int, str, pd.DataFrame]] = []  # (has_required, ncols, sep, df)

    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            ncols = len(df.columns)
            has_req = 1 if (required is not None and required.issubset(set(df.columns))) else 0
            candidates.append((has_req, ncols, sep, df))
        except Exception:
            continue

    try:
        df_sniff = pd.read_csv(path, sep=None, engine="python")
        ncols = len(df_sniff.columns)
        has_req = 1 if (required is not None and required.issubset(set(df_sniff.columns))) else 0
        candidates.append((has_req, ncols, "sniff", df_sniff))
    except Exception:
        pass

    if not candidates:
        raise ValueError(f"Could not read CSV with common separators: {path}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_has_req, best_ncols, best_sep, best_df = candidates[0]

    if required is not None and best_has_req == 0:
        info = [(sep, ncols) for (has_req, ncols, sep, _df) in candidates]
        raise ValueError(
            f"{path}: could not find required cols {sorted(required)}. "
            f"Parsed candidates (sep,ncols)={info}. "
            f"Best sep was '{best_sep}' with {best_ncols} cols: {list(best_df.columns)}"
        )
    return best_df


# =========================
# Percent conversion
# =========================
def to_percent_series(s: pd.Series) -> pd.Series:
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
    "n": ["Num Questions", "Num_Questions", "N", "n", "Total", "Total Questions", "Total_Questions"],
    "invalid": ["Invalid Answer", "Invalid Answers", "Invalid", "Invalid_Answer", "Invalid_Count"],
    "exact": ["Exact Match", "Exact", "Exact_Match", "ExactMatch", "Exact_Count"],
    "partial": ["Partial Match Code", "Partial Match", "Partial", "Partial_Match", "Partial_Count"],
    "false_code": ["False Code", "False", "False_Code", "FalseCount", "False_Count"],
}

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def recompute_acc_invalid_as_false(
    df: pd.DataFrame,
    strict_total_denominator: bool = True,
    fold_invalid_into_false_code: bool = True,
    set_invalid_to_zero: bool = False,
    verbose_label: str = ""
) -> pd.DataFrame:
    out = df.copy()

    col_n       = _first_existing_col(out, COUNT_SYNONYMS["n"])
    col_invalid = _first_existing_col(out, COUNT_SYNONYMS["invalid"])
    col_exact   = _first_existing_col(out, COUNT_SYNONYMS["exact"])
    col_partial = _first_existing_col(out, COUNT_SYNONYMS["partial"])
    col_false   = _first_existing_col(out, COUNT_SYNONYMS["false_code"])

    have_counts = all([col_n, col_exact, col_partial])
    if not have_counts:
        return out

    n       = pd.to_numeric(out[col_n], errors="coerce")
    exact   = pd.to_numeric(out[col_exact], errors="coerce").fillna(0)
    partial = pd.to_numeric(out[col_partial], errors="coerce").fillna(0)

    if col_invalid is not None:
        invalid = pd.to_numeric(out[col_invalid], errors="coerce").fillna(0)
    else:
        invalid = pd.Series(0, index=out.index)

    denom = n if strict_total_denominator else (n - invalid)
    denom = denom.replace(0, np.nan)

    out["Acc_exact"]   = (exact / denom) * 100.0
    out["Acc_partial"] = ((exact + partial) / denom) * 100.0

    if fold_invalid_into_false_code and col_false is not None and col_invalid is not None:
        false_code = pd.to_numeric(out[col_false], errors="coerce").fillna(0)
        out[col_false] = false_code + invalid
        if set_invalid_to_zero:
            out[col_invalid] = 0

    if verbose_label:
        print(f"[INFO] Recomputed Acc from counts (invalid treated as false): {verbose_label}")

    return out


# =========================
# Normalize metrics (kept for compatibility)
# =========================
REQ_PARTIAL_ONLY = {"Partial Accuracy"}
REQ_EXACT_ONLY   = {"Accuracy"}
REQ_FULL         = {"Accuracy", "Partial Accuracy"}

def normalize_partial_only(df: pd.DataFrame, include_invalid: bool, verbose_label: str = "") -> pd.DataFrame:
    out = df.copy()

    if include_invalid:
        out2 = recompute_acc_invalid_as_false(
            out,
            strict_total_denominator=True,
            fold_invalid_into_false_code=True,
            set_invalid_to_zero=False,
            verbose_label=verbose_label
        )
        if "Acc_partial" in out2.columns and out2["Acc_partial"].notna().any():
            return out2

    missing = REQ_PARTIAL_ONLY - set(out.columns)
    if missing:
        raise ValueError(f"Partial CSV missing required columns: {sorted(missing)}. Found: {list(out.columns)}")

    out["Acc_partial"] = to_percent_series(out["Partial Accuracy"])
    return out

def normalize_exact_only(df: pd.DataFrame, include_invalid: bool, verbose_label: str = "") -> pd.DataFrame:
    out = df.copy()

    if include_invalid:
        out2 = recompute_acc_invalid_as_false(
            out,
            strict_total_denominator=True,
            fold_invalid_into_false_code=True,
            set_invalid_to_zero=False,
            verbose_label=verbose_label
        )
        if "Acc_exact" in out2.columns and out2["Acc_exact"].notna().any():
            return out2

    missing = REQ_EXACT_ONLY - set(out.columns)
    if missing:
        raise ValueError(f"Exact CSV missing required columns: {sorted(missing)}. Found: {list(out.columns)}")

    out["Acc_exact"] = to_percent_series(out["Accuracy"])
    return out

def normalize_full(df: pd.DataFrame, include_invalid: bool, verbose_label: str = "") -> pd.DataFrame:
    out = df.copy()

    if include_invalid:
        out2 = recompute_acc_invalid_as_false(
            out,
            strict_total_denominator=True,
            fold_invalid_into_false_code=True,
            set_invalid_to_zero=False,
            verbose_label=verbose_label
        )
        if ("Acc_exact" in out2.columns and out2["Acc_exact"].notna().any()) and \
           ("Acc_partial" in out2.columns and out2["Acc_partial"].notna().any()):
            return out2

    missing = REQ_FULL - set(out.columns)
    if missing:
        raise ValueError(f"Full metrics CSV missing required columns: {sorted(missing)}. Found: {list(out.columns)}")

    out["Acc_exact"]   = to_percent_series(out["Accuracy"])
    out["Acc_partial"] = to_percent_series(out["Partial Accuracy"])
    return out

def pick_best_row(df: pd.DataFrame, col: str) -> pd.Series:
    tmp = df.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    if tmp[col].notna().sum() == 0:
        raise ValueError(f"No numeric values in '{col}' after conversion.")
    return tmp.loc[tmp[col].idxmax()]


# =========================
# Infer / filter embedding model (kept for compatibility)
# =========================
def infer_embedding_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Source_Path" in out.columns:
        src = out["Source_Path"].astype(str)
    elif "Source_Folder" in out.columns:
        src = out["Source_Folder"].astype(str)
    else:
        out["Embedding_Model_Inferred"] = "UNKNOWN"
        return out

    m = src.str.extract(r"/Filtered/([^/]+?)(?:_ICD10|_ICDO|_shots|$)", expand=False)
    out["Embedding_Model_Inferred"] = m.fillna("UNKNOWN")
    return out

def filter_to_embedding(df: pd.DataFrame, embedding_name: str) -> pd.DataFrame:
    df = df.copy()

    for col in ["Model_Name", "Model"]:
        if col in df.columns:
            sub = df[df[col].astype(str) == embedding_name].copy()
            if not sub.empty:
                return sub
            sub = df[df[col].astype(str).str.contains(embedding_name, case=False, na=False)].copy()
            if not sub.empty:
                return sub

    if "Embedding_Model_Inferred" in df.columns:
        sub = df[df["Embedding_Model_Inferred"].astype(str) == embedding_name].copy()
        if not sub.empty:
            return sub
        sub = df[df["Embedding_Model_Inferred"].astype(str).str.contains(embedding_name, case=False, na=False)].copy()
        if not sub.empty:
            return sub

    return df.iloc[0:0].copy()


# =========================
# Loaders (NEW: from consolidated table)
# =========================
def _llm_to_key(llm: str) -> str:
    llm = str(llm).strip()
    mapping = {
        "Llama 3.1 base": "L31_base",
        "Llama 3.1 PEFT": "L31_peft",
        "Llama 3.3 base": "L33_base",
        "Llama 3.3 PEFT": "L33_peft",
        "—": "L31_base",   # baseline placeholder
        "-": "L31_base",
    }
    return mapping.get(llm, "L31_base")

def _method_to_group(method: str) -> str:
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
    df = read_csv_flexible(csv_path, required_cols={"Method", "LLM", "Accuracy", "Partial Accuracy"})
    df.columns = [c.strip() for c in df.columns]

    # Remove visual separator rows ;;;;;;;
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
        mk = _llm_to_key(row["LLM"])

        acc_e = row["Accuracy"]
        acc_p = row["Partial Accuracy"]

        if pd.isna(acc_e) and pd.isna(acc_p):
            continue

        values_exact[(group, mk)] = float(acc_e) if not pd.isna(acc_e) else np.nan
        values_partial[(group, mk)] = float(acc_p) if not pd.isna(acc_p) else np.nan
        source_paths[(group, mk)] = str(csv_path)

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
    GTDS_color = "#4ec47b"

    desired_model_order = ["L31_base", "L31_peft", "L33_base", "L33_peft"]

    BAR_W     = 0.30
    PAIR_GAP  = 0.0
    MODEL_GAP = 0.16
    GROUP_GAP = 0.55
    EXTRA_GAP_AFTER = {"Similarity search": 0.95}

    model_span = 1.6 * BAR_W + PAIR_GAP
    slot       = model_span + MODEL_GAP

    method_to_models = {m: [] for m in methods}
    for (meth, mk) in set(list(values_exact.keys()) + list(values_partial.keys())):
        if meth in method_to_models:
            method_to_models[meth].append(mk)

    for m in methods:
        method_to_models[m] = [k for k in desired_model_order if k in method_to_models[m]]

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
        group_end   = x0 + (len(models_here) - 1) * slot + model_span
        centers.append(((group_start + group_end) / 2.0, meth))

        for i, mk in enumerate(models_here):
            model_left = x0 + i * slot
            xe = model_left
            xp = model_left + BAR_W + PAIR_GAP

            exact_x.append(xe)
            partial_x.append(xp)

            exact_y.append(values_exact.get((meth, mk), np.nan))
            partial_y.append(values_partial.get((meth, mk), np.nan))

            if meth in ["Similarity search", "Classifier head"]:
                exact_c.append(similarity_color)
                partial_c.append(similarity_color)
            else:
                exact_c.append(model_colors[mk])
                partial_c.append(model_colors[mk])

        x0 = group_end + GROUP_GAP + EXTRA_GAP_AFTER.get(meth, 0.0)

    exact_x   = np.array(exact_x, dtype=float)
    partial_x = np.array(partial_x, dtype=float)
    exact_y   = np.array(exact_y, dtype=float)
    partial_y = np.array(partial_y, dtype=float)

    fig_h = 9.5
    fig_w = max(16.0, len(centers) * 2.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.30, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 125)

    ax.axhspan(81, 94, facecolor=GTDS_color, alpha=0.15, hatch="//",
               edgecolor=(0,0,0,0), linewidth=0, zorder=1)
    ax.axhspan(60, 79, facecolor=GTDS_color, alpha=0.12,
               edgecolor=(0,0,0,0), linewidth=0, zorder=1)

    bars_exact = ax.bar(
        exact_x, exact_y, width=BAR_W, align="edge",
        color=exact_c, edgecolor="black", linewidth=0.5, zorder=3
    )
    bars_part = ax.bar(
        partial_x, partial_y, width=BAR_W, align="edge",
        color=partial_c, edgecolor="black", linewidth=0.5, hatch="//", zorder=3
    )

    def label_above(container, dy=1.2):
        for r in container:
            h = r.get_height()
            if np.isnan(h):
                continue
            x = r.get_x() + r.get_width() / 2.0
            ax.text(x, h + dy, f"{h:.1f}", ha="center", va="bottom", fontsize=10, clip_on=True)

    label_above(bars_exact)
    label_above(bars_part)

    ax.set_xticks([c for c, _ in centers], [lab for _, lab in centers], fontsize=14)
    ax.tick_params(axis="x", pad=6, length=0)

    ax.set_ylabel("Score (%)", fontsize=14)
    ax.set_title(
        "\nComparison of methods (similarity search, classifier, zero-shot, three-shot, RAG) for ICD-10 coding\n",
        fontsize=17, pad=10, fontweight="bold"
    )

    metric_handles = [
        Patch(facecolor="white", edgecolor="black", label="Exact match"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Partial match"),
    ]
    model_handles = [
        Patch(facecolor=model_colors[k], edgecolor="black", label=model_labels[k])
        for k in ["L31_base", "L31_peft", "L33_base", "L33_peft"]
    ]
    header_metric = Line2D([], [], linestyle="none", label=r"$\bf{Metric}$")
    header_model  = Line2D([], [], linestyle="none", label=r"$\bf{Model}$")
    blank         = Line2D([], [], linestyle="none", label="")
    handles = [header_metric] + metric_handles + [blank] + [blank] + [header_model] + model_handles

    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.01, 1.00),
        ncol=2,
        frameon=True, framealpha=1.0, edgecolor="black",
        fontsize=14,
        handlelength=1.6,
        handletextpad=0.8,
        borderpad=0.8,
        labelspacing=0.55,
    )

    # (Optional) keep your paths box text building, but not drawn in this snippet
    fig.tight_layout(rect=[0, 0.12, 0.78, 0.95])


    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot gespeichert unter: {out_png}")

# =========================
# Main
# =========================
def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    methods = [
        "Similarity search",
        "Classifier head",
        "Zero-shot",
        "Three-shot",
        "RAG (similarity search)",
        "RAG (classifier head)",
    ]

    values_exact, values_partial, source_paths = load_values_from_results_table(IN_RESULTS_CSV)

    build_plot_grouped(methods, values_exact, values_partial, OUT_PNG, source_paths)


if __name__ == "__main__":
    main()
