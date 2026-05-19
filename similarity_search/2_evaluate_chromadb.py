"""Evaluate ICD-10 semantic-similarity result files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/alic/RAG")
RESULTS_DIR = PROJECT_ROOT / "similarity_search" / "results"

CSV_PATH = RESULTS_DIR / "filtered_embedding_models_icd10"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results_icd10"

METRICS_SUMMARY_PATH = OUTPUT_PATH / "ChromaDB_metrics_summary_icd10.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

CSV_SEP = ";"

COL_GT = "ICD-10-Code"
COL_PRED = "suggestedMetadata1"

ICD_CODE_PATTERN = re.compile(
    r"^[A-Z]\d{1,2}(?:\.\d{1,2})?$",
    re.IGNORECASE,
)


def extract_icd_code(icd_code: str) -> str | None:
    """
    Extract the ICD-10 code without the description part.

    Args:
        icd_code (str): Full ICD-10 code with optional description.

    Returns:
        str | None: Extracted ICD-10 code, for example 'C50.1',
        or None if the input is not a string.
    """
    if isinstance(icd_code, str):
        return icd_code.split(" ")[0]

    return None


def enlarge_icd_code(icd_code: str) -> str | None:
    """
    Reduce an ICD-10 code to its broader base code.

    Example:
        'C50.1' becomes 'C50'.

    Args:
        icd_code (str): Full ICD-10 code.

    Returns:
        str | None: Broader ICD-10 code, or None if the input is not a string.
    """
    if isinstance(icd_code, str):
        return icd_code.split(".")[0]

    return None


def is_code_valid(code: Optional[str]) -> bool:
    """
    Check whether a code has a valid ICD-style format.

    Args:
        code (str | None): Input code.

    Returns:
        bool: True if the code is valid, otherwise False.
    """
    if code is None:
        return False

    normalized_code = str(code).strip()

    if not normalized_code or normalized_code.lower() == "nan":
        return False

    return bool(ICD_CODE_PATTERN.match(normalized_code))


def norm_code(code: str) -> str:
    """
    Normalize a code by trimming whitespace and converting it to uppercase.

    Args:
        code (str): Input code.

    Returns:
        str: Normalized code.
    """
    return str(code).strip().upper()


def get_base_code(code: str) -> str:
    """
    Return the base code before the decimal point.

    Example:
        'C50.1' becomes 'C50'.

    Args:
        code (str): Input code.

    Returns:
        str: Base code.
    """
    return norm_code(code).split(".")[0]


def precision_at_k(row: pd.Series, k: int) -> int:
    """
    Calculate Precision@k for one DataFrame row.

    Args:
        row (pd.Series): Row with ground-truth ICD-10 code and predictions.
        k (int): Number of top predictions to check.

    Returns:
        int: 1 if the ground-truth code appears in the top-k predictions,
        otherwise 0.
    """
    ground_truth_code = extract_icd_code(row["ICD-10-Code"])
    predictions = [row.get(f"suggestedMetadata{i + 1}") for i in range(k)]

    return int(ground_truth_code in predictions)


def reciprocal_rank(row: pd.Series) -> float:
    """
    Calculate reciprocal rank for one DataFrame row.

    Args:
        row (pd.Series): Row with ground-truth ICD-10 code and predictions.

    Returns:
        float: Reciprocal rank value.
    """
    ground_truth_code = extract_icd_code(row["ICD-10-Code"])

    for i in range(5):
        if row.get(f"suggestedMetadata{i + 1}") == ground_truth_code:
            return 1.0 / (i + 1)

    return 0.0


def compute_f1_per_class(
    gt_list: List[str],
    pr_list: List[Optional[str]],
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Compute per-class F1 scores, macro F1, and weighted F1.

    Missing or invalid predictions are counted as false negatives for the
    corresponding ground-truth class.

    Args:
        gt_list (list[str]): Ground-truth codes.
        pr_list (list[str | None]): Predicted codes.

    Returns:
        tuple[dict[str, dict[str, float]], float, float]:
        Per-class metrics, macro F1, and weighted F1.
    """
    labels = sorted(set(gt_list))
    support = defaultdict(int)

    for ground_truth in gt_list:
        support[ground_truth] += 1

    per_class: Dict[str, Dict[str, float]] = {}

    for label in labels:
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for ground_truth, prediction in zip(gt_list, pr_list):
            if ground_truth == label and prediction == label:
                true_positive += 1
            elif ground_truth != label and prediction == label:
                false_positive += 1
            elif ground_truth == label and prediction != label:
                false_negative += 1
            else:
                true_negative += 1

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[label] = {
            "support": float(support[label]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
            "tp": float(true_positive),
            "fp": float(false_positive),
            "tn": float(true_negative),
            "fn": float(false_negative),
        }

    macro_f1 = (
        sum(metrics["f1"] for metrics in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )

    total_support = sum(support.values())
    weighted_f1 = (
        sum(per_class[label]["f1"] * support[label] for label in labels)
        / total_support
        if total_support > 0
        else 0.0
    )

    return per_class, float(macro_f1), float(weighted_f1)


def compare_icd_with_metadata_exact_match(
    df: pd.DataFrame,
) -> Tuple[Dict[str, int], int, int, Dict[str, float]]:
    """
    Calculate exact ICD-10 code matches.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted ICD-10 codes.

    Returns:
        tuple[dict[str, int], int, int, dict[str, float]]:
        Comparison results per prediction column, total row count,
        number of rows with at least one match, and precision metrics.
    """
    df["ICD-10-Pure-Code"] = df["ICD-10-Code"].apply(extract_icd_code)

    metadata_columns = [f"suggestedMetadata{i}" for i in range(1, 6)]
    df_cleaned = df.dropna(
        subset=["ICD-10-Pure-Code"] + metadata_columns
    ).copy()

    df_cleaned["any_match"] = df_cleaned.apply(
        lambda row: any(
            row["ICD-10-Pure-Code"] == row[column]
            for column in metadata_columns
        ),
        axis=1,
    )

    total_any_match = df_cleaned["any_match"].sum()

    comparison_results = {
        column: (
            df_cleaned["ICD-10-Pure-Code"] == df_cleaned[column]
        ).sum()
        for column in metadata_columns
    }

    total_rows = len(df_cleaned)

    for k in [1, 3, 5]:
        df_cleaned[f"precision@{k}"] = df_cleaned.apply(
            lambda row: precision_at_k(row, k),
            axis=1,
        )

    df_cleaned["mrr"] = df_cleaned.apply(reciprocal_rank, axis=1)

    precision_metrics = {
        f"Precision@{k}": df_cleaned[f"precision@{k}"].mean() * 100
        for k in [1, 3, 5]
    }
    precision_metrics["MRR"] = df_cleaned["mrr"].mean()

    return comparison_results, total_rows, total_any_match, precision_metrics


def compare_icd_with_metadata(
    df: pd.DataFrame,
) -> Tuple[Dict[str, int], int, int, Dict[str, float]]:
    """
    Calculate matches based on broader ICD-10 base codes.

    Example:
        'C50.1' and 'C50.9' are both reduced to 'C50' before comparison.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted ICD-10 codes.

    Returns:
        tuple[dict[str, int], int, int, dict[str, float]]:
        Comparison results per prediction column, total row count,
        number of rows with at least one match, and precision metrics.
    """
    df["ICD-10-Pure-Code"] = df["ICD-10-Code"].apply(extract_icd_code)
    df["ICD-10-core-Code"] = df["ICD-10-Pure-Code"].apply(enlarge_icd_code)

    metadata_columns = [f"suggestedMetadata{i}" for i in range(1, 6)]

    for column in metadata_columns:
        df[f"{column}_core_code"] = df[column].apply(enlarge_icd_code)

    core_columns = [f"{column}_core_code" for column in metadata_columns]
    df_cleaned = df.dropna(
        subset=["ICD-10-core-Code"] + core_columns
    ).copy()

    df_cleaned["any_match"] = df_cleaned.apply(
        lambda row: any(
            row["ICD-10-core-Code"] == row[f"{column}_core_code"]
            for column in metadata_columns
        ),
        axis=1,
    )

    total_any_match = df_cleaned["any_match"].sum()

    comparison_results = {
        column: (
            df_cleaned["ICD-10-core-Code"]
            == df_cleaned[f"{column}_core_code"]
        ).sum()
        for column in metadata_columns
    }

    total_rows = len(df_cleaned)

    for k in [1, 3, 5]:
        df_cleaned[f"precision@{k}"] = df_cleaned.apply(
            lambda row: precision_at_k(row, k),
            axis=1,
        )

    df_cleaned["mrr"] = df_cleaned.apply(reciprocal_rank, axis=1)

    precision_metrics = {
        f"Precision@{k}": df_cleaned[f"precision@{k}"].mean() * 100
        for k in [1, 3, 5]
    }
    precision_metrics["MRR"] = df_cleaned["mrr"].mean()

    return comparison_results, total_rows, total_any_match, precision_metrics


def compute_accuracy_f1_metrics(csv_file: str | Path) -> Dict[str, object]:
    """
    Compute exact and partial accuracy and F1 metrics for one result file.

    This uses the first suggested metadata column as the top-1 prediction.

    Args:
        csv_file (str | Path): Path to the result CSV file.

    Returns:
        dict[str, object]: Accuracy and F1 metrics.
    """
    csv_file = Path(csv_file)

    df = pd.read_csv(
        csv_file,
        sep=CSV_SEP,
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = [
        column for column in (COL_GT, COL_PRED) if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"[{csv_file}] Missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    ground_truth_codes: List[str] = []
    predicted_codes: List[Optional[str]] = []
    ground_truth_base_codes: List[str] = []
    predicted_base_codes: List[Optional[str]] = []

    for _, row in df.iterrows():
        ground_truth_raw = row.get(COL_GT, "")
        prediction_raw = row.get(COL_PRED, "")

        if not is_code_valid(ground_truth_raw):
            continue

        ground_truth = norm_code(ground_truth_raw)
        ground_truth_base = get_base_code(ground_truth)

        if is_code_valid(prediction_raw):
            prediction = norm_code(prediction_raw)
            prediction_base = get_base_code(prediction)
        else:
            prediction = None
            prediction_base = None

        ground_truth_codes.append(ground_truth)
        predicted_codes.append(prediction)
        ground_truth_base_codes.append(ground_truth_base)
        predicted_base_codes.append(prediction_base)

    rows_evaluated = len(ground_truth_codes)

    if rows_evaluated == 0:
        raise ValueError(f"[{csv_file}] No valid ground-truth codes found.")

    exact_correct = sum(
        1
        for ground_truth, prediction in zip(
            ground_truth_codes,
            predicted_codes,
        )
        if prediction is not None and prediction == ground_truth
    )

    partial_correct = sum(
        1
        for ground_truth, prediction in zip(
            ground_truth_base_codes,
            predicted_base_codes,
        )
        if prediction is not None and prediction == ground_truth
    )

    accuracy_exact = exact_correct / rows_evaluated
    accuracy_partial = partial_correct / rows_evaluated

    _, macro_exact, weighted_exact = compute_f1_per_class(
        ground_truth_codes,
        predicted_codes,
    )
    _, macro_partial, weighted_partial = compute_f1_per_class(
        ground_truth_base_codes,
        predicted_base_codes,
    )

    return {
        "Model": csv_file.stem,
        "File": csv_file.name,
        "Path": str(csv_file),
        "Prediction Column": COL_PRED,
        "Rows Evaluated": rows_evaluated,
        "Accuracy Exact": accuracy_exact,
        "Accuracy Partial": accuracy_partial,
        "F1 Macro Exact": macro_exact,
        "F1 Weighted Exact": weighted_exact,
        "F1 Macro Partial": macro_partial,
        "F1 Weighted Partial": weighted_partial,
    }


def evaluate_icd_predictions(
    csv_file: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate ICD-10 predictions from one semantic-similarity result file.

    Args:
        csv_file (str | Path): Path to the CSV file with predictions.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Overall evaluation results and
        detailed neighbor-level results.
    """
    csv_file = Path(csv_file)

    df = pd.read_csv(csv_file, sep=CSV_SEP)
    model_name = csv_file.stem

    (
        _,
        total_rows,
        total_any_match,
        precision_metrics,
    ) = compare_icd_with_metadata(df)

    overall_result = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Total Rows": total_rows,
                "Rows with At Least One Match": total_any_match,
                "Percentage": (
                    (total_any_match / total_rows) * 100
                    if total_rows > 0
                    else 0
                ),
                **precision_metrics,
            }
        ]
    )

    (
        comparison_results,
        total_rows,
        _,
        _,
    ) = compare_icd_with_metadata(df)

    neighbor_results = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Neighbor": column,
                "Matches": matches,
                "Total Rows": total_rows,
                "Percentage": (
                    (matches / total_rows) * 100
                    if total_rows > 0
                    else 0
                ),
            }
            for column, matches in comparison_results.items()
        ]
    )

    neighbor_results["Neighbor"] = neighbor_results["Neighbor"].str.replace(
        r"suggestedMetadata(\d+)",
        lambda match: f"{match.group(1)}_nearest_neighbor",
        regex=True,
    )

    return overall_result.round(2), neighbor_results.round(2)


def main() -> None:
    """
    Evaluate all ICD-10 result files and save summary CSV files.
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    all_overall = []
    all_neighbors = []
    all_accuracy_f1_metrics = []

    for csv_file in CSV_PATH.rglob("results_*.csv"):
        print(f"Evaluating: {csv_file}")

        try:
            overall_df, neighbor_df = evaluate_icd_predictions(csv_file)
            accuracy_f1_metrics = compute_accuracy_f1_metrics(csv_file)

            model_name = csv_file.stem

            overall_df.to_csv(
                OUTPUT_PATH / f"{model_name}_overall.csv",
                index=False,
            )
            neighbor_df.to_csv(
                OUTPUT_PATH / f"{model_name}_neighbors.csv",
                index=False,
            )

            all_overall.append(overall_df)
            all_neighbors.append(neighbor_df)
            all_accuracy_f1_metrics.append(accuracy_f1_metrics)

        except Exception as error:
            print(f"Error while processing file {csv_file.name}: {error}")

    if all_overall:
        pd.concat(all_overall).to_csv(
            OUTPUT_PATH / "all_models_overall.csv",
            index=False,
        )

    if all_neighbors:
        pd.concat(all_neighbors).to_csv(
            OUTPUT_PATH / "all_models_neighbors.csv",
            index=False,
        )

    if all_accuracy_f1_metrics:
        pd.DataFrame(all_accuracy_f1_metrics).to_csv(
            METRICS_SUMMARY_PATH,
            sep=CSV_SEP,
            index=False,
        )

    print("All files were evaluated.")


if __name__ == "__main__":
    main()