"""Evaluate ICD-10 classifier prediction files with top-k and F1 metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import f1_score


PROJECT_ROOT = Path("/home/alic/RAG")
RESULTS_DIR = PROJECT_ROOT / "/classifier_results"

INPUT_FOLDER = RESULTS_DIR / "filtered_embedding_models_icd10"
OUTPUT_CSV = RESULTS_DIR / "evaluation_results_icd10" / "exact_match_metrics.csv"


def extract_icd_code(icd_code: str) -> str | None:
    """
    Extract the ICD code before the first space.

    Args:
        icd_code (str): Original ICD code text.

    Returns:
        str | None: Extracted ICD code, or None if the input is not a string.
    """
    if isinstance(icd_code, str):
        return icd_code.split(" ")[0]

    return None


def enlarge_icd_code(icd_code: str) -> str | None:
    """
    Extract the ICD code before the first dot.

    Args:
        icd_code (str): Original ICD code text.

    Returns:
        str | None: Shortened ICD code, or None if the input is not a string.
    """
    if isinstance(icd_code, str):
        return icd_code.split(".")[0]

    return None


def precision_at_k(row: pd.Series, k: int) -> int:
    """
    Check whether the true ICD code appears in the top-k predictions.

    Args:
        row (pd.Series): One DataFrame row.
        k (int): Number of top predictions to check.

    Returns:
        int: 1 if the true code is found, otherwise 0.
    """
    ground_truth_code = extract_icd_code(row["ICD-10-Code"])
    predictions = [row.get(f"suggestedICD{i + 1}") for i in range(k)]

    return int(ground_truth_code in predictions)


def reciprocal_rank(row: pd.Series) -> float:
    """
    Calculate the reciprocal rank of the correct ICD code.

    Args:
        row (pd.Series): One DataFrame row.

    Returns:
        float: Reciprocal rank value.
    """
    ground_truth_code = extract_icd_code(row["ICD-10-Code"])

    for i in range(5):
        if row.get(f"suggestedICD{i + 1}") == ground_truth_code:
            return 1.0 / (i + 1)

    return 0.0


def compare_icd_with_metadata_exact_match(
    df: pd.DataFrame,
) -> Tuple[Dict[str, int], int, int, Dict[str, float]]:
    """
    Compare the true ICD code with the predicted ICD codes.

    Args:
        df (pd.DataFrame): Input DataFrame with true and predicted ICD codes.

    Returns:
        tuple[dict[str, int], int, int, dict[str, float]]:
        Match counts per suggested ICD column, number of usable rows,
        number of rows with any top-5 match, and precision/MRR metrics.
    """
    df["ICD-10-Pure-Code"] = df["ICD-10-Code"].apply(extract_icd_code)
    metadata_columns = [f"suggestedICD{i}" for i in range(1, 6)]

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


def compute_f1_top1(df: pd.DataFrame) -> tuple[float, float, float, int]:
    """
    Calculate micro, macro, and weighted F1 based on the top-1 prediction.

    Args:
        df (pd.DataFrame): Input DataFrame with true and predicted ICD codes.

    Returns:
        tuple[float, float, float, int]: Micro F1, macro F1, weighted F1,
        and number of used rows.
    """
    work = df.copy()
    work["ICD-10-Pure-Code"] = work["ICD-10-Code"].apply(extract_icd_code)

    work = work.dropna(subset=["ICD-10-Pure-Code", "suggestedICD1"]).copy()

    work["ICD-10-Pure-Code"] = work["ICD-10-Pure-Code"].astype(str)
    work["suggestedICD1"] = work["suggestedICD1"].astype(str)

    if len(work) == 0:
        return 0.0, 0.0, 0.0, 0

    micro = f1_score(
        work["ICD-10-Pure-Code"],
        work["suggestedICD1"],
        average="micro",
        zero_division=0,
    )
    macro = f1_score(
        work["ICD-10-Pure-Code"],
        work["suggestedICD1"],
        average="macro",
        zero_division=0,
    )
    weighted = f1_score(
        work["ICD-10-Pure-Code"],
        work["suggestedICD1"],
        average="weighted",
        zero_division=0,
    )

    return micro, macro, weighted, len(work)


def evaluate_folder(
    input_folder: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Evaluate all matching CSV files in one folder.

    Args:
        input_folder (str | Path): Folder containing prediction CSV files.
        output_csv (str | Path): File path for saving the final results CSV.

    Returns:
        pd.DataFrame: DataFrame with all calculated metrics.
    """
    input_folder = Path(input_folder)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    csv_files = sorted(
        input_folder.rglob("*classifier_predictions.csv")
    )

    required_columns = ["ICD-10-Code"] + [
        f"suggestedICD{i}" for i in range(1, 6)
    ]

    for csv_file in csv_files:
        model_name = csv_file.stem

        try:
            df = pd.read_csv(csv_file, sep=";", engine="python")
        except Exception:
            df = pd.read_csv(csv_file)

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            print(
                f"[WARNING] {model_name}: skipped - "
                f"missing columns: {missing_columns}"
            )
            continue

        _, total_rows, total_any_match, precision_metrics = (
            compare_icd_with_metadata_exact_match(df)
        )

        micro_f1, macro_f1, weighted_f1, f1_rows = compute_f1_top1(df)

        rows.append(
            {
                "Model": model_name,
                "Rows_usable_for_metrics": total_rows,
                "AnyMatch_Top5_Count": total_any_match,
                "Top1 (%)": precision_metrics.get(
                    "Precision@1",
                    float("nan"),
                ),
                "Top3 (%)": precision_metrics.get(
                    "Precision@3",
                    float("nan"),
                ),
                "Top5 (%)": precision_metrics.get(
                    "Precision@5",
                    float("nan"),
                ),
                "MRR": precision_metrics.get("MRR", float("nan")),
                "MicroF1_Top1 (%)": micro_f1 * 100,
                "MacroF1_Top1 (%)": macro_f1 * 100,
                "WeightedF1_Top1 (%)": weighted_f1 * 100,
                "Rows_used_for_F1": f1_rows,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)

    print(f"Results saved: {output_csv}")

    return results


def main() -> None:
    """Run the evaluation and print the resulting metrics table."""
    results_df = evaluate_folder(
        input_folder=INPUT_FOLDER,
        output_csv=OUTPUT_CSV,
    )

    print(results_df)


if __name__ == "__main__":
    main()