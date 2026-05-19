"""Evaluate ICD-10 predictions from one GTDS prediction file."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/alic/RAG")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "classifier/results"

PREDICTION_FILE = (
    DATA_DIR / "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv"
)

OUTPUT_DIR = RESULTS_DIR / "evaluation_results_untrained"

OVERALL_OUTPUT_FILE = OUTPUT_DIR / "gtds_overall.csv"
NEIGHBORS_OUTPUT_FILE = OUTPUT_DIR / "gtds_neighbors.csv"


def extract_icd_code(icd_code: str) -> str | None:
    """
    Extract the ICD-10 code without the description part.

    Args:
        icd_code (str): Full ICD-10 code string with optional description.

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
    predictions = [row.get(f"suggestedICD{i + 1}") for i in range(k)]

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
        if row.get(f"suggestedICD{i + 1}") == ground_truth_code:
            return 1.0 / (i + 1)

    return 0.0


def validate_prediction_columns(df: pd.DataFrame) -> None:
    """
    Check whether the required columns exist.

    Args:
        df (pd.DataFrame): Input prediction DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = ["ICD-10-Code"] + [
        f"suggestedICD{i}" for i in range(1, 6)
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


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

    metadata_columns = [f"suggestedICD{i}" for i in range(1, 6)]

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


def evaluate_icd_predictions(
    prediction_file: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate ICD-10 predictions from one prediction CSV file.

    Args:
        prediction_file (str | Path): Path to the CSV file containing
            ground-truth ICD-10 codes and suggested ICD predictions.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Overall evaluation results and
        detailed neighbor-level results.
    """
    prediction_file = Path(prediction_file)

    df = pd.read_csv(prediction_file, sep=";")
    validate_prediction_columns(df)

    (
        comparison_results,
        total_rows,
        total_any_match,
        precision_metrics,
    ) = compare_icd_with_metadata(df)

    overall_result = pd.DataFrame(
        [
            {
                "File": prediction_file.name,
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

    neighbor_results = pd.DataFrame(
        [
            {
                "File": prediction_file.name,
                "Neighbor": column,
                "Matches": matches,
                "Total Rows": total_rows,
                "Match %": (
                    (matches / total_rows) * 100
                    if total_rows > 0
                    else 0
                ),
            }
            for column, matches in comparison_results.items()
        ]
    )

    neighbor_results["Neighbor"] = neighbor_results["Neighbor"].str.replace(
        r"suggestedICD(\d+)",
        lambda match: f"{match.group(1)}_nearest_neighbor",
        regex=True,
    )

    return overall_result.round(2), neighbor_results.round(2)


def main() -> None:
    """Evaluate one GTDS prediction file and save the result CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_df, neighbor_df = evaluate_icd_predictions(PREDICTION_FILE)

    overall_df.to_csv(
        OVERALL_OUTPUT_FILE,
        sep=";",
        index=False,
    )
    neighbor_df.to_csv(
        NEIGHBORS_OUTPUT_FILE,
        sep=";",
        index=False,
    )

    print(f"Evaluated file: {PREDICTION_FILE}")
    print(f"Overall results saved: {OVERALL_OUTPUT_FILE}")
    print(f"Neighbor results saved: {NEIGHBORS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()