"""Analyze LLM ICD-code responses and calculate accuracy and F1 metrics."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/alic/RAG")
RESULTS_DIR = PROJECT_ROOT / "rag" /"results"

RESPONSES_DIRECTORY = RESULTS_DIR / "llm_responses"
OUTPUT_DIRECTORY = RESULTS_DIR / "llm_response_analysis"

LOG_FILE = OUTPUT_DIRECTORY / "analyzation.log"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# Patterns
# -----------------------------------------------------------------------------

ICD_CODE_PATTERN = re.compile(
    r"^[A-Z]\d{1,2}(?:\.\d{1,2})?$",
    re.IGNORECASE,
)

FIRST_CODE_IN_TEXT_PATTERN = re.compile(
    r"\b(?<![\w/])[A-Z](?![A-Z])\d{1,2}(?:\.\d{1,2})?"
    r"(?!\.\d|\w|-|/)"
)


def is_code_valid(code: str) -> bool:
    """
    Check whether a string looks like an ICD-style code.

    Examples are 'A00' and 'C50.9'.

    Args:
        code (str): Candidate ICD-style code.

    Returns:
        bool: True if the string matches the ICD pattern, otherwise False.
    """
    if not code:
        return False

    return bool(ICD_CODE_PATTERN.match(code.strip()))


def norm_code(code: str) -> str:
    """
    Normalize a code by trimming whitespace and converting it to uppercase.

    Args:
        code (str): Input code.

    Returns:
        str: Normalized code.
    """
    return code.strip().upper()


def get_base_code(code: str) -> str:
    """
    Return the base code before the decimal point.

    Example:
        'C50.9' becomes 'C50'.

    Args:
        code (str): Full ICD-style code.

    Returns:
        str: Base code used for partial-match evaluation.
    """
    return norm_code(code).split(".")[0]


def find_first_code(text: str) -> Optional[str]:
    """
    Find the first valid ICD-style code in free text.

    Args:
        text (str): Free-text model answer.

    Returns:
        str | None: First detected code in uppercase, or None if no code
        was found.
    """
    if not text:
        return None

    match = FIRST_CODE_IN_TEXT_PATTERN.search(text)

    return match.group(0).upper() if match else None


def fmt4(value: Any) -> Any:
    """
    Format numeric values with four decimal places.

    Args:
        value (Any): Input value.

    Returns:
        Any: Formatted string for numeric values, otherwise the original value.
    """
    try:
        return f"{float(value):.4f}"
    except Exception:
        return value


def initialize_counter() -> Dict[str, int]:
    """
    Initialize the global metric counter.

    Returns:
        dict[str, int]: Counter dictionary for response categories and lengths.
    """
    return {
        "num questions": 0,
        "invalid answer": 0,
        "exact match": 0,
        "partial match code": 0,
        "false code": 0,
        "false bool": 0,
        "answer length": 0,
    }


def calculate_further_metrics(counter: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate derived metrics from the raw counter values.

    Args:
        counter (dict[str, int]): Raw response-category counts.

    Returns:
        dict[str, float]: Accuracy, partial accuracy, invalid rate,
        error rate, and average answer length.
    """
    num_questions = counter.get("num questions", 0)
    exact_match = counter.get("exact match", 0)
    partial_match = counter.get("partial match code", 0)
    invalid = counter.get("invalid answer", 0)
    false = counter.get("false code", 0) + counter.get("false bool", 0)
    length = counter.get("answer length", 0)

    if num_questions > 0:
        accuracy = round(exact_match / num_questions, 4)
        partial_accuracy = round(
            (exact_match + partial_match) / num_questions,
            4,
        )
        invalid_rate = round(invalid / num_questions, 4)
        error_rate = round((invalid + false) / num_questions, 4)
        average_answer_length = round(length / num_questions, 4)
    else:
        logging.warning("No questions found in counter. All metrics set to 0.")
        accuracy = 0.0
        partial_accuracy = 0.0
        invalid_rate = 0.0
        error_rate = 0.0
        average_answer_length = 0.0

    return {
        "accuracy": accuracy,
        "partial accuracy": partial_accuracy,
        "invalid rate": invalid_rate,
        "error rate": error_rate,
        "average answer length": average_answer_length,
    }


def check_answer(true_answer: str, generated_answer: str) -> str:
    """
    Categorize a generated answer for ICD-code or yes/no questions.

    Args:
        true_answer (str): Expected answer, either an ICD-style code or
            'Ja'/'Nein'.
        generated_answer (str): Free-text answer generated by the model.

    Returns:
        str: One of the categories:
        'exact match', 'partial match code', 'false code', 'false bool',
        or 'invalid answer'.
    """

    def find_first_bool(text: str) -> Optional[str]:
        match = re.search(r"\b(ja|nein)\b", (text or "").lower())

        return match.group(0).capitalize() if match else None

    if not generated_answer or not generated_answer.strip():
        logging.warning("Empty or invalid answer: '%s'", generated_answer)
        return "invalid answer"

    generated_answer = generated_answer.strip()

    if is_code_valid(true_answer):
        first_code = find_first_code(generated_answer)

        if not first_code:
            return "invalid answer"

        if not is_code_valid(first_code):
            return "invalid answer"

        if norm_code(first_code) == norm_code(true_answer):
            return "exact match"

        if get_base_code(first_code) == get_base_code(true_answer):
            return "partial match code"

        return "false code"

    if true_answer in ["Ja", "Nein"]:
        first_bool = find_first_bool(generated_answer)

        if not first_bool:
            return "invalid answer"

        if first_bool.lower() == true_answer.lower():
            return "exact match"

        return "false bool"

    return "invalid answer"


def write_csv(
    output_base_dir: str | Path,
    dataset_name: str,
    metrics: List[Dict[str, Any]],
) -> None:
    """
    Write epoch-wise metrics to a CSV file.

    Args:
        output_base_dir (str | Path): Base output directory.
        dataset_name (str): Dataset folder name.
        metrics (list[dict[str, Any]]): Metric dictionaries, one per epoch.
    """
    output_base_dir = Path(output_base_dir)
    dataset_name = Path(dataset_name).stem
    csv_filename = f"analysis_{dataset_name}_with_F1.csv"
    file_path = output_base_dir / dataset_name / csv_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    "Num Questions",
                    "Invalid Answer",
                    "Exact Match",
                    "Partial Match Code",
                    "False Code",
                    "False Bool",
                    "Accuracy",
                    "Partial Accuracy",
                    "Invalid Rate",
                    "Error Rate",
                    "Average Answer Length",
                    "F1 Exact (macro)",
                    "F1 Exact (weighted)",
                    "F1 Partial (macro)",
                    "F1 Partial (weighted)",
                ]
            )

            for index, metric in enumerate(metrics):
                writer.writerow(
                    [
                        index,
                        metric.get("num questions", 0),
                        metric.get("invalid answer", 0),
                        metric.get("exact match", 0),
                        metric.get("partial match code", 0),
                        metric.get("false code", 0),
                        metric.get("false bool", 0),
                        metric.get("accuracy", 0),
                        metric.get("partial accuracy", 0),
                        metric.get("invalid rate", 0),
                        metric.get("error rate", 0),
                        metric.get("average answer length", 0),
                        metric.get("f1_exact_macro", 0.0),
                        metric.get("f1_exact_weighted", 0.0),
                        metric.get("f1_partial_macro", 0.0),
                        metric.get("f1_partial_weighted", 0.0),
                    ]
                )
    except Exception as error:
        logging.error("Error while writing CSV file: %s", error)
        print(f"Error while writing CSV file: {error}")


def save_json(obj: Any, path: str | Path) -> None:
    """
    Save a Python object as a JSON file.

    Args:
        obj (Any): JSON-serializable object.
        path (str | Path): Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(obj, file, ensure_ascii=False, indent=2)
    except Exception as error:
        logging.error("Error while saving %s: %s", path, error)


def save_detailed_results(
    detailed_results: List[Dict[str, Any]],
    epoch: int,
    output_dir: str | Path,
) -> None:
    """
    Save detailed epoch results as JSON.

    Args:
        detailed_results (list[dict[str, Any]]): Detailed response results.
        epoch (int): Epoch index.
        output_dir (str | Path): Output directory.
    """
    output_path = Path(output_dir) / f"analysis_epoch_{epoch}.json"
    save_json(detailed_results, output_path)


def compute_f1_per_class(
    items: List[Dict[str, Optional[str]]],
    mode: str = "exact",
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Compute per-class F1, macro F1, and weighted F1.

    Args:
        items (list[dict[str, str | None]]): Entries with ground-truth and
            predicted codes.
        mode (str): 'exact' for full-code comparison or 'partial' for
            base-code comparison.

    Returns:
        tuple[dict[str, dict[str, float]], float, float]: Per-class metrics,
        macro F1, and weighted F1.
    """
    if mode not in ("exact", "partial"):
        raise ValueError("mode must be either 'exact' or 'partial'.")

    gt_key = "gt_code" if mode == "exact" else "gt_base"
    pred_key = "pred_code" if mode == "exact" else "pred_base"

    filtered = [item for item in items if item.get(gt_key)]

    if not filtered:
        return {}, 0.0, 0.0

    labels = sorted(set(item[gt_key] for item in filtered))
    per_class: Dict[str, Dict[str, float]] = {}

    support = defaultdict(int)

    for item in filtered:
        support[item[gt_key]] += 1

    for label in labels:
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for item in filtered:
            ground_truth = item.get(gt_key)
            prediction = item.get(pred_key)

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
            "support": support[label],
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1_score, 6),
            "tp": true_positive,
            "fp": false_positive,
            "tn": true_negative,
            "fn": false_negative,
        }

    macro_f1 = (
        sum(class_metrics["f1"] for class_metrics in per_class.values())
        / len(per_class)
        if per_class
        else 0.0
    )

    total_support = sum(support.values())
    weighted_f1 = (
        sum(
            class_metrics["f1"] * class_metrics["support"]
            for class_metrics in per_class.values()
        )
        / total_support
        if total_support > 0
        else 0.0
    )

    return per_class, round(macro_f1, 6), round(weighted_f1, 6)


def analyze_responses(file_path: str | Path) -> Dict[str, Any]:
    """
    Analyze one JSON file with model responses.

    Args:
        file_path (str | Path): Path to a JSON file containing model responses.

    Returns:
        dict[str, Any]: Metrics, detailed results, F1 items, and per-code F1
        results.
    """
    file_path = Path(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as error:
        logging.error("Error while loading file %s: %s", file_path, error)
        return {}

    counter = initialize_counter()
    detailed_results: List[Dict[str, Any]] = []
    icd_items: List[Dict[str, Optional[str]]] = []

    for entry in data:
        qa_list = entry.get("qa", [])
        last_qa = qa_list[-1] if qa_list else {}
        question = last_qa.get("question", "")
        true_answer = entry.get("true_answer", "")
        generated_answer = entry.get("generated_answer", "")

        category = check_answer(true_answer, generated_answer)

        counter["num questions"] += 1
        counter[category] += 1
        counter["answer length"] += len(generated_answer or "")

        if is_code_valid(true_answer):
            ground_truth_code = norm_code(true_answer)
            ground_truth_base = get_base_code(ground_truth_code)

            predicted_code = None
            first_code = find_first_code(generated_answer or "")

            if first_code and is_code_valid(first_code):
                predicted_code = norm_code(first_code)

            predicted_base = (
                get_base_code(predicted_code) if predicted_code else None
            )

            icd_items.append(
                {
                    "gt_code": ground_truth_code,
                    "gt_base": ground_truth_base,
                    "pred_code": predicted_code,
                    "pred_base": predicted_base,
                }
            )

        detailed_results.append(
            {
                "question": question,
                "ground_truth": true_answer,
                "generated_answer": generated_answer,
                "category": category,
            }
        )

    metrics = {**counter, **calculate_further_metrics(counter)}

    per_code_exact, macro_exact, weighted_exact = compute_f1_per_class(
        icd_items,
        mode="exact",
    )
    per_code_partial, macro_partial, weighted_partial = compute_f1_per_class(
        icd_items,
        mode="partial",
    )

    metrics["f1_exact_macro"] = macro_exact
    metrics["f1_exact_weighted"] = weighted_exact
    metrics["f1_partial_macro"] = macro_partial
    metrics["f1_partial_weighted"] = weighted_partial

    return {
        "metrics": metrics,
        "detailed_results": detailed_results,
        "icd_items": icd_items,
        "per_code_exact": per_code_exact,
        "per_code_partial": per_code_partial,
    }


def write_single_final_row_csv(
    dataset_output_path: str | Path,
    dataset_name: str,
    metrics: Dict[str, Any],
    source_file: str | Path,
) -> Path:
    """
    Write one final metric row to a CSV file.

    Args:
        dataset_output_path (str | Path): Output folder.
        dataset_name (str): Dataset name used in the output file name.
        metrics (dict[str, Any]): Metrics for the row.
        source_file (str | Path): Source JSON file name.

    Returns:
        Path: Path to the written CSV file.
    """
    dataset_output_path = Path(dataset_output_path)
    dataset_output_path.mkdir(parents=True, exist_ok=True)

    output_csv = (
        dataset_output_path / f"final_results_{dataset_name}_with_F1.csv"
    )

    headers = [
        "File",
        "Num Questions",
        "Invalid Answer",
        "Exact Match",
        "Partial Match Code",
        "False Code",
        "False Bool",
        "Accuracy",
        "Partial Accuracy",
        "Invalid Rate",
        "Error Rate",
        "Average Answer Length",
        "F1 Exact (macro)",
        "F1 Exact (weighted)",
        "F1 Partial (macro)",
        "F1 Partial (weighted)",
    ]

    row = [
        Path(source_file).name,
        fmt4(metrics.get("num questions", 0)),
        fmt4(metrics.get("invalid answer", 0)),
        fmt4(metrics.get("exact match", 0)),
        fmt4(metrics.get("partial match code", 0)),
        fmt4(metrics.get("false code", 0)),
        fmt4(metrics.get("false bool", 0)),
        fmt4(metrics.get("accuracy", 0.0)),
        fmt4(metrics.get("partial accuracy", 0.0)),
        fmt4(metrics.get("invalid rate", 0.0)),
        fmt4(metrics.get("error rate", 0.0)),
        fmt4(metrics.get("average answer length", 0.0)),
        fmt4(metrics.get("f1_exact_macro", 0.0)),
        fmt4(metrics.get("f1_exact_weighted", 0.0)),
        fmt4(metrics.get("f1_partial_macro", 0.0)),
        fmt4(metrics.get("f1_partial_weighted", 0.0)),
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(row)

    return output_csv


def analyze_multiple_responses_on_model(
    responses_directory: str | Path,
    output_directory: str | Path,
) -> None:
    """
    Analyze all dataset folders in one model response folder.

    Args:
        responses_directory (str | Path): Root directory containing one
            subfolder per dataset.
        output_directory (str | Path): Directory where CSV results are written.
    """
    responses_directory = Path(responses_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_name = responses_directory.name
    model_output_path = output_directory / model_name
    model_output_path.mkdir(parents=True, exist_ok=True)

    dataset_paths = [
        path for path in responses_directory.iterdir() if path.is_dir()
    ]

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name

        logging.info(
            "Starting analysis for model: %s | dataset: %s",
            model_name,
            dataset_name,
        )

        dataset_output_path = model_output_path / dataset_name
        dataset_output_path.mkdir(parents=True, exist_ok=True)

        metrics_list: List[Dict[str, Any]] = []

        epoch_files = [
            path
            for path in dataset_path.iterdir()
            if path.is_file()
            and path.name.startswith("responses_epoch_")
            and path.suffix == ".json"
        ]

        epochs = sorted(
            set(
                int(re.search(r"epoch_(\d+)", path.name).group(1))
                for path in epoch_files
            )
        )

        if epochs:
            for epoch in epochs:
                file_path = dataset_path / f"responses_epoch_{epoch}.json"

                if not file_path.exists():
                    logging.warning("File not found: %s", file_path)
                    continue

                analysis_result = analyze_responses(file_path)

                if analysis_result:
                    metrics_list.append(analysis_result["metrics"])

            write_csv(model_output_path, dataset_name, metrics_list)

            logging.info(
                "Analysis for dataset %s with epochs completed.",
                dataset_name,
            )

        else:
            candidates = []
            priority_candidates = []

            for file_path in dataset_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json":
                    candidates.append(file_path)

                    if file_path.name == "responses.json":
                        priority_candidates.append((0, file_path))
                    elif file_path.name.startswith("responses_"):
                        priority_candidates.append((1, file_path))
                    else:
                        priority_candidates.append((2, file_path))

            if not candidates:
                logging.warning(
                    "No JSON file found in dataset folder: %s",
                    dataset_path,
                )
                continue

            priority_candidates.sort(key=lambda item: item[0])
            json_path = priority_candidates[0][1]

            analysis_result = analyze_responses(json_path)

            if not analysis_result:
                continue

            output_csv = write_single_final_row_csv(
                dataset_output_path=dataset_output_path,
                dataset_name=dataset_name,
                metrics=analysis_result["metrics"],
                source_file=json_path,
            )

            logging.info("Final single-row CSV written: %s", output_csv)
            logging.info(
                "Analysis for dataset %s single-JSON mode completed.",
                dataset_name,
            )


def format_value(value: Any) -> str:
    """
    Format numeric values as percentage strings.

    Args:
        value (Any): Input value.

    Returns:
        str: Percentage string for numeric values, otherwise string value.
    """
    if isinstance(value, (int, float)):
        percentage_value = value * 100

        if percentage_value == 0:
            return "0.00%"

        return f"{percentage_value:.2f}%"

    return str(value)


def main() -> None:
    """Run response analysis for one model response directory."""
    analyze_multiple_responses_on_model(
        responses_directory=RESPONSES_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()