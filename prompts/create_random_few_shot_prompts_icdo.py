"""Generate random few-shot ICD-O question blocks as a JSON file."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/alic/RAG")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "similarity_search" / "results"

ICD_O_CSV = DATA_DIR / "ICDO3_LE_dataset.csv"

INPUT_RESULTS_CSV = RESULTS_DIR / "word2vec" / "outputs_w2v_icdo" /"nn_results_topk.csv"

OUTPUT_JSON = (
    RESULTS_DIR
    / "random" / "random3_from_catalog_ICDO_shot_questions.json"
)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

COL_GT_CODE = "ICD-O-Code"
COL_GT_TEXT = "Text extended"

PROMPT_INSTRUCTION = "Antworte nur mit dem ICD-O-Topographie-Code."
QUESTION_TEMPLATE = (
    'Wie lautet der ICD-O-Code für die Lokalisation zur Tumordiagnose "{diag}"?'
)


def prep_icdm(
    icd_csv: str | Path = ICD_O_CSV,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the ICD-O catalog and return unique ICD-O codes.

    Args:
        icd_csv (str | Path): Path to the ICD-O catalog CSV file.

    Returns:
        tuple[pd.DataFrame, list[str]]: Cleaned ICD-O DataFrame and list of
        unique ICD-O codes.

    Raises:
        ValueError: If required columns are missing.
    """
    data = pd.read_csv(
        icd_csv,
        encoding="utf-8",
        quotechar='"',
        sep=";",
        engine="python",
    )

    if "Prompttext" not in data.columns or "ICD-O-Code" not in data.columns:
        raise ValueError(
            "ICD-O catalog must contain columns 'Prompttext' and "
            "'ICD-O-Code'. "
            f"Found columns: {list(data.columns)}"
        )

    data["Prompttext"] = data["Prompttext"].astype(str).str.strip().str.strip('"')
    data["ICD-O-Code"] = data["ICD-O-Code"].astype(str).str.strip()

    icdm_dataset = pd.DataFrame(
        {
            "Prompttext": data["Prompttext"],
            "ICD-O-Code": data["ICD-O-Code"],
        }
    ).dropna().reset_index(drop=True)

    icd_unique = (
        icdm_dataset["ICD-O-Code"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return icdm_dataset, icd_unique


def diagnosis_icdm(
    actual_icd_code: str,
    icdm_dataset: pd.DataFrame,
) -> Optional[str]:
    """
    Return the prompt text for one ICD-O code.

    Args:
        actual_icd_code (str): ICD-O code to search for.
        icdm_dataset (pd.DataFrame): ICD-O catalog DataFrame.

    Returns:
        str | None: Prompt text if found, otherwise None.
    """
    code = str(actual_icd_code).strip()
    match = icdm_dataset[icdm_dataset["ICD-O-Code"] == code]

    if match.empty:
        return None

    text = match["Prompttext"].iloc[0]

    return None if pd.isna(text) else str(text).strip()


def fallback_diagnosis_lookup(
    code: str,
    alpha_df: pd.DataFrame,
) -> Optional[str]:
    """
    Try to find a prompt text in a fallback DataFrame.

    Args:
        code (str): ICD-O code to search for.
        alpha_df (pd.DataFrame): Fallback DataFrame.

    Returns:
        str | None: Prompt text if found, otherwise None.
    """
    code = str(code).strip()

    possible_code_columns = ["ICD_Code", "ICD-O-Code"]
    possible_label_columns = ["Prompttext"]

    code_column = next(
        (
            column
            for column in possible_code_columns
            if column in alpha_df.columns
        ),
        None,
    )
    label_column = next(
        (
            column
            for column in possible_label_columns
            if column in alpha_df.columns
        ),
        None,
    )

    if code_column is None or label_column is None:
        return None

    match = alpha_df[alpha_df[code_column].astype(str).str.strip() == code]

    if match.empty:
        return None

    text = match[label_column].iloc[0]

    return None if pd.isna(text) else str(text).strip().strip('"')


def safe_diagnosis_lookup(
    code: str,
    icdm_dataset: pd.DataFrame,
    alpha_df: pd.DataFrame,
) -> Optional[str]:
    """
    Find a prompt text using the main catalog first, then the fallback catalog.

    Args:
        code (str): ICD-O code to search for.
        icdm_dataset (pd.DataFrame): Main ICD-O catalog DataFrame.
        alpha_df (pd.DataFrame): Fallback DataFrame.

    Returns:
        str | None: Prompt text if found, otherwise None.
    """
    text = diagnosis_icdm(code, icdm_dataset)

    if text is None:
        text = fallback_diagnosis_lookup(code, alpha_df)

    return text


def sample_random_icd_codes(
    icd_unique: List[str],
    n: int,
    exclude: Optional[set] = None,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Sample random ICD-O codes that are not in the exclude set.

    Args:
        icd_unique (list[str]): List of available ICD-O codes.
        n (int): Number of codes to sample.
        exclude (set | None): Codes that should not be sampled.
        rng (random.Random | None): Random generator.

    Returns:
        list[str]: Randomly sampled ICD-O codes.

    Raises:
        ValueError: If not enough ICD-O codes are available for sampling.
    """
    rng = rng or random.Random()
    exclude = exclude or set()

    pool = [code for code in icd_unique if code not in exclude]

    if len(pool) < n:
        raise ValueError(
            f"Not enough ICD-O codes to sample {n} unique values after "
            "exclusion."
        )

    return rng.sample(pool, n)


def build_questions_block(
    gt_text: str,
    gt_code: str,
    icdm_dataset: pd.DataFrame,
    icd_unique: List[str],
    alpha_df: pd.DataFrame,
    n_random_examples: int = 3,
    rng: Optional[random.Random] = None,
    max_resample_trials: int = 50,
) -> List[Dict[str, str]]:
    """
    Build one question block with random examples and one ground-truth question.

    Args:
        gt_text (str): Ground-truth diagnosis text.
        gt_code (str): Ground-truth ICD-O code.
        icdm_dataset (pd.DataFrame): Main ICD-O catalog DataFrame.
        icd_unique (list[str]): List of unique ICD-O codes.
        alpha_df (pd.DataFrame): Fallback DataFrame.
        n_random_examples (int): Number of random examples to add.
        rng (random.Random | None): Random generator.
        max_resample_trials (int): Maximum number of resampling attempts.

    Returns:
        list[dict[str, str]]: List of question-answer dictionaries.

    Raises:
        RuntimeError: If not enough valid random examples can be sampled.
    """
    rng = rng or random.Random()
    gt_text = str(gt_text).strip()
    gt_code = str(gt_code).strip()

    exclude = {gt_code}
    questions = []
    used = set(exclude)

    trials = 0

    while len(questions) < n_random_examples and trials < max_resample_trials:
        trials += 1

        candidate = sample_random_icd_codes(
            icd_unique,
            1,
            exclude=used,
            rng=rng,
        )[0]
        used.add(candidate)

        label = safe_diagnosis_lookup(candidate, icdm_dataset, alpha_df)

        if not label:
            continue

        questions.append(
            {
                "question": (
                    f"{PROMPT_INSTRUCTION} "
                    f"{QUESTION_TEMPLATE.format(diag=label)}"
                ),
                "answer": candidate,
            }
        )

    if len(questions) < n_random_examples:
        raise RuntimeError(
            f"Could not sample {n_random_examples} random ICD-O examples with "
            f"valid labels after {max_resample_trials} trials."
        )

    questions.append(
        {
            "question": (
                f"{PROMPT_INSTRUCTION} "
                f"{QUESTION_TEMPLATE.format(diag=gt_text)}"
            ),
            "answer": gt_code,
        }
    )

    return questions


def generate_random_shot_questions_json(
    results_csv: Optional[str | Path] = None,
    results_df: Optional[pd.DataFrame] = None,
    output_json: str | Path = OUTPUT_JSON,
    n_random_examples: int = 3,
    seed: int = 42,
) -> None:
    """
    Generate a JSON file with random few-shot ICD-O question blocks.

    Args:
        results_csv (str | Path | None): Path to the input results CSV file.
        results_df (pd.DataFrame | None): Input DataFrame as an alternative
            to a CSV file.
        output_json (str | Path): Output path for the JSON file.
        n_random_examples (int): Number of random examples per question block.
        seed (int): Random seed for reproducible sampling.

    Raises:
        ValueError: If no input data is provided or required columns are missing.
    """
    if results_df is None:
        if results_csv is None:
            raise ValueError("Provide either results_csv or results_df.")

        results_df = pd.read_csv(
            results_csv,
            encoding="utf-8",
            sep=";",
            engine="python",
        )

    if COL_GT_CODE not in results_df.columns or COL_GT_TEXT not in results_df.columns:
        raise ValueError(
            f"Results must contain columns '{COL_GT_CODE}' and '{COL_GT_TEXT}'. "
            f"Found columns: {list(results_df.columns)}"
        )

    icdm_dataset, icd_unique = prep_icdm(ICD_O_CSV)

    alpha_df = pd.read_csv(
        ICD_O_CSV,
        encoding="utf-8",
        sep=";",
        engine="python",
    )

    rng = random.Random(seed)

    output_data = []
    total = 0
    skipped = 0

    for _, row in results_df.iterrows():
        total += 1

        gt_code = str(row.get(COL_GT_CODE, "") or "").strip()
        gt_text = str(row.get(COL_GT_TEXT, "") or "").strip()

        if not gt_code or not gt_text:
            skipped += 1
            continue

        try:
            question_block = build_questions_block(
                gt_text=gt_text,
                gt_code=gt_code,
                icdm_dataset=icdm_dataset,
                icd_unique=icd_unique,
                alpha_df=alpha_df,
                n_random_examples=n_random_examples,
                rng=rng,
            )
        except Exception:
            skipped += 1
            continue

        output_data.append({"questions": question_block})

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("======================================")
    print(f"[OK] Input rows      : {total}")
    print(f"[OK] Written prompts : {len(output_data)}")
    print(f"[OK] Skipped         : {skipped}")
    print(f"[OK] Output JSON     : {output_path}")
    print("======================================")


def main() -> None:
    """Generate random three-shot ICD-O question blocks."""
    generate_random_shot_questions_json(
        results_csv=INPUT_RESULTS_CSV,
        output_json=OUTPUT_JSON,
        n_random_examples=3,
        seed=42,
    )


if __name__ == "__main__":
    main()