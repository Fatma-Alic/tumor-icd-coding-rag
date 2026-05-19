import json
import os
from typing import Any, Dict, List

import pandas as pd

RESULTS_TOP5_CSV = "/home/alic/RAG/similarity_search/results/filtered_embedding_models_icdo"
ICD_GM_CSV = "/home/alic/ChromaDB/datasets/knowledgebase_icdo_codes_ids.csv"


def pick_unique_suggestions(
    row: pd.Series,
    k: int = 3,
    max_fields: int = 15,
) -> List[str]:
    """
    Collects, in order, the first up to k unique values from
    'suggestedICD' and 'suggestedICD1..max_fields'.

    Exact comparison is used, without normalization.
    """
    cols: List[str] = []

    if "suggestedICD" in row.index:
        cols.append("suggestedICD")

    for i in range(1, max_fields + 1):
        col = f"suggestedICD{i}"
        if col in row.index:
            cols.append(col)

    seen = set()
    unique_codes: List[str] = []

    for col in cols:
        val = row[col]

        if pd.isna(val) or val is None:
            continue

        if isinstance(val, str) and val == "":
            continue

        if val not in seen:
            seen.add(val)
            unique_codes.append(val)

            if len(unique_codes) == k:
                break

    return unique_codes


def prep_icdm():
    """
    Loads the ICD-O catalog as a DataFrame and extracts unique ICD-O codes.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with 'Label' and 'ICD_Dia'
            - np.ndarray: Array with unique ICD-O codes
    """
    data = pd.read_csv(
        ICD_GM_CSV,
        encoding="utf-8",
        quotechar='"',
        sep=";",
        engine="python",
    )
    data["Extended Label"] = data["Extended Label"].str.strip('"')
    data.columns = data.columns.str.replace(
        "\ufeff",
        "",
        regex=False,
    ).str.strip()

    icdm_dataset = pd.DataFrame(
        {
            "Label": data["Extended Label"],
            "ICD_Dia": data["ICD-O-Code"],
        }
    ).reset_index(drop=True)

    icd_unique = icdm_dataset["ICD_Dia"].unique()

    return icdm_dataset, icd_unique


def diagnosis_icdm(actual_icd_code, icdm_dataset):
    """
    Returns the diagnosis text for the given ICD code.

    Args:
        actual_icd_code (str): ICD-O code
        icdm_dataset (pd.DataFrame): ICDM catalog with columns
            'ICD_Dia' and 'Label'

    Returns:
        diagnosis_text_icdm (str): Diagnosis description from the ICDM catalog
    """
    diagnosis_text_icdm = None
    matching_row = icdm_dataset[icdm_dataset["ICD_Dia"] == actual_icd_code]

    if not matching_row.empty:
        diagnosis_text_icdm = matching_row["Label"].iloc[0]

    return diagnosis_text_icdm


def extract_icd_codes(df, i):
    """
    Extracts relevant ICD codes and the ground-truth diagnosis from a row.

    Args:
        df (pd.DataFrame): DataFrame with top-5 predictions
        i (int): Row index

    Returns:
        tuple:
            - str: suggestedICD1
            - str: suggestedICD2
            - str: suggestedICD3
            - str: suggestedICD4
            - str: suggestedICD5
            - str: actual ICD-O code
            - str: ground-truth diagnosis text
    """
    row = df.iloc[i]
    icd_code_1 = row["suggestedICD1"]
    icd_code_2 = row["suggestedICD2"]
    icd_code_3 = row["suggestedICD3"]
    icd_code_4 = row["suggestedICD4"]
    icd_code_5 = row["suggestedICD5"]
    actual_icd_code = row["ICD-O-Code"]
    diagnosis_text_gtds = row["Text extended"]

    return (
        icd_code_1,
        icd_code_2,
        icd_code_3,
        icd_code_4,
        icd_code_5,
        actual_icd_code,
        diagnosis_text_gtds,
    )


def safe_diagnosis_lookup(
    code: str,
    icdm_dataset: pd.DataFrame,
    alpha_df: pd.DataFrame,
) -> str | None:
    txt = diagnosis_icdm(code, icdm_dataset)

    if txt is None:
        txt = fallback_diagnosis_lookup(code, alpha_df)

    return txt


def generating_shots_questions(
    results_top5: pd.DataFrame,
    model_dir: str,
    model_name: str,
) -> None:
    """
    Creates few-shot QA files:
      - takes up to 3 unique suggestion codes per row,
      - reorders them: last -> ... -> first,
      - appends the ground-truth QA at the end,
      - saves everything as JSON.
    """
    icdm_dataset, _ = prep_icdm()

    alpha_df = pd.read_csv(
        ICD_GM_CSV,
        encoding="utf-8",
        sep=";",
        engine="python",
    )
    alpha_df.columns = alpha_df.columns.str.replace(
        "\ufeff",
        "",
        regex=False,
    ).str.strip()

    questions_file: List[Dict[str, Any]] = []

    for i, row in results_top5.iterrows():
        picked_codes: List[str] = pick_unique_suggestions(row, k=3)

        if not picked_codes:
            print(f"Row {i} skipped: no unique codes found.")
            continue

        diag_texts: List[str] = [
            safe_diagnosis_lookup(code, icdm_dataset, alpha_df)
            for code in picked_codes
        ]

        filtered = [
            (code, text)
            for code, text in zip(picked_codes, diag_texts)
            if text is not None and isinstance(text, str) and text.strip() != ""
        ]

        if not filtered:
            print(f"Row {i} skipped: no valid diagnosis mapping found.")
            continue

        picked_codes, diag_texts = zip(*filtered)
        picked_codes = list(picked_codes)
        diag_texts = list(diag_texts)

        actual_icd_code = row.get("ICD-O-Code", "")
        diagnosis_text_gtds = row.get("Text extended", "")

        if (
            pd.isna(actual_icd_code)
            or not isinstance(diagnosis_text_gtds, str)
            or diagnosis_text_gtds.strip() == ""
        ):
            print(f"Row {i} skipped: ground truth incomplete.")
            continue

        order = list(range(len(picked_codes) - 1, -1, -1))

        question_array: List[Dict[str, str]] = []

        for idx in order:
            code = picked_codes[idx]
            label = diag_texts[idx]
            question_array.append(
                {
                    "question": (
                        "Answer only with the ICD-O topography code. "
                        "What is the ICD-O code for the location of the "
                        f'tumor diagnosis "{label}"?'
                    ),
                    "answer": f"{code}",
                }
            )

        question_array.append(
            {
                "question": (
                    "Answer only with the ICD-O topography code. "
                    "What is the ICD-O code for the location of the "
                    f'tumor diagnosis "{diagnosis_text_gtds}"?'
                ),
                "answer": f"{actual_icd_code}",
            }
        )

        questions_file.append({"questions": question_array})

    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(
        model_dir,
        f"{model_name}_ICDO_shots_questions_2.json",
    )

    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(questions_file, output, ensure_ascii=False, indent=4)

    print(f"Saved: {output_path}")


def fallback_diagnosis_lookup(code: str, alpha_df: pd.DataFrame) -> str | None:
    """
    Calls the Alpha catalog if the diagnosis is not found in the ICD-GM catalog.

    Args:
        code (str): ICD code
        alpha_df (pd.DataFrame): Alpha catalog

    Returns:
        match (str): Diagnosis from the Alpha catalog
    """
    if pd.isna(code) or code == "":
        return None

    match = alpha_df[alpha_df["ICD-O-Code"] == code]

    if not match.empty:
        return match["Extended Label"].iloc[0]

    return None


def main():
    model_dir = (
        "/home/alic/ChromaDB/icdo_scripts/evaluation_results_icdo/"
        "text extended/Extended Label"
    )

    for root, _, files in os.walk(RESULTS_TOP5_CSV):
        for file in files:
            if file.startswith("result_"):
                path = os.path.join(root, file)
                print(f"Creating prompts: {path}")

                try:
                    data = pd.read_csv(
                        path,
                        encoding="utf-8",
                        quotechar='"',
                        sep=";",
                        engine="python",
                    )
                    data.columns = data.columns.str.replace(
                        "\ufeff",
                        "",
                        regex=False,
                    ).str.strip()

                    model_name = os.path.splitext(file)[0].replace(
                        "result_",
                        "",
                    )
                    generating_shots_questions(data, model_dir, model_name)

                except Exception as error:
                    print(f"Error processing file {file}: {error}")


if __name__ == "__main__":
    main()