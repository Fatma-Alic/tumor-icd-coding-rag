#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

from models.embedding_models import (
    list_of_embedding_models,
    list_of_input_dim_models,
)

CSV_PATH = (
    "/home/alic/RAG/data/"
    "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv"
)
EMBEDDING_MODELS_DIR = "/home/alic/RAG/classifier/embedding_models_icdo"
OUTPUT_DIR = "/home/alic/RAG/classifier/results"

TEXT_COLUMN = "Text extended"
ICD_COLUMN = "ICD-O-Code"
TOP_K = 20
GPU_DEVICE = torch.device("cuda:0")

MAPPING_PATH = "/home/alic/PredictICD10/ID_ICDO_Mapping.csv"


def load_icd_mapping(path: str):
    mapping_df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str)

    # Important: ICD-O-Num must really be numeric.
    mapping_df["ICD-O-Num"] = pd.to_numeric(
        mapping_df["ICD-O-Num"],
        errors="raise",
    )

    icd_to_number = {
        row["ICD-O-Code"]: int(row["ICD-O-Num"])
        for _, row in mapping_df.iterrows()
    }
    number_to_icd = {v: k for k, v in icd_to_number.items()}
    num_classes = len(icd_to_number)

    return icd_to_number, number_to_icd, num_classes


def load_classifier_state_dict(weights_path: str):
    obj = torch.load(weights_path, map_location="cpu")

    # Case A: state_dict saved directly.
    if isinstance(obj, dict) and "classifier_state_dict" not in obj:
        if any(k.endswith("weight") or k.endswith("bias") for k in obj.keys()):
            return obj

    # Case B: checkpoint dictionary saved.
    if isinstance(obj, dict) and "classifier_state_dict" in obj:
        return obj["classifier_state_dict"]

    raise ValueError(f"Unknown format in {weights_path}: type={type(obj)}")



class EmbeddingClassifier(nn.Module):
    """
    Feedforward neural network for ICD-10 classification.

    Args:
        input_dim (int): Input dimension of the embeddings.
        num_classes (int): Number of target ICD-10 classes.
        hidden_dim (int): Hidden layer dimension.
    """

    def __init__(
        self,
        input_dim: int = 0,
        num_classes: int = 2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def predict_classifier(
    embedding_model: SentenceTransformer,
    classifier: torch.nn.Module,
    texts: list,
    top_k: int,
):
    embeddings = embedding_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    embeddings = embeddings.to(dtype=torch.float32, device=GPU_DEVICE)

    classifier.eval()
    with torch.no_grad():
        outputs = classifier(embeddings)
        probs = F.softmax(outputs, dim=1)

        k = min(top_k, probs.shape[1])
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)

    # Important: CUDA -> CPU, then numpy.
    return (
        topk_indices.detach().cpu().numpy(),
        topk_probs.detach().cpu().numpy(),
    )


def evaluate_all_models():
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")

    icd_to_number, number_to_icd, num_classes = load_icd_mapping(MAPPING_PATH)

    all_codes = set(df[ICD_COLUMN].astype(str).str.strip())
    mapped_codes = set(icd_to_number.keys())
    missing = sorted(all_codes - mapped_codes)

    print(f"[INFO] Codes in GTDS: {len(all_codes)}")
    print(f"[INFO] Codes in mapping: {len(mapped_codes)}")
    print(f"[WARN] Missing codes: {len(missing)}")
    print("[WARN] Example missing:", missing[:30])

    # The class C42.1 is not available in the catalog and is added.
    """
    df = df[df[ICD_COLUMN].isin(icd_to_number)].copy()
    df["ICD_Num"] = df[ICD_COLUMN].map(icd_to_number)
    texts = df[TEXT_COLUMN].astype(str).tolist()
    """
    df = df.copy()
    df[ICD_COLUMN] = df[ICD_COLUMN].astype(str).str.strip()
    texts = df[TEXT_COLUMN].astype(str).tolist()  # All rows.

    for i, model_name in enumerate(list_of_embedding_models):
        print(f"Evaluating classifier model: {model_name}")

        # ---- Embedding model ----
        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
        try:
            embedding_model.to(GPU_DEVICE)
        except Exception:
            pass

        safe_name = model_name.replace("/", "_")

        # ---- Search checkpoint ----
        candidates = [
            os.path.join(EMBEDDING_MODELS_DIR, f"{safe_name}_checkpoint.pt"),
        ]
        weights_path = next((p for p in candidates if os.path.exists(p)), None)

        if weights_path is None:
            print(
                f"[SKIP] No checkpoint found for {model_name}. "
                f"Tried: {candidates}"
            )
            continue

        print(f"[INFO] Loading weights from: {weights_path}")

        # ---- Load state_dict ----
        state_dict = load_classifier_state_dict(weights_path)

        # Derive architecture from checkpoint, no retraining needed.
        # These keys exist in your setup: fc1.weight, fc2.weight, ...
        ckpt_input_dim = state_dict["fc1.weight"].shape[1]
        ckpt_hidden_dim = state_dict["fc1.weight"].shape[0]
        ckpt_num_classes = state_dict["fc2.weight"].shape[0]

        # Optional: check input_dim against list, warning only.
        expected_input_dim = list_of_input_dim_models[i]
        if ckpt_input_dim != expected_input_dim:
            print(
                f"[WARN] input_dim mismatch for {model_name}: "
                f"list={expected_input_dim}, ckpt={ckpt_input_dim} "
                "(using ckpt)"
            )

        # If num_classes does not match, the mapping basis is not identical
        # to training.
        if ckpt_num_classes != num_classes:
            print(
                f"[SKIP] num_classes mismatch for {model_name}: "
                f"mapping={num_classes}, ckpt={ckpt_num_classes}. "
                "Use the same mapping file/order as during training."
            )
            continue

        # ---- Instantiate classifier exactly as trained ----
        classifier = EmbeddingClassifier(
            input_dim=ckpt_input_dim,
            num_classes=ckpt_num_classes,
            hidden_dim=ckpt_hidden_dim,
        ).to(GPU_DEVICE)

        classifier.load_state_dict(state_dict, strict=True)

        # ---- Prediction ----
        topk_indices, topk_probs = predict_classifier(
            embedding_model=embedding_model,
            classifier=classifier,
            texts=texts,
            top_k=TOP_K,
        )

        # ---- Build output ----
        predictions = df.copy()
        k_used = topk_indices.shape[1]

        for k in range(k_used):
            predictions[f"suggestedIDs{k + 1}"] = topk_indices[:, k]
            predictions[f"suggestedICD{k + 1}"] = [
                number_to_icd.get(int(idx), "UNKNOWN")
                for idx in topk_indices[:, k]
            ]
            predictions[f"suggestedProbs{k + 1}"] = topk_probs[:, k]

        # ---- Save ----
        output_folder = os.path.join(OUTPUT_DIR, safe_name)
        os.makedirs(output_folder, exist_ok=True)

        out_path = os.path.join(
            output_folder,
            f"{safe_name}_classifier_icdo_predictions.csv",
        )
        predictions.to_csv(out_path, sep=";", index=False)
        print(f"Saved: {out_path}")


def main():
    evaluate_all_models()


if __name__ == "__main__":
    main()