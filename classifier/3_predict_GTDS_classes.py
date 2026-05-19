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
    "/home/alic/PredictICD10/FINAL/"
    "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv"
)
EMBEDDING_MODELS_DIR = (
    "/home/alic/PredictICD10/FINAL/filtered/embedding_models_icdo"
)
OUTPUT_DIR = "/home/alic/PredictICD10/FINAL/evaluation_results_icdo/text extended"

TEXT_COLUMN = "Text extended"
ICD_COLUMN = "ICD-O-Code"
TOP_K = 20
GPU_DEVICE = torch.device("cuda:0")

MAPPING_PATH = "/home/alic/PredictICD10/ID_ICDO_Mapping.csv"


def load_icd_mapping(path: str):
    mapping_df = pd.read_csv(path, sep=";", encoding="utf-8")
    icd_to_number = {
        row["ICD-O-Code"]: int(row["ICD-O-Num"])
        for _, row in mapping_df.iterrows()
    }
    number_to_icd = {v: k for k, v in icd_to_number.items()}
    num_classes = len(icd_to_number)

    return icd_to_number, number_to_icd, num_classes


def load_classifier_state_dict(weights_path: str):
    obj = torch.load(weights_path)

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
    classifier,
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

    return topk_indices.numpy(), topk_probs.numpy()


def evaluate_all_models():
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")

    icd_to_number, number_to_icd, num_classes = load_icd_mapping(MAPPING_PATH)

    df = df[df[ICD_COLUMN].isin(icd_to_number)].copy()
    df["ICD_Num"] = df[ICD_COLUMN].map(icd_to_number)

    texts = df[TEXT_COLUMN].astype(str).tolist()

    for i, model_name in enumerate(list_of_embedding_models):
        print(f"Evaluating classifier model: {model_name}")

        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
        try:
            embedding_model.to(GPU_DEVICE)
        except Exception:
            pass

        input_dim = list_of_input_dim_models[i]
        classifier = EmbeddingClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
        ).to(GPU_DEVICE)

        safe_name = model_name.replace("/", "_")

        # Important: load checkpoint, which contains classifier_state_dict.
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
        state_dict = load_classifier_state_dict(weights_path)
        classifier.load_state_dict(state_dict, strict=False)

        topk_indices, topk_probs = predict_classifier(
            embedding_model,
            classifier,
            texts,
            TOP_K,
        )

        predictions = df.copy()
        k_used = topk_indices.shape[1]

        for k in range(k_used):
            predictions[f"suggestedIDs{k + 1}"] = topk_indices[:, k]
            predictions[f"suggestedICD{k + 1}"] = [
                number_to_icd.get(int(idx), "UNKNOWN")
                for idx in topk_indices[:, k]
            ]
            predictions[f"suggestedProbs{k + 1}"] = topk_probs[:, k]

        output_folder = os.path.join(OUTPUT_DIR, safe_name)
        os.makedirs(output_folder, exist_ok=True)

        out_path = os.path.join(
            output_folder,
            f"{safe_name}_classifier_icdo_predictions_2.csv",
        )
        predictions.to_csv(out_path, sep=";", index=False)
        print(f"Saved: {out_path}")


def main():
    evaluate_all_models()


if __name__ == "__main__":
    main()