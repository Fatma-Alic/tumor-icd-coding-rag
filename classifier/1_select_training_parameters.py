"""Run Optuna hyperparameter tuning for embedding-based ICD classifiers."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path("/home/alic/RAG")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.embedding_models import list_of_embedding_models  # noqa: E402
from preprocessing.prep_dataset import create_icd_dataframe  # noqa: E402


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "classifier/results"

RESULTS_PATH = DATA_DIR / "Alpha_ID_dataset.csv"
OUT_ROOT = RESULTS_DIR / "optuna_tuning_embeddings_classifier"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

VAL_FRACTION = 0.15
SEED = 42
USE_GPU = torch.cuda.is_available()

N_TRIALS_PER_MODEL = 25
ENCODE_BATCH_SIZE = 128

BATCH_CHOICES = [16, 32, 64]
EPOCHS_RANGE = (3, 12)
LR_LOG_RANGE = (1e-5, 5e-3)
WEIGHT_DECAY_RANGE = (0.0, 0.1)
HIDDEN_CHOICES = [128, 256, 384, 512]
DROPOUT_RANGE = (0.1, 0.6)


def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str) -> str:
    """
    Create a filesystem-safe name.

    Args:
        name (str): Original model name.

    Returns:
        str: Sanitized model name.
    """
    return name.replace("/", "__").replace(":", "_")


def load_dataframe(results_path: str | Path) -> pd.DataFrame:
    """
    Load the input dataset.

    The dataset must contain:
    - 'Label'
    - one ICD column, such as 'ICD-10-Code', 'ICD-10', or 'ICD-10-Num'

    Args:
        results_path (str | Path): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded input dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(results_path, sep=";")

    if "Label" not in df.columns:
        raise ValueError("Column 'Label' is missing in the input dataset.")

    valid_code_columns = {"ICD-10-Code", "ICD-10", "ICD-10-Num"}
    if not valid_code_columns.intersection(df.columns):
        raise ValueError(
            "The input dataset must contain one of these columns: "
            "'ICD-10-Code', 'ICD-10', or 'ICD-10-Num'."
        )

    return df


def detect_code_column(df: pd.DataFrame) -> str:
    """
    Detect the ICD code column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        str: Detected ICD code column name.

    Raises:
        ValueError: If no supported ICD code column is found.
    """
    for column in ["ICD-10-Code", "ICD-10", "ICD-10-Num"]:
        if column in df.columns:
            return column

    raise ValueError(
        "No supported ICD code column found. Expected one of: "
        "'ICD-10-Code', 'ICD-10', or 'ICD-10-Num'."
    )


def prepare_labels(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Map ICD classes to numeric indices.

    Priority is:
    - 'ICD-10-Code'
    - 'ICD-10'
    - 'ICD-10-Num'

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple[np.ndarray, dict[str, int], list[str]]: Numeric labels,
        class-to-index mapping, and sorted class names.
    """
    code_column = detect_code_column(df)
    labels_raw = df[code_column].astype(str)

    classes = sorted(labels_raw.unique().tolist())
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    y = labels_raw.map(class_to_index).to_numpy(dtype=np.int64)

    return y, class_to_index, classes


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 128,
) -> np.ndarray:
    """
    Calculate sentence embeddings once for all input texts.

    Args:
        model (SentenceTransformer): Embedding model.
        texts (list[str]): Input texts.
        batch_size (int): Encoding batch size.

    Returns:
        np.ndarray: Embedding matrix with shape [number of texts, embedding size].
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
        device=model.device,
    )

    return embeddings.astype(np.float32)


def make_torch_datasets(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create train and validation TensorDatasets.

    Args:
        x (np.ndarray): Embedding matrix.
        y (np.ndarray): Numeric labels.
        val_fraction (float): Fraction of data used for validation.
        seed (int): Random seed for the split.

    Returns:
        tuple[TensorDataset, TensorDataset]: Training and validation datasets.
    """
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)

    if len(dataset) < 10:
        return dataset, dataset

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset


class EmbeddingClassifier(nn.Module):
    """
    Simple multilayer perceptron classifier for embeddings.

    Args:
        input_dim (int): Embedding dimension.
        num_classes (int): Number of target classes.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Class logits.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_one_trial(
    trial: optuna.Trial,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    input_dim: int,
    num_classes: int,
) -> float:
    """
    Train the classifier with one sampled Optuna hyperparameter combination.

    The returned score is the weighted F1 score on the validation split.

    Args:
        trial (optuna.Trial): Optuna trial object.
        train_ds (TensorDataset): Training dataset.
        val_ds (TensorDataset): Validation dataset.
        input_dim (int): Embedding dimension.
        num_classes (int): Number of classes.

    Returns:
        float: Best validation weighted F1 score for this trial.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", HIDDEN_CHOICES)
    dropout = trial.suggest_float("dropout", DROPOUT_RANGE[0], DROPOUT_RANGE[1])
    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    lr = trial.suggest_float("lr", LR_LOG_RANGE[0], LR_LOG_RANGE[1], log=True)
    weight_decay = trial.suggest_float(
        "weight_decay",
        WEIGHT_DECAY_RANGE[0],
        WEIGHT_DECAY_RANGE[1],
    )
    epochs = trial.suggest_int("epochs", EPOCHS_RANGE[0], EPOCHS_RANGE[1])

    device = torch.device("cuda" if USE_GPU else "cpu")

    model = EmbeddingClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    y_train = torch.tensor([label.item() for _, label in train_ds], dtype=torch.long)
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_weighted_f1 = 0.0
    patience = 3
    no_improve = 0

    for epoch in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        predictions_all: List[int] = []
        labels_all: List[int] = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                predictions = logits.argmax(dim=1).cpu().numpy().tolist()

                predictions_all.extend(predictions)
                labels_all.extend(yb.numpy().tolist())

        val_weighted_f1 = f1_score(
            labels_all,
            predictions_all,
            average="weighted",
        )
        trial.report(val_weighted_f1, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_weighted_f1 > best_weighted_f1 + 1e-6:
            best_weighted_f1 = val_weighted_f1
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= patience:
                break

    del model

    if USE_GPU:
        torch.cuda.empty_cache()

    return best_weighted_f1


def run_optuna_for_embedding_model(
    emb_model_name: str,
    texts: List[str],
    y: np.ndarray,
    out_root: Path,
) -> Dict[str, object]:
    """
    Run Optuna tuning for one embedding model.

    Steps:
    - calculate frozen embeddings once
    - create train and validation datasets
    - run Optuna hyperparameter tuning
    - save best parameters and all trial results

    Args:
        emb_model_name (str): SentenceTransformer model name.
        texts (list[str]): Input texts.
        y (np.ndarray): Numeric labels.
        out_root (Path): Root output directory.

    Returns:
        dict[str, object]: Best tuning result and metadata.
    """
    print(f"\n=== Starting model: {emb_model_name} ===")

    device = "cuda" if USE_GPU else "cpu"
    sentence_model = SentenceTransformer(
        emb_model_name,
        device=device,
        trust_remote_code=True,
    )

    x = encode_texts(
        sentence_model,
        texts,
        batch_size=ENCODE_BATCH_SIZE,
    )
    input_dim = x.shape[1]
    num_classes = int(y.max()) + 1

    train_ds, val_ds = make_torch_datasets(x, y, VAL_FRACTION, SEED)

    model_dir = out_root / sanitize_name(emb_model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    )
    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=sanitize_name(emb_model_name),
    )

    def objective(trial: optuna.Trial) -> float:
        return train_one_trial(
            trial,
            train_ds,
            val_ds,
            input_dim,
            num_classes,
        )

    study.optimize(
        objective,
        n_trials=N_TRIALS_PER_MODEL,
        gc_after_trial=True,
    )

    best_params = study.best_params if study.best_trial else {}
    best_value = study.best_value if study.best_trial else None

    val_accuracy = evaluate_best_on_val(
        best_params,
        train_ds,
        val_ds,
        input_dim,
        num_classes,
    )

    payload = {
        "embedding_model": emb_model_name,
        "best_params": best_params,
        "best_val_weighted_f1": best_value,
        "val_accuracy_at_best": val_accuracy,
        "direction": "maximize",
        "notes": (
            "Classifier on frozen embeddings. "
            "Optimization metric is weighted F1."
        ),
        "embedding_dim": input_dim,
        "num_classes": num_classes,
        "n_samples_total": len(texts),
        "val_fraction": VAL_FRACTION,
    }

    with open(
        model_dir / "best_hyperparameter_combination_optuna.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    try:
        trials_df = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "state",
                "params",
                "user_attrs",
                "system_attrs",
            )
        )
        trials_df.to_csv(model_dir / "optuna_trials.csv", index=False)
    except Exception as error:
        print(
            f"[WARNING] Could not save trials dataframe for "
            f"{emb_model_name}: {error}"
        )

    del sentence_model

    if USE_GPU:
        torch.cuda.empty_cache()

    return payload


def evaluate_best_on_val(
    best_params: Dict[str, object],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    input_dim: int,
    num_classes: int,
) -> float:
    """
    Train the classifier once with the best parameters and report accuracy.

    Args:
        best_params (dict[str, object]): Best Optuna hyperparameters.
        train_ds (TensorDataset): Training dataset.
        val_ds (TensorDataset): Validation dataset.
        input_dim (int): Embedding dimension.
        num_classes (int): Number of classes.

    Returns:
        float: Validation accuracy after training with the best parameters.
    """
    if not best_params:
        return float("nan")

    device = torch.device("cuda" if USE_GPU else "cpu")

    model = EmbeddingClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=int(best_params["hidden_dim"]),
        dropout=float(best_params["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
    )

    batch_size = int(best_params["batch_size"])
    epochs = int(best_params["epochs"])

    y_train = torch.tensor([label.item() for _, label in train_ds], dtype=torch.long)
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    for _ in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions_all: List[int] = []
    labels_all: List[int] = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()

            predictions_all.extend(predictions)
            labels_all.extend(yb.numpy().tolist())

    return float(accuracy_score(labels_all, predictions_all))


def main() -> None:
    """Load data, run Optuna for all embedding models, and save results."""
    set_all_seeds(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataframe(RESULTS_PATH)
    code_column = detect_code_column(df_raw)

    df_icd, _ = create_icd_dataframe(
        data=df_raw,
        label_column="Label",
        code_column=code_column,
    )

    texts = df_icd["Label"].astype(str).tolist()
    y = df_icd["ICD-10-Num"].to_numpy(dtype=np.int64)

    summary_rows: List[Dict[str, object]] = []

    for embedding_model in list_of_embedding_models:
        try:
            result = run_optuna_for_embedding_model(
                embedding_model,
                texts,
                y,
                OUT_ROOT,
            )
            summary_rows.append(
                {
                    "embedding_model": embedding_model,
                    "best_val_weighted_f1": result.get("best_val_weighted_f1"),
                    "val_accuracy_at_best": result.get("val_accuracy_at_best"),
                    **{
                        f"best_{key}": value
                        for key, value in result.get("best_params", {}).items()
                    },
                }
            )
        except Exception as error:
            print(f"[ERROR] Model {embedding_model} failed: {error}")
            summary_rows.append(
                {
                    "embedding_model": embedding_model,
                    "error": str(error),
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        if "best_val_weighted_f1" in summary_df.columns:
            summary_df = summary_df.sort_values(
                by="best_val_weighted_f1",
                ascending=False,
                na_position="last",
            )

        summary_path = OUT_ROOT / "summary_best_params_all_models.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()