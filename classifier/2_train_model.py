"""Train ICD-10 classifiers using SentenceTransformer embeddings."""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.embedding_models import (
    list_of_embedding_models,
    list_of_input_dim_models,
)
from preprocessing.prep_dataset import create_icd_dataframe

PROJECT_ROOT = Path("/home/alic/RAG")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "classifier/results"

LOG_DIR = RESULTS_DIR / "logs_untrained"
MODEL_DIR = RESULTS_DIR / "embedding_models"

INPUT_CSV = DATA_DIR / "Alpha_ID_dataset.csv"

GPU_DEVICE = torch.device("cuda:0")

def load_model(model_index: int) -> SentenceTransformer:
    """
    Load a selected SentenceTransformer model.

    Args:
        model_index (int): Index of the selected embedding model.

    Returns:
        SentenceTransformer: Loaded SentenceTransformer model.
    """
    if model_index < 0 or model_index >= len(list_of_embedding_models):
        raise ValueError(
            "Invalid model index. Please choose an index within the "
            "available model range."
        )

    model_name = list_of_embedding_models[model_index]

    return SentenceTransformer(model_name, trust_remote_code=True)


def load_input_dim(model_index: int) -> int:
    """
    Return the input dimension for the selected embedding model.

    Args:
        model_index (int): Index of the selected embedding model.

    Returns:
        int: Input dimension of the selected model.
    """
    return list_of_input_dim_models[model_index]


class ClassificationDataset(Dataset):
    """
    PyTorch dataset for ICD-10 classification.

    The dataset returns only text and labels. Embeddings are calculated
    batch-wise in the collate function.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'Label' and
            'ICD-10-Num'.
        model (SentenceTransformer): Embedding model.
        input_dim (int): Embedding input dimension.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        model: SentenceTransformer,
        input_dim: int,
    ) -> None:
        self.texts = dataframe["Label"].tolist()
        self.icd = dataframe["ICD-10-Num"].tolist()
        self.model = model
        self.max_length = input_dim

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one text and its ICD-10 label.

        Embeddings are not calculated here. They are calculated batch-wise
        inside the collate function.
        """
        text = self.texts[idx]
        icd = self.icd[idx]

        return {
            "text": text,
            "input_ids": torch.tensor(icd, dtype=torch.long),
        }


def make_collate_fn(embedding_model: SentenceTransformer):
    """
    Create a collate function that calculates embeddings batch-wise.
    """

    def collate_fn(batch: list[dict]) -> dict:
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["input_ids"] for item in batch])

        embeddings = embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=embedding_model.device,
        ).clone().detach().float()

        return {
            "embedding": embeddings,
            "input_ids": labels,
            "text": texts,
        }

    return collate_fn


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


class CombinedModel(nn.Module):
    """
    Combined embedding and classifier model.

    Args:
        embedding_model (SentenceTransformer): SentenceTransformer model.
        classifier (nn.Module): Trained classifier model.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = classifier

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode texts and classify the resulting embeddings."""
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
        )
        outputs = self.classifier(embeddings)

        return outputs


def init_criterion(
    dataframe: pd.DataFrame,
    icd_unique_labels: list,
) -> nn.CrossEntropyLoss:
    """
    Initialize weighted cross-entropy loss.

    Class weights are based on the ICD-10 class distribution.

    Args:
        dataframe (pd.DataFrame): Training data with 'ICD-10-Num'.
        icd_unique_labels (list): Unique numeric ICD-10 labels.

    Returns:
        nn.CrossEntropyLoss: Weighted loss function.
    """
    class_counts = dataframe["ICD-10-Num"].value_counts()
    class_weights = np.ones(len(icd_unique_labels))
    class_weights[0] = 1

    for class_id in class_counts.index:
        if class_id != 0:
            class_weights[class_id] = (
                class_counts[0] / class_counts[class_id]
            )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            list(class_weights),
            dtype=torch.float32,
        ).to(GPU_DEVICE)
    )

    return criterion


def init_dataloader(
    data: pd.DataFrame,
    embedding_model: SentenceTransformer,
    batch_size: int,
    input_dim: int,
) -> DataLoader:
    """
    Initialize the DataLoader for training.

    Args:
        data (pd.DataFrame): Training dataset.
        embedding_model (SentenceTransformer): Embedding model.
        batch_size (int): Batch size.
        input_dim (int): Embedding input dimension.

    Returns:
        DataLoader: DataLoader using batch-wise embedding calculation.
    """
    dataset = ClassificationDataset(data, embedding_model, input_dim)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(embedding_model),
    )

    return dataloader


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    log_path: str | Path,
    number_to_icd: dict,
) -> tuple[nn.Module, list, list, float]:
    """
    Train the classifier model.

    Args:
        model (nn.Module): Classifier model.
        dataloader (DataLoader): Training DataLoader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        log_path (str | Path): Output path for the training log.
        number_to_icd (dict): Mapping from numeric labels to ICD-10 codes.

    Returns:
        tuple[nn.Module, list, list, float]: Trained model, accuracies,
        losses, and training duration.
    """
    train_losses = []
    train_accuracies = []
    model.train()

    start_time = time.time()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, mode="w", newline="", encoding="utf-8") as logfile:
        writer = csv.writer(logfile, delimiter=";")
        writer.writerow(["Epoch", "Loss", "Accuracy"])

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0

            epoch_pred_path = log_path.with_name(
                f"{log_path.stem}_ep{epoch + 1}_predictions.csv"
            )

            with open(
                epoch_pred_path,
                mode="w",
                newline="",
                encoding="utf-8",
            ) as predfile:
                pred_writer = csv.writer(predfile, delimiter=";")
                pred_writer.writerow(
                    [
                        "Epoch",
                        "Label",
                        "Actual ICD-10-Code",
                        "Predicted ICD-10-Code",
                    ]
                )

                for batch in dataloader:
                    inputs = batch["embedding"].to(GPU_DEVICE)
                    labels = batch["input_ids"].to(GPU_DEVICE)
                    texts = batch["text"]

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    for i, text in enumerate(texts):
                        actual_icd = number_to_icd[labels[i].item()]
                        predicted_icd = number_to_icd[preds[i].item()]
                        pred_writer.writerow(
                            [
                                epoch + 1,
                                text,
                                actual_icd,
                                predicted_icd,
                            ]
                        )

            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Loss: {avg_loss:.4f} - "
                f"Accuracy: {accuracy * 100:.2f}%"
            )
            writer.writerow([epoch + 1, avg_loss, accuracy])

    duration = time.time() - start_time
    print(f"Training duration: {duration:.2f} seconds")

    return model, train_accuracies, train_losses, duration


def log_model_info(
    model_name: str,
    train_losses: list,
    train_accuracies: list,
    lr: float,
    batch_size: int,
    weight_decay: float,
    num_epochs: int,
    training_time: float,
    output_dir: str | Path = LOG_DIR,
) -> None:
    """
    Save training metrics to a CSV file.

    Args:
        model_name (str): Name of the embedding model.
        train_losses (list): Training loss values per epoch.
        train_accuracies (list): Training accuracy values per epoch.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        weight_decay (float): Weight decay value.
        num_epochs (int): Number of epochs.
        training_time (float): Total training time in seconds.
        output_dir (str | Path): Output directory for the log file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{model_name.replace('/', '_')}_ep{num_epochs}_bs{batch_size}.csv"

    df = pd.DataFrame(
        {
            "Epoch": list(range(1, num_epochs + 1)),
            "Loss": train_losses,
            "Accuracy": train_accuracies,
            "LearningRate": [lr] * num_epochs,
            "BatchSize": [batch_size] * num_epochs,
            "WeightDecay": [weight_decay] * num_epochs,
            "Model": [model_name] * num_epochs,
            "TrainingTime": [training_time] * num_epochs,
        }
    )

    df.to_csv(output_dir / file_name, index=False, sep=";")
    print(f"Log saved to: {output_dir / file_name}")


def main_1() -> None:
    """Run ICD-10 classifier training for all configured embedding models."""
    icd_raw_data = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8",
        sep=";",
        engine="python",
    )

    dataset, label_map = create_icd_dataframe(
        icd_raw_data,
        "Label",
        "ICD-10-Code",
    )
    icd_labels = dataset["ICD-10-Num"].unique()

    num_epochs = 70
    lr = 0.00021700394405050138
    batch_size = 16
    weight_decay = 0.0034388521115218396

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for i, model_name in enumerate(list_of_embedding_models):
        print(f"Starting training for model: {model_name}")
        embedding_model = load_model(i)

        try:
            embedding_model.to(GPU_DEVICE)
        except Exception:
            pass

        input_dim = load_input_dim(i)
        dataloader = init_dataloader(
            dataset,
            embedding_model,
            batch_size,
            input_dim,
        )

        classifier = EmbeddingClassifier(
            input_dim,
            len(icd_labels),
        ).to(GPU_DEVICE)

        criterion = init_criterion(dataset, icd_labels)
        optimizer = AdamW(
            classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(timestamp)

        safe_model_name = model_name.replace("/", "_")
        log_path = LOG_DIR / (
            f"{safe_model_name}_ep{num_epochs}_bs{batch_size}.csv"
        )

        trained_model, losses, accuracies, duration = train_model(
            classifier,
            dataloader,
            criterion,
            optimizer,
            num_epochs,
            log_path,
            number_to_icd=label_map,
        )

        model_filename = MODEL_DIR / f"{safe_model_name}_checkpoint.pt"
        combined_model = CombinedModel(embedding_model, trained_model)

        torch.save(
            combined_model.state_dict(),
            MODEL_DIR / f"{safe_model_name}_combined_model.pt",
        )

        torch.save(
            {
                "embedding_model_name": model_name,
                "classifier_state_dict": trained_model.state_dict(),
            },
            model_filename,
        )

        print(f"Model saved to: {model_filename}")

        log_model_info(
            model_name,
            losses,
            accuracies,
            lr,
            batch_size,
            weight_decay,
            num_epochs,
            duration,
        )


if __name__ == "__main__":
    main_1()