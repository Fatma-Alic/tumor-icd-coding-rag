"""Run ICD-10 semantic similarity search with ChromaDB and embeddings."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
import pandas as pd
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path("/home/alic/RAG")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.embedding_models import list_of_embedding_models  # noqa: E402


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "similarity_search/results"

KB_FILE = DATA_DIR / "Alpha_ID_dataset.csv"
QRY_FILE = DATA_DIR / "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv"

CHROMA_PATH = RESULTS_DIR / "chromadb_icd10"
OUT_ROOT = RESULTS_DIR / "filtered_embedding_models_icd10"
CACHE_ROOT = RESULTS_DIR / "emb_cache_icd10"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

VAR_NR = "var1"
LIB = "libA"

TARGET_DEVICES = ["cuda:0", "cuda:1", "cuda:2"]
ENCODE_BATCH_SIZE = 128
UPSERT_BATCH_SIZE = 500
N_RESULTS = 15

USE_CACHE = True


def cuda_soft_cleanup() -> None:
    """
    Free unused CPU and GPU memory.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def sanitize_model_name(model_name: str) -> str:
    """
    Replace unsafe characters in the model name.

    Args:
        model_name (str): Original model name.

    Returns:
        str: Filesystem-safe model name.
    """
    return model_name.replace("/", "-").replace("\\", "-").replace(":", "_")


def init_chroma_client() -> chromadb.PersistentClient:
    """
    Create a ChromaDB client.

    Returns:
        chromadb.PersistentClient: ChromaDB client object.
    """
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )


def setup_chroma_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    reset: bool = True,
):
    """
    Create or reset a ChromaDB collection.

    Args:
        client (chromadb.PersistentClient): ChromaDB client.
        collection_name (str): Name of the collection.
        reset (bool): If True, delete the old collection first.

    Returns:
        Collection: ChromaDB collection.
    """
    if reset:
        for collection in client.list_collections():
            if collection.name == collection_name:
                client.delete_collection(collection_name)
                break

    return client.get_or_create_collection(name=collection_name)


def load_icd10_data(filepath: Path) -> pd.DataFrame:
    """
    Load the ICD-10 knowledge base file.

    Args:
        filepath (Path): Path to the knowledge base CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath, sep=";")


def prepare_kb_texts(df: pd.DataFrame, var_nr: str) -> List[str]:
    """
    Prepare text entries from the knowledge base.

    Args:
        df (pd.DataFrame): Knowledge base DataFrame.
        var_nr (str): Variant name that controls text formatting.

    Returns:
        list[str]: Prepared knowledge base texts.
    """
    texts = df["Label"].astype(str).tolist()

    if var_nr == "var1":
        texts = [f"Tumordiagnose: {text}" for text in texts]

    return texts


def load_query_data(filepath: Path) -> pd.DataFrame:
    """
    Load the query file.

    Args:
        filepath (Path): Path to the query CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath, sep=";")


def get_or_compute_embeddings_multi_gpu(
    model: SentenceTransformer,
    pool,
    texts: List[str],
    cache_path: Path,
    batch_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load embeddings from cache or compute them with multi-GPU encoding.

    Args:
        model (SentenceTransformer): Embedding model.
        pool: Multi-process pool for multi-GPU encoding.
        texts (list[str]): Input texts for embedding.
        cache_path (Path): File path for cached embeddings.
        batch_size (int): Number of texts per batch.
        normalize (bool): If True, normalize embeddings.

    Returns:
        np.ndarray: Embedding matrix.
    """
    if USE_CACHE and cache_path.exists():
        return np.load(cache_path)

    embeddings = model.encode_multi_process(
        texts,
        pool,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if USE_CACHE:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings


def upsert_kb(
    collection,
    kb_df: pd.DataFrame,
    kb_texts: List[str],
    kb_emb: np.ndarray,
    batch_size: int = 500,
) -> None:
    """
    Insert the knowledge base into ChromaDB in batches.

    Args:
        collection: ChromaDB collection.
        kb_df (pd.DataFrame): Knowledge base DataFrame.
        kb_texts (list[str]): Knowledge base texts.
        kb_emb (np.ndarray): Knowledge base embeddings.
        batch_size (int): Number of items per upsert batch.
    """
    ids = kb_df["Alpha-ID"].astype(str).tolist()
    metadatas = [
        {"ICD-10-Code": str(code)}
        for code in kb_df["ICD-10-Code"].astype(str).tolist()
    ]

    assert len(kb_texts) == len(ids) == len(metadatas) == kb_emb.shape[0], (
        "Length mismatch in knowledge base data."
    )

    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=kb_texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            embeddings=kb_emb[i : i + batch_size].tolist(),
        )


def query_chroma(
    collection,
    qry_emb: np.ndarray,
    n_results: int = 15,
) -> Dict[str, Any]:
    """
    Query ChromaDB with query embeddings.

    Args:
        collection: ChromaDB collection.
        qry_emb (np.ndarray): Query embeddings.
        n_results (int): Number of results per query.

    Returns:
        dict[str, Any]: Query results.
    """
    return collection.query(
        query_embeddings=qry_emb.tolist(),
        n_results=n_results,
    )


def process_query_results(
    query_df: pd.DataFrame,
    results: Dict[str, Any],
    n_results: int = 15,
) -> pd.DataFrame:
    """
    Add ChromaDB results to the query DataFrame.

    Args:
        query_df (pd.DataFrame): Original query DataFrame.
        results (dict[str, Any]): Results returned by ChromaDB.
        n_results (int): Number of result columns to create.

    Returns:
        pd.DataFrame: DataFrame with result columns.
    """
    query_df = query_df.copy()

    query_df["suggestedIDs"] = results.get("ids", [])
    query_df["suggestedDocuments"] = results.get("documents", [])
    query_df["suggestedMetadata"] = results.get("metadatas", [])
    query_df["distances"] = results.get("distances", [])

    query_df["suggestedMetadata"] = query_df["suggestedMetadata"].apply(
        lambda items: (
            [item.get("ICD-10-Code") for item in items]
            if isinstance(items, list)
            else []
        )
    )

    for i in range(n_results):
        query_df[f"suggestedIDs{i + 1}"] = query_df["suggestedIDs"].apply(
            lambda values: (
                values[i]
                if isinstance(values, list) and len(values) > i
                else None
            )
        )
        query_df[f"suggestedDocuments{i + 1}"] = query_df[
            "suggestedDocuments"
        ].apply(
            lambda values: (
                values[i]
                if isinstance(values, list) and len(values) > i
                else None
            )
        )
        query_df[f"suggestedMetadata{i + 1}"] = query_df[
            "suggestedMetadata"
        ].apply(
            lambda values: (
                values[i]
                if isinstance(values, list) and len(values) > i
                else None
            )
        )
        query_df[f"distances{i + 1}"] = query_df["distances"].apply(
            lambda values: (
                values[i]
                if isinstance(values, list) and len(values) > i
                else None
            )
        )

    return query_df.drop(
        columns=[
            "suggestedIDs",
            "suggestedDocuments",
            "suggestedMetadata",
            "distances",
        ],
        errors="ignore",
    )


def save_results(df: pd.DataFrame, out_file: Path) -> None:
    """
    Save the result DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        out_file (Path): Output CSV file path.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False, sep=";")
    print(f"[OK] Saved file: {out_file}")


def perform_semantic_similarity(
    model_name: str,
    var_nr: str,
    lib: str,
) -> None:
    """
    Run the full semantic similarity workflow for one model.

    Args:
        model_name (str): Name of the embedding model.
        var_nr (str): Variant name for text preparation.
        lib (str): Library name used in the collection name.
    """
    safe_name = sanitize_model_name(model_name)

    kb_df = load_icd10_data(KB_FILE)
    kb_texts = prepare_kb_texts(kb_df, var_nr)

    qry_df = load_query_data(QRY_FILE)
    qry_texts = qry_df["Text extended"].astype(str).tolist()

    client = init_chroma_client()
    collection_name = f"icd10_{var_nr}_{lib}_{safe_name}"
    collection = setup_chroma_collection(client, collection_name, reset=True)

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device="cuda:0",
    )

    pool = None

    try:
        pool = model.start_multi_process_pool(target_devices=TARGET_DEVICES)

        kb_cache = CACHE_ROOT / safe_name / f"kb_{var_nr}.npy"
        kb_emb = get_or_compute_embeddings_multi_gpu(
            model=model,
            pool=pool,
            texts=kb_texts,
            cache_path=kb_cache,
            batch_size=ENCODE_BATCH_SIZE,
            normalize=True,
        )

        print(f"[INFO] Upserting KB: {len(kb_texts):,} documents")
        upsert_kb(
            collection=collection,
            kb_df=kb_df,
            kb_texts=kb_texts,
            kb_emb=kb_emb,
            batch_size=UPSERT_BATCH_SIZE,
        )

        qry_cache = CACHE_ROOT / safe_name / "qry.npy"
        qry_emb = get_or_compute_embeddings_multi_gpu(
            model=model,
            pool=pool,
            texts=qry_texts,
            cache_path=qry_cache,
            batch_size=ENCODE_BATCH_SIZE,
            normalize=True,
        )

    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    print(f"[INFO] Querying: {len(qry_texts):,} queries x top-{N_RESULTS}")
    results = query_chroma(collection, qry_emb, n_results=N_RESULTS)

    out_df = process_query_results(qry_df, results, n_results=N_RESULTS)
    out_file = OUT_ROOT / safe_name / f"results_{safe_name}.csv"

    save_results(out_df, out_file)


def main() -> None:
    """
    Run the semantic similarity workflow for all embedding models.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    for model_name in list_of_embedding_models:
        print(f"\n=== MODEL: {model_name} ===")

        try:
            perform_semantic_similarity(model_name, VAR_NR, LIB)
        except Exception as error:
            print(f"[ERROR] {model_name}: {error}")


if __name__ == "__main__":
    main()