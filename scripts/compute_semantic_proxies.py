import argparse
import json
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, pipeline


def read_csv_with_texts(path: str) -> Tuple[pd.DataFrame, Union[str, int], Union[str, int]]:
    """Read results CSV and return dataframe plus columns for original and rewritten texts.
    Tries header first; falls back to positional columns.
    """
    # Try with header
    try:
        df = pd.read_csv(path)
        if {"original_text", "rewritten_text"}.issubset(df.columns):
            return df, "original_text", "rewritten_text"
    except Exception:
        pass

    # Fallback: assume no header and fixed positions (3=original, 4=rewritten)
    df = pd.read_csv(path, header=None)
    # Basic sanity: ensure at least 5 columns
    if df.shape[1] < 5:
        raise ValueError(f"CSV at {path} does not have expected columns; got shape {df.shape}")
    return df, 3, 4


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def compute_embeddings(texts, tokenizer, model, device: str = "cpu") -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])  # [batch, hidden]
    return pooled.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_norm * b_norm).sum(axis=1)


def process_dataset(csv_path: str, limit: int, device: str) -> dict:
    df, oc, rc = read_csv_with_texts(csv_path)
    df = df.dropna(subset=[oc, rc])
    if limit and limit > 0:
        df = df.head(limit)

    originals = df[oc].astype(str).tolist()
    rewrites = df[rc].astype(str).tolist()

    # Embedding similarity (RoBERTa-base, mean pooled)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base").to(device)
    emb_o = compute_embeddings(originals, tokenizer, model, device)
    emb_r = compute_embeddings(rewrites, tokenizer, model, device)
    sims = cosine_similarity(emb_o, emb_r)

    # Label invariance (sentiment) on a small subset for speed
    n_label = min(50, len(originals))
    invariance = None
    try:
        senti = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        )
        ori_labels = [senti(originals[i])[0]["label"] for i in range(n_label)]
        rew_labels = [senti(rewrites[i])[0]["label"] for i in range(n_label)]
        invariance = float(np.mean(np.array(ori_labels) == np.array(rew_labels))) if n_label > 0 else None
    except Exception as e:
        invariance = None

    return {
        "embedding_similarity_roberta_mean": float(np.mean(sims)),
        "embedding_similarity_roberta_median": float(np.median(sims)),
        "label_invariance_rewrite": invariance,
        "sample_size": int(len(df)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute semantic preservation proxies (embedding similarity, label invariance)")
    parser.add_argument("--existing_csv", default="data/existing_datasets_test/existing_datasets_test_results.csv")
    parser.add_argument("--new_csv", default="data/new_scraped_dataset_test/new_scraped_dataset_test_results.csv")
    parser.add_argument("--limit", type=int, default=100, help="Max rows to process per dataset (to keep runtime reasonable)")
    parser.add_argument("--out", default="data/semantic_proxies.json", help="Output JSON path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    try:
        results["existing"] = process_dataset(args.existing_csv, args.limit, device)
    except Exception as e:
        results["existing_error"] = str(e)

    try:
        results["new"] = process_dataset(args.new_csv, args.limit, device)
    except Exception as e:
        results["new_error"] = str(e)

    # Write JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()