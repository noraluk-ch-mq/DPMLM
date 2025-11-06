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


def process_dataset(csv_path: str, limit: int, device: str, label_n: int = 200, return_samples: bool = False):
    df, oc, rc = read_csv_with_texts(csv_path)
    df = df.dropna(subset=[oc, rc])
    if limit and limit > 0:
        df = df.head(limit)

    originals = df[oc].astype(str).tolist()
    rewrites = df[rc].astype(str).tolist()

    # Embedding similarity (RoBERTa-base, mean pooled)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base").to(device)
    model.eval()
    emb_o = compute_embeddings(originals, tokenizer, model, device)
    emb_r = compute_embeddings(rewrites, tokenizer, model, device)
    sims = cosine_similarity(emb_o, emb_r)

    # Label invariance (sentiment) on a subset for speed (configurable)
    n_label = min(label_n, len(originals))
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

    result = {
        "embedding_similarity_roberta_mean": float(np.mean(sims)),
        "embedding_similarity_roberta_median": float(np.median(sims)),
        "embedding_similarity_roberta_std": float(np.std(sims)),
        "label_invariance_rewrite": invariance,
        "label_invariance_sample_size": int(n_label),
        "label_invariance_std_estimate": (float(np.sqrt(invariance * (1 - invariance) / n_label)) if invariance is not None and n_label > 0 else None),
        "sample_size": int(len(df)),
    }
    if return_samples:
        result["_embedding_similarity_samples"] = sims.tolist()
        if invariance is not None:
            flags = (np.array(ori_labels) == np.array(rew_labels)).astype(int).tolist()
            result["_label_invariance_flags"] = flags
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute semantic preservation proxies (embedding similarity, label invariance)")
    parser.add_argument("--existing_csv", default="data/existing_datasets_test/existing_datasets_test_results.csv")
    parser.add_argument("--new_csv", default="data/new_scraped_dataset_test/new_scraped_dataset_test_results.csv")
    parser.add_argument("--limit", type=int, default=200, help="Max rows to process per dataset (to keep runtime reasonable)")
    parser.add_argument("--label_n", type=int, default=200, help="Subset size for label invariance evaluation")
    parser.add_argument("--out", default="data/semantic_proxies.json", help="Output JSON path")
    parser.add_argument("--save_figures", action="store_true", help="Also generate distribution plots for embeddings and label invariance")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reproducibility safeguards
    try:
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except Exception:
        pass

    results = {}
    try:
        results["existing"] = process_dataset(args.existing_csv, args.limit, device, label_n=args.label_n, return_samples=args.save_figures)
    except Exception as e:
        results["existing_error"] = str(e)

    try:
        results["new"] = process_dataset(args.new_csv, args.limit, device, label_n=args.label_n, return_samples=args.save_figures)
    except Exception as e:
        results["new_error"] = str(e)

    # Write JSON
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    # Optional figures: embedding similarity distribution and label invariance distribution
    try:
        if args.save_figures and "existing" in results and "new" in results:
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
            fig_dir = Path("figures")
            fig_dir.mkdir(parents=True, exist_ok=True)
            # Embedding similarity distributions (overlay)
            ex_sims = results["existing"].get("_embedding_similarity_samples")
            new_sims = results["new"].get("_embedding_similarity_samples")
            if ex_sims and new_sims:
                plt.figure(figsize=(10, 6))
                bins = np.linspace(0.5, 1.0, 30)
                plt.hist(ex_sims, bins=bins, alpha=0.6, label="Existing", density=True)
                plt.hist(new_sims, bins=bins, alpha=0.6, label="New", density=True)
                plt.xlabel("Embedding cosine similarity (RoBERTa)")
                plt.ylabel("Density")
                plt.title("Embedding Similarity Distribution: Existing vs New")
                plt.legend()
                out_path = fig_dir / "embedding_similarity_distribution.png"
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                try:
                    plt.savefig(out_path.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
                except Exception:
                    pass
                plt.close()
            # Label invariance flags distribution (unchanged vs changed)
            ex_flags = results["existing"].get("_label_invariance_flags")
            new_flags = results["new"].get("_label_invariance_flags")
            if ex_flags is not None and new_flags is not None:
                plt.figure(figsize=(8, 5))
                counts = {
                    "Existing": {
                        "unchanged": int(np.sum(np.array(ex_flags) == 1)),
                        "changed": int(np.sum(np.array(ex_flags) == 0)),
                    },
                    "New": {
                        "unchanged": int(np.sum(np.array(new_flags) == 1)),
                        "changed": int(np.sum(np.array(new_flags) == 0)),
                    },
                }
                labels = ["Unchanged", "Changed"]
                x = np.arange(len(labels))
                width = 0.35
                ex_vals = [counts["Existing"]["unchanged"], counts["Existing"]["changed"]]
                new_vals = [counts["New"]["unchanged"], counts["New"]["changed"]]
                plt.bar(x - width/2, ex_vals, width, label="Existing", alpha=0.7)
                plt.bar(x + width/2, new_vals, width, label="New", alpha=0.7)
                plt.xticks(x, labels)
                plt.ylabel("Count")
                plt.title("Label Invariance (sentiment) Distribution: Existing vs New")
                plt.legend()
                out2 = fig_dir / "label_invariance_distribution.png"
                plt.savefig(out2, dpi=300, bbox_inches="tight")
                try:
                    plt.savefig(out2.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
                except Exception:
                    pass
                plt.close()
    except Exception as e:
        print(f"Warning: could not generate semantic distribution figures: {e}")


if __name__ == "__main__":
    main()