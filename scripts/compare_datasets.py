#!/usr/bin/env python3

"""
Compare DP-MLM performance on existing UCI datasets vs new scraped datasets.

Inputs (expected to exist):
- data/existing_datasets_test/existing_datasets_test_results.csv
- data/existing_datasets_test/existing_datasets_test_report.json
- data/new_scraped_dataset_test/new_scraped_dataset_test_results.csv
- data/new_scraped_dataset_test/new_scraped_dataset_test_report.json
- data/new_scraped_dataset_test/data_quality_report.json (optional)

Outputs:
- figures/comparison_change_dist.png
- figures/epsilon_vs_change_overlay.png
- figures/length_time_existing_vs_new.png
- figures/lexical_diversity_comparison.png
- figures/readability_comparison.png
- data/comparative_analysis.json

This script computes additional metrics:
- Readability (Flesch Reading Ease) and lexical diversity for both datasets
- Correlations: epsilon vs change %, text length vs processing time
- Error breakdown for new dataset
"""

import os
import json
import math
import statistics
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = Path("figures")
DATA_DIR = Path("data")


class DataQualityAnalyzer:
    """Lightweight text quality analyzer (heuristic syllable counting)."""

    VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

    @staticmethod
    def calculate(text: str):
        if not isinstance(text, str) or not text:
            return {
                "word_count": 0,
                "flesch_reading_ease": 0.0,
                "lexical_diversity": 0.0,
                "sentence_count": 0,
                "avg_sentence_length": 0.0,
            }

        words = re.findall(r"\w+", text.lower())
        word_count = len(words)

        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        if word_count == 0 or sentence_count == 0:
            flesch_score = 0.0
            avg_sentence_length = 0.0
        else:
            syllables = 0
            for w in words:
                matches = DataQualityAnalyzer.VOWEL_RE.findall(w)
                syllables += len(matches)
                if w.endswith("e") and not w.endswith("le"):
                    syllables -= 1
                if syllables == 0:
                    syllables = 1
            avg_sentence_length = word_count / sentence_count
            try:
                flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count)
            except ZeroDivisionError:
                flesch_score = 0.0

        lexical_diversity = (len(set(words)) / word_count) if word_count > 0 else 0.0
        return {
            "word_count": word_count,
            "flesch_reading_ease": float(flesch_score),
            "lexical_diversity": float(lexical_diversity),
            "sentence_count": sentence_count,
            "avg_sentence_length": float(avg_sentence_length),
        }


def safe_mean(series):
    s = [x for x in series if pd.notnull(x)]
    return float(np.mean(s)) if s else 0.0


def load_existing():
    csv_path = DATA_DIR / "existing_datasets_test" / "existing_datasets_test_results.csv"
    json_path = DATA_DIR / "existing_datasets_test" / "existing_datasets_test_report.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    meta = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


def load_new():
    csv_path = DATA_DIR / "new_scraped_dataset_test" / "new_scraped_dataset_test_results.csv"
    json_path = DATA_DIR / "new_scraped_dataset_test" / "new_scraped_dataset_test_report.json"
    qual_path = DATA_DIR / "new_scraped_dataset_test" / "data_quality_report.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    meta = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    qual = {}
    if qual_path.exists():
        with open(qual_path, "r", encoding="utf-8") as f:
            qual = json.load(f)
    return df, meta, qual


def add_quality_metrics_existing(df):
    analyzer = DataQualityAnalyzer()
    metrics = df["original_text"].fillna("").apply(analyzer.calculate)
    dfq = pd.DataFrame(list(metrics))
    for col in dfq.columns:
        df[f"quality_{col}"] = dfq[col]
    return df


def compute_correlations(df, label):
    result = {"label": label}
    # epsilon vs change
    try:
        eps = df["epsilon"].astype(float)
        chg = df["change_percentage"].astype(float)
        result["epsilon_change_pearson"] = float(np.corrcoef(eps, chg)[0, 1]) if len(eps) > 1 else 0.0
    except Exception:
        result["epsilon_change_pearson"] = 0.0

    # length vs time
    try:
        ln = df["text_length"].astype(float)
        tm = df["processing_time"].astype(float)
        result["length_time_pearson"] = float(np.corrcoef(ln, tm)[0, 1]) if len(ln) > 1 else 0.0
    except Exception:
        result["length_time_pearson"] = 0.0

    # lexical diversity vs change (when available)
    if "quality_lexical_diversity" in df.columns:
        try:
            ld = df["quality_lexical_diversity"].astype(float)
            chg = df["change_percentage"].astype(float)
            result["lexicaldiv_change_pearson"] = float(np.corrcoef(ld, chg)[0, 1]) if len(ld) > 1 else 0.0
        except Exception:
            result["lexicaldiv_change_pearson"] = 0.0
    return result


def make_figures(existing_df, new_df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1) Change % distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(existing_df[existing_df["success"] == True]["change_percentage"], label="Existing", linewidth=2)
    sns.kdeplot(new_df[new_df["success"] == True]["change_percentage"], label="New", linewidth=2)
    plt.xlabel("Change (%)")
    plt.ylabel("Density")
    plt.title("Distribution of Change Percentage")
    plt.legend()
    out1 = FIG_DIR / "comparison_change_dist.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    try:
        plt.savefig(out1.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
    except Exception as e:
        print(f"Warning: PDF save failed for {out1}: {e}")
    plt.close()

    # 2) Epsilon vs average change overlay
    plt.figure(figsize=(10, 6))
    ex_eps = existing_df.groupby("epsilon")["change_percentage"].mean()
    new_eps = new_df.groupby("epsilon")["change_percentage"].mean()
    plt.plot(ex_eps.index, ex_eps.values, "o-", label="Existing")
    plt.plot(new_eps.index, new_eps.values, "o-", label="New")
    plt.xscale("log")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Average Change (%)")
    plt.title("Privacy–Utility: ε vs Avg Change (Overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out2 = FIG_DIR / "epsilon_vs_change_overlay.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    try:
        plt.savefig(out2.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
    except Exception as e:
        print(f"Warning: PDF save failed for {out2}: {e}")
    plt.close()

    # 3) Text length vs processing time (both datasets)
    plt.figure(figsize=(10, 6))
    plt.scatter(existing_df["text_length"], existing_df["processing_time"], alpha=0.5, label="Existing")
    plt.scatter(new_df["text_length"], new_df["processing_time"], alpha=0.5, label="New")
    plt.xlabel("Text Length (chars)")
    plt.ylabel("Processing Time (s)")
    plt.title("Text Length vs Processing Time")
    plt.legend()
    out3 = FIG_DIR / "length_time_existing_vs_new.png"
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    try:
        plt.savefig(out3.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
    except Exception as e:
        print(f"Warning: PDF save failed for {out3}: {e}")
    plt.close()

    # 4) Lexical diversity comparison (boxplots)
    plt.figure(figsize=(10, 6))
    ex_ld = existing_df["quality_lexical_diversity"] if "quality_lexical_diversity" in existing_df.columns else pd.Series(dtype=float)
    new_ld = new_df["quality_lexical_diversity"] if "quality_lexical_diversity" in new_df.columns else pd.Series(dtype=float)
    plot_df = pd.DataFrame({
        "lexical_diversity": pd.concat([ex_ld, new_ld], ignore_index=True),
        "dataset": (['Existing'] * len(ex_ld)) + (['New'] * len(new_ld))
    }) if len(ex_ld) and len(new_ld) else None
    if plot_df is not None and not plot_df.empty:
        sns.boxplot(data=plot_df, x="dataset", y="lexical_diversity")
        plt.title("Lexical Diversity by Dataset")
        out4 = FIG_DIR / "lexical_diversity_comparison.png"
        plt.savefig(out4, dpi=300, bbox_inches="tight")
        try:
            plt.savefig(out4.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
        except Exception as e:
            print(f"Warning: PDF save failed for {out4}: {e}")
        plt.close()
    else:
        out4 = None

    # 5) Readability comparison (Flesch)
    plt.figure(figsize=(10, 6))
    ex_fr = existing_df["quality_flesch_reading_ease"] if "quality_flesch_reading_ease" in existing_df.columns else pd.Series(dtype=float)
    new_fr = new_df["quality_flesch_reading_ease"] if "quality_flesch_reading_ease" in new_df.columns else pd.Series(dtype=float)
    plot_df2 = pd.DataFrame({
        "flesch_reading_ease": pd.concat([ex_fr, new_fr], ignore_index=True),
        "dataset": (['Existing'] * len(ex_fr)) + (['New'] * len(new_fr))
    }) if len(ex_fr) and len(new_fr) else None
    if plot_df2 is not None and not plot_df2.empty:
        sns.boxplot(data=plot_df2, x="dataset", y="flesch_reading_ease")
        plt.title("Readability (Flesch) by Dataset")
        out5 = FIG_DIR / "readability_comparison.png"
        plt.savefig(out5, dpi=300, bbox_inches="tight")
        try:
            plt.savefig(out5.with_suffix('.pdf'), format='pdf', bbox_inches="tight")
        except Exception as e:
            print(f"Warning: PDF save failed for {out5}: {e}")
        plt.close()
    else:
        out5 = None

    return {
        "comparison_change_dist": str(out1),
        "epsilon_vs_change_overlay": str(out2),
        "length_time_existing_vs_new": str(out3),
        "lexical_diversity_comparison": str(out4) if out4 else None,
        "readability_comparison": str(out5) if out5 else None,
    }


def main():
    existing_df, existing_meta = load_existing()
    new_df, new_meta, quality_meta = load_new()

    # Clean basic types
    for df in (existing_df, new_df):
        if "success" in df.columns:
            df["success"] = df["success"].astype(bool)
        for col in ("change_percentage", "processing_time", "text_length", "epsilon"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Augment with quality metrics
    existing_df = add_quality_metrics_existing(existing_df)

    # For new_df, infer quality metrics if not already present
    # new_df has only test results; derive quality metrics from original_text
    new_df = add_quality_metrics_existing(new_df)

    # Compute correlations
    corr_existing = compute_correlations(existing_df, label="existing")
    corr_new = compute_correlations(new_df, label="new")

    # Aggregate stats
    ex_success = existing_df["success"].mean() * 100.0 if "success" in existing_df.columns else None
    new_success = new_df["success"].mean() * 100.0 if "success" in new_df.columns else None

    summary = {
        "success_rate": {
            "existing": float(ex_success) if ex_success is not None else None,
            "new": float(new_success) if new_success is not None else None,
        },
        "avg_change": {
            "existing": safe_mean(existing_df["change_percentage"]).__float__(),
            "new": safe_mean(new_df["change_percentage"]).__float__(),
        },
        "avg_processing_time": {
            "existing": safe_mean(existing_df["processing_time"]).__float__(),
            "new": safe_mean(new_df["processing_time"]).__float__(),
        },
        "avg_text_length": {
            "existing": safe_mean(existing_df["text_length"]).__float__(),
            "new": safe_mean(new_df["text_length"]).__float__(),
        },
        "avg_readability_flesch": {
            "existing": safe_mean(existing_df.get("quality_flesch_reading_ease", pd.Series(dtype=float))).__float__(),
            "new": safe_mean(new_df.get("quality_flesch_reading_ease", pd.Series(dtype=float))).__float__(),
        },
        "avg_lexical_diversity": {
            "existing": safe_mean(existing_df.get("quality_lexical_diversity", pd.Series(dtype=float))).__float__(),
            "new": safe_mean(new_df.get("quality_lexical_diversity", pd.Series(dtype=float))).__float__(),
        },
        "correlations": {
            "existing": corr_existing,
            "new": corr_new,
        },
        "notes": {
            "quality_meta_available": bool(quality_meta),
            "existing_meta_available": bool(existing_meta),
            "new_meta_available": bool(new_meta),
        },
    }

    # Figures
    fig_paths = make_figures(existing_df, new_df)
    summary["figures"] = fig_paths

    # Save summary JSON
    out_json = DATA_DIR / "comparative_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Comparative analysis saved:")
    print(f"  {out_json}")
    for name, p in fig_paths.items():
        if p:
            print(f"  {name}: {p}")


if __name__ == "__main__":
    main()