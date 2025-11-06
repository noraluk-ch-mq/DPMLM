import os
import argparse
import json
import random
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DPMLM import DPMLM


def run_dpmlm(text, epsilon=1.0, cfg=None):
    if cfg is None:
        cfg = {
            "FILTER": True,
            "POS": True,
            "STOP": False,
            "TEMP": True,
            "CONCAT": True,
        }
    model = DPMLM()
    try:
        rewritten, perturbed, total = model.dpmlm_rewrite(
            text,
            epsilon,
            REPLACE=False,
            FILTER=cfg.get("FILTER", True),
            STOP=cfg.get("STOP", False),
            TEMP=cfg.get("TEMP", True),
            POS=cfg.get("POS", True),
            CONCAT=cfg.get("CONCAT", True),
        )
        success = int(perturbed > 0)
        change_ratio = float(perturbed) / float(total) if total > 0 else 0.0
        return {"ok": True, "success": success, "perturbed": perturbed, "total": total, "change_ratio": change_ratio}
    except Exception as e:
        return {"ok": False, "error": str(e), "success": 0, "perturbed": 0, "total": 0, "change_ratio": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Ablation on constraints for long texts")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "new_scraped_dataset_test", "new_scraped_dataset_test_results.csv"),
        help="Path to new dataset results CSV",
    )
    parser.add_argument("--per_cat", type=int, default=10, help="Number of long samples per category")
    parser.add_argument("--min_length", type=int, default=600, help="Minimum char length to consider long")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    need_cols = {"category", "original_text", "text_length", "epsilon"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(need_cols)}")

    cats = sorted(df["category"].dropna().unique().tolist())

    ablations = {
        "base": {"FILTER": True, "POS": True, "STOP": False, "TEMP": True, "CONCAT": True},
        "no_POS": {"FILTER": True, "POS": False, "STOP": False, "TEMP": True, "CONCAT": True},
        "no_FILTER": {"FILTER": False, "POS": True, "STOP": False, "TEMP": True, "CONCAT": True},
        "stop_included": {"FILTER": True, "POS": True, "STOP": True, "TEMP": True, "CONCAT": True},
        "no_CONCAT": {"FILTER": True, "POS": True, "STOP": False, "TEMP": True, "CONCAT": False},
    }

    results = {name: {} for name in ablations.keys()}

    for cat in cats:
        long_df = df[(df["category"] == cat) & (df["text_length"] >= args.min_length)]
        if long_df.empty:
            continue
        # sample
        sample = long_df.sample(n=min(args.per_cat, len(long_df)), random_state=args.seed)
        texts = sample["original_text"].tolist()
        epsilons = sample["epsilon"].tolist()

        for ab_name, cfg in ablations.items():
            succ = 0
            total_n = 0
            mean_change = []
            for t, eps in zip(texts, epsilons):
                r = run_dpmlm(t, epsilon=eps, cfg=cfg)
                total_n += 1
                succ += (1 if r.get("success", 0) == 1 else 0)
                mean_change.append(r.get("change_ratio", 0.0))
            rate = (succ / total_n) if total_n > 0 else 0.0
            results[ab_name][cat] = {
                "n": total_n,
                "success_rate": rate,
                "avg_change_ratio": float(np.mean(mean_change)) if mean_change else 0.0,
            }

    out_json = os.path.join("data", "statistical_tests", "ablation_long_texts.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # also emit a small LaTeX table for quick inclusion
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Ablation & n & Success(%) & Avg change \\ ", "\\midrule"]
    for ab_name, by_cat in results.items():
        total_n = sum(v["n"] for v in by_cat.values())
        avg_succ = np.mean([v["success_rate"] for v in by_cat.values()]) if by_cat else 0.0
        avg_change = np.mean([v["avg_change_ratio"] for v in by_cat.values()]) if by_cat else 0.0
        lines.append(f"{ab_name} & {total_n} & {avg_succ*100:.1f} & {avg_change:.3f} \\ ")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path = os.path.join("report", "ablation_long_texts_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print("Saved:", out_json)
    print("Table:", tex_path)


if __name__ == "__main__":
    main()