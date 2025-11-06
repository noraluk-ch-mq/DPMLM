import os
import argparse
import json
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "pos"
    elif c <= -0.05:
        return "neg"
    else:
        return "neu"


def main():
    parser = argparse.ArgumentParser(description="Quick sentiment check using VADER for Reviews/Social")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "new_scraped_dataset_test", "new_scraped_dataset_test_results.csv"),
        help="Path to new dataset results CSV",
    )
    parser.add_argument("--per_cat", type=int, default=10, help="Number of samples per category")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    df = pd.read_csv(args.csv)
    required = {"category", "original_text", "rewritten_text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")

    # Normalize category casing/whitespace (lowercase)
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    cats = ["reviews", "social"]
    rows = []
    for cat in cats:
        sub_all = df[df["category"] == cat]
        n_pick = min(args.per_cat, len(sub_all))
        sub = sub_all.sample(n=n_pick, random_state=args.seed) if n_pick > 0 else sub_all
        invariances = []
        total_n = 0
        for _, r in sub.iterrows():
            lo = sia.polarity_scores(str(r["original_text"]))["compound"]
            lw = sia.polarity_scores(str(r["rewritten_text"]))["compound"]
            invariances.append(1 if label_from_compound(lo) == label_from_compound(lw) else 0)
            total_n += 1
        if total_n:
            p_hat = float(np.mean(invariances))
            # Bootstrap 95% CI
            rng = np.random.default_rng(42)
            B = 5000
            boots = [np.mean(rng.choice(invariances, size=total_n, replace=True)) for _ in range(B)]
            ci_low = float(np.percentile(boots, 2.5))
            ci_high = float(np.percentile(boots, 97.5))
        else:
            p_hat, ci_low, ci_high = 0.0, 0.0, 0.0
        rows.append({
            "category": cat,
            "n": total_n,
            "invariance": p_hat,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    out_json = os.path.join("data", "semantic_proxies", "human_eval_vader_reviews_social.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    # Write LaTeX table
    disp = {"reviews": "Reviews", "social": "Social"}
    lines = [
        '\\begin{table}[H]',
        '  \\centering',
        '  \\caption{Sentiment invariance (VADER) on small samples}',
        '  \\small',
        '  \\begin{tabular}{lrr}',
        '    \\toprule',
        '    Category & n & \\% [95\\% CI] \\\\',
        '    \\midrule',
    ]
    for r in rows:
        cat_name = disp.get(r["category"], r["category"].capitalize())
        percent = r["invariance"] * 100.0
        ci_low = r["ci_low"] * 100.0
        ci_high = r["ci_high"] * 100.0
        lines.append(f'    {cat_name} & {r["n"]} & {percent:.1f} [{ci_low:.1f}, {ci_high:.1f}] \\\\')
    lines += [
        '    \\bottomrule',
        '  \\end{tabular}',
        '\\end{table}',
    ]

    tex_path = os.path.join("report", "human_eval_reviews_social_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    print("Saved:", out_json)
    print("Table:", tex_path)


if __name__ == "__main__":
    main()