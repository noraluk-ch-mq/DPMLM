import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_loglog(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    # guard against non-positive values
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return None, None, None
    lx = np.log(x)
    ly = np.log(y)
    # linear regression in log space
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    # R^2
    y_pred = slope * lx + intercept
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, intercept, r2


def main():
    parser = argparse.ArgumentParser(description="Fit log-log time vs length per category and plot")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "new_scraped_dataset_test", "new_scraped_dataset_test_results.csv"),
        help="Path to new dataset results CSV",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("figures", "length_time_per_category_fit.pdf"),
        help="Output figure path (.pdf or .png)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if not {"category", "text_length", "processing_time"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: category, text_length, processing_time")

    cats = sorted(df["category"].dropna().unique().tolist())

    ncols = 3
    nrows = int(np.ceil(len(cats) / ncols))
    plt.figure(figsize=(ncols * 4.0, nrows * 3.5))

    summary = {}
    for idx, cat in enumerate(cats, 1):
        sub = df[df["category"] == cat]
        x = sub["text_length"].values
        y = sub["processing_time"].values
        slope, intercept, r2 = fit_loglog(x, y)
        summary[cat] = {"slope": slope, "r2": r2, "n": len(x)}
        ax = plt.subplot(nrows, ncols, idx)
        ax.scatter(x, y, s=8, alpha=0.6, label=f"{cat} data")
        ax.set_title(f"{cat} (n={len(x)})")
        ax.set_xlabel("Length (chars)")
        ax.set_ylabel("Time (s)")
        # plot fit curve in original space if available
        if slope is not None:
            xx = np.linspace(max(1, np.min(x)), np.max(x), 100)
            yy = np.exp(slope * np.log(xx) + intercept)
            ax.plot(xx, yy, color="orange", lw=2, label=f"fit slope={slope:.2f}, R²={r2:.2f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    # also emit png for convenience
    base, ext = os.path.splitext(args.out)
    plt.savefig(base + ".png")

    # write summary to JSON for report references
    import json
    json_path = os.path.join("data", "statistical_tests", "length_time_fit_summary.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", args.out, "and", base + ".png")
    print("Summary:", json_path)


if __name__ == "__main__":
    main()