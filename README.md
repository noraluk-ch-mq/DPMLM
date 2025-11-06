# DP-MLM

Code​‍​‌‍​‍‌ repository for the ACL Findings paper: "DP-MLM: Differentially Private Text Rewriting Using Masked Language ​‍​‌‍​‍‌Models".

## Setup

- Python 3.10+ (the code has been tested on macOS with Python 3.13.4).
- To install necessary package dependencies, run the following command:
 `pip install -r requirements.txt`.
- Resources:
 - WordNet 2022: `python -m wn download oewn:2022`.
 - NLTK VADER (for super-fast sentiment checks):
 `python -c "import nltk; nltk.download('vader_lexicon')"`.
 - Optional GPU acceleration (PyTorch + CUDA) for embedding computations.

## File Overview and ​‍​‌‍​‍‌Commands

-​‍​‌‍​‍‌ `test_existing_datasets.py` — Execute DP-MLM on various standard datasets and save the results in `data/existing_datasets_test/`.
- Run:
 ```bash
 python test_existing_datasets.py
 ```
- `test_new_scraped_dataset.py` — Fetch data (Quotes/News/Social) and evaluate DP-MLM, writing results under `data/new_scraped_dataset_test/`.
- Run:
 ```bash
 python test_new_scraped_dataset.py
 ```
- `scripts/compute_semantic_proxies.py` — Create semantic proxies (embedding similarity, sentiment label invariance).
- Run:
 ```bash
 python scripts/compute_semantic_proxies.py
 ```
- `scripts/compare_datasets.py` — Compare existing vs new datasets; produce summary metrics and figures.
- Run:
 ```bash
 python scripts/compare_datasets.py --existing data/existing_datasets_test/existing_datasets_test_results.csv --new data/new_scraped_dataset_test/new_scraped_dataset_test_results.csv
 ```
- `scripts/length_time_fit.py` — Fit and visualize processing time vs writing length per category.
- Run:
 ```bash
 python scripts/length_time_fit.py
 ```
- `scripts/human_eval_vader.py` — Quick sentiment invariance evaluation for Reviews/Social; outputs table for the report.
- Run:
 ```bash
 python scripts/human_eval_vader.py
 ```
- `scripts/ablation_long_texts.py` — Ablation study for long-text constraints; outputs JSON and LaTeX table.
- Run:
 ```bash
 python scripts/ablation_long_texts.py
 ```
- `scripts/generate_appendix_examples.py` — Generate qualitative appendix (LaTeX) from selected examples.
- Run:
 ```bash
 python scripts/generate_appendix_examples.py
 ```
- `libs/dataset_manager.py` — Dataset loading/preparation utilities used by tests and scripts.
- `libs/scraper.py` — Web scraping utilities to build the new dataset.
- `libs/utils.py` — General helpers (text processing, constraints, metrics, ​‍​‌‍​‍‌I/O).

-​‍​‌‍​‍‌ `data/` — data in CSV/JSON/TeX formats that are used by the scripts and the report.
- `figures/` — figures that have been generated and are included in the report.
- `requirements.txt` — Python libraries and packages required.
- `.gitignore` — Git ignored ​‍​‌‍​‍‌files/directories.
