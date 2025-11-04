import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def save_plot(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print(f"Saved: {path}")
    # Also save a PDF variant for LaTeX (XeLaTeX) compatibility
    try:
        if name.lower().endswith('.png'):
            pdf_name = name[:-4] + '.pdf'
            pdf_path = os.path.join(FIG_DIR, pdf_name)
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Saved: {pdf_path}")
    except Exception as e:
        print(f"Warning: could not save PDF variant for {name}: {e}")

def plot_existing_change_vs_epsilon():
    csv_path = os.path.join(ROOT, 'data', 'existing_datasets_test', 'existing_datasets_test_results.csv')
    df = pd.read_csv(csv_path)
    # Aggregate mean change by dataset and epsilon
    agg = df.groupby(['dataset', 'epsilon'])['change_percentage'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, sub in agg.groupby('dataset'):
        sub = sub.sort_values('epsilon')
        ax.plot(sub['epsilon'], sub['change_percentage'], marker='o', label=dataset)

    ax.set_title('Existing Datasets: Average Change vs Epsilon')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('average change (%)')
    ax.legend(title='dataset')
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'existing_change_vs_epsilon.png')

def plot_new_change_vs_epsilon():
    csv_path = os.path.join(ROOT, 'data', 'new_scraped_dataset_test', 'new_scraped_dataset_test_results.csv')
    df = pd.read_csv(csv_path)
    # Aggregate mean by category and epsilon
    agg = df.groupby(['category', 'epsilon'])['change_percentage'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for category, sub in agg.groupby('category'):
        sub = sub.sort_values('epsilon')
        ax.plot(sub['epsilon'], sub['change_percentage'], marker='o', linewidth=2, markersize=6, label=category)

    ax.set_title('New Data: Average Change vs Epsilon')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('average change (%)')
    ax.legend(title='category')
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'new_change_vs_epsilon.png')

def plot_new_change_vs_epsilon_facet():
    """Facet plot per category to magnify small variations across epsilon."""
    csv_path = os.path.join(ROOT, 'data', 'new_scraped_dataset_test', 'new_scraped_dataset_test_results.csv')
    df = pd.read_csv(csv_path)

    grouped = df.groupby(['category', 'epsilon'])['change_percentage']
    agg = grouped.agg(['mean', 'std', 'count']).reset_index()
    agg['ci95'] = agg.apply(lambda r: (1.96 * r['std'] / np.sqrt(r['count'])) if r['count'] > 1 and pd.notnull(r['std']) else 0.0, axis=1)

    cats = sorted(agg['category'].unique())
    n = len(cats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.6), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    eps_sorted = sorted(df['epsilon'].unique())
    for i, cat in enumerate(cats):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        sub = agg[agg['category'] == cat].sort_values('epsilon')
        ax.errorbar(sub['epsilon'], sub['mean'], yerr=sub['ci95'], marker='o', capsize=3, label=cat, color='#1f77b4')
        # tighten y-limits for visibility
        ymin = (sub['mean'] - sub['ci95']).min()
        ymax = (sub['mean'] + sub['ci95']).max()
        pad = max(0.5, (ymax - ymin) * 0.15)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(cat)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(eps_sorted)

    # remove unused axes if any
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r, c])

    fig.suptitle('New Data: Average Change vs Epsilon by Category (95% CI)', fontsize=13)
    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel('epsilon')
    axes[0,0].set_ylabel('average change (%)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, 'new_change_vs_epsilon_facet.png')

def plot_success_rate_by_category():
    import json
    
    # Load data from JSON report
    json_path = os.path.join(ROOT, 'data', 'new_scraped_dataset_test', 'new_scraped_dataset_test_report.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract category-wise success rates
    category_stats = data['category_wise_stats']
    
    # Get categories and their success rates
    categories = []
    success_rates = []
    
    for cat, stats in category_stats.items():
        categories.append(cat.title())  # Capitalize category names
        success_rates.append(stats['success_rate'])
    
    # Sort by success rate for better visualization
    sorted_data = sorted(zip(categories, success_rates), key=lambda x: x[1])
    categories, success_rates = zip(*sorted_data)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on success rate (red for low, green for high)
    colors = ['#d62728' if rate < 50 else '#ff7f0e' if rate < 90 else '#2ca02c' for rate in success_rates]
    
    bars = ax.bar(categories, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('New Data: Success Rate by Text Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Text Category', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_plot(fig, 'success_rate_by_category.png')

def plot_processing_time_vs_length():
    csv_path = os.path.join(ROOT, 'data', 'new_scraped_dataset_test', 'new_scraped_dataset_test_results.csv')
    df = pd.read_csv(csv_path)
    # Drop rows with zero time/length to avoid skew
    df = df[(df['processing_time'] > 0) & (df['text_length'] > 0)]
    r = np.corrcoef(df['text_length'], df['processing_time'])[0, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['text_length'], df['processing_time'], alpha=0.4, s=18)
    ax.set_title(f'New Data: Processing Time vs Text Length (r={r:.3f})')
    ax.set_xlabel('text length (chars)')
    ax.set_ylabel('processing time (s)')
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'processing_time_vs_length.png')

def plot_combined_privacy_utility():
    # Existing
    df_e = pd.read_csv(os.path.join(ROOT, 'data', 'existing_datasets_test', 'existing_datasets_test_results.csv'))
    agg_e = df_e.groupby(['epsilon'])['change_percentage'].mean().reset_index()
    agg_e['source'] = 'existing'

    # New
    df_n = pd.read_csv(os.path.join(ROOT, 'data', 'new_scraped_dataset_test', 'new_scraped_dataset_test_results.csv'))
    agg_n = df_n.groupby(['epsilon'])['change_percentage'].mean().reset_index()
    agg_n['source'] = 'new'

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sub in [('existing', agg_e), ('new', agg_n)]:
        sub = sub.sort_values('epsilon')
        ax.plot(sub['epsilon'], sub['change_percentage'], marker='o', label=name)

    ax.set_title('Privacy–Utility Trade-off: Average Change vs Epsilon')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('average change (%)')
    ax.legend(title='source')
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'combined_privacy_utility.png')

def main():
    plot_existing_change_vs_epsilon()
    plot_new_change_vs_epsilon()
    plot_new_change_vs_epsilon_facet()
    plot_success_rate_by_category()
    plot_processing_time_vs_length()
    plot_combined_privacy_utility()

if __name__ == '__main__':
    main()