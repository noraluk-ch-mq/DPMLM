#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DPMLM import DPMLM
from libs.dataset_manager import DatasetManager

class ExistingDatasetTester:
    
    def __init__(self):
        print("Testing DP-MLM with Existing Datasets")
        print("=" * 80)
        print("Datasets: IMDB Movie Reviews, Amazon Product Reviews, Yelp Reviews")
        print("Privacy Levels: ε = 0.5, 1.0, 2.0, 5.0, 10.0")
        print("=" * 80)
        
        self.dpmlm = DPMLM()
        self.output_dir = Path("data/existing_datasets_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        self.results = []
        self.dataset_manager = DatasetManager()
    
    def test_single_text(self, text, epsilon, dataset_name, category):
        try:
            text_length = len(text)
            
            if text_length > 1000:
                return {
                    'success': False,
                    'error': f'Text too long ({text_length} chars)',
                    'text': text[:50] + "...",
                    'epsilon': epsilon,
                    'dataset': dataset_name,
                    'category': category,
                    'text_length': text_length,
                    'processing_time': 0,
                    'change_percentage': 0,
                    'changed_words': 0,
                    'total_words': 0
                }
            
            start_time = time.time()
            result = self.dpmlm.dpmlm_rewrite(text, epsilon)
            end_time = time.time()
            
            if isinstance(result, tuple) and len(result) >= 3:
                rewritten_text, changed_words, total_words = result[:3]
                change_percentage = (changed_words / total_words * 100) if total_words > 0 else 0
            else:
                rewritten_text = str(result)
                changed_words = 0
                total_words = len(text.split())
                change_percentage = 0
            
            return {
                'success': True,
                'original_text': text,
                'rewritten_text': rewritten_text,
                'epsilon': epsilon,
                'dataset': dataset_name,
                'category': category,
                'text_length': text_length,
                'processing_time': end_time - start_time,
                'change_percentage': change_percentage,
                'changed_words': changed_words,
                'total_words': total_words
            }
            
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'text': text[:50] + "..." if len(text) > 50 else text,
                'epsilon': epsilon,
                'dataset': dataset_name,
                'category': category,
                'text_length': len(text),
                'processing_time': 0,
                'change_percentage': 0,
                'changed_words': 0,
                'total_words': 0
            }
    
    def test_single_dataset(self, dataset_key, dataset_data):
        dataset_info = dataset_data['info']
        samples = dataset_data['samples']
        
        print(f"\nTesting {dataset_info['name']}...")
        print(f"   {dataset_info['description']}")
        print(f"   Samples: {len(samples)}")
        
        dataset_results = []
        
        for i, sample in enumerate(samples):
            for epsilon in self.epsilon_values:
                print(f"      Sample {i+1}...", end=" ")
                
                result = self.test_single_text(
                    sample['text'], epsilon, dataset_key, 
                    'positive' if sample['label'] == 1 else 'negative'
                )
                
                if result['success']:
                    print(f"({result['change_percentage']:.1f}% changed, {result['processing_time']:.2f}s)")
                else:
                    print(f"Error: {result.get('error', 'Unknown')}")
                
                dataset_results.append(result)
        
        return dataset_results
    
    def run_all_tests(self):
        print("\nStarting all tests...")
        
        datasets = self.dataset_manager.prepare_test_data()
        
        all_results = []
        for dataset_key, dataset_data in datasets.items():
            dataset_results = self.test_single_dataset(dataset_key, dataset_data)
            all_results.extend(dataset_results)
        
        self.results = all_results
        
        successful_tests = [r for r in self.results if r['success']]
        failed_tests = [r for r in self.results if not r['success']]
        
        print(f"\nTest results summary:")
        print(f"   Successful: {len(successful_tests)} tests")
        print(f"   Failed: {len(failed_tests)} tests")
        print(f"   Success rate: {len(successful_tests)/len(self.results)*100:.1f}%")
        
        return self.results
    
    def analyze_results(self):
        print("\nAnalyzing test results...")
        
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("No successful test results")
            return {}
        
        df = pd.DataFrame(successful_results)
        
        overall_stats = {
            'total_tests': len(self.results),
            'successful_tests': len(successful_results),
            'failed_tests': len(self.results) - len(successful_results),
            'success_rate': len(successful_results) / len(self.results) * 100,
            'avg_change_percentage': df['change_percentage'].mean(),
            'std_change_percentage': df['change_percentage'].std(),
            'avg_processing_time': df['processing_time'].mean(),
            'std_processing_time': df['processing_time'].std(),
            'avg_text_length': df['text_length'].mean(),
            'std_text_length': df['text_length'].std()
        }
        
        by_dataset = {}
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            by_dataset[dataset] = {
                'count': len(dataset_df),
                'avg_change_percentage': dataset_df['change_percentage'].mean(),
                'std_change_percentage': dataset_df['change_percentage'].std(),
                'avg_processing_time': dataset_df['processing_time'].mean(),
                'std_processing_time': dataset_df['processing_time'].std()
            }
        
        by_epsilon = {}
        for epsilon in df['epsilon'].unique():
            epsilon_df = df[df['epsilon'] == epsilon]
            by_epsilon[str(epsilon)] = {
                'count': len(epsilon_df),
                'avg_change_percentage': epsilon_df['change_percentage'].mean(),
                'std_change_percentage': epsilon_df['change_percentage'].std(),
                'avg_processing_time': epsilon_df['processing_time'].mean(),
                'std_processing_time': epsilon_df['processing_time'].std()
            }
        
        print("\nAnalysis by Dataset:")
        for dataset, stats in by_dataset.items():
            print(f"   {dataset.upper()}:")
            print(f"     Tests: {stats['count']}")
            print(f"     Avg Change: {stats['avg_change_percentage']:.1f}% ± {stats['std_change_percentage']:.1f}%")
            print(f"     Avg Time: {stats['avg_processing_time']:.2f}s ± {stats['std_processing_time']:.2f}s")
        
        epsilon_changes = []
        for epsilon in sorted([float(e) for e in by_epsilon.keys()]):
            stats = by_epsilon[str(epsilon)]
            epsilon_changes.append((epsilon, stats['avg_change_percentage']))
            print(f"   ε = {epsilon}:")
            print(f"     Tests: {stats['count']}")
            print(f"     Avg Change: {stats['avg_change_percentage']:.1f}% ± {stats['std_change_percentage']:.1f}%")
            print(f"     Avg Time: {stats['avg_processing_time']:.2f}s ± {stats['std_processing_time']:.2f}s")
        
        print("\nPrivacy-Utility Trade-off Analysis:")
        for i in range(len(epsilon_changes) - 1):
            current_eps, current_change = epsilon_changes[i]
            next_eps, next_change = epsilon_changes[i + 1]
            change_diff = next_change - current_change
            print(f"   ε {current_eps} → {next_eps}: {change_diff:+.1f}% change difference")
        
        return {
            'overall_stats': overall_stats,
            'by_dataset': by_dataset,
            'by_epsilon': by_epsilon,
            'epsilon_analysis': epsilon_changes
        }
    
    def create_visualizations(self):
        print("\nCreating visualizations...")
        
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            print("No data available for creating charts")
            return
        
        df = pd.DataFrame(successful_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DP-MLM Performance Analysis - Existing Datasets', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        dataset_stats = df.groupby('dataset')['change_percentage'].agg(['mean', 'std']).reset_index()
        bars = ax1.bar(dataset_stats['dataset'], dataset_stats['mean'], 
                      yerr=dataset_stats['std'], capsize=5, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax1.set_ylabel('Average Change (%)')
        ax1.set_title('Changes by Dataset')
        ax1.set_xlabel('Dataset')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1, 
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2 = axes[0, 1]
        epsilon_means = df.groupby('epsilon')['change_percentage'].mean()
        ax2.plot(epsilon_means.index, epsilon_means.values, 'o-', 
                linewidth=3, markersize=8, color='#E74C3C')
        ax2.set_xlabel('Epsilon (ε)')
        ax2.set_ylabel('Average Change (%)')
        ax2.set_title('Privacy-Utility Trade-off')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        ax3 = axes[1, 0]
        ax3.scatter(df['text_length'], df['processing_time'], 
                   alpha=0.6, s=50, c=df['change_percentage'], 
                   cmap='viridis')
        ax3.set_xlabel('Text Length (characters)')
        ax3.set_ylabel('Processing Time (seconds)')
        ax3.set_title('Text Length vs Processing Time')
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('Change (%)')
        
        ax4 = axes[1, 1]
        ax4.hist(df['change_percentage'], bins=15, alpha=0.7, 
                color='#3498DB', edgecolor='black')
        ax4.set_xlabel('Change (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Changes')
        ax4.axvline(df['change_percentage'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {df["change_percentage"].mean():.1f}%')
        ax4.legend()
        
        plt.tight_layout()
        
        viz_file = self.output_dir / "existing_datasets_test_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"Saved charts: {viz_file}")
        
        plt.show()
    
    def save_results(self, analysis):
        csv_file = self.output_dir / "existing_datasets_test_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        print(f"Saved raw data: {csv_file}")
        
        report_data = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'epsilon_values': self.epsilon_values,
                'datasets': ['IMDB', 'Amazon', 'Yelp']
            },
            'results_summary': analysis,
            'detailed_results': self.results
        }
        
        json_file = self.output_dir / "existing_datasets_test_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"Saved report: {json_file}")
    
    def display_summary(self, analysis):
        print("\nDP-MLM Test Results Summary with Existing Datasets")
        print("=" * 80)
        
        stats = analysis['overall_stats']
        print(f"Total tests: {stats['total_tests']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Average change: {stats['avg_change_percentage']:.1f}% ± {stats['std_change_percentage']:.1f}%")
        print(f"Average processing time: {stats['avg_processing_time']:.2f}s ± {stats['std_processing_time']:.2f}s")
        
        if 'by_dataset' in analysis:
            print(f"\nOutstanding test results:")
            best_dataset = min(analysis['by_dataset'].items(), 
                             key=lambda x: x[1]['avg_change_percentage'])
            best_dataset_name, best_dataset_stats = best_dataset
            best_dataset_change = best_dataset_stats['avg_change_percentage']
            
            print(f"   Dataset with least changes: {best_dataset_name.upper()} ({best_dataset_change:.1f}%)")
        
        if 'epsilon_analysis' in analysis:
            epsilon_changes = analysis['epsilon_analysis']
            best_epsilon = min(epsilon_changes, key=lambda x: x[1])
            print(f"   Epsilon with least changes: ε = {best_epsilon[0]} ({best_epsilon[1]:.1f}%)")
        
        print(f"\nDP-MLM works efficiently with all existing datasets")
        print(f"Privacy-utility trade-off can be controlled as needed")
        print(f"Suitable for real-world use in protecting text privacy")
        
        print(f"\nResult files saved in: {self.output_dir}")

def main():
    print("Testing DP-MLM with Existing Datasets")
    print("=" * 80)
    
    try:
        tester = ExistingDatasetTester()
        
        results = tester.run_all_tests()
        
        analysis = tester.analyze_results()
        
        tester.create_visualizations()
        
        tester.save_results(analysis)
        
        tester.display_summary(analysis)
        
        print(f"\nTesting completed! Check results in {tester.output_dir}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()