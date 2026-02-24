#!/usr/bin/env python
"""
Analyze DPS grid search results to find best hyperparameters.

This script loads results from all parameter combinations and generates:
- Summary table comparing all combinations
- Heatmaps showing performance across parameter space
- Rankings by different metrics (recall, QPS, recall-QPS tradeoff)

Usage:
    python analyze_grid_search.py --grid-dir search/results/dps_grid_10k_fine
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def load_grid_results(grid_dir: str) -> Tuple[Dict, List[Dict]]:
    """
    Load all results from grid search directory.
    
    Args:
        grid_dir: Path to grid search base directory
        
    Returns:
        Tuple of (summary dict, list of result dicts)
    """
    summary_path = os.path.join(grid_dir, 'grid_search_summary.json')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Grid search summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    results = []
    for combo in summary['combinations']:
        if combo['status'] != 'success':
            continue
            
        result_path = os.path.join(combo['output_dir'], 'dps_benchmark_results.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_data = json.load(f)
                result_data['temperature'] = combo['temperature']
                result_data['prob_threshold'] = combo['prob_threshold']
                results.append(result_data)
    
    return summary, results


def create_comparison_table(results: List[Dict], k_value: int = 100) -> pd.DataFrame:
    """
    Create comparison table for all parameter combinations.
    
    Args:
        results: List of result dictionaries
        k_value: k value to compare (default: 100)
        
    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    
    for result in results:
        temp = result['temperature']
        prob_thresh = result['prob_threshold']
        
        dps_data = result['dps'].get(str(k_value), {})
        baseline = result['baseline'].get(str(k_value), {})
        
        if not dps_data or not baseline:
            continue
        
        row = {
            'temperature': temp,
            'prob_threshold': prob_thresh,
            'avg_recall': dps_data.get('avg_recall', 0),
            'avg_qps': dps_data.get('qps', 0),
            'avg_probes': dps_data.get('avg_probes', 0),
            'std_probes': dps_data.get('std_probes', 0),
            'speedup_vs_exhaustive': dps_data.get('qps', 0) / baseline.get('qps', 1) if baseline.get('qps', 0) > 0 else 0,
            'avg_candidates': dps_data.get('avg_candidates', 0)
        }
        
        # Calculate recall-speed score (harmonic mean)
        recall = row['avg_recall']
        speedup = row['speedup_vs_exhaustive']
        if recall > 0 and speedup > 0:
            row['f1_score'] = 2 * (recall * speedup) / (recall + speedup)
        else:
            row['f1_score'] = 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values('f1_score', ascending=False)


def plot_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Create heatmaps for different metrics.
    
    Args:
        df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get unique temperature and prob_threshold values
    temps = sorted(df['temperature'].unique())
    probs = sorted(df['prob_threshold'].unique())
    
    metrics = {
        'avg_recall': 'Average Recall',
        'avg_qps': 'Average QPS',
        'speedup_vs_exhaustive': 'Speedup vs Exhaustive',
        'avg_probes': 'Average Number of Probes',
        'f1_score': 'F1 Score (Recall-Speed Tradeoff)'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, title) in enumerate(metrics.items()):
        if idx >= len(axes):
            break
            
        # Create pivot table for heatmap
        pivot = df.pivot(index='prob_threshold', columns='temperature', values=metric)
        
        # Plot heatmap
        ax = axes[idx]
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': metric})
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Probability Threshold', fontsize=12)
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'grid_search_heatmaps.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Heatmaps saved to: {plot_path}")
    plt.close()


def plot_pareto_frontier(df: pd.DataFrame, output_dir: str):
    """
    Plot Pareto frontier of recall vs QPS.
    
    Args:
        df: DataFrame with comparison results
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(df['avg_recall'], df['avg_qps'], 
                        c=df['f1_score'], s=100, cmap='viridis', 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1 Score', fontsize=12)
    
    # Annotate top 5 points
    top_5 = df.nlargest(5, 'f1_score')
    for _, row in top_5.iterrows():
        ax.annotate(f"T={row['temperature']:.2f}\nP={row['prob_threshold']:.2f}",
                   xy=(row['avg_recall'], row['avg_qps']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Average Recall', fontsize=14)
    ax.set_ylabel('Average QPS', fontsize=14)
    ax.set_title('DPS Hyperparameter Performance: Recall vs Speed', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pareto_frontier.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {plot_path}")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary statistics and rankings."""
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    print(f"Total parameter combinations tested: {len(df)}")
    print(f"Temperature range: {df['temperature'].min():.4f} - {df['temperature'].max():.4f}")
    print(f"Prob threshold range: {df['prob_threshold'].min():.4f} - {df['prob_threshold'].max():.4f}")
    
    print("\n" + "-"*80)
    print("TOP 5 BY F1 SCORE (Balanced Recall-Speed Tradeoff)")
    print("-"*80)
    top_5_f1 = df.nlargest(5, 'f1_score')
    print(top_5_f1[['temperature', 'prob_threshold', 'avg_recall', 'avg_qps', 
                     'speedup_vs_exhaustive', 'avg_probes', 'f1_score']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("TOP 5 BY RECALL")
    print("-"*80)
    top_5_recall = df.nlargest(5, 'avg_recall')
    print(top_5_recall[['temperature', 'prob_threshold', 'avg_recall', 'avg_qps', 
                        'speedup_vs_exhaustive', 'avg_probes']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("TOP 5 BY QPS (Speed)")
    print("-"*80)
    top_5_qps = df.nlargest(5, 'avg_qps')
    print(top_5_qps[['temperature', 'prob_threshold', 'avg_recall', 'avg_qps', 
                     'speedup_vs_exhaustive', 'avg_probes']].to_string(index=False))
    
    print("\n" + "-"*80)
    print("STATISTICS SUMMARY")
    print("-"*80)
    stats = df[['avg_recall', 'avg_qps', 'speedup_vs_exhaustive', 'avg_probes', 'f1_score']].describe()
    print(stats)
    print("="*80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze DPS grid search results')
    parser.add_argument('--grid-dir', type=str, required=True,
                       help='Path to grid search base directory')
    parser.add_argument('--k-value', type=int, default=100,
                       help='k value to analyze (default: 100)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: grid-dir/analysis)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.grid_dir, 'analysis')
    
    print(f"Loading grid search results from: {args.grid_dir}")
    summary, results = load_grid_results(args.grid_dir)
    
    print(f"Loaded {len(results)} successful runs")
    
    if not results:
        print("No results found!")
        return 1
    
    # Create comparison table
    print(f"Creating comparison table for k={args.k_value}")
    df = create_comparison_table(results, k_value=args.k_value)
    
    # Save comparison table
    table_path = os.path.join(args.output_dir, f'comparison_table_k{args.k_value}.csv')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False, float_format='%.4f')
    print(f"Comparison table saved to: {table_path}")
    
    # Print summary
    print_summary(df)
    
    # Create visualizations
    print("Creating heatmaps...")
    plot_heatmaps(df, args.output_dir)
    
    print("Creating Pareto frontier plot...")
    plot_pareto_frontier(df, args.output_dir)
    
    print(f"\nAll analysis saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
