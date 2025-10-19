#!/usr/bin/env python
"""
Create plots from benchmark results JSON file.

Usage:
    python plot_results.py [path_to_benchmark_results.json]
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from pathlib import Path

def load_results(json_path):
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_ivf_results(results, output_dir):
    """Plot IVF search results."""
    ivf_results = results.get('ivf_search_results', {})
    flat_results = results.get('flat_search_results', {})
    
    if not ivf_results:
        print("No IVF results found in the JSON file.")
        return
    
    # Get the first method's results
    method = list(ivf_results.keys())[0]
    method_results = ivf_results[method]
    
    # Extract k values and n_probe values
    k_values = sorted([int(k) for k in method_results.keys()])
    n_probe_values = sorted([int(n) for n in method_results[str(k_values[0])].keys()])
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    # Plot 1: QPS vs k for different n_probe values
    for i, n_probe in enumerate(n_probe_values):
        qps_values = [method_results[str(k)][str(n_probe)]['qps'] for k in k_values]
        color = colors[i % len(colors)]
        ax1.plot(k_values, qps_values, 'o-', label=f'IVF (n_probe={n_probe})', 
                color=color, linewidth=2, markersize=6)
    
    # Add flat search baseline if available
    if flat_results and method in flat_results:
        flat_qps = [flat_results[method][str(k)]['qps'] for k in k_values]
        ax1.plot(k_values, flat_qps, 's-', label='Flat Search', 
                color='black', linewidth=3, markersize=8)
    
    ax1.set_xlabel('k (top-k results)')
    ax1.set_ylabel('Queries Per Second (QPS)')
    ax1.set_title('Query Throughput vs k')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Recall vs k for different n_probe values
    for i, n_probe in enumerate(n_probe_values):
        recall_values = [method_results[str(k)][str(n_probe)]['avg_recall'] for k in k_values]
        color = colors[i % len(colors)]
        ax2.plot(k_values, recall_values, 'o-', label=f'IVF (n_probe={n_probe})', 
                color=color, linewidth=2, markersize=6)
    
    # Add flat search baseline (recall = 1.0)
    if flat_results and method in flat_results:
        flat_recall = [1.0] * len(k_values)  # Flat search has perfect recall
        ax2.plot(k_values, flat_recall, 's-', label='Flat Search', 
                color='black', linewidth=3, markersize=8)
    
    ax2.set_xlabel('k (top-k results)')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall vs k')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: QPS vs Recall trade-off (for largest k)
    largest_k = k_values[-1]
    
    # Collect data for curve
    recall_vals = []
    qps_vals = []
    
    for i, n_probe in enumerate(n_probe_values):
        qps = method_results[str(largest_k)][str(n_probe)]['qps']
        recall = method_results[str(largest_k)][str(n_probe)]['avg_recall']
        recall_vals.append(recall)
        qps_vals.append(qps)
        color = colors[i % len(colors)]
        ax3.scatter([recall], [qps], s=100, c=color, marker='s', 
                   label=f'IVF (n_probe={n_probe})')
    
    # Draw curve connecting IVF points
    if len(recall_vals) > 1:
        # Sort by recall for proper curve
        sorted_data = sorted(zip(recall_vals, qps_vals))
        sorted_recalls, sorted_qps = zip(*sorted_data)
        ax3.plot(sorted_recalls, sorted_qps, 'b--', alpha=0.6, linewidth=2, label='IVF Curve')
    
    # Add flat search point if available
    if flat_results and method in flat_results:
        flat_qps = flat_results[method][str(largest_k)]['qps']
        ax3.scatter([1.0], [flat_qps], s=150, c='black', marker='o', 
                   label='Flat Search', edgecolors='white', linewidth=2)
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Queries Per Second (QPS)')
    ax3.set_title(f'QPS vs Recall Trade-off (k={largest_k})')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Speedup vs Recall (if flat results available)
    if flat_results and method in flat_results:
        flat_qps = flat_results[method][str(largest_k)]['qps']
        
        # Collect data for curve
        speedup_recalls = []
        speedups = []
        
        for i, n_probe in enumerate(n_probe_values):
            ivf_qps = method_results[str(largest_k)][str(n_probe)]['qps']
            recall = method_results[str(largest_k)][str(n_probe)]['avg_recall']
            speedup = ivf_qps / flat_qps if flat_qps > 0 else 0
            speedup_recalls.append(recall)
            speedups.append(speedup)
            color = colors[i % len(colors)]
            ax4.scatter([recall], [speedup], s=100, c=color, marker='s', 
                       label=f'IVF (n_probe={n_probe})')
        
        # Draw curve connecting IVF points
        if len(speedup_recalls) > 1:
            # Sort by recall for proper curve
            sorted_data = sorted(zip(speedup_recalls, speedups))
            sorted_recalls, sorted_speedups = zip(*sorted_data)
            ax4.plot(sorted_recalls, sorted_speedups, 'b--', alpha=0.6, linewidth=2, label='IVF Curve')
        
        # Add flat search baseline
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Flat Search (1.0x)')
        ax4.scatter([1.0], [1.0], s=150, c='black', marker='o', 
                   edgecolors='white', linewidth=2, zorder=5)
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Speedup vs Flat Search')
        ax4.set_title(f'Speedup vs Recall Trade-off (k={largest_k})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        # Just show n_probe vs recall if no flat results
        recalls = [method_results[str(largest_k)][str(n_probe)]['avg_recall'] for n_probe in n_probe_values]
        ax4.plot(n_probe_values, recalls, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('n_probe')
        ax4.set_ylabel('Recall')
        ax4.set_title(f'Recall vs n_probe (k={largest_k})')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    dataset_name = results.get('dataset', 'unknown')
    combined_plot_path = os.path.join(output_dir, f'benchmark_results_{dataset_name}.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    
    # Save upper row plot (QPS and Recall vs k)
    fig_upper, (ax1_new, ax2_new) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recreate upper plots
    for i, n_probe in enumerate(n_probe_values):
        qps_values = [method_results[str(k)][str(n_probe)]['qps'] for k in k_values]
        recall_values = [method_results[str(k)][str(n_probe)]['avg_recall'] for k in k_values]
        color = colors[i % len(colors)]
        ax1_new.plot(k_values, qps_values, 'o-', label=f'IVF (n_probe={n_probe})', 
                    color=color, linewidth=2, markersize=6)
        ax2_new.plot(k_values, recall_values, 'o-', label=f'IVF (n_probe={n_probe})', 
                    color=color, linewidth=2, markersize=6)
    
    if flat_results and method in flat_results:
        flat_qps = [flat_results[method][str(k)]['qps'] for k in k_values]
        flat_recall = [1.0] * len(k_values)
        ax1_new.plot(k_values, flat_qps, 's-', label='Flat Search', 
                    color='black', linewidth=3, markersize=8)
        ax2_new.plot(k_values, flat_recall, 's-', label='Flat Search', 
                    color='black', linewidth=3, markersize=8)
    
    ax1_new.set_xlabel('k (top-k results)')
    ax1_new.set_ylabel('Queries Per Second (QPS)')
    ax1_new.set_title('Query Throughput vs k')
    ax1_new.grid(True, alpha=0.3)
    ax1_new.legend()
    
    ax2_new.set_xlabel('k (top-k results)')
    ax2_new.set_ylabel('Recall')
    ax2_new.set_title('Recall vs k')
    ax2_new.grid(True, alpha=0.3)
    ax2_new.legend()
    
    plt.tight_layout()
    upper_plot_path = os.path.join(output_dir, f'benchmark_throughput_recall_{dataset_name}.png')
    plt.savefig(upper_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save lower row plot (Trade-offs)
    fig_lower, (ax3_new, ax4_new) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recreate lower plots
    largest_k = k_values[-1]
    recall_vals = []
    qps_vals = []
    
    for i, n_probe in enumerate(n_probe_values):
        qps = method_results[str(largest_k)][str(n_probe)]['qps']
        recall = method_results[str(largest_k)][str(n_probe)]['avg_recall']
        recall_vals.append(recall)
        qps_vals.append(qps)
        color = colors[i % len(colors)]
        ax3_new.scatter([recall], [qps], s=100, c=color, marker='s', 
                       label=f'IVF (n_probe={n_probe})')
    
    if len(recall_vals) > 1:
        sorted_data = sorted(zip(recall_vals, qps_vals))
        sorted_recalls, sorted_qps = zip(*sorted_data)
        ax3_new.plot(sorted_recalls, sorted_qps, 'b--', alpha=0.6, linewidth=2, label='IVF Curve')
    
    if flat_results and method in flat_results:
        flat_qps = flat_results[method][str(largest_k)]['qps']
        ax3_new.scatter([1.0], [flat_qps], s=150, c='black', marker='o', 
                       label='Flat Search', edgecolors='white', linewidth=2)
        
        # Speedup plot
        speedup_recalls = []
        speedups = []
        
        for i, n_probe in enumerate(n_probe_values):
            ivf_qps = method_results[str(largest_k)][str(n_probe)]['qps']
            recall = method_results[str(largest_k)][str(n_probe)]['avg_recall']
            speedup = ivf_qps / flat_qps if flat_qps > 0 else 0
            speedup_recalls.append(recall)
            speedups.append(speedup)
            color = colors[i % len(colors)]
            ax4_new.scatter([recall], [speedup], s=100, c=color, marker='s', 
                           label=f'IVF (n_probe={n_probe})')
        
        if len(speedup_recalls) > 1:
            sorted_data = sorted(zip(speedup_recalls, speedups))
            sorted_recalls, sorted_speedups = zip(*sorted_data)
            ax4_new.plot(sorted_recalls, sorted_speedups, 'b--', alpha=0.6, linewidth=2, label='IVF Curve')
        
        ax4_new.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Flat Search (1.0x)')
        ax4_new.scatter([1.0], [1.0], s=150, c='black', marker='o', 
                       edgecolors='white', linewidth=2, zorder=5)
        
        ax4_new.set_xlabel('Recall')
        ax4_new.set_ylabel('Speedup vs Flat Search')
        ax4_new.set_title(f'Speedup vs Recall Trade-off (k={largest_k})')
        ax4_new.grid(True, alpha=0.3)
        ax4_new.legend()
    
    ax3_new.set_xlabel('Recall')
    ax3_new.set_ylabel('Queries Per Second (QPS)')
    ax3_new.set_title(f'QPS vs Recall Trade-off (k={largest_k})')
    ax3_new.grid(True, alpha=0.3)
    ax3_new.legend()
    
    plt.tight_layout()
    lower_plot_path = os.path.join(output_dir, f'benchmark_tradeoffs_{dataset_name}.png')
    plt.savefig(lower_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create new plot: QPS and Average Query Time vs Recall
    fig_qps_time, (ax_qps, ax_time) = plt.subplots(1, 2, figsize=(12, 5))
    
    largest_k = k_values[-1]
    recall_vals = []
    qps_vals = []
    avg_time_vals = []
    
    for i, n_probe in enumerate(n_probe_values):
        qps = method_results[str(largest_k)][str(n_probe)]['qps']
        recall = method_results[str(largest_k)][str(n_probe)]['avg_recall']
        avg_time = method_results[str(largest_k)][str(n_probe)]['avg_query_time'] * 1000  # Convert to ms
        
        recall_vals.append(recall)
        qps_vals.append(qps)
        avg_time_vals.append(avg_time)
        
        color = colors[i % len(colors)]
        
        # QPS vs Recall
        ax_qps.scatter([recall], [qps], s=100, c=color, marker='s', 
                      label=f'IVF (n_probe={n_probe})')
        
        # Average Time vs Recall  
        ax_time.scatter([recall], [avg_time], s=100, c=color, marker='s',
                       label=f'IVF (n_probe={n_probe})')
    
    # Draw connecting lines
    if len(recall_vals) > 1:
        sorted_data_qps = sorted(zip(recall_vals, qps_vals))
        sorted_recalls_qps, sorted_qps = zip(*sorted_data_qps)
        ax_qps.plot(sorted_recalls_qps, sorted_qps, 'b--', alpha=0.6, linewidth=2, label='IVF Curve')
        
        sorted_data_time = sorted(zip(recall_vals, avg_time_vals))
        sorted_recalls_time, sorted_times = zip(*sorted_data_time)
        ax_time.plot(sorted_recalls_time, sorted_times, 'b--', alpha=0.6, linewidth=2, label='IVF Curve')
    
    # Add flat search reference points
    if flat_results and method in flat_results:
        flat_qps = flat_results[method][str(largest_k)]['qps']
        flat_time = flat_results[method][str(largest_k)]['avg_query_time'] * 1000  # Convert to ms
        
        ax_qps.scatter([1.0], [flat_qps], s=150, c='black', marker='o', 
                      label='Flat Search', edgecolors='white', linewidth=2)
        ax_time.scatter([1.0], [flat_time], s=150, c='black', marker='o',
                       label='Flat Search', edgecolors='white', linewidth=2)
    
    # Configure QPS plot
    ax_qps.set_xlabel('Recall')
    ax_qps.set_ylabel('Queries Per Second (QPS)')
    ax_qps.set_title(f'QPS vs Recall (k={largest_k})')
    ax_qps.grid(True, alpha=0.3)
    ax_qps.legend()
    
    # Configure time plot  
    ax_time.set_xlabel('Recall')
    ax_time.set_ylabel('Average Query Time (ms)')
    ax_time.set_title(f'Query Time vs Recall (k={largest_k})')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()
    
    plt.tight_layout()
    qps_time_plot_path = os.path.join(output_dir, f'benchmark_qps_time_recall_{dataset_name}.png')
    plt.savefig(qps_time_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.close('all')  # Close the original figure
    
    print(f"Combined plot saved to: {combined_plot_path}")
    print(f"Throughput/Recall plot saved to: {upper_plot_path}")
    print(f"Trade-offs plot saved to: {lower_plot_path}")
    print(f"QPS/Time vs Recall plot saved to: {qps_time_plot_path}")
    
    return combined_plot_path, upper_plot_path, lower_plot_path, qps_time_plot_path

def export_to_csv(results, output_dir):
    """Export benchmark results to CSV files."""
    ivf_results = results.get('ivf_search_results', {})
    flat_results = results.get('flat_search_results', {})
    
    if not ivf_results:
        print("No IVF results found for CSV export.")
        return None, None
    
    method = list(ivf_results.keys())[0]
    method_results = ivf_results[method]
    k_values = sorted([int(k) for k in method_results.keys()])
    n_probe_values = sorted([int(n) for n in method_results[str(k_values[0])].keys()])
    
    # Export 1: Trade-off Analysis (detailed results for all k and n_probe)
    tradeoff_data = []
    
    for k in k_values:
        for n_probe in n_probe_values:
            result = method_results[str(k)][str(n_probe)]
            row = {
                'method': 'IVF',
                'k': k,
                'n_probe': n_probe,
                'qps': round(result['qps'], 2),
                'recall': round(result['avg_recall'], 3),
                'query_time_ms': round(result['avg_query_time'] * 1000, 2)
            }
            
            # Add speedup if flat results available
            if flat_results and method in flat_results:
                flat_qps = flat_results[method][str(k)]['qps']
                row['speedup_vs_flat'] = round(result['qps'] / flat_qps, 2)
            
            tradeoff_data.append(row)
    
    # Add flat search results to trade-off data
    if flat_results and method in flat_results:
        for k in k_values:
            result = flat_results[method][str(k)]
            row = {
                'method': 'Flat',
                'k': k,
                'n_probe': None,
                'qps': round(result['qps'], 2),
                'recall': round(result['avg_recall'], 3),
                'query_time_ms': round(result['avg_query_time'] * 1000, 2),
                'speedup_vs_flat': 1.0
            }
            tradeoff_data.append(row)
    
    # Save trade-off CSV
    dataset_name = results.get('dataset', 'unknown')
    tradeoff_csv_path = os.path.join(output_dir, f'tradeoff_analysis_{dataset_name}.csv')
    
    df_tradeoff = pd.DataFrame(tradeoff_data)
    df_tradeoff.to_csv(tradeoff_csv_path, index=False)
    
    # Export 2: Key Results Summary (best results for each k)
    summary_data = []
    
    # Add flat search results
    if flat_results and method in flat_results:
        for k in k_values:
            result = flat_results[method][str(k)]
            summary_data.append({
                'method': 'Flat Search',
                'k': k,
                'qps': round(result['qps'], 2),
                'recall': round(result['avg_recall'], 3),
                'query_time_ms': round(result['avg_query_time'] * 1000, 2),
                'speedup_vs_flat': 1.0,
                'configuration': 'Exhaustive search'
            })
    
    # Add best IVF results (n_probe with recall >= 0.95 and highest QPS)
    for k in k_values:
        best_ivf = None
        best_qps = 0
        
        for n_probe in n_probe_values:
            result = method_results[str(k)][str(n_probe)]
            if result['avg_recall'] >= 0.95 and result['qps'] > best_qps:
                best_qps = result['qps']
                best_ivf = (n_probe, result)
        
        if best_ivf:
            n_probe, result = best_ivf
            speedup = result['qps'] / flat_results[method][str(k)]['qps'] if flat_results else None
            summary_data.append({
                'method': 'IVF Search (Best)',
                'k': k,
                'qps': round(result['qps'], 2),
                'recall': round(result['avg_recall'], 3),
                'query_time_ms': round(result['avg_query_time'] * 1000, 2),
                'speedup_vs_flat': round(speedup, 2) if speedup else None,
                'configuration': f'n_probe={n_probe}'
            })
    
    # Save summary CSV
    summary_csv_path = os.path.join(output_dir, f'key_results_summary_{dataset_name}.csv')
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_csv_path, index=False)
    
    print(f"Trade-off analysis CSV saved to: {tradeoff_csv_path}")
    print(f"Key results summary CSV saved to: {summary_csv_path}")
    
    return tradeoff_csv_path, summary_csv_path

def print_performance_summary(results):
    """Print performance comparison summary."""
    ivf_results = results.get('ivf_search_results', {})
    flat_results = results.get('flat_search_results', {})
    
    print("\n=== PERFORMANCE ANALYSIS ===")
    print(f"Dataset: {results.get('dataset', 'unknown')}")
    print(f"Data size: {results.get('data_size', 'unknown'):,} molecules")
    print(f"Queries: {results.get('n_queries', 'unknown')}")
    print(f"Fingerprint: {results['parameters']['fp_type']} ({results['parameters']['fp_size']} bits)")
    
    if not ivf_results:
        print("No IVF results found.")
        return
    
    method = list(ivf_results.keys())[0]
    method_results = ivf_results[method]
    k_values = sorted([int(k) for k in method_results.keys()])
    
    print(f"\n--- IVF Search Performance ({method.upper()}) ---")
    
    for k in k_values:
        print(f"\nk={k}:")
        n_probe_results = method_results[str(k)]
        
        for n_probe in sorted([int(n) for n in n_probe_results.keys()]):
            result = n_probe_results[str(n_probe)]
            qps = result['qps']
            recall = result['avg_recall']
            avg_time = result['avg_query_time']
            
            if flat_results and method in flat_results:
                flat_qps = flat_results[method][str(k)]['qps']
                speedup = qps / flat_qps
                print(f"  n_probe={n_probe:2d}: {qps:6.1f} QPS, Recall={recall:.3f}, "
                      f"Time={avg_time*1000:.2f}ms, Speedup={speedup:.1f}x")
            else:
                print(f"  n_probe={n_probe:2d}: {qps:6.1f} QPS, Recall={recall:.3f}, "
                      f"Time={avg_time*1000:.2f}ms")
    
    if flat_results and method in flat_results:
        print(f"\n--- Flat Search Performance ({method.upper()}) ---")
        for k in k_values:
            result = flat_results[method][str(k)]
            qps = result['qps']
            recall = result['avg_recall']
            avg_time = result['avg_query_time']
            print(f"k={k:3d}: {qps:6.1f} QPS, Recall={recall:.3f}, Time={avg_time*1000:.2f}ms")
        
        print(f"\n--- Best IVF Performance vs Flat ---")
        for k in k_values:
            flat_qps = flat_results[method][str(k)]['qps']
            
            # Find best IVF result (highest QPS with recall >= 0.95)
            best_ivf = None
            best_speedup = 0
            
            for n_probe in sorted([int(n) for n in method_results[str(k)].keys()]):
                result = method_results[str(k)][str(n_probe)]
                if result['avg_recall'] >= 0.95:
                    speedup = result['qps'] / flat_qps
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_ivf = (n_probe, result)
            
            if best_ivf:
                n_probe, result = best_ivf
                print(f"k={k:3d}: Best IVF (n_probe={n_probe}) = {result['qps']:.1f} QPS, "
                      f"Recall={result['avg_recall']:.3f}, Speedup={best_speedup:.1f}x")
            else:
                print(f"k={k:3d}: No IVF result with recall >= 0.95")
    else:
        print("\n⚠️  No flat search results found for comparison!")
        print("   Run the full benchmark to get baseline flat search performance:")
        print("   python search/improved_benchmark.py --dataset data/chembl_33_np.smi --methods rdkit --verbose")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = 'search/results/data/benchmark_results.json'
    
    if not os.path.exists(json_path):
        print(f"Error: Benchmark results file not found: {json_path}")
        print("Usage: python plot_results.py [path_to_benchmark_results.json]")
        return 1
    
    # Load results
    print(f"Loading benchmark results from: {json_path}")
    results = load_results(json_path)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(json_path), '..', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    try:
        plot_paths = plot_ivf_results(results, output_dir)
        csv_paths = export_to_csv(results, output_dir)
        print_performance_summary(results)

        print(f"\nAnalysis complete!")
        print(f"Plots saved:")
        if isinstance(plot_paths, tuple):
            if len(plot_paths) == 4:
                combined_plot, upper_plot, lower_plot, qps_time_plot = plot_paths
                print(f"   - Combined: {combined_plot}")
                print(f"   - Throughput/Recall: {upper_plot}")
                print(f"   - Trade-offs: {lower_plot}")
                print(f"   - QPS/Time vs Recall: {qps_time_plot}")
            else:
                combined_plot, upper_plot, lower_plot = plot_paths
                print(f"   - Combined: {combined_plot}")
                print(f"   - Throughput/Recall: {upper_plot}")
                print(f"   - Trade-offs: {lower_plot}")
        else:
            print(f"   - {plot_paths}")

        if csv_paths and csv_paths[0]:
            print(f"CSV files saved:")
            print(f"   - Trade-off analysis: {csv_paths[0]}")
            print(f"   - Key results summary: {csv_paths[1]}")

    except Exception as e:
        print(f"Error generating plots/CSV: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())