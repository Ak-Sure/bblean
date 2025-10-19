#!/usr/bin/env python3
"""
Plot comparison between different dataset sizes (10K, 100K, 1M) for BitBirch performance analysis.
Creates visualizations showing scalability trends and performance differences.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_benchmark_data(base_path):
    """Load benchmark data for all three datasets."""
    datasets = ['subset_10k', 'subset_100k', 'subset_1M']
    data = {}
    
    for dataset in datasets:
        result_path = os.path.join(base_path, dataset, 'data', 'benchmark_results.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data[dataset] = json.load(f)
            print(f"Loaded data for {dataset}: {data[dataset]['data_size']:,} molecules")
        else:
            print(f"Warning: {result_path} not found")
    
    return data

def extract_performance_metrics(data):
    """Extract key performance metrics for comparison."""
    metrics = {}
    
    for dataset_name, dataset_data in data.items():
        dataset_size = dataset_data['data_size']
        
        # Extract IVF results
        ivf_results = dataset_data.get('ivf_search_results', {})
        flat_results = dataset_data.get('flat_search_results', {})
        
        if 'rdkit' in ivf_results and 'rdkit' in flat_results:
            method_results = ivf_results['rdkit']
            flat_method_results = flat_results['rdkit']
            
            # Get k=100 results for consistency
            k_str = '100'
            if k_str in method_results and k_str in flat_method_results:
                ivf_k_results = method_results[k_str]
                flat_k_result = flat_method_results[k_str]
                
                # Extract metrics for different n_probe values
                n_probe_metrics = {}
                for n_probe_str, result in ivf_k_results.items():
                    n_probe = int(n_probe_str)
                    n_probe_metrics[n_probe] = {
                        'qps': result['qps'],
                        'recall': result['avg_recall'],
                        'avg_time_ms': result['avg_query_time'] * 1000,
                        'speedup': result['qps'] / flat_k_result['qps'] if flat_k_result['qps'] > 0 else 0
                    }
                
                metrics[dataset_name] = {
                    'dataset_size': dataset_size,
                    'flat_qps': flat_k_result['qps'],
                    'flat_time_ms': flat_k_result['avg_query_time'] * 1000,
                    'n_probe_results': n_probe_metrics
                }
    
    return metrics

def create_comparison_plots(metrics, output_dir):
    """Create comparison plots between datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_names = ['subset_10k', 'subset_100k', 'subset_1M']
    dataset_labels = ['10K', '100K', '1M']
    dataset_sizes = [metrics[name]['dataset_size'] for name in dataset_names if name in metrics]
    
    # Colors for datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Scalability Overview: Max Speedup vs Dataset Size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Maximum Speedup vs Dataset Size
    max_speedups = []
    moderate_speedups = []  # speedup at ~85% recall
    high_recall_speedups = []  # speedup at >95% recall
    
    for dataset_name in dataset_names:
        if dataset_name not in metrics:
            continue
            
        n_probe_results = metrics[dataset_name]['n_probe_results']
        
        # Find maximum speedup
        max_speedup = max(result['speedup'] for result in n_probe_results.values())
        max_speedups.append(max_speedup)
        
        # Find speedup at moderate recall (~85%)
        moderate_speedup = 0
        for result in n_probe_results.values():
            if 0.80 <= result['recall'] <= 0.90:
                moderate_speedup = max(moderate_speedup, result['speedup'])
        moderate_speedups.append(moderate_speedup)
        
        # Find speedup at high recall (>95%)
        high_recall_speedup = 0
        for result in n_probe_results.values():
            if result['recall'] >= 0.95:
                high_recall_speedup = max(high_recall_speedup, result['speedup'])
        high_recall_speedups.append(high_recall_speedup)
    
    x_pos = np.arange(len(dataset_labels))
    width = 0.25
    
    ax1.bar(x_pos - width, max_speedups, width, label='Max Speedup', color=colors[0], alpha=0.8)
    ax1.bar(x_pos, moderate_speedups, width, label='~85% Recall', color=colors[1], alpha=0.8)
    ax1.bar(x_pos + width, high_recall_speedups, width, label='>95% Recall', color=colors[2], alpha=0.8)
    
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Speedup vs Flat Search')
    ax1.set_title('IVF Speedup Scaling by Dataset Size')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Flat Search Performance Degradation
    flat_qps_values = [metrics[name]['flat_qps'] for name in dataset_names if name in metrics]
    flat_time_values = [metrics[name]['flat_time_ms'] for name in dataset_names if name in metrics]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(dataset_sizes, flat_qps_values, 'o-', color=colors[0], linewidth=3, markersize=8, label='Flat Search QPS')
    line2 = ax2_twin.plot(dataset_sizes, flat_time_values, 's-', color=colors[1], linewidth=3, markersize=8, label='Flat Search Time (ms)')
    
    ax2.set_xlabel('Dataset Size (molecules)')
    ax2.set_ylabel('Queries Per Second (QPS)', color=colors[0])
    ax2_twin.set_ylabel('Average Query Time (ms)', color=colors[1])
    ax2.set_title('Flat Search Performance vs Dataset Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2_twin.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax2_twin.tick_params(axis='y', labelcolor=colors[1])
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    # Plot 3: IVF Performance at Different Recall Levels
    recall_thresholds = [0.4, 0.6, 0.8, 0.9, 0.95]
    threshold_colors = plt.cm.viridis(np.linspace(0, 1, len(recall_thresholds)))
    
    for i, recall_threshold in enumerate(recall_thresholds):
        qps_at_recall = []
        
        for dataset_name in dataset_names:
            if dataset_name not in metrics:
                continue
                
            n_probe_results = metrics[dataset_name]['n_probe_results']
            
            # Find best QPS at or above this recall threshold
            best_qps = 0
            for result in n_probe_results.values():
                if result['recall'] >= recall_threshold:
                    best_qps = max(best_qps, result['qps'])
            
            qps_at_recall.append(best_qps if best_qps > 0 else None)
        
        # Only plot if we have data for all datasets
        if all(qps is not None for qps in qps_at_recall):
            ax3.plot(dataset_sizes, qps_at_recall, 'o-', 
                    color=threshold_colors[i], linewidth=2, markersize=6,
                    label=f'Recall ≥ {recall_threshold:.1f}')
    
    ax3.set_xlabel('Dataset Size (molecules)')
    ax3.set_ylabel('Best IVF QPS')
    ax3.set_title('IVF Performance at Different Recall Thresholds')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recall vs Performance Trade-off Curves
    n_probe_values = [1, 2, 4, 8, 16, 32, 64]
    
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in metrics:
            continue
            
        n_probe_results = metrics[dataset_name]['n_probe_results']
        
        recalls = []
        speedups = []
        
        for n_probe in n_probe_values:
            if n_probe in n_probe_results:
                recalls.append(n_probe_results[n_probe]['recall'])
                speedups.append(n_probe_results[n_probe]['speedup'])
        
        ax4.plot(recalls, speedups, 'o-', color=colors[i], linewidth=3, 
                markersize=8, label=f'{dataset_labels[i]} molecules', alpha=0.8)
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Speedup vs Flat Search')
    ax4.set_title('Recall vs Speedup Trade-off Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'dataset_size_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Performance Metrics Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Query Time Comparison at Different Recall Levels
    recall_levels = [0.4, 0.8, 0.95]
    recall_labels = ['Low (40%)', 'Medium (80%)', 'High (95%)']
    
    x_pos = np.arange(len(dataset_labels))
    width = 0.25
    
    for i, recall_level in enumerate(recall_levels):
        times_at_recall = []
        
        for dataset_name in dataset_names:
            if dataset_name not in metrics:
                continue
                
            n_probe_results = metrics[dataset_name]['n_probe_results']
            
            # Find best (lowest) time at or above this recall level
            best_time = float('inf')
            for result in n_probe_results.values():
                if result['recall'] >= recall_level:
                    best_time = min(best_time, result['avg_time_ms'])
            
            times_at_recall.append(best_time if best_time != float('inf') else None)
        
        # Replace None with 0 for plotting
        times_at_recall = [t if t is not None else 0 for t in times_at_recall]
        
        ax1.bar(x_pos + i*width - width, times_at_recall, width, 
               label=recall_labels[i], alpha=0.8)
    
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Average Query Time (ms)')
    ax1.set_title('Query Time at Different Recall Levels')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Index Building Efficiency (if timing data available)
    # Placeholder - would need index building times from benchmark data
    ax2.text(0.5, 0.5, 'Index Building Times\n(Data not available\nin current benchmark)', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Index Building Performance')
    
    # Plot 3: Memory Usage Scaling
    # Extract memory usage if available
    memory_usage = []
    for dataset_name in dataset_names:
        if dataset_name not in metrics:
            continue
        # This would need memory data from the benchmark results
        # For now, use dataset size as proxy
        memory_usage.append(dataset_sizes[dataset_names.index(dataset_name)] * 0.001)  # Rough estimate
    
    ax3.plot(dataset_sizes, memory_usage, 'o-', linewidth=3, markersize=8, color=colors[0])
    ax3.set_xlabel('Dataset Size (molecules)')
    ax3.set_ylabel('Memory Usage (GB)')
    ax3.set_title('Memory Usage Scaling (Estimated)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency Ratio (QPS per molecule)
    efficiency_ratios = []
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in metrics:
            continue
            
        # Calculate efficiency as best QPS divided by dataset size
        n_probe_results = metrics[dataset_name]['n_probe_results']
        best_qps = max(result['qps'] for result in n_probe_results.values())
        efficiency = best_qps / dataset_sizes[i] * 1000000  # QPS per million molecules
        efficiency_ratios.append(efficiency)
    
    ax4.bar(dataset_labels, efficiency_ratios, color=colors, alpha=0.8)
    ax4.set_xlabel('Dataset Size')
    ax4.set_ylabel('Max QPS per Million Molecules')
    ax4.set_title('Search Efficiency by Dataset Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, 'performance_metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_plot_path, metrics_plot_path

def main():
    """Main function to generate comparison plots."""
    base_path = 'search/results'
    output_dir = 'search/results/comparison_plots'
    
    print("Loading benchmark data for all datasets...")
    data = load_benchmark_data(base_path)
    
    if len(data) < 2:
        print("Error: Need at least 2 datasets for comparison")
        return 1
    
    print("Extracting performance metrics...")
    metrics = extract_performance_metrics(data)
    
    print("Creating comparison plots...")
    plot_paths = create_comparison_plots(metrics, output_dir)
    
    print(f"\n✅ Comparison analysis complete!")
    print(f"📊 Plots saved:")
    print(f"   - Dataset Size Comparison: {plot_paths[0]}")
    print(f"   - Performance Metrics: {plot_paths[1]}")
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    for dataset_name in ['subset_10k', 'subset_100k', 'subset_1M']:
        if dataset_name in metrics:
            dataset_size = metrics[dataset_name]['dataset_size']
            flat_qps = metrics[dataset_name]['flat_qps']
            n_probe_results = metrics[dataset_name]['n_probe_results']
            
            max_speedup = max(result['speedup'] for result in n_probe_results.values())
            best_qps = max(result['qps'] for result in n_probe_results.values())
            
            print(f"   {dataset_name}: {dataset_size:,} molecules, "
                  f"Flat: {flat_qps:.1f} QPS, Best IVF: {best_qps:.1f} QPS, "
                  f"Max Speedup: {max_speedup:.1f}x")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())