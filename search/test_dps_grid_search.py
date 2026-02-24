#!/usr/bin/env python
"""
Grid search script for DPS hyperparameter tuning.

This script runs the DPS benchmark across a grid of temperature and prob_threshold
values, saving results to separate directories for each parameter combination.

Usage:
    python test_dps_grid_search.py --dataset data/subset_10k.csv \
        --temperatures 0.05 0.1 0.2 \
        --prob-thresholds 0.3 0.5 0.7 \
        --output-base search/results/dps_grid_search
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List
import itertools

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.test_dps import DPSBenchmark


def run_grid_search(
    dataset_path: str,
    output_base: str,
    temperatures: List[float],
    prob_thresholds: List[float],
    k_values: List[int] = [10, 50, 100],
    n_probe_values: List[int] = [1, 2, 4, 8, 16, 32, 64],
    max_probe: int = None,
    n_queries: int = 100,
    n_runs: int = 3,
    fp_type: str = 'morgan',
    fp_size: int = 2048,
    radius: int = 2,
    n_clusters: int = None,
    threshold: float = 0.65
):
    """
    Run grid search over temperature and prob_threshold parameters.
    
    Args:
        dataset_path: Path to dataset
        output_base: Base output directory
        temperatures: List of temperature values to test
        prob_thresholds: List of probability threshold values to test
        k_values: List of k values for search
        n_probe_values: List of fixed n_probe values to compare
        max_probe: Maximum probes for DPS
        n_queries: Number of test queries
        n_runs: Number of timing runs per query
        fp_type: Fingerprint type
        fp_size: Fingerprint size
        radius: Morgan fingerprint radius
        n_clusters: Number of clusters
        threshold: BitBIRCH threshold
    """
    
    # Create base output directory
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    # Create parameter combinations
    param_combinations = list(itertools.product(temperatures, prob_thresholds))
    total_combinations = len(param_combinations)
    
    print("="*80)
    print("DPS HYPERPARAMETER GRID SEARCH")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Output base: {output_base}")
    print(f"Temperatures: {temperatures}")
    print(f"Probability thresholds: {prob_thresholds}")
    print(f"Total combinations: {total_combinations}")
    print(f"Queries per combination: {n_queries}")
    print("="*80)
    
    # Track results
    grid_results = {
        'dataset': dataset_path,
        'temperatures': temperatures,
        'prob_thresholds': prob_thresholds,
        'k_values': k_values,
        'n_queries': n_queries,
        'n_runs': n_runs,
        'combinations': []
    }
    
    # Run benchmark for each parameter combination
    for idx, (temp, prob_thresh) in enumerate(param_combinations, 1):
        print(f"\n{'='*80}")
        print(f"GRID SEARCH [{idx}/{total_combinations}]")
        print(f"Temperature: {temp}, Probability Threshold: {prob_thresh}")
        print(f"{'='*80}\n")
        
        # Create output directory for this combination
        output_dir = os.path.join(
            output_base,
            f"temp_{temp:.4f}_prob_{prob_thresh:.4f}"
        )
        
        combination_start = time.time()
        
        try:
            # Create benchmark instance
            benchmark = DPSBenchmark(
                dataset_path=dataset_path,
                output_dir=output_dir,
                fp_type=fp_type,
                fp_size=fp_size,
                radius=radius,
                n_clusters=n_clusters,
                threshold=threshold
            )
            
            # Run benchmark with current parameters
            benchmark.run_full_benchmark(
                k_values=k_values,
                n_probe_values=n_probe_values,
                max_probe=max_probe,
                prob_threshold=prob_thresh,
                temperature=temp,
                n_queries=n_queries,
                n_runs=n_runs
            )
            
            combination_time = time.time() - combination_start
            
            # Record results
            combination_result = {
                'temperature': temp,
                'prob_threshold': prob_thresh,
                'output_dir': output_dir,
                'time_seconds': combination_time,
                'status': 'success'
            }
            
            print(f"\n✓ Combination completed in {combination_time:.1f}s")
            
        except Exception as e:
            combination_time = time.time() - combination_start
            combination_result = {
                'temperature': temp,
                'prob_threshold': prob_thresh,
                'output_dir': output_dir,
                'time_seconds': combination_time,
                'status': 'failed',
                'error': str(e)
            }
            print(f"\n✗ Combination failed after {combination_time:.1f}s: {e}")
        
        grid_results['combinations'].append(combination_result)
    
    # Save grid search summary
    summary_path = os.path.join(output_base, 'grid_search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(grid_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Successful: {sum(1 for r in grid_results['combinations'] if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in grid_results['combinations'] if r['status'] == 'failed')}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    return grid_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Grid search for DPS hyperparameter tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic grid search
  python test_dps_grid_search.py --dataset data/subset_10k.csv \\
      --temperatures 0.05 0.1 0.2 \\
      --prob-thresholds 0.3 0.5 0.7
  
  # With custom ranges
  python test_dps_grid_search.py --dataset data/subset_10k.csv \\
      --temperatures 0.01 0.05 0.1 0.15 0.2 \\
      --prob-thresholds 0.1 0.3 0.5 0.7 0.9 \\
      --n-queries 50 \\
      --output-base results/dps_grid
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to dataset (.smi or .csv)')
    
    # Grid search parameters
    parser.add_argument('--temperatures', type=float, nargs='+', required=True,
                        help='List of temperature values to test')
    parser.add_argument('--prob-thresholds', type=float, nargs='+', required=True,
                        help='List of probability threshold values to test')
    
    # Output directory
    parser.add_argument('--output-base', type=str, default='search/results/dps_grid_search',
                        help='Base output directory for grid search results')
    
    # Dataset parameters
    parser.add_argument('--fp-type', type=str, default='morgan', choices=['morgan', 'rdkit'],
                        help='Fingerprint type')
    parser.add_argument('--fp-size', type=int, default=2048,
                        help='Fingerprint size')
    parser.add_argument('--radius', type=int, default=2,
                        help='Morgan fingerprint radius')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters (default: sqrt(n))')
    parser.add_argument('--threshold', type=float, default=0.65,
                        help='BitBIRCH threshold')
    
    # Benchmark parameters
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 50, 100],
                        help='k values to test')
    parser.add_argument('--n-probe-values', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64],
                        help='n_probe values to compare')
    parser.add_argument('--max-probe', type=int, default=None,
                        help='Maximum probes for DPS (default: sqrt(n_samples))')
    parser.add_argument('--n-queries', type=int, default=100,
                        help='Number of test queries')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of timing runs per query')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not args.temperatures:
        parser.error("At least one temperature value is required")
    if not args.prob_thresholds:
        parser.error("At least one probability threshold value is required")
    
    # Run grid search
    grid_results = run_grid_search(
        dataset_path=args.dataset,
        output_base=args.output_base,
        temperatures=args.temperatures,
        prob_thresholds=args.prob_thresholds,
        k_values=args.k_values,
        n_probe_values=args.n_probe_values,
        max_probe=args.max_probe,
        n_queries=args.n_queries,
        n_runs=args.n_runs,
        fp_type=args.fp_type,
        fp_size=args.fp_size,
        radius=args.radius,
        n_clusters=args.n_clusters,
        threshold=args.threshold
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
