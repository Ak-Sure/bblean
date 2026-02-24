#!/usr/bin/env python
"""
Parallelized grid search script for DPS hyperparameter tuning.

This script runs the DPS benchmark across a grid of temperature and prob_threshold
values in parallel, saving results to separate directories for each parameter combination.

Usage:
    python test_dps_grid_search_parallel.py --dataset data/subset_10k.csv \
        --temperatures 0.05 0.1 0.2 \
        --prob-thresholds 0.3 0.5 0.7 \
        --n-workers 4 \
        --output-base search/results/dps_grid_search
"""

import os
import sys
import argparse
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import itertools
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.test_dps import DPSBenchmark


def prepare_shared_index(
    dataset_path: str,
    cache_dir: str,
    fp_type: str,
    fp_size: int,
    radius: int,
    n_clusters: int,
    threshold: float,
    branching_factor: int,
    k_values: List[int],
    n_queries: int
) -> str:
    """
    Build IVF index once and save to disk for workers to load.
    
    Returns:
        Path to the cache directory with saved index
    """
    print("\n" + "="*80)
    print("PREPARING SHARED INDEX (ONE-TIME COST)")
    print("="*80)
    
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'shared_index.pkl')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Found existing cache at {cache_file}")
        print("Loading cached index...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"✓ Cache loaded successfully")
            print(f"  - {len(cached_data['smiles'])} molecules")
            print(f"  - {cached_data['ivf_index'].n_clusters} clusters")
            print("="*80 + "\n")
            return cache_file
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            print("Rebuilding index...")
    
    # Build index from scratch
    start_time = time.time()
    
    # Create temporary benchmark instance
    temp_benchmark = DPSBenchmark(
        dataset_path=dataset_path,
        output_dir=cache_dir,
        fp_type=fp_type,
        fp_size=fp_size,
        radius=radius,
        n_clusters=n_clusters,
        threshold=threshold,
        branching_factor=branching_factor
    )
    
    # Load data and build index
    print("Loading dataset...")
    temp_benchmark.load_data()
    print(f"✓ Loaded {len(temp_benchmark.smiles)} molecules")
    
    print("Building IVF index...")
    temp_benchmark.build_index()
    print(f"✓ Built index with {temp_benchmark.ivf_index.n_clusters} clusters")
    
    print("Computing ground truth...")
    temp_benchmark.compute_ground_truth(k_values, n_queries)
    print(f"✓ Ground truth computed for k={k_values}")
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    cache_data = {
        'smiles': temp_benchmark.smiles,
        'fingerprints': temp_benchmark.fingerprints,
        'fingerprints_rdkit': temp_benchmark.fingerprints_rdkit,
        'ivf_index': temp_benchmark.ivf_index,
        'query_indices': temp_benchmark.query_indices,
        'ground_truth': temp_benchmark.ground_truth,
        'dataset_path': dataset_path,
        'fp_type': fp_type,
        'fp_size': fp_size,
        'radius': radius,
        'n_clusters': n_clusters,
        'threshold': threshold
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    build_time = time.time() - start_time
    print(f"✓ Index prepared and cached in {build_time:.1f}s")
    print("="*80 + "\n")
    
    return cache_file


def run_single_combination(args_tuple: Tuple) -> Dict:
    """
    Run benchmark for a single parameter combination.
    
    This function is designed to be called by multiprocessing workers.
    Loads pre-built index from cache instead of rebuilding.
    
    Args:
        args_tuple: Tuple containing all necessary arguments
        
    Returns:
        Dictionary with combination results
    """
    (
        temp, prob_thresh, cache_file, output_base,
        k_values, n_probe_values, max_probe, n_queries, n_runs, idx, total
    ) = args_tuple
    
    print(f"\n{'='*80}")
    print(f"[Worker {os.getpid()}] COMBINATION [{idx}/{total}]")
    print(f"Temperature: {temp}, Probability Threshold: {prob_thresh}")
    print(f"{'='*80}\n")
    
    # Create output directory for this combination
    output_dir = os.path.join(
        output_base,
        f"temp_{temp:.4f}_prob_{prob_thresh:.4f}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    combination_start = time.time()
    
    try:
        # Load pre-built index from cache
        print(f"[Worker {os.getpid()}] Loading pre-built index from cache...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Create benchmark instance and populate with cached data
        benchmark = DPSBenchmark(
            dataset_path=cache_data['dataset_path'],
            output_dir=output_dir,
            fp_type=cache_data['fp_type'],
            fp_size=cache_data['fp_size'],
            radius=cache_data['radius'],
            n_clusters=cache_data['n_clusters'],
            threshold=cache_data['threshold']
        )
        
        # Restore cached data (skip expensive loading/building)
        benchmark.smiles = cache_data['smiles']
        benchmark.fingerprints = cache_data['fingerprints']
        benchmark.fingerprints_rdkit = cache_data['fingerprints_rdkit']
        benchmark.ivf_index = cache_data['ivf_index']
        benchmark.query_indices = cache_data['query_indices']
        benchmark.ground_truth = cache_data['ground_truth']
        
        print(f"[Worker {os.getpid()}] ✓ Loaded index with {benchmark.ivf_index.n_clusters} clusters")
        print(f"[Worker {os.getpid()}] Running DPS search with temp={temp}, prob_thresh={prob_thresh}...")
        
        # Run ONLY the search benchmarks (skip data loading and index building)
        # Compute exhaustive baseline
        baseline_results = benchmark.compute_exhaustive_search_baseline(k_values, n_runs)
        
        # Run fixed n_probe benchmark
        fixed_results = benchmark.run_fixed_nprobe_benchmark(k_values, n_probe_values, n_runs)
        
        # Run DPS benchmark with current parameters
        dps_results = benchmark.run_dps_benchmark(k_values, max_probe, prob_thresh, temp, n_runs)
        
        # Generate plots and save results
        benchmark.plot_results(fixed_results, dps_results, baseline_results, k_values, n_probe_values)
        benchmark.print_summary(fixed_results, dps_results, baseline_results, k_values, n_probe_values)
        benchmark.save_results(fixed_results, dps_results, baseline_results)
        benchmark.save_per_query_csv(dps_results)
        
        combination_time = time.time() - combination_start
        
        # Record results
        combination_result = {
            'temperature': temp,
            'prob_threshold': prob_thresh,
            'output_dir': output_dir,
            'time_seconds': combination_time,
            'status': 'success'
        }
        
        print(f"\n[Worker] ✓ Combination completed in {combination_time:.1f}s")
        
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
        print(f"\n[Worker] ✗ Combination failed after {combination_time:.1f}s: {e}")
        import traceback
        traceback.print_exc()
    
    return combination_result


def run_grid_search_parallel(
    dataset_path: str,
    output_base: str,
    temperatures: List[float],
    prob_thresholds: List[float],
    n_workers: int = None,
    k_values: List[int] = [10, 50, 100],
    n_probe_values: List[int] = [1, 2, 4, 8, 16, 32, 64],
    max_probe: int = None,
    n_queries: int = 100,
    n_runs: int = 3,
    fp_type: str = 'morgan',
    fp_size: int = 2048,
    radius: int = 2,
    n_clusters: int = None,
    threshold: float = 0.25,
    branching_factor: int = 50
):
    """
    Run parallelized grid search over temperature and prob_threshold parameters.
    
    Args:
        dataset_path: Path to dataset
        output_base: Base output directory
        temperatures: List of temperature values to test
        prob_thresholds: List of probability threshold values to test
        n_workers: Number of parallel workers (None = use all CPUs)
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
    cache_dir = os.path.join(output_base, '.cache')
    
    # Create parameter combinations
    param_combinations = list(itertools.product(temperatures, prob_thresholds))
    total_combinations = len(param_combinations)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = cpu_count()
    n_workers = min(n_workers, total_combinations)  # Don't use more workers than combinations
    
    print("="*80)
    print("DPS HYPERPARAMETER GRID SEARCH (PARALLELIZED + CACHED)")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Output base: {output_base}")
    print(f"Temperatures: {temperatures}")
    print(f"Probability thresholds: {prob_thresholds}")
    print(f"Total combinations: {total_combinations}")
    print(f"Parallel workers: {n_workers}")
    print(f"Queries per combination: {n_queries}")
    print(f"Estimated speedup: {n_workers}x (ideal)")
    print("="*80)
    
    # Step 1: Prepare shared index (one-time cost)
    cache_file = prepare_shared_index(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        fp_type=fp_type,
        fp_size=fp_size,
        radius=radius,
        n_clusters=n_clusters,
        threshold=threshold,
        branching_factor=branching_factor,
        k_values=k_values,
        n_queries=n_queries
    )
    
    # Track results
    grid_results = {
        'dataset': dataset_path,
        'temperatures': temperatures,
        'prob_thresholds': prob_thresholds,
        'k_values': k_values,
        'n_queries': n_queries,
        'n_runs': n_runs,
        'n_workers': n_workers,
        'combinations': []
    }
    
    # Prepare arguments for each combination (now much lighter - just params + cache path)
    worker_args = []
    for idx, (temp, prob_thresh) in enumerate(param_combinations, 1):
        args_tuple = (
            temp, prob_thresh, cache_file, output_base,
            k_values, n_probe_values, max_probe, n_queries, n_runs, idx, total_combinations
        )
        worker_args.append(args_tuple)
    
    # Run combinations in parallel
    print(f"\nStarting parallel execution with {n_workers} workers...")
    print(f"Each worker will load pre-built index (fast) and run DPS search only")
    overall_start = time.time()
    
    with Pool(processes=n_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = pool.map(run_single_combination, worker_args)
    
    overall_time = time.time() - overall_start
    
    # Collect results
    grid_results['combinations'] = results
    grid_results['total_time_seconds'] = overall_time
    
    # Save grid search summary
    summary_path = os.path.join(output_base, 'grid_search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(grid_results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    avg_time = sum(r['time_seconds'] for r in results if r['status'] == 'success') / max(successful, 1)
    
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total wall time: {overall_time/60:.1f} minutes")
    print(f"Average time per combination: {avg_time:.1f}s")
    print(f"Effective speedup: {(total_combinations * avg_time) / overall_time:.1f}x")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    return grid_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Parallelized grid search for DPS hyperparameter tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic parallel grid search
  python test_dps_grid_search_parallel.py --dataset data/subset_10k.csv \\
      --temperatures 0.05 0.1 0.2 \\
      --prob-thresholds 0.3 0.5 0.7 \\
      --n-workers 4
  
  # Use all available CPUs
  python test_dps_grid_search_parallel.py --dataset data/subset_10k.csv \\
      --temperatures 0.01 0.05 0.1 0.15 0.2 \\
      --prob-thresholds 0.1 0.3 0.5 0.7 0.9 \\
      --n-workers -1
  
  # Custom ranges with 8 workers
  python test_dps_grid_search_parallel.py --dataset data/subset_10k.csv \\
      --temperatures 0.01 0.05 0.1 0.15 0.2 \\
      --prob-thresholds 0.1 0.3 0.5 0.7 0.9 \\
      --n-queries 50 \\
      --n-workers 8 \\
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
    
    # Parallelization
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs, -1: all CPUs)')
    
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
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='BitBIRCH threshold')
    parser.add_argument('--branching-factor', type=int, default=50,
                        help='BitBIRCH branching factor')
    
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
    
    # Handle -1 for n_workers (use all CPUs)
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = None
    
    # Run parallel grid search
    grid_results = run_grid_search_parallel(
        dataset_path=args.dataset,
        output_base=args.output_base,
        temperatures=args.temperatures,
        prob_thresholds=args.prob_thresholds,
        n_workers=n_workers,
        k_values=args.k_values,
        n_probe_values=args.n_probe_values,
        max_probe=args.max_probe,
        n_queries=args.n_queries,
        n_runs=args.n_runs,
        fp_type=args.fp_type,
        fp_size=args.fp_size,
        radius=args.radius,
        n_clusters=args.n_clusters,
        threshold=args.threshold,
        branching_factor=args.branching_factor
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
