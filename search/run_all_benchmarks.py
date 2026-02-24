#!/usr/bin/env python
"""
Batch runner for enhanced benchmarks on all subset CSV files.

This script runs benchmarks on all available subset files with different configurations
and generates a comprehensive report comparing performance across different dataset sizes.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import shlex, subprocess, sys

def find_subset_files(data_dir: str) -> List[str]:
    """Find all subset CSV files in the data directory."""
    subset_files = []
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return subset_files
    
    for filename in os.listdir(data_dir):
        if filename.startswith('subset_') and filename.endswith('.csv'):
            subset_files.append(os.path.join(data_dir, filename))
    
    # Sort by size (extract number from filename)
    def extract_size(filename):
        """Extract size from filename like 'subset_10k.csv'."""
        base = os.path.basename(filename)
        size_part = base.replace('subset_', '').replace('.csv', '')
        
        # Convert size strings to numbers for sorting
        size_mapping = {
            '10k': 10000,
            '100k': 100000,
            '1M': 1000000,
            '10M': 10000000
        }
        return size_mapping.get(size_part, 0)
    
    subset_files.sort(key=extract_size)
    return subset_files


def run_benchmark(dataset_path: str, output_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Run benchmark on a single dataset."""
    print(f"\n{'='*80}")
    print(f"Running benchmark on: {os.path.basename(dataset_path)}")
    print(f"{'='*80}")
    
    # Prepare command
    cmd = [
        sys.executable, 
        'enhanced_benchmark.py',
        '--dataset', dataset_path,
        '--output-dir', output_dir,
        '--fp-type', args.fp_type,
        '--fp-size', str(args.fp_size),
        '--n-runs', str(args.n_runs),
        '--n-queries', str(args.n_queries),
        '--threshold', str(args.threshold)
    ]
    
    # Add optional arguments
    if args.methods:
        cmd.extend(['--methods'] + args.methods)
    if args.k_values:
        cmd.extend(['--k-values'] + [str(k) for k in args.k_values])
    if args.n_probe_values:
        cmd.extend(['--n-probe-values'] + [str(n) for n in args.n_probe_values])
    if args.n_clusters:
        cmd.extend(['--n-clusters', str(args.n_clusters)])
    if args.verbose:
        cmd.append('--verbose')
    
    # Add stage control arguments
    if args.skip_fingerprints:
        cmd.append('--skip-fingerprints')
    if args.skip_ground_truth:
        cmd.append('--skip-ground-truth')
    if args.skip_index_build:
        cmd.append('--skip-index-build')
    if args.skip_benchmarks:
        cmd.append('--skip-benchmarks')
    if args.force_reload:
        cmd.append('--force-reload')
    
    # Run the benchmark
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Benchmark completed successfully in {end_time - start_time:.2f}s")
            
            # Try to load the results
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            results_path = os.path.join(output_dir, dataset_name, 'data', 'benchmark_results.json')
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    benchmark_results = json.load(f)
                return {
                    'status': 'success',
                    'dataset': dataset_path,
                    'runtime': end_time - start_time,
                    'results': benchmark_results
                }
            else:
                print(f"⚠ Results file not found: {results_path}")
                return {
                    'status': 'success_no_results',
                    'dataset': dataset_path,
                    'runtime': end_time - start_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        else:
            print(f"✗ Benchmark failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {
                'status': 'failed',
                'dataset': dataset_path,
                'runtime': end_time - start_time,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ Benchmark timed out after {args.timeout}s")
        return {
            'status': 'timeout',
            'dataset': dataset_path,
            'runtime': args.timeout
        }
    except Exception as e:
        print(f"✗ Benchmark failed with error: {e}")
        return {
            'status': 'error',
            'dataset': dataset_path,
            'runtime': time.time() - start_time,
            'error': str(e)
        }


def generate_summary_report(batch_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate a comprehensive summary report."""
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Prepare summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_benchmarks': len(batch_results),
        'successful_benchmarks': sum(1 for r in batch_results if r['status'] == 'success'),
        'failed_benchmarks': sum(1 for r in batch_results if r['status'] in ['failed', 'timeout', 'error']),
        'total_runtime': sum(r['runtime'] for r in batch_results),
        'benchmark_details': []
    }
    
    # Extract performance metrics for successful benchmarks
    for result in batch_results:
        if result['status'] == 'success' and 'results' in result:
            dataset_name = os.path.splitext(os.path.basename(result['dataset']))[0]
            benchmark_data = result['results']

            # Prefer explicit dataset_info if present, otherwise fall back to older keys
            ds_size = benchmark_data.get('dataset_info', {}).get('size') if isinstance(benchmark_data.get('dataset_info'), dict) else None
            if ds_size is None:
                ds_size = benchmark_data.get('dataset_info', {}).get('size') if isinstance(benchmark_data.get('dataset_info'), dict) else None
            if ds_size is None:
                ds_size = benchmark_data.get('data_size') or benchmark_data.get('dataset_size') or 'Unknown'

            n_queries_val = benchmark_data.get('dataset_info', {}).get('n_queries') if isinstance(benchmark_data.get('dataset_info'), dict) else None
            if n_queries_val is None:
                n_queries_val = benchmark_data.get('n_queries') or benchmark_data.get('num_queries') or 'Unknown'

            detail = {
                'dataset': dataset_name,
                'dataset_size': ds_size,
                'n_queries': n_queries_val,
                'runtime': result['runtime'],
                'timing_breakdown': benchmark_data.get('timing', {}),
                'memory_usage': benchmark_data.get('memory', {}),
                'performance_metrics': {}
            }
            
            # Extract performance metrics for each method
            # Support two possible result layouts:
            # 1) New layout: benchmark_data['benchmark_results'][method]['flat_search'/'ivf_search']
            # 2) Older layout: benchmark_data['flat_search_results'][method], benchmark_data['ivf_search_results'][method]
            benchmark_results = benchmark_data.get('benchmark_results')
            if benchmark_results is None:
                # Build a compatible structure from older layout if present
                flat_results_all = benchmark_data.get('flat_search_results', {})
                ivf_results_all = benchmark_data.get('ivf_search_results', {})
                benchmark_results = {}
                # Collect methods from either flat or ivf results
                methods_set = set(list(flat_results_all.keys()) + list(ivf_results_all.keys()))
                for method in methods_set:
                    benchmark_results[method] = {
                        'flat_search': flat_results_all.get(method, {}),
                        'ivf_search': ivf_results_all.get(method, {})
                    }

            for method, method_results in benchmark_results.items():
                detail['performance_metrics'][method] = {
                    'flat_search': {},
                    'ivf_search': {}
                }

                # Flat search metrics
                if 'flat_search' in method_results:
                    for k, k_results in method_results['flat_search'].items():
                        detail['performance_metrics'][method]['flat_search'][k] = {
                            'avg_query_time': k_results.get('avg_query_time', 0),
                            'qps': k_results.get('qps', 0),
                            'avg_recall': k_results.get('avg_recall', 0)
                        }

                # IVF search metrics
                if 'ivf_search' in method_results:
                    for k, k_results in method_results['ivf_search'].items():
                        detail['performance_metrics'][method]['ivf_search'][k] = {}
                        for n_probe, probe_results in k_results.items():
                            detail['performance_metrics'][method]['ivf_search'][k][n_probe] = {
                                'avg_query_time': probe_results.get('avg_query_time', 0),
                                'qps': probe_results.get('qps', 0),
                                'avg_recall': probe_results.get('avg_recall', 0)
                            }
            
            summary['benchmark_details'].append(detail)
        else:
            # Add failed benchmark info
            dataset_name = os.path.splitext(os.path.basename(result['dataset']))[0]
            summary['benchmark_details'].append({
                'dataset': dataset_name,
                'status': result['status'],
                'runtime': result['runtime'],
                'error_info': result.get('stderr', result.get('error', 'Unknown error'))
            })
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'batch_benchmark_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Summary report saved to: {summary_path}")
    
    # Generate human-readable summary
    print(f"\n{'='*80}")
    print("BATCH BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Successful: {summary['successful_benchmarks']}")
    print(f"Failed: {summary['failed_benchmarks']}")
    print(f"Total runtime: {summary['total_runtime']:.2f}s ({summary['total_runtime']/3600:.2f}h)")
    
    if summary['successful_benchmarks'] > 0:
        print(f"\nSuccessful benchmarks:")
        for detail in summary['benchmark_details']:
            if 'dataset_size' in detail:
                ds = detail.get('dataset_size', 'Unknown')
                # Safely format dataset size whether it's numeric or a string like 'Unknown'
                try:
                    ds_int = int(ds)
                    ds_str = f"{ds_int:,}"
                except (ValueError, TypeError):
                    ds_str = str(ds)

                runtime = detail.get('runtime', 0.0)
                print(f"  {detail['dataset']}: {ds_str} molecules, {runtime:.1f}s")
    
    if summary['failed_benchmarks'] > 0:
        print(f"\nFailed benchmarks:")
        for detail in summary['benchmark_details']:
            if 'dataset_size' not in detail:
                print(f"  {detail['dataset']}: {detail['status']}")


def main():
    """Main function for batch benchmark runner."""
    parser = argparse.ArgumentParser(
        description='Run enhanced benchmarks on all subset CSV files'
    )
    
    # Required arguments
    parser.add_argument('--data-dir', default='../data', help='Directory containing subset CSV files')
    parser.add_argument('--output-dir', default='search/results', help='Base output directory')
    
    # Benchmark parameters
    parser.add_argument('--fp-type', default='morgan', choices=['morgan', 'rdkit'], help='Fingerprint type')
    parser.add_argument('--fp-size', type=int, default=2048, help='Fingerprint size in bits')
    parser.add_argument('--methods', nargs='+', default=['rdkit'], choices=['rdkit', 'fpsim2'], help='Methods to test')
    parser.add_argument('--k-values', nargs='+', type=int, default=[10, 50, 100], help='k values for top-k search')
    parser.add_argument('--n-probe-values', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64], help='n_probe values for IVF search')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs per query')
    parser.add_argument('--n-queries', type=int, default=100, help='Number of query molecules')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters for IVF index')
    parser.add_argument('--threshold', type=float, default=0.65, help='Similarity threshold')
    
    # Execution parameters
    parser.add_argument('--timeout', type=int, default=36000, help='Timeout per benchmark in seconds')
    parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    
    # Stage control arguments
    parser.add_argument('--skip-fingerprints', action='store_true', help='Skip fingerprint generation (load from cache)')
    parser.add_argument('--skip-ground-truth', action='store_true', help='Skip ground truth computation (load from cache)')
    parser.add_argument('--skip-index-build', action='store_true', help='Skip index building (load from cache)')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip benchmark execution')
    parser.add_argument('--force-reload', action='store_true', help='Force regeneration of all cached data')
    parser.add_argument('--run-plots', action='store_true', help='Generate plots after benchmarks')
    
    args = parser.parse_args()
    
    # Find all subset files
    subset_files = find_subset_files(args.data_dir)
    
    if not subset_files:
        print(f"No subset CSV files found in {args.data_dir}")
        return 1
    
    print(f"Found {len(subset_files)} subset files:")
    for f in subset_files:
        print(f"  {os.path.basename(f)}")
    
    if args.dry_run:
        print("\nDry run mode - would run benchmarks on these files.")
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    print(f"\nStarting batch benchmarks...")
    batch_start_time = time.time()
    batch_results = []
    
    for i, dataset_path in enumerate(subset_files):
        print(f"\nProgress: {i+1}/{len(subset_files)}")
        result = run_benchmark(dataset_path, args.output_dir, args)
        batch_results.append(result)
        
        # Save intermediate results
        intermediate_path = os.path.join(args.output_dir, 'batch_results_intermediate.json')
        with open(intermediate_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
    
    batch_end_time = time.time()
    print(f"\nBatch benchmarks completed in {batch_end_time - batch_start_time:.2f}s")
    
    # Generate summary report
    generate_summary_report(batch_results, args.output_dir)
    
    # Save final results
    final_results_path = os.path.join(args.output_dir, 'batch_results_final.json')
    with open(final_results_path, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"\nFinal results saved to: {final_results_path}")
    
    # Generate plots if requested
    if args.run_plots:
        print(f"\n{'='*80}")
        print("GENERATING PLOTS")
        print(f"{'='*80}")
        
        successful_result_paths = []
        for result in batch_results:
            if result.get('status') == 'success' and 'results' in result:
                dataset_name = os.path.splitext(os.path.basename(result['dataset']))[0]
                results_dir = os.path.join(args.output_dir, dataset_name, 'data')
                results_path = os.path.join(results_dir, 'benchmark_results.json')
                if os.path.exists(results_path):
                    successful_result_paths.append(results_path)
                    print(f"Preparing results for plotting: {results_path}")
                
                else:
                    # If the file is missing but we have results in memory, write them out
                    try:
                        os.makedirs(results_dir, exist_ok=True)
                        with open(results_path, 'w') as wf:
                            json.dump(result['results'], wf, indent=2, default=str)
                        print(f"Wrote in-memory results to expected path: {results_path}")
                        successful_result_paths.append(results_path)
                    except Exception as e:
                        print(f"Warning: failed to write results file for plotting: {results_path} ({e})")
        
        if successful_result_paths:
            for results_path in successful_result_paths:
                plot_cmd = [sys.executable, 'plot_results.py', results_path]
                # print a shell-safe representation for logs
                try:
                    cmd_display = shlex.join(plot_cmd)
                except AttributeError:
                    cmd_display = " ".join(shlex.quote(p) for p in plot_cmd)
                print(f"Running plot generation: {cmd_display}")
                try:
                    plot_result = subprocess.run(plot_cmd, capture_output=True, text=True, timeout=600)
                    if plot_result.returncode == 0:
                        print(f"✓ Plots generated for: {results_path}")
                        if plot_result.stdout:
                            print(plot_result.stdout)
                    else:
                        print(f"✗ Plot generation failed for: {results_path}")
                        print(f"STDOUT: {plot_result.stdout}")
                        print(f"STDERR: {plot_result.stderr}")
                except Exception as e:
                    print(f"✗ Plot generation failed for {results_path}: {e}")
        else:
             print("No successful benchmarks found, skipping plot generation")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())