#!/usr/bin/env python
"""
Test script for Dynamic Probe Selection (DPS) in IVF search.

This script compares the DPS approach with fixed n_probe values and generates
comprehensive performance metrics and visualizations.

Usage:
    python test_dps.py --dataset ../data/subset_10k.csv [options]
"""

import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.ivf_index import IVFIndex
from search.utils import load_smiles_file, generate_fingerprints, load_fingerprints


class DPSBenchmark:
    """Benchmark for Dynamic Probe Selection."""
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = 'search/results/dps_test',
        fp_type: str = 'morgan',
        fp_size: int = 2048,
        radius: int = 2,
        n_clusters: int = None,
        threshold: float = 0.65,
        branching_factor: int = 50
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.fp_type = fp_type
        self.fp_size = fp_size
        self.radius = radius
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.branching_factor = branching_factor
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.smiles = None
        self.fingerprints = None
        self.fingerprints_rdkit = None
        self.ivf_index = None
        self.query_indices = None
        self.ground_truth = {}
        
    def log(self, message: str):
        """Log a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def load_data(self):
        """Load and prepare dataset."""
        self.log(f"Loading dataset from {self.dataset_path}")
        
        # Load SMILES
        self.smiles = load_smiles_file(self.dataset_path)
        self.log(f"Loaded {len(self.smiles)} SMILES strings")
        
        # Generate fingerprints
        self.log(f"Generating {self.fp_type} fingerprints (size={self.fp_size}, radius={self.radius})")
        fp_params = {'nBits': self.fp_size, 'radius': self.radius}
        self.fingerprints = generate_fingerprints(
            self.smiles,
            fp_type=self.fp_type,
            fp_params=fp_params
        )
        self.log(f"Generated fingerprints shape: {self.fingerprints.shape}")
        
        # Generate RDKit fingerprint objects for ground truth computation
        from rdkit import Chem
        from rdkit.Chem import AllChem
        self.log("Generating RDKit fingerprint objects for ground truth")
        self.fingerprints_rdkit = []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.fp_size)
                self.fingerprints_rdkit.append(fp)
            else:
                # Handle invalid SMILES
                self.fingerprints_rdkit.append(None)
        
    def build_index(self):
        """Build IVF index."""
        self.log("Building IVF index")
        
        # Determine n_clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = int(np.sqrt(len(self.smiles)))
            self.log(f"Using sqrt heuristic: n_clusters = {self.n_clusters}")
        
        # Build index
        self.ivf_index = IVFIndex(
            n_clusters=self.n_clusters,
            similarity_method='rdkit',
            threshold=self.threshold,
            branching_factor=self.branching_factor
        )
        
        start_time = time.time()
        self.ivf_index.build_index(
            self.fingerprints,
            smiles=self.smiles,
            fingerprints_rdkit=self.fingerprints_rdkit
        )
        build_time = time.time() - start_time
        
        self.log(f"Index built in {build_time:.2f}s with {self.n_clusters} clusters")
        
    def compute_ground_truth(self, k_values: List[int], n_queries: int = 100):
        """Compute ground truth using exhaustive search."""
        self.log(f"Computing ground truth for {n_queries} random queries")
        
        from rdkit import DataStructs
        
        # Select random query indices
        np.random.seed(42)
        self.query_indices = np.random.choice(len(self.smiles), size=n_queries, replace=False)
        
        # Compute ground truth for each k
        for k in k_values:
            self.ground_truth[k] = {}
            
            for query_idx in self.query_indices:
                query_fp = self.fingerprints_rdkit[query_idx]
                
                # Compute similarities to all fingerprints
                similarities = DataStructs.BulkTanimotoSimilarity(query_fp, self.fingerprints_rdkit)
                
                # Get top-k indices
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                
                self.ground_truth[k][query_idx] = [
                    {'index': int(idx), 'similarity': float(similarities[idx])}
                    for idx in top_k_indices
                ]
        
        self.log(f"Ground truth computed for k values: {k_values}")
        
    def run_fixed_nprobe_benchmark(
        self,
        k_values: List[int],
        n_probe_values: List[int],
        n_runs: int = 3
    ) -> Dict:
        """Run benchmark with fixed n_probe values."""
        self.log("Running fixed n_probe benchmark")
        
        results = {}
        
        for k in k_values:
            results[k] = {}
            
            for n_probe in n_probe_values:
                results[k][n_probe] = {
                    'query_times': [],
                    'recalls': [],
                    'num_candidates': []
                }
                
                for query_idx in self.query_indices:
                    query_fp = self.fingerprints_rdkit[query_idx]
                    gt_indices = set(r['index'] for r in self.ground_truth[k][query_idx])
                    
                    # Run multiple times for timing
                    query_times = []
                    for _ in range(n_runs):
                        start_time = time.time()
                        search_results = self.ivf_index.search(query_fp, k=k, n_probe=n_probe)
                        query_times.append(time.time() - start_time)
                    
                    # Calculate recall
                    result_indices = set(r['index'] for r in search_results)
                    recall = len(gt_indices.intersection(result_indices)) / len(gt_indices)
                    
                    # Count candidates
                    nearest_clusters = self.ivf_index._find_nearest_clusters(query_fp, n_probe)
                    num_candidates = sum(len(self.ivf_index.cluster_members[cid]) for cid in nearest_clusters)
                    
                    results[k][n_probe]['query_times'].extend(query_times)
                    results[k][n_probe]['recalls'].append(recall)
                    results[k][n_probe]['num_candidates'].append(num_candidates)
                
                # Calculate averages
                results[k][n_probe]['avg_query_time'] = float(np.mean(results[k][n_probe]['query_times']))
                results[k][n_probe]['avg_recall'] = float(np.mean(results[k][n_probe]['recalls']))
                results[k][n_probe]['avg_candidates'] = float(np.mean(results[k][n_probe]['num_candidates']))
                results[k][n_probe]['qps'] = float(1.0 / results[k][n_probe]['avg_query_time'])
        
        return results
    
    def run_dps_benchmark(
        self,
        k_values: List[int],
        max_probe: Optional[int] = None,
        prob_threshold: float = 0.1,
        temperature: float = 0.1,
        n_runs: int = 3
    ) -> Dict:
        """Run benchmark with DPS cluster selection."""
        effective_max_probe = max_probe if max_probe is not None else self.ivf_index.default_max_probe
        self.log(f"Running DPS benchmark (max_probe={max_probe} [effective: {effective_max_probe}], prob_threshold={prob_threshold}, temp={temperature})")
        
        results = {}
        
        for k in k_values:
            results[k] = {
                'query_times': [],
                'recalls': [],
                'num_candidates': [],
                'num_probes': [],
                'per_query_details': []  # Store detailed info for CSV export
            }
            
            for query_idx in self.query_indices:
                query_fp = self.fingerprints_rdkit[query_idx]
                gt_indices = set(r['index'] for r in self.ground_truth[k][query_idx])
                
                # Run search multiple times for timing using search_dps method
                query_times = []
                all_probes = []
                for _ in range(n_runs):
                    start_time = time.time()
                    search_results = self.ivf_index.search_dps(
                        query_fp,
                        k=k,
                        max_probe=max_probe,
                        prob_threshold=prob_threshold,
                        temperature=temperature
                    )
                    query_times.append(time.time() - start_time)
                    
                    # Get number of probes from result (same for all runs with same query)
                    if search_results:
                        all_probes.append(search_results[0]['n_probes'])
                
                # Calculate recall
                result_indices = set(r['index'] for r in search_results)
                recall = len(gt_indices.intersection(result_indices)) / len(gt_indices)
                
                # Get actual number of probes used (should be same across runs for same query)
                num_probes = all_probes[0] if all_probes else 0
                
                # Count actual candidates searched
                # Re-select clusters to count candidates (without timing)
                selected_clusters = self.ivf_index._select_clusters_dps(
                    query_fp, k=k, max_probe=max_probe,
                    prob_threshold=prob_threshold, temperature=temperature
                )
                num_candidates = sum(len(self.ivf_index.cluster_members[cid]) for cid in selected_clusters)
                
                # Store per-query details for CSV export
                query_detail = {
                    'query_idx': int(query_idx),
                    'query_smiles': self.smiles[query_idx] if self.smiles else '',
                    'k': k,
                    'n_probes': num_probes,
                    'num_candidates': num_candidates,
                    'recall': recall,
                    'avg_query_time_ms': np.mean(query_times) * 1000,
                    'prob_threshold': prob_threshold,
                    'temperature': temperature,
                    'max_probe': max_probe
                }
                results[k]['per_query_details'].append(query_detail)
                
                results[k]['query_times'].extend(query_times)
                results[k]['recalls'].append(recall)
                results[k]['num_candidates'].append(num_candidates)
                results[k]['num_probes'].append(num_probes)
            
            # Calculate averages
            results[k]['avg_query_time'] = float(np.mean(results[k]['query_times']))
            results[k]['avg_recall'] = float(np.mean(results[k]['recalls']))
            results[k]['avg_candidates'] = float(np.mean(results[k]['num_candidates']))
            results[k]['avg_probes'] = float(np.mean(results[k]['num_probes']))
            results[k]['std_probes'] = float(np.std(results[k]['num_probes']))  # Variability in probes
            results[k]['qps'] = float(1.0 / results[k]['avg_query_time'])
            
            self.log(f"  k={k}: avg_probes={results[k]['avg_probes']:.1f} ± {results[k]['std_probes']:.1f}, "
                    f"avg_recall={results[k]['avg_recall']:.3f}, qps={results[k]['qps']:.1f}")
        
        return results
    
    def compute_exhaustive_search_baseline(self, k_values: List[int], n_runs: int = 3) -> Dict:
        """Compute exhaustive search baseline."""
        self.log("Computing exhaustive search baseline")
        
        from rdkit import DataStructs
        
        results = {}
        
        for k in k_values:
            query_times = []
            
            for query_idx in self.query_indices:
                query_fp = self.fingerprints_rdkit[query_idx]
                
                for _ in range(n_runs):
                    start_time = time.time()
                    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, self.fingerprints_rdkit)
                    top_k_indices = np.argsort(similarities)[-k:][::-1]
                    query_times.append(time.time() - start_time)
            
            results[k] = {
                'avg_query_time': float(np.mean(query_times)),
                'qps': float(1.0 / np.mean(query_times)),
                'recall': 1.0
            }
        
        return results
    
    def plot_results(
        self,
        fixed_results: Dict,
        dps_results: Dict,
        baseline_results: Dict,
        k_values: List[int],
        n_probe_values: List[int]
    ):
        """Create comprehensive comparison plots."""
        self.log("Creating comparison plots")
        
        # Create a 2x3 grid of plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dynamic Probe Selection (DPS) vs Fixed n_probe', fontsize=16, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(n_probe_values)))
        
        # For each k value, create plots
        for k_idx, k in enumerate(k_values):
            if k_idx >= 2:  # Only plot first 2 k values to avoid clutter
                break
                
            row = k_idx
            
            # Plot 1: Recall comparison
            ax1 = axes[row, 0]
            fixed_recalls = [fixed_results[k][n_probe]['avg_recall'] for n_probe in n_probe_values]
            dps_recall = dps_results[k]['avg_recall']
            dps_nprobe = dps_results[k]['avg_probes']
            
            ax1.plot(n_probe_values, fixed_recalls, 'o-', label='Fixed n_probe', linewidth=2, markersize=8)
            ax1.axhline(y=dps_recall, color='red', linestyle='--', linewidth=2, label=f'DPS (avg n_probe={dps_nprobe:.1f})')
            ax1.scatter([dps_nprobe], [dps_recall], color='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=5, label='DPS Point')
            ax1.set_xlabel('n_probe', fontsize=11)
            ax1.set_ylabel('Recall', fontsize=11)
            ax1.set_title(f'Recall vs n_probe (k={k})', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: QPS comparison
            ax2 = axes[row, 1]
            fixed_qps = [fixed_results[k][n_probe]['qps'] for n_probe in n_probe_values]
            dps_qps = dps_results[k]['qps']
            baseline_qps = baseline_results[k]['qps']
            
            ax2.plot(n_probe_values, fixed_qps, 'o-', label='Fixed n_probe', linewidth=2, markersize=8)
            ax2.axhline(y=dps_qps, color='red', linestyle='--', linewidth=2, label='DPS')
            ax2.axhline(y=baseline_qps, color='gray', linestyle=':', linewidth=2, label='Exhaustive')
            ax2.scatter([dps_nprobe], [dps_qps], color='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=5)
            ax2.set_xlabel('n_probe', fontsize=11)
            ax2.set_ylabel('Queries Per Second (QPS)', fontsize=11)
            ax2.set_title(f'Throughput vs n_probe (k={k})', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Recall vs QPS tradeoff
            ax3 = axes[row, 2]
            ax3.plot(fixed_recalls, fixed_qps, 'o-', label='Fixed n_probe', linewidth=2, markersize=8)
            ax3.scatter([dps_recall], [dps_qps], color='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=5, label='DPS')
            ax3.scatter([baseline_results[k]['recall']], [baseline_qps], color='gray', s=150, marker='s', edgecolors='black', linewidth=2, label='Exhaustive')
            ax3.set_xlabel('Recall', fontsize=11)
            ax3.set_ylabel('QPS', fontsize=11)
            ax3.set_title(f'Recall-Speed Tradeoff (k={k})', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'dps_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.log(f"Saved comparison plot to {plot_path}")
        plt.close()
        
        # Create additional detailed plots
        self._plot_detailed_metrics(fixed_results, dps_results, baseline_results, k_values, n_probe_values)
        self._plot_probe_variability(dps_results, k_values)
        
    def _plot_detailed_metrics(
        self,
        fixed_results: Dict,
        dps_results: Dict,
        baseline_results: Dict,
        k_values: List[int],
        n_probe_values: List[int]
    ):
        """Create detailed metric plots."""
        
        # Plot 1: Average number of candidates
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for k in k_values:
            fixed_candidates = [fixed_results[k][n_probe]['avg_candidates'] for n_probe in n_probe_values]
            dps_candidates = dps_results[k]['avg_candidates']
            dps_nprobe = dps_results[k]['avg_probes']
            
            axes[0].plot(n_probe_values, fixed_candidates, 'o-', label=f'Fixed (k={k})', linewidth=2, markersize=6)
            axes[0].scatter([dps_nprobe], [dps_candidates], s=150, marker='*', edgecolors='black', linewidth=2, zorder=5)
            axes[0].annotate(f'DPS k={k}', xy=(dps_nprobe, dps_candidates), xytext=(10, 10),
                           textcoords='offset points', fontsize=9, color='red')
        
        axes[0].set_xlabel('n_probe', fontsize=12)
        axes[0].set_ylabel('Avg Number of Candidates', fontsize=12)
        axes[0].set_title('Search Space Size', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Speedup vs baseline
        for k in k_values:
            # FIXED: Speedup = method_qps / baseline_qps (should be > 1.0 for faster methods)
            fixed_speedups = [fixed_results[k][n_probe]['qps'] / baseline_results[k]['qps']
                             if baseline_results[k]['qps'] > 0 else 0 
                             for n_probe in n_probe_values]
            fixed_recalls = [fixed_results[k][n_probe]['avg_recall'] for n_probe in n_probe_values]
            
            dps_speedup = dps_results[k]['qps'] / baseline_results[k]['qps'] if baseline_results[k]['qps'] > 0 else 0
            dps_recall = dps_results[k]['avg_recall']
            
            axes[1].plot(fixed_recalls, fixed_speedups, 'o-', label=f'Fixed (k={k})', linewidth=2, markersize=6)
            axes[1].scatter([dps_recall], [dps_speedup], s=150, marker='*', edgecolors='black', linewidth=2, zorder=5)
            axes[1].annotate(f'DPS k={k}', xy=(dps_recall, dps_speedup), xytext=(10, -10),
                           textcoords='offset points', fontsize=9, color='red')
        
        axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Speedup vs Exhaustive', fontsize=12)
        axes[1].set_title('Efficiency Gain', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'dps_detailed_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.log(f"Saved detailed metrics plot to {plot_path}")
        plt.close()
    
    def _plot_probe_variability(self, dps_results: Dict, k_values: List[int]):
        """Plot the variability in number of probes per query to show adaptivity."""
        
        fig, axes = plt.subplots(1, len(k_values), figsize=(6*len(k_values), 5))
        if len(k_values) == 1:
            axes = [axes]
        
        for idx, k in enumerate(k_values):
            if k not in dps_results:
                continue
            
            probes_per_query = dps_results[k]['num_probes']
            
            # Create histogram
            axes[idx].hist(probes_per_query, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            
            # Add statistics
            mean_probes = np.mean(probes_per_query)
            std_probes = np.std(probes_per_query)
            min_probes = np.min(probes_per_query)
            max_probes = np.max(probes_per_query)
            
            axes[idx].axvline(mean_probes, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_probes:.1f}')
            axes[idx].axvline(mean_probes - std_probes, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            axes[idx].axvline(mean_probes + std_probes, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
                            label=f'±1 Std: {std_probes:.1f}')
            
            axes[idx].set_xlabel('Number of Probes', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'DPS Probe Distribution (k={k})\nRange: [{min_probes}, {max_probes}]', 
                              fontsize=13, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'dps_probe_variability.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.log(f"Saved probe variability plot to {plot_path}")
        plt.close()
        
    def print_summary(
        self,
        fixed_results: Dict,
        dps_results: Dict,
        baseline_results: Dict,
        k_values: List[int],
        n_probe_values: List[int]
    ):
        """Print summary statistics."""
        self.log("\n" + "="*80)
        self.log("BENCHMARK SUMMARY")
        self.log("="*80)
        
        for k in k_values:
            self.log(f"\n--- Results for k={k} ---")
            
            # Baseline
            self.log(f"Exhaustive Search: {baseline_results[k]['qps']:.2f} QPS, Recall=1.000")
            
            # DPS
            dps_qps = dps_results[k]['qps']
            dps_recall = dps_results[k]['avg_recall']
            dps_nprobe = dps_results[k]['avg_probes']
            dps_std_nprobe = dps_results[k]['std_probes']
            dps_speedup = dps_qps / baseline_results[k]['qps'] if baseline_results[k]['qps'] > 0 else 0
            
            # Show variability in probes (demonstrates adaptivity)
            min_probes = min(dps_results[k]['num_probes'])
            max_probes = max(dps_results[k]['num_probes'])
            
            self.log(f"DPS: {dps_qps:.2f} QPS, Recall={dps_recall:.3f}, Speedup={dps_speedup:.2f}x")
            self.log(f"  → Avg probes={dps_nprobe:.1f} ± {dps_std_nprobe:.1f} (range: [{min_probes}, {max_probes}]) - ADAPTIVE!")
            
            # Best fixed n_probe at similar recall
            best_fixed = None
            min_recall_diff = float('inf')
            for n_probe in n_probe_values:
                recall_diff = abs(fixed_results[k][n_probe]['avg_recall'] - dps_recall)
                if recall_diff < min_recall_diff:
                    min_recall_diff = recall_diff
                    best_fixed = n_probe
            
            if best_fixed:
                fixed_qps = fixed_results[k][best_fixed]['qps']
                fixed_recall = fixed_results[k][best_fixed]['avg_recall']
                fixed_speedup = fixed_qps / baseline_results[k]['qps'] if baseline_results[k]['qps'] > 0 else 0
                self.log(f"Best Fixed (n_probe={best_fixed}): {fixed_qps:.2f} QPS, Recall={fixed_recall:.3f}, Speedup={fixed_speedup:.2f}x")
                
                # Compare DPS vs best fixed
                qps_improvement = ((dps_qps - fixed_qps) / fixed_qps * 100) if fixed_qps > 0 else 0
                self.log(f"DPS vs Best Fixed: {qps_improvement:+.1f}% QPS difference")
        
        self.log("="*80 + "\n")
    
    def save_results(
        self,
        fixed_results: Dict,
        dps_results: Dict,
        baseline_results: Dict
    ):
        """Save results to JSON file."""
        results = {
            'dataset': self.dataset_path,
            'n_clusters': self.n_clusters,
            'dataset_size': len(self.smiles),
            'fixed_nprobe': fixed_results,
            'dps': dps_results,
            'baseline': baseline_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = os.path.join(self.output_dir, 'dps_benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"Results saved to {results_path}")
    
    def save_per_query_csv(self, dps_results: Dict):
        """Save per-query DPS metrics to CSV files."""
        self.log("Saving per-query CSV files")
        
        for k in dps_results.keys():
            if 'per_query_details' not in dps_results[k]:
                continue
            
            per_query_data = dps_results[k]['per_query_details']
            
            if not per_query_data:
                continue
            
            # Create DataFrame
            df = pd.DataFrame(per_query_data)
            
            # Sort by number of probes for easier analysis
            df = df.sort_values('n_probes', ascending=False)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f'dps_per_query_k{k}.csv')
            df.to_csv(csv_path, index=False, float_format='%.4f')
            
            self.log(f"  Saved per-query data for k={k} to {csv_path}")
            
            # Also create a summary showing probe distribution
            probe_counts = df['n_probes'].value_counts().sort_index()
            summary_path = os.path.join(self.output_dir, f'dps_probe_summary_k{k}.csv')
            
            summary_df = pd.DataFrame({
                'n_probes': probe_counts.index,
                'query_count': probe_counts.values,
                'percentage': (probe_counts.values / len(df) * 100)
            })
            summary_df.to_csv(summary_path, index=False, float_format='%.2f')
            
            self.log(f"  Saved probe count summary for k={k} to {summary_path}")
        
        # Also create a combined CSV with all k values
        all_data = []
        for k in dps_results.keys():
            if 'per_query_details' in dps_results[k]:
                all_data.extend(dps_results[k]['per_query_details'])
        
        if all_data:
            combined_df = pd.DataFrame(all_data)
            combined_df = combined_df.sort_values(['k', 'n_probes'], ascending=[True, False])
            combined_path = os.path.join(self.output_dir, 'dps_per_query_all.csv')
            combined_df.to_csv(combined_path, index=False, float_format='%.4f')
            self.log(f"  Saved combined per-query data to {combined_path}")
    
    def run_full_benchmark(
        self,
        k_values: List[int] = [10, 50, 100],
        n_probe_values: List[int] = [1, 2, 4, 8, 16, 32],
        max_probe: Optional[int] = None,
        prob_threshold: float = 0.1,
        temperature: float = 0.1,
        n_queries: int = 100,
        n_runs: int = 3
    ):
        """Run complete benchmark pipeline."""
        self.log("Starting DPS benchmark pipeline")
        self.log(f"Parameters: k_values={k_values}, n_probe_values={n_probe_values}")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Build index
        self.build_index()
        
        # Show effective max_probe after index is built
        effective_max_probe = max_probe if max_probe is not None else self.ivf_index.default_max_probe
        self.log(f"DPS params: max_probe={max_probe} (effective: {effective_max_probe}), prob_threshold={prob_threshold}, temperature={temperature}")
        
        # Step 3: Compute ground truth
        self.compute_ground_truth(k_values, n_queries)
        
        # Step 4: Run exhaustive baseline
        baseline_results = self.compute_exhaustive_search_baseline(k_values, n_runs)
        
        # Step 5: Run fixed n_probe benchmark
        fixed_results = self.run_fixed_nprobe_benchmark(k_values, n_probe_values, n_runs)
        
        # Step 6: Run DPS benchmark
        dps_results = self.run_dps_benchmark(k_values, max_probe, prob_threshold, temperature, n_runs)
        
        # Step 7: Generate plots
        self.plot_results(fixed_results, dps_results, baseline_results, k_values, n_probe_values)
        
        # Step 8: Print summary
        self.print_summary(fixed_results, dps_results, baseline_results, k_values, n_probe_values)
        
        # Step 9: Save results
        self.save_results(fixed_results, dps_results, baseline_results)
        
        # Step 10: Save per-query CSV files
        self.save_per_query_csv(dps_results)
        
        self.log("Benchmark complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test Dynamic Probe Selection for IVF search')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset (.smi or .csv)')
    parser.add_argument('--output-dir', type=str, default='search/results/dps_test', help='Output directory')
    parser.add_argument('--fp-type', type=str, default='morgan', choices=['morgan', 'rdkit'], help='Fingerprint type')
    parser.add_argument('--fp-size', type=int, default=2048, help='Fingerprint size')
    parser.add_argument('--radius', type=int, default=2, help='Morgan fingerprint radius')
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters (default: sqrt(n))')
    parser.add_argument('--threshold', type=float, default=0.65, help='BitBIRCH threshold')
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 50, 100], help='k values to test')
    parser.add_argument('--n-probe-values', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64], help='n_probe values to compare')
    parser.add_argument('--max-probe', type=int, default=None, help='Maximum probes for DPS (default: sqrt(n_samples))')
    parser.add_argument('--prob-threshold', type=float, default=0.6, help='Probability threshold for DPS')
    parser.add_argument('--temperature', type=float, default=0.1, help='Softmax temperature for DPS')
    parser.add_argument('--n-queries', type=int, default=100, help='Number of test queries')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of timing runs per query')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = DPSBenchmark(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        fp_type=args.fp_type,
        fp_size=args.fp_size,
        radius=args.radius,
        n_clusters=args.n_clusters,
        threshold=args.threshold
    )
    
    # Run benchmark
    benchmark.run_full_benchmark(
        k_values=args.k_values,
        n_probe_values=args.n_probe_values,
        max_probe=args.max_probe,
        prob_threshold=args.prob_threshold,
        temperature=args.temperature,
        n_queries=args.n_queries,
        n_runs=args.n_runs
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
