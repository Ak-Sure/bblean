#!/usr/bin/env python
"""
Enhanced benchmark runner for BitBIRCH IVF search with detailed timing and memory monitoring.

Features:
- Support for both .smi and .csv file formats
- Detailed timing measurements for each step
- Memory usage monitoring
- Organized results directory structure for multiple datasets
- Comprehensive reporting and visualization

Usage:
    python enhanced_benchmark.py --dataset ../data/subset_10k.csv [options]
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
import psutil
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("tqdm not available, falling back to standard progress reporting")
    TQDM_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.ivf_index import IVFIndex
from search.utils import load_smiles_file, generate_fingerprints, load_fingerprints
from search.similarity_engines import RDKitEngine, FPSim2Engine


class MemoryMonitor:
    """Simple memory monitoring utility."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        try:
            return self.process.memory_info().peak_wset / 1024 / 1024
        except AttributeError:
            # peak_wset not available on all platforms
            return self.get_memory_usage()['rss_mb']


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, logger_func=None):
        self.name = name
        self.logger_func = logger_func or print
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger_func(f"Starting: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.logger_func(f"Completed: {self.name} in {self.duration:.3f}s")


class EnhancedBenchmark:
    """Enhanced benchmark runner with detailed timing and memory monitoring."""
    
    def __init__(
        self,
        dataset_path: str,
        base_output_dir: str = 'search/results',
        fp_type: str = 'morgan',
        fp_size: int = 2048,
        radius: int = 2,
        threshold: float = 0.65,
        verbose: bool = True
    ):
        """Initialize the enhanced benchmark."""
        self.dataset_path = dataset_path
        self.base_output_dir = base_output_dir
        self.fp_type = fp_type
        self.fp_size = fp_size
        self.radius = radius
        self.threshold = threshold
        self.verbose = verbose
        
        # Extract dataset name and create specific output directory
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(base_output_dir, dataset_name)
        
        # Set up fingerprint parameters
        if self.fp_type.lower() == 'morgan':
            self.fp_params = {'radius': self.radius, 'nBits': self.fp_size}
        else:
            self.fp_params = {'fpSize': self.fp_size}
            
        # Create output directories
        self.data_dir = os.path.join(self.output_dir, 'data')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()
        
        # Initialize timing records
        self.timing_records = {}
        self.memory_records = {}
        
        # Initialize data structures
        self.smiles = []
        self.fingerprints = None
        self.fingerprints_rdkit = None
        self.query_indices = None
        self.query_fps = None
        self.query_fps_rdkit = None
        self.ground_truth = {}
        
        # Paths for saved files (will use .npy for dense, .npz for sparse)
        self.fps_path = os.path.join(self.data_dir, 'fingerprints.npz')  # Default, may change to .npy
        self.fps_rdkit_path = os.path.join(self.data_dir, 'fingerprints_rdkit.pickle')
        self.smiles_path = os.path.join(self.data_dir, 'smiles.json')
        self.queries_path = os.path.join(self.data_dir, 'queries.json')
        self.ground_truth_path = os.path.join(self.data_dir, 'ground_truth.pickle')
        self.results_path = os.path.join(self.data_dir, 'benchmark_results.json')
        self.ivf_index_path = os.path.join(self.data_dir, 'ivf_index.pickle')
        self.timing_path = os.path.join(self.data_dir, 'timing_results.json')
        self.memory_path = os.path.join(self.data_dir, 'memory_results.json')
        
    def log(self, message: str) -> None:
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def record_timing(self, step_name: str, duration: float) -> None:
        """Record timing for a step."""
        self.timing_records[step_name] = duration
        
    def record_memory(self, step_name: str, memory_usage: Dict[str, float]) -> None:
        """Record memory usage for a step."""
        self.memory_records[step_name] = memory_usage
        
    def load_data(self, force_reload: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Load SMILES and generate fingerprints with detailed timing and memory monitoring."""
        # Check if we can load from cache (try both .npz and .npy formats)
        fps_npz_path = self.fps_path
        fps_npy_path = self.fps_path.replace('.npz', '.npy')
        
        if (not force_reload and 
            (os.path.exists(fps_npz_path) or os.path.exists(fps_npy_path)) and 
            os.path.exists(self.smiles_path)):
            
            with TimingContext("Loading cached data", self.log) as timing_ctx:
                # Try sparse format first, then dense
                if os.path.exists(fps_npz_path):
                    from scipy import sparse as sp
                    self.log(f"Loading cached sparse fingerprints from {fps_npz_path}")
                    self.fingerprints = sp.load_npz(fps_npz_path)
                    is_sparse = True
                elif os.path.exists(fps_npy_path):
                    self.log(f"Loading cached dense fingerprints from {fps_npy_path}")
                    self.fingerprints = np.load(fps_npy_path)
                    is_sparse = False
                
                # Try to load RDKit fingerprints if available
                if os.path.exists(self.fps_rdkit_path):
                    self.log(f"Loading cached RDKit fingerprints from {self.fps_rdkit_path}")
                    with open(self.fps_rdkit_path, 'rb') as f:
                        self.fingerprints_rdkit = pickle.load(f)
                else:
                    self.log("RDKit fingerprints cache not found, will regenerate both formats")
                    force_reload = True
                
                if not force_reload:
                    self.log(f"Loading cached SMILES from {self.smiles_path}")
                    with open(self.smiles_path, 'r') as f:
                        self.smiles = json.load(f)
                        
                    if is_sparse:
                        self.log(f"Loaded {len(self.smiles)} molecules from cache")
                        self.log(f"Sparse matrix shape: {self.fingerprints.shape}, nnz: {self.fingerprints.nnz}")
                    else:
                        self.log(f"Loaded {len(self.smiles)} molecules from cache")
                        self.log(f"Dense matrix shape: {self.fingerprints.shape}")
                    
            if not force_reload:
                self.record_timing("load_cached_data", timing_ctx.duration)
                self.record_memory("load_cached_data", self.memory_monitor.get_memory_usage())
                return self.fingerprints, self.smiles
        
        # Load from file with timing
        with TimingContext("Loading SMILES file", self.log) as timing_ctx:
            self.log(f"Loading SMILES from {self.dataset_path}")
            self.smiles = load_smiles_file(self.dataset_path)
            self.log(f"Loaded {len(self.smiles)} SMILES")
            
        self.record_timing("load_smiles", timing_ctx.duration)
        self.record_memory("load_smiles", self.memory_monitor.get_memory_usage())
        
        # Generate fingerprints using default dense matrices
        self.log(f"Generating {self.fp_type} fingerprints ({self.fp_size} bits)")
        
        with TimingContext("Generating fingerprints", self.log) as timing_ctx:
            self.fingerprints = generate_fingerprints(
                self.smiles, 
                fp_type=self.fp_type, 
                fp_params=self.fp_params
                # Uses sparse=False by default (dense matrices)
            )
            
            self.log(f"Generated fingerprint matrix: {self.fingerprints.shape}")
        
        self.record_timing("generate_fingerprints", timing_ctx.duration)
        self.record_memory("generate_fingerprints", self.memory_monitor.get_memory_usage())
        
        # For backward compatibility, also generate RDKit format for similarity engines
        # This is still needed for RDKit similarity calculations
        with TimingContext("Generating RDKit fingerprints for similarity engines", self.log) as timing_ctx:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            
            # Convert SMILES to molecules  
            mols = [Chem.MolFromSmiles(s) for s in self.smiles]
            mols = [m for m in mols if m is not None]
            
            if self.fp_type.lower() == 'morgan':
                mgen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.fp_params.get('radius', 2),
                    fpSize=self.fp_params.get('nBits', 2048)
                )
                self.fingerprints_rdkit = [mgen.GetFingerprint(mol) for mol in mols]
            elif self.fp_type.lower() == 'rdkit':
                rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                    fpSize=self.fp_params.get('fpSize', 2048)
                )
                self.fingerprints_rdkit = [rdgen.GetFingerprint(mol) for mol in mols]
            else:
                raise ValueError(f"Unsupported fingerprint type: {self.fp_type}")
                
            self.log(f"Generated {len(self.fingerprints_rdkit)} RDKit fingerprints for similarity engines")
            
        self.record_timing("generate_rdkit_fingerprints", timing_ctx.duration)
        self.record_memory("generate_rdkit_fingerprints", self.memory_monitor.get_memory_usage())
        
        # Save to cache with timing
        with TimingContext("Saving fingerprints to cache", self.log) as timing_ctx:
            # Save fingerprints (check if sparse or dense)
            if hasattr(self.fingerprints, 'nnz'):  # sparse matrix
                from scipy import sparse as sp
                self.log(f"Saving sparse fingerprints to {self.fps_path}")
                sp.save_npz(self.fps_path, self.fingerprints)
            else:  # dense array
                self.log(f"Saving dense fingerprints to {self.fps_path.replace('.npz', '.npy')}")
                np.save(self.fps_path.replace('.npz', '.npy'), self.fingerprints)
            
            self.log(f"Saving RDKit fingerprints to {self.fps_rdkit_path}")
            with open(self.fps_rdkit_path, 'wb') as f:
                pickle.dump(self.fingerprints_rdkit, f)
            
            self.log(f"Saving SMILES to {self.smiles_path}")
            with open(self.smiles_path, 'w') as f:
                json.dump(self.smiles, f)
                
        self.record_timing("save_cache", timing_ctx.duration)
        self.record_memory("save_cache", self.memory_monitor.get_memory_usage())
            
        return self.fingerprints, self.smiles
    
    def build_ivf_index(
        self,
        method: str = 'rdkit',
        n_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
        force_rebuild: bool = False
    ) -> IVFIndex:
        """Build IVF index with BitBIRCH clustering and detailed timing."""
        # Check if we can load from cache
        if not force_rebuild and os.path.exists(self.ivf_index_path):
            with TimingContext("Loading cached IVF index", self.log) as timing_ctx:
                self.log(f"Loading cached IVF index from {self.ivf_index_path}")
                with open(self.ivf_index_path, 'rb') as f:
                    ivf_index = pickle.load(f)
                    
                # Check if the loaded index matches our parameters
                n_samples = self.fingerprints.shape[0] if hasattr(self.fingerprints, 'shape') else len(self.fingerprints)
                expected_n_clusters = n_clusters if n_clusters is not None else int(np.sqrt(n_samples))
                if (ivf_index.similarity_method == method and 
                    ivf_index.n_clusters == expected_n_clusters and
                    (threshold is None or abs(ivf_index.threshold - (threshold or self.threshold)) < 1e-6)):
                    self.log(f"Loaded IVF index with {ivf_index.n_clusters} clusters")
                    self.record_timing("load_cached_ivf_index", timing_ctx.duration)
                    return ivf_index
                else:
                    self.log(f"Cached index parameters don't match, rebuilding")
        
        # Make sure fingerprints are loaded
        if self.fingerprints is None:
            self.load_data()
            
        # Use default threshold if not specified
        if threshold is None:
            threshold = self.threshold
            
        # Build index with detailed timing
        with TimingContext("Building IVF index with BitBIRCH clustering", self.log) as timing_ctx:
            # Determine n_clusters if not specified - use sqrt(n_samples) as default
            if n_clusters is None:
                n_samples = self.fingerprints.shape[0] if hasattr(self.fingerprints, 'shape') else len(self.fingerprints)
                n_clusters = int(np.sqrt(n_samples))
                self.log(f"n_clusters not specified, using sqrt({n_samples}) = {n_clusters}")
            
            self.log(f"Building IVF index with BitBIRCH clustering (method={method}, n_clusters={n_clusters})")
            ivf_index = IVFIndex(n_clusters=n_clusters, similarity_method=method, threshold=threshold)
            
            # Monitor memory during index building
            initial_memory = self.memory_monitor.get_memory_usage()
            
            ivf_index.build_index(self.fingerprints, self.smiles, self.fingerprints_rdkit)
            
            final_memory = self.memory_monitor.get_memory_usage()
            
            self.log(f"Built IVF index with {ivf_index.n_clusters} clusters")
            
        self.record_timing("build_ivf_index", timing_ctx.duration)
        self.record_memory("build_ivf_index", final_memory)
        
        # Save to cache
        with TimingContext("Saving IVF index to cache", self.log) as save_timing_ctx:
            self.log(f"Saving IVF index to {self.ivf_index_path}")
            with open(self.ivf_index_path, 'wb') as f:
                pickle.dump(ivf_index, f)
                
        self.record_timing("save_ivf_index", save_timing_ctx.duration)
            
        return ivf_index
    
    def run_enhanced_benchmark(
        self,
        methods: List[str] = ['rdkit'],
        k_values: List[int] = [10, 50, 100],
        n_probe_values: List[int] = [1, 2, 4, 8, 16],
        threshold: float = 0.0,
        n_runs: int = 3,
        n_queries: int = 100,
        n_clusters: Optional[int] = None,
        skip_fingerprints: bool = False,
        skip_ground_truth: bool = False,
        skip_index_build: bool = False,
        skip_benchmarks: bool = False,
        force_reload: bool = False
    ) -> Dict:
        """Run the complete enhanced benchmark with detailed timing and memory monitoring."""
        
        self.log("=== Enhanced Benchmark Pipeline Starting ===")
        benchmark_start_time = time.time()
        
        # Stage 1: Load data and generate fingerprints
        if not skip_fingerprints:
            self.log("=== Stage 1: Loading data and generating fingerprints ===")
            self.load_data(force_reload=force_reload)
        else:
            self.log("=== Stage 1: Skipping fingerprint generation (loading from cache) ===")
            self.load_data(force_reload=False)
        
        # Stage 2: Select query molecules
        self.log("=== Stage 2: Selecting query molecules ===")
        with TimingContext("Selecting query molecules", self.log) as timing_ctx:
            self.select_queries(n_queries=n_queries, force_reload=force_reload)
        self.record_timing("select_queries", timing_ctx.duration)
        
        # Stage 3: Compute ground truth
        if not skip_ground_truth:
            self.log("=== Stage 3: Computing ground truth ===")
            with TimingContext("Computing ground truth", self.log) as timing_ctx:
                self.compute_ground_truth(k_values=k_values, threshold=threshold, force_reload=force_reload)
            self.record_timing("compute_ground_truth", timing_ctx.duration)
            self.record_memory("compute_ground_truth", self.memory_monitor.get_memory_usage())
        else:
            self.log("=== Stage 3: Skipping ground truth computation (loading from cache) ===")
            self.compute_ground_truth(k_values=k_values, threshold=threshold, force_reload=False)
        
        # Stage 4: Build IVF index
        if not skip_index_build:
            self.log("=== Stage 4: Building IVF index ===")
            ivf_index = self.build_ivf_index(
                method=methods[0],  # Use first method for index building
                n_clusters=n_clusters,
                threshold=threshold,
                force_rebuild=force_reload
            )
        else:
            self.log("=== Stage 4: Skipping index building (loading from cache) ===")
            ivf_index = self.build_ivf_index(
                method=methods[0],  # Use first method for index building
                n_clusters=n_clusters,
                threshold=threshold,
                force_rebuild=False
            )
        
        # Stage 5: Run benchmarks
        self.log("=== Stage 5: Running benchmarks ===")
        results = {
            'dataset': self.dataset_name,
            'dataset_path': self.dataset_path,
            'data_size': len(self.smiles),
            'n_queries': len(self.query_indices) if self.query_indices else 0,
            'parameters': {
                'fp_type': self.fp_type,
                'fp_size': self.fp_size,
                'n_clusters': ivf_index.n_clusters,
                'threshold': threshold
            },
            'timestamp': datetime.now().isoformat(),
            'timing': self.timing_records,
            'memory': self.memory_records,
            'flat_search_results': {},
            'ivf_search_results': {}
        }
        
        # Run flat and IVF search benchmarks for each method
        if not skip_benchmarks:
            for method in methods:
                self.log(f"Running benchmarks for method: {method}")
                method_results = {}
                
                # Flat search benchmark
                with TimingContext(f"Flat search benchmark ({method})", self.log) as timing_ctx:
                    flat_results = self.run_flat_search_benchmark(
                        method=method, k_values=k_values, threshold=threshold, n_runs=n_runs
                    )
                results['flat_search_results'][method] = flat_results
                self.record_timing(f"flat_search_{method}", timing_ctx.duration)
                
                # IVF search benchmark
                with TimingContext(f"IVF search benchmark ({method})", self.log) as timing_ctx:
                    ivf_results = self.run_ivf_search_benchmark(
                        ivf_index=ivf_index, k_values=k_values, n_probe_values=n_probe_values,
                        threshold=threshold, n_runs=n_runs
                    )
                results['ivf_search_results'][method] = ivf_results
                self.record_timing(f"ivf_search_{method}", timing_ctx.duration)
        else:
            self.log("=== Skipping benchmark execution ===")
            # Create empty results structure for consistency
            for method in methods:
                results['flat_search_results'][method] = {}
                results['ivf_search_results'][method] = {}
        
        # Stage 6: Save results
        self.log("=== Stage 6: Saving results ===")
        with TimingContext("Saving results", self.log) as timing_ctx:
            self.save_enhanced_results(results)
        self.record_timing("save_results", timing_ctx.duration)
        
        benchmark_total_time = time.time() - benchmark_start_time
        self.record_timing("total_benchmark", benchmark_total_time)
        
        self.log(f"=== Enhanced Benchmark Completed in {benchmark_total_time:.2f}s ===")
        
        return results
    
    # Include the essential methods from the original benchmark class
    def select_queries(self, n_queries: int = 100, seed: int = 42, force_reload: bool = False) -> List[int]:
        """Select random molecules to use as queries."""
        if not force_reload and os.path.exists(self.queries_path):
            self.log(f"Loading cached query indices from {self.queries_path}")
            with open(self.queries_path, 'r') as f:
                data = json.load(f)
                self.query_indices = data['query_indices']
                
            if self.fingerprints is not None:
                self.query_fps = self.fingerprints[self.query_indices]
            if self.fingerprints_rdkit is not None:
                self.query_fps_rdkit = [self.fingerprints_rdkit[i] for i in self.query_indices]
                
            self.log(f"Loaded {len(self.query_indices)} query indices from cache")
            return self.query_indices
            
        if self.fingerprints is None:
            self.load_data()
            
        self.log(f"Selecting {n_queries} random query molecules with seed {seed}")
        np.random.seed(seed)
        # Handle sparse matrix shape properly
        n_mols = self.fingerprints.shape[0] if hasattr(self.fingerprints, 'shape') else len(self.fingerprints)
        self.query_indices = np.random.choice(n_mols, size=min(n_queries, n_mols), replace=False).tolist()
        self.query_fps = self.fingerprints[self.query_indices]
        self.query_fps_rdkit = [self.fingerprints_rdkit[i] for i in self.query_indices]
        
        self.log(f"Saving query indices to {self.queries_path}")
        with open(self.queries_path, 'w') as f:
            json.dump({
                'query_indices': self.query_indices,
                'seed': seed,
                'n_queries': len(self.query_indices)
            }, f)
            
        return self.query_indices
    
    def compute_ground_truth(self, k_values: List[int] = [10, 50, 100], threshold: float = 0.0, force_reload: bool = False) -> Dict:
        """Compute ground truth results using exhaustive search."""
        if not force_reload and os.path.exists(self.ground_truth_path):
            self.log(f"Loading cached ground truth from {self.ground_truth_path}")
            with open(self.ground_truth_path, 'rb') as f:
                self.ground_truth = pickle.load(f)
            return self.ground_truth
        
        if self.fingerprints is None:
            self.load_data()
        if self.query_indices is None:
            self.select_queries()
            
        self.log("Computing ground truth using exhaustive search")
        rdkit_engine = RDKitEngine()
        
        self.ground_truth = {}
        
        for k in k_values:
            self.log(f"Computing ground truth for k={k}")
            self.ground_truth[k] = {}
            
            if TQDM_AVAILABLE:
                for query_idx in tqdm(self.query_indices, desc=f"Ground truth for k={k}"):
                    query_fp_rdkit = self.fingerprints_rdkit[query_idx]
                    results = rdkit_engine.top_k_similarity(
                        query_fp_rdkit, self.fingerprints_rdkit, k=k, threshold=threshold
                    )
                    self.ground_truth[k][query_idx] = results
            else:
                for i, query_idx in enumerate(self.query_indices):
                    if i % 10 == 0:
                        self.log(f"Processing query {i+1}/{len(self.query_indices)} for k={k}")
                    query_fp_rdkit = self.fingerprints_rdkit[query_idx]
                    results = rdkit_engine.top_k_similarity(
                        query_fp_rdkit, self.fingerprints_rdkit, k=k, threshold=threshold
                    )
                    self.ground_truth[k][query_idx] = results
        
        self.log(f"Saving ground truth to {self.ground_truth_path}")
        with open(self.ground_truth_path, 'wb') as f:
            pickle.dump(self.ground_truth, f)
            
        return self.ground_truth
    
    def run_flat_search_benchmark(self, method: str = 'rdkit', k_values: List[int] = [10, 50, 100], threshold: float = 0.0, n_runs: int = 3) -> Dict:
        """Run flat search benchmark with detailed timing."""
        results = {}
        
        if method.lower() == 'rdkit':
            engine = RDKitEngine()
        elif method.lower() == 'fpsim2':
            engine = FPSim2Engine()
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        for k in k_values:
            results[k] = {'query_times': [], 'recalls': []}
            
            for query_idx in self.query_indices:
                if method.lower() == 'rdkit':
                    query_fp = self.fingerprints_rdkit[query_idx]
                    target_fps = self.fingerprints_rdkit
                else:
                    query_fp = self.fingerprints[query_idx]
                    target_fps = self.fingerprints
                    
                gt_results = self.ground_truth[k][query_idx]
                gt_indices = set(result['index'] for result in gt_results)
                
                query_times = []
                for run in range(n_runs):
                    start_time = time.time()
                    run_results = engine.top_k_similarity(query_fp, target_fps, k=k, threshold=threshold)
                    end_time = time.time()
                    query_times.append(end_time - start_time)
                
                result_indices = set(result['index'] for result in run_results)
                recall = len(gt_indices.intersection(result_indices)) / len(gt_indices) if gt_indices else 1.0
                
                results[k]['query_times'].extend(query_times)
                results[k]['recalls'].append(recall)
                
            results[k]['avg_query_time'] = float(np.mean(results[k]['query_times']))
            results[k]['avg_recall'] = float(np.mean(results[k]['recalls']))
            results[k]['qps'] = float(1.0 / results[k]['avg_query_time'])
            
        return results
    
    def run_ivf_search_benchmark(self, ivf_index: IVFIndex, k_values: List[int] = [10, 50, 100], n_probe_values: List[int] = [1, 2, 4, 8, 16], threshold: float = 0.0, n_runs: int = 3) -> Dict:
        """Run IVF search benchmark with detailed timing."""
        results = {}
        
        for k in k_values:
            results[k] = {}
            
            for n_probe in n_probe_values:
                results[k][n_probe] = {'query_times': [], 'recalls': []}
                
                for query_idx in self.query_indices:
                    if ivf_index.similarity_method.lower() == 'rdkit':
                        query_fp = self.fingerprints_rdkit[query_idx]
                    else:
                        query_fp = self.fingerprints[query_idx]
                        
                    gt_results = self.ground_truth[k][query_idx]
                    gt_indices = set(result['index'] for result in gt_results)
                    
                    query_times = []
                    for run in range(n_runs):
                        start_time = time.time()
                        run_results = ivf_index.search(query_fp, k=k, n_probe=n_probe, threshold=threshold)
                        end_time = time.time()
                        query_times.append(end_time - start_time)
                    
                    result_indices = set(result['index'] for result in run_results)
                    recall = len(gt_indices.intersection(result_indices)) / len(gt_indices) if gt_indices else 1.0
                    
                    results[k][n_probe]['query_times'].extend(query_times)
                    results[k][n_probe]['recalls'].append(recall)
                    
                results[k][n_probe]['avg_query_time'] = float(np.mean(results[k][n_probe]['query_times']))
                results[k][n_probe]['avg_recall'] = float(np.mean(results[k][n_probe]['recalls']))
                results[k][n_probe]['qps'] = float(1.0 / results[k][n_probe]['avg_query_time'])
                
        return results
    
    def save_enhanced_results(self, results: Dict) -> None:
        """Save enhanced benchmark results."""
        # Update timing and memory records
        results['timing'] = self.timing_records
        results['memory'] = self.memory_records
        results['timestamp'] = datetime.now().isoformat()
        
        # Save main results
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save timing results separately
        with open(self.timing_path, 'w') as f:
            json.dump(self.timing_records, f, indent=2)
            
        # Save memory results separately
        with open(self.memory_path, 'w') as f:
            json.dump(self.memory_records, f, indent=2, default=str)
            
        self.log(f"Enhanced results saved to {self.results_path}")
        self.log(f"Timing results saved to {self.timing_path}")
        self.log(f"Memory results saved to {self.memory_path}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run enhanced BitBIRCH IVF search benchmarks with detailed timing and memory monitoring'
    )
    
    parser.add_argument('--dataset', required=True, help='Path to dataset file (.smi or .csv)')
    parser.add_argument('--output-dir', default='search/results', help='Base output directory')
    parser.add_argument('--fp-type', default='morgan', choices=['morgan', 'rdkit'], help='Fingerprint type')
    parser.add_argument('--fp-size', type=int, default=2048, help='Fingerprint size in bits')
    parser.add_argument('--methods', nargs='+', default=['rdkit'], choices=['rdkit', 'fpsim2'], help='Methods to test')
    parser.add_argument('--k-values', nargs='+', type=int, default=[10, 50, 100], help='k values for top-k search')
    parser.add_argument('--n-probe-values', nargs='+', type=int, default=[1, 2, 4, 8, 16], help='n_probe values for IVF search')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of runs per query')
    parser.add_argument('--n-queries', type=int, default=100, help='Number of query molecules')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters for IVF index')
    parser.add_argument('--threshold', type=float, default=0.65, help='Similarity threshold')
    parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')
    
    # Stage control arguments
    parser.add_argument('--skip-fingerprints', action='store_true', help='Skip fingerprint generation (load from cache)')
    parser.add_argument('--skip-ground-truth', action='store_true', help='Skip ground truth computation (load from cache)')
    parser.add_argument('--skip-index-build', action='store_true', help='Skip index building (load from cache)')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip benchmark execution')
    parser.add_argument('--force-reload', action='store_true', help='Force regeneration of all cached data')
    
    args = parser.parse_args()
    
    # Create enhanced benchmark instance
    benchmark = EnhancedBenchmark(
        dataset_path=args.dataset,
        base_output_dir=args.output_dir,
        fp_type=args.fp_type,
        fp_size=args.fp_size,
        threshold=args.threshold,
        verbose=args.verbose
    )
    
    try:
        results = benchmark.run_enhanced_benchmark(
            methods=args.methods,
            k_values=args.k_values,
            n_probe_values=args.n_probe_values,
            threshold=args.threshold,
            n_runs=args.n_runs,
            n_queries=args.n_queries,
            n_clusters=args.n_clusters,
            skip_fingerprints=args.skip_fingerprints,
            skip_ground_truth=args.skip_ground_truth,
            skip_index_build=args.skip_index_build,
            skip_benchmarks=args.skip_benchmarks,
            force_reload=args.force_reload
        )
        
        print("\n=== ENHANCED BENCHMARK SUMMARY ===")
        print(f"Dataset: {results['dataset']}")
        print(f"Size: {results['data_size']} molecules")
        print(f"Queries: {results['n_queries']}")
        print(f"Total time: {results['timing']['total_benchmark']:.2f}s")
        
        print("\nStep timings:")
        for step, duration in results['timing'].items():
            if step != 'total_benchmark' and duration is not None:
                print(f"  {step}: {duration:.3f}s")
        
        print(f"\nResults saved to: {benchmark.output_dir}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())