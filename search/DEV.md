# Development Commands for BitBIRCH IVF Search Benchmarks

This document contains the commands needed to run comprehensive benchmarks on the BitBIRCH IVF search implementation.

## TL;DR

Last command i did was 
```
python ./run_all_benchmarks.py --data-dir ../data --k-values 1 5 10 50 100 --n-probe-values 1 2 4 8 16 32 64 --n-runs 1 --n-queries 100 --run-plots --verbose --skip-fingerprints
```
(fingerprints were computed from previous rund, if you dont have them, remove `--skip-fingerprints`)

## Quick Test Command

Test the implementation on 10k dataset with minimal parameters:

```bash
python search/enhanced_benchmark.py \
  --dataset data/subset_10k.csv \
  --k-values 10 50 \
  --n-probe-values 1 4 16 \
  --n-runs 1 \
  --n-queries 20 \
  --verbose
```

## Single Dataset - Full Benchmark

Run comprehensive benchmark on 10k dataset:

```bash
python search/enhanced_benchmark.py \
  --dataset data/subset_10k.csv \
  --k-values 1 5 10 50 100 \
  --n-probe-values 1 2 4 8 16 32 64 \
  --n-runs 1 \
  --n-queries 100 \
  --verbose
```

## All Datasets - Full Benchmark with Plots

Run comprehensive benchmarks on all datasets (10k, 100k, 1M) and generate plots:

```bash
python search/run_all_benchmarks.py \
  --data-dir data \
  --k-values 1 5 10 50 100 \
  --n-probe-values 1 2 4 8 16 32 64 \
  --n-runs 1 \
  --n-queries 100 \
  --run-plots \
  --verbose
```

## Index Building Only

Just build indexes without running benchmarks (useful for debugging):

```bash
python search/run_all_benchmarks.py \
  --data-dir data \
  --skip-fingerprints \
  --skip-ground-truth \
  --skip-benchmarks \
  --verbose
```

## Force Rebuild Everything

Force regeneration of all cached data:

```bash
python search/run_all_benchmarks.py \
  --data-dir data \
  --force-reload \
  --k-values 1 10 50 100 \
  --n-probe-values 1 2 4 8 16 32 64 \
  --n-runs 3 \
  --n-queries 100 \
  --run-plots \
  --verbose
```

## Generate Plots Only

Generate plots from existing benchmark results:

```bash
python search/plot_results.py subset_10k subset_100k subset_1M
```

## Parameter Explanations

- `--skip-fingerprints`: Load fingerprints from cache (saves time)
- `--skip-ground-truth`: Load ground truth from cache
- `--skip-index-build`: Load IVF index from cache
- `--skip-benchmarks`: Skip benchmark execution (useful for index building only)
- `--force-reload`: Force regeneration of all cached data
- `--k-values`: Number of nearest neighbors to retrieve (1, 10, 50, 100)
- `--n-probe-values`: Number of clusters to search (1, 2, 4, 8, 16, 32, 64)
- `--n-runs`: Number of runs per query for timing accuracy
- `--n-queries`: Number of query molecules to test
- `--n-clusters`: Exact number of clusters for IVF index (optional, defaults to sqrt(dataset_size))
- `--run-plots`: Generate plots after all benchmarks complete
- `--verbose`: Enable detailed logging

## Notes

- Fingerprints are cached after first generation, so subsequent runs with `--skip-fingerprints` are much faster
- Ground truth computation is also cached
- IVF indexes are cached and will be rebuilt if parameters change
- Use `--dry-run` to see what would be executed without actually running benchmarks