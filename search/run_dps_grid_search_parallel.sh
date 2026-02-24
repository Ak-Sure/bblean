#!/bin/bash
#SBATCH --job-name=DPS_grid_parallel  # Job name
#SBATCH --mail-user=ak.surendran@ufl.edu  # Email for notifications
#SBATCH --mail-type=FAIL,END  # When to send email notifications
#SBATCH --error=DPS_grid_parallel_%j.err  # File to log errors (%j = job ID)
#SBATCH --output=DPS_grid_parallel_%j.out  # File to log standard output
#SBATCH --cpus-per-task=20  # Request 20 CPUs for parallel execution
#SBATCH --mem=200G  # Request 200 GB of memory
#SBATCH --time=150:00:00  # Maximum runtime (150 hours - increased due to larger dataset and parallelization)
#SBATCH --nodes=1  # Use a single node
#SBATCH --account=rmirandaquintana  # Account name
#SBATCH --qos=rmirandaquintana  # Quality of service

# Activate your virtual environment
source /home/ak.surendran/blue/ak.surendran/cdr_bench/cdr_env/bin/activate

# Load necessary modules
module load python/3.11  

echo "=================================="
echo "DPS Grid Search (PARALLELIZED)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Set dataset path
DATASET="data/subset_1M.csv"

# Set output base directory
OUTPUT_BASE="search/results/norm/dps_grid_1M_parallel_0.25"

# ========================================
# CHOOSE ONE OF THE FOLLOWING CONFIGURATIONS
# ========================================

# ------- FINE GRID (5x5 = 25 combinations) -------
# With 20 workers: ~25/20 = 1.25 sequential batches
# Estimated time: 1-2 hours (vs 10-12 hours sequential)
TEMPERATURES=(0.01 0.05 0.08 0.1 0.2 0.5 0.8 1)
PROB_THRESHOLDS=(0.4 0.5 0.6 0.7 0.8)

# ------- COARSE GRID (3x3 = 9 combinations) -------
# With 20 workers: ~9/20 = 0.45 sequential batches
# Estimated time: <1 hour (vs 3-4 hours sequential)
# TEMPERATURES=(0.05 0.1 0.2)
# PROB_THRESHOLDS=(0.3 0.5 0.7)

# ------- FOCUSED GRID (4x4 = 16 combinations) -------
# Focused on middle range values
# With 20 workers: ~16/20 = 0.8 sequential batches
# Estimated time: <1 hour (vs 5-7 hours sequential)
# TEMPERATURES=(0.08 0.1 0.12 0.15)
# PROB_THRESHOLDS=(0.4 0.5 0.6 0.7)

# ------- EXTENSIVE GRID (7x7 = 49 combinations) -------
# Very comprehensive search
# With 20 workers: ~49/20 = 2.45 sequential batches
# Estimated time: 2-3 hours (vs 20+ hours sequential)
# TEMPERATURES=(0.01 0.05 0.08 0.1 0.15 0.2 0.3)
# PROB_THRESHOLDS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)

echo "Configuration:"
echo "  Temperatures: ${TEMPERATURES[@]}"
echo "  Prob thresholds: ${PROB_THRESHOLDS[@]}"
echo "  Number of workers: $SLURM_CPUS_PER_TASK"
echo "  Output directory: $OUTPUT_BASE"
echo ""

# Run parallelized grid search
python3.11 test_dps_grid_search_parallel.py \
    --dataset "$DATASET" \
    --temperatures ${TEMPERATURES[@]} \
    --prob-thresholds ${PROB_THRESHOLDS[@]} \
    --n-workers $SLURM_CPUS_PER_TASK \
    --output-base "$OUTPUT_BASE" \
    --n-queries 100 \
    --k-values 10 50 100 \
    --n-probe-values 1 2 4 8 16 32 64

echo ""
echo "=================================="
echo "Grid search complete!"
echo "End time: $(date)"
echo ""
echo "Results saved in: $OUTPUT_BASE"
echo "Summary file: $OUTPUT_BASE/grid_search_summary.json"
echo ""
echo "Next steps:"
echo "  1. Run analysis: sbatch run_analyze_grid.sh"
echo "  2. Update GRID_DIR in run_analyze_grid.sh to: $OUTPUT_BASE"
echo "=================================="
