#!/bin/bash
#SBATCH --job-name=DPS_analyze  # Job name
#SBATCH --mail-user=ak.surendran@ufl.edu  # Email for notifications
#SBATCH --mail-type=FAIL,END  # When to send email notifications
#SBATCH --error=DPS_analyze_%j.err  # File to log errors (%j = job ID)
#SBATCH --output=DPS_analyze_%j.out  # File to log standard output
#SBATCH --cpus-per-task=1  # Number of CPUs to allocate
#SBATCH --mem=32G  # Request 32 GB of memory (analysis is lighter than grid search)
#SBATCH --time=02:00:00  # Maximum runtime (2 hours should be plenty)
#SBATCH --nodes=1  # Use a single node
#SBATCH --account=rmirandaquintana  # Account name
#SBATCH --qos=rmirandaquintana  # Quality of service

# Activate your virtual environment
source /home/ak.surendran/blue/ak.surendran/cdr_bench/cdr_env/bin/activate

# Load necessary modules
module load python/3.11  

echo "=================================="
echo "DPS Grid Search Results Analysis"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Set the grid directory to analyze
# MODIFY THIS to match your grid search output directory
GRID_DIR="search/results/dps_grid_100k_parallel_0.25"  # Example: "search/results/norm/dps_grid_100k_parallel_0.25"

# Check if directory exists
if [ ! -d "$GRID_DIR" ]; then
    echo "ERROR: Grid directory not found: $GRID_DIR"
    echo "Please update GRID_DIR variable in this script"
    exit 1
fi

echo "Analyzing results from: $GRID_DIR"
echo ""

# Run analysis for different k values
for K_VALUE in 10 50 100; do
    echo "=================================="
    echo "Analyzing k=${K_VALUE}"
    echo "=================================="
    
    python3.11 analyze_grid_search.py \
        --grid-dir "$GRID_DIR" \
        --k-value $K_VALUE \
        --output-dir "$GRID_DIR/analysis_k${K_VALUE}"
    
    echo ""
done

echo "=================================="
echo "Analysis Complete!"
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - $GRID_DIR/analysis_k10/"
echo "  - $GRID_DIR/analysis_k50/"
echo "  - $GRID_DIR/analysis_k100/"
echo "=================================="
