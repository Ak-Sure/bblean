#!/bin/bash
#SBATCH --job-name=DPS_grid  # Job name
#SBATCH --mail-user=ak.surendran@ufl.edu  # Email for notifications
#SBATCH --mail-type=FAIL,END  # When to send email notifications
#SBATCH --error=DPS_grid_%j.err  # File to log errors (%j = job ID)
#SBATCH --output=DPS_grid_%j.out  # File to log standard output
#SBATCH --cpus-per-task=10  # Number of CPUs to allocate
#SBATCH --mem=50G  # Request 50 GB of memory
#SBATCH --time=20:00:00  # Maximum runtime (hh:mm:ss) - longer for grid search
#SBATCH --nodes=1  # Use a single node
#SBATCH --account=rmirandaquintana  # Account name
#SBATCH --qos=rmirandaquintana  # Quality of service

# Activate your virtual environment
source /home/ak.surendran/blue/ak.surendran/cdr_bench/cdr_env/bin/activate

# Load necessary modules
module load python/3.11  

# Run DPS hyperparameter grid search
echo "=================================="
echo "DPS Hyperparameter Grid Search"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Example 1: Fine-grained grid search (5x5 = 25 combinations)
python3.11 test_dps_grid_search.py \
    --dataset data/subset_10k.csv \
    --temperatures 0.01 0.05 0.08 0.1 0.2 0.5 0.8 1 \
    --prob-thresholds 0.4 0.5 0.6 0.7 0.8 \
    --output-base search/results/kmeans/dps_grid_10k_fine \
    --n-queries 100 \
    --n-runs 1

# Example 2: Coarse grid search (3x3 = 9 combinations) - Faster
# python3.11 test_dps_grid_search.py \
#     --dataset data/subset_10k.csv \
#     --temperatures 0.05 0.1 0.2 \
#     --prob-thresholds 0.3 0.5 0.7 \
#     --output-base search/results/kmeans/dps_grid_10k_coarse \
#     --n-queries 100 \
#     --n-runs 3

# Example 3: Focused search around best parameters
# python3.11 test_dps_grid_search.py \
#     --dataset data/subset_10k.csv \
#     --temperatures 0.06 0.08 0.10 0.12 \
#     --prob-thresholds 0.45 0.50 0.55 0.60 \
#     --output-base search/results/kmeans/dps_grid_10k_focused \
#     --n-queries 100 \
#     --n-runs 3

echo ""
echo "End time: $(date)"
echo "=================================="
