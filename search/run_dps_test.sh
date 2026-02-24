#!/bin/bash
#SBATCH --job-name=DPS10k  # Job name
#SBATCH --mail-user=ak.surendran@ufl.edu  # Email for notifications
#SBATCH --mail-type=FAIL,END  # When to send email notifications
#SBATCH --error=DPS10k.err  # File to log errors
#SBATCH --output=DPS10k.out  # File to log standard output
#SBATCH --cpus-per-task=1  # Number of CPUs to allocate
#SBATCH --mem=100G  # Request 100 GB of memory
#SBATCH --time=20:00:00  # Maximum runtime (hh:mm:ss)
#SBATCH --nodes=1  # Use a single node
#SBATCH --account=rmirandaquintana  # Account name
#SBATCH --qos=rmirandaquintana  # Quality of service

# Activate your virtual environment
source /home/ak.surendran/blue/ak.surendran/cdr_bench/cdr_env/bin/activate

# Load necessary modules
module load python/3.11  

# Example script to run DPS benchmark

# Usage examples:

# Basic test with default parameters
echo "Running DPS test with default parameters..."
python3.11 test_dps.py --dataset data/subset_10k.csv --output-dir search/results/dps_test/t0.08/0.5/10k --n-queries 100 --prob-threshold 0.5 --temperature 0.08

# Test with custom parameters
# After running tests, you can analyze the per-query CSV files:
echo ""
echo "To analyze per-query results, run:"
echo "python analyze_dps_csv.py --csv-file search/results/dps_test/verbose/0.6/10k/dps_per_query_k100.csv --create-plots"
    