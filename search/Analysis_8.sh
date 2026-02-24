#!/bin/bash
#SBATCH --job-name=BBlean8  # Job name
#SBATCH --mail-user=ak.surendran@ufl.edu  # Email for notifications
#SBATCH --mail-type=FAIL,END  # When to send email notifications
#SBATCH --error=BBlean8.err  # File to log errors
#SBATCH --output=BBlean8.out  # File to log standard output
#SBATCH --cpus-per-task=10  # Number of CPUs to allocate
#SBATCH --mem=200G  # Request 400 GB of memory
#SBATCH --time=20:00:00  # Maximum runtime (hh:mm:ss)
#SBATCH --nodes=1  # Use a single node
#SBATCH --account=rmirandaquintana  # Account name
#SBATCH --qos=rmirandaquintana  # Quality of service

# Activate your virtual environment
source /home/ak.surendran/blue/ak.surendran/cdr_bench/cdr_env/bin/activate

# Load necessary modules
module load python/3.11  

python3.11 run_all_benchmarks.py --data-dir data --output-dir search/results/RDKIT8 --fp-type rdkit --k-values 1 5 10 50 100 --n-probe-values 1 2 4 8 16 32 64 --n-runs 1 --n-queries 100 --run-plots --threshold 0.8 --verbose --force-reload