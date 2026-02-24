#!/bin/bash
#SBATCH --job-name=Comparison_RDKIT  # Job name
#SBATCH --mail-user=ak.surendran@ufl.edu  # Email for notifications
#SBATCH --mail-type=FAIL,END  # When to send email notifications
#SBATCH --error=Comparison_RDKIT.err  # File to log errors
#SBATCH --output=Comparison_RDKIT.out  # File to log standard output
#SBATCH --cpus-per-task=10  # Number of CPUs to allocate
#SBATCH --mem=20G  # Request 400 GB of memory
#SBATCH --time=20:00:00  # Maximum runtime (hh:mm:ss)
#SBATCH --nodes=1  # Use a single node
#SBATCH --account=rmirandaquintana  # Account name
#SBATCH --qos=rmirandaquintana  # Quality of service

# Activate your virtual environment
source /home/ak.surendran/blue/ak.surendran/cdr_bench/cdr_env/bin/activate

# Load necessary modules
module load python/3.11  

python3.11 plot_comparison.py