#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory (8 GB)
#SBATCH -p gpu  # Partition for GPU nodes
#SBATCH -G 1  # Number of GPUs requested
#SBATCH -t 01:00:00  # Job time limit (1 hour)
#SBATCH -o slurm-%j.out  # Output file (%j = job ID)

module load cuda/11.8  # Load CUDA module
module load python/3.x  # Load the Python module (update "3.x" to the specific version if needed)

# Run the Python script with the desired arguments
python mf_simple_new.py --data Data/ml-1m --epochs 256 --embedding_dim 16 --regularization 0.005 --negatives 8 --learning_rate 0.002 --stddev 0.1
