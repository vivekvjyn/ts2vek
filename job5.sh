#!/bin/bash
# SCRIPT NAME: job.sh

# Partition type
#SBATCH --partition=high

# Number of nodes
#SBATCH --nodes=1

# Number of tasks
#SBATCH --ntasks=1

# Number of tasks per node
#SBATCH --tasks-per-node=1

# Memory per node. 5 GB (In total, 10 GB)
#SBATCH --mem=16g

# Number of GPUs per node
#SBATCH --gres=gpu:1

# Select Intel nodes (with Infiniband)
#SBATCH --constraint=tesla

#SBATCH --cpus-per-task=8

# Modules
module load CUDA

cd /home/tnuttall/vivek/ts2vek

singularity run --nv svartok_latest.sif python exp5.py

