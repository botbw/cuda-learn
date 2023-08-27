#!/bin/bash

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --job-name=cuda
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# module load anaconda
# source activate TestEnv
# python test.py

cd ~/cuda-learn/chap3
PATH="/usr/local/cuda-12.1/bin:$PATH"

nvcc reduceInteger.cu
nvprof --query-metrics
nvprof ./a.out
nvprof --metrics gld_throughput ./a.out
nvprof --metrics gld_efficiency ./a.out
