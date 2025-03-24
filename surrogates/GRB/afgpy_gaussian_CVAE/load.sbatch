#!/bin/bash

#SBATCH --partition cpu
#SBATCH --job-name=load
#SBATCH -o ./log_cpu
#SBATCH -e ./err_cpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G


#SBATCH --time=24:00:00

module load miniforge3
eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate fiesta

python train_afterglowpy_gaussian.py
