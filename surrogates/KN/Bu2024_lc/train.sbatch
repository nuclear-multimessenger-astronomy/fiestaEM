#!/bin/bash

#SBATCH --partition gpu
#SBATCH --job-name=train_Bu2024
#SBATCH -o ./log
#SBATCH -e ./err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=24:00:00

module load miniforge3
eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate fiesta

python train_Bu2024.py
