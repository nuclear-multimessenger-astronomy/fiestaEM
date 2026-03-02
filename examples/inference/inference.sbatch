#!/bin/bash

#SBATCH -J AT2017gfo
#SBATCH -o ./log
#SBATCH -e ./err

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

srun python inference_joint.py
