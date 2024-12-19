#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=5G
#SBATCH --output=outdir_GRB170817_tophat/log.out
#SBATCH --job-name=GRB170817

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/ninjax

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
python run_GRB170817_tophat.py

echo "DONE"