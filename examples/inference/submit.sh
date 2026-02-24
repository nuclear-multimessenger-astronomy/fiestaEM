#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=40G
#SBATCH --output="./outdir_KN_SMC/log.out"
#SBATCH --job-name="fiesta_SMC"

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
source /home/twouters2/projects/13_fiesta_dev/.venv/bin/activate

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=================================="
echo "=== Running inference (SMC-RW) ==="
echo "=================================="

python inference_KN_SMC.py