#!/bin/bash
#SBATCH --job-name=bw_data
#SBATCH --account=oz480
#SBATCH --partition=milan
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/logs/blastwave_data_%A_%a.out
#SBATCH --error=/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/logs/blastwave_data_%A_%a.err

module load gcc/13.2.0 python/3.11.5
source /fred/oz480/mcoughli/envs/fiesta_test/bin/activate

# Work from /fred so no core dumps or temp files land in $HOME
cd /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/
ulimit -c 0  # disable core dumps

python /home/mcoughli/fiestaEM/surrogates/GRB/_create_data/create_data_blastwave_gaussian_chunk.py ${SLURM_ARRAY_TASK_ID} 1000 16
