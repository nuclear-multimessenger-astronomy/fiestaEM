#!/bin/bash
#SBATCH --job-name=bw_rs_data
#SBATCH --array=0-219
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/logs/bw_rs_%a.out
#SBATCH --error=/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/logs/bw_rs_%a.err

# 220 chunks x 100 samples = 22,000 total (20K train + 2K val)
# At ~2s/sample, each chunk takes ~200s

module load gcccore/13.2.0 python/3.11.5
source /fred/oz480/mcoughli/envs/blastwave/bin/activate

python /fred/oz480/mcoughli/fiestaEM/surrogates/GRB/_create_data/create_data_blastwave_rs_gaussian_chunk.py \
    ${SLURM_ARRAY_TASK_ID} 100 4
