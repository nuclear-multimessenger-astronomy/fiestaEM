#!/bin/bash

#SBATCH --partition=milan-gpu
#SBATCH --job-name=train_bw_CVAE
#SBATCH --account=oz480
#SBATCH -o /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_gaussian_CVAE/logs/log
#SBATCH -e /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_gaussian_CVAE/logs/err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

#SBATCH --time=2-00:00:00

module load gcc/13.2.0 python/3.11.5 cuda/12.8.0
source /fred/oz480/mcoughli/envs/fiesta_test/bin/activate

ulimit -c 0
export LD_PRELOAD="$CUDA_HOME/targets/x86_64-linux/lib/libcusparse.so.12"

python train_blastwave_gaussian.py
