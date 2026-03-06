#!/bin/bash
#SBATCH --job-name=bw_rs_cvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=milan-gpu
#SBATCH --time=02:00:00
#SBATCH --output=/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_rs_gaussian_CVAE/train.out
#SBATCH --error=/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_rs_gaussian_CVAE/train.err

module load gcccore/13.2.0 python/3.11.5
source /fred/oz480/mcoughli/envs/fiesta_test/bin/activate

nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices())"

mkdir -p /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_rs_gaussian_CVAE/model
mkdir -p /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_rs_gaussian_CVAE/benchmarks

python /fred/oz480/mcoughli/fiestaEM/surrogates/GRB/blastwave_rs_gaussian_CVAE/train_blastwave_rs_gaussian.py
