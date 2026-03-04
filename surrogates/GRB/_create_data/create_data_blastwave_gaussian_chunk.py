"""Generate one chunk of blastwave Gaussian data. Intended for SLURM array jobs.

Usage:
    python create_data_blastwave_gaussian_chunk.py <chunk_id> <n_samples> <n_pool>

Writes to /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/blastwave_gaussian_chunk_<chunk_id>.h5
"""
import sys
import numpy as np
from fiesta.train.AfterglowData import BlastwaveData

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <chunk_id> <n_samples> <n_pool>", file=sys.stderr)
        sys.exit(1)
    chunk_id = int(sys.argv[1])
    n_samples = int(sys.argv[2])
    n_pool = int(sys.argv[3])

    tmin = 1e-4  # days
    tmax = 2000  # days
    n_times = 250

    numin = 1e9   # Hz
    numax = 2.5e19  # Hz (100 keV)
    n_nu = 256

    parameter_distributions = {
        'inclination_EM': (0, np.pi/2, "uniform"),
        'log10_E0': (47, 57, "uniform"),
        'thetaCore': (0.01, np.pi/5, "loguniform"),
        'log10_n0': (-6, 2, "uniform"),
        'p': (2.01, 3, "uniform"),
        'log10_epsilon_e': (-4, 0, "uniform"),
        'log10_epsilon_B': (-8, 0, "uniform"),
        'log10_lf': (1.0, 3.5, "uniform"),
    }

    jet_type = 2

    outfile = f"/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/blastwave_gaussian_chunk_{chunk_id:03d}.h5"

    # Use different random seed per chunk to avoid duplicate samples
    np.random.seed(42 + chunk_id)

    creator = BlastwaveData(outfile=outfile,
                           jet_type=jet_type,
                           n_training=n_samples,
                           n_val=0,
                           n_test=0,
                           parameter_distributions=parameter_distributions,
                           n_pool=n_pool,
                           tmin=tmin,
                           tmax=tmax,
                           n_times=n_times,
                           numin=numin,
                           numax=numax,
                           n_nu=n_nu)

    print(f"Chunk {chunk_id} complete: {n_samples} samples written to {outfile}")
