"""Generate one chunk of blastwave+RS Gaussian data. Intended for SLURM array jobs.

Usage:
    python create_data_blastwave_rs_gaussian_chunk.py <chunk_id> <n_samples> <n_pool>

Writes to /fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/blastwave_rs_gaussian_chunk_<chunk_id>.h5
"""
import sys
import numpy as np
from fiesta.train.AfterglowData import BlastwaveRSData

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

    # Parameter ranges following Japelj+ 2014 (1402.3701)
    # RS microphysics tied to FS: eps_e_rs = eps_e, eps_b_rs = RB * eps_b, p_rs = p
    parameter_distributions = {
        'inclination_EM': (0, 1.5707963267948966, "uniform"),
        'log10_E0': (50, 56, "uniform"),              # EK in [1e50, 1e56] erg
        'thetaCore': (0.01, 0.6283185307179586, "loguniform"),
        'log10_n0': (-1, 4, "uniform"),                # n in [0.1, 1e4] cm^-3
        'p': (2.01, 3, "uniform"),
        'log10_epsilon_e': (-4, -0.3010299956639812, "uniform"),  # eps_e in [1e-4, 0.5]
        'log10_epsilon_B': (-5, -1, "uniform"),        # eps_B,f in [1e-5, 0.1]
        'log10_lf': (1.6989700043360187, 4, "uniform"),  # Gamma0 in [50, 1e4]
        'log10_RB': (0, 5, "uniform"),                 # RB = eps_B,r/eps_B,f in [1, 1e5]
        'log10_duration': (1, 5, "uniform"),           # engine duration in [10, 1e5] s
    }

    jet_type = 2

    outfile = f"/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/blastwave_rs_gaussian_chunk_{chunk_id:03d}.h5"

    # Use different random seed per chunk to avoid duplicate samples
    np.random.seed(42 + chunk_id)

    creator = BlastwaveRSData(outfile=outfile,
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
                              n_nu=n_nu,
                              fixed_parameters={'sigma': 0.0})

    print(f"Chunk {chunk_id} complete: {n_samples} samples written to {outfile}")
