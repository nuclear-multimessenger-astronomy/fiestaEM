import numpy as np 

from fiesta.train.AfterglowData import PyblastafterglowData
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


pbag_kwargs = dict(tb0=1e-3,
                   tb1=1e11,
                   loglevel='info',
                   ntb=1200)


outfile = f"../training_data/pyblastafterglow_gaussian_raw_data_{rank}.h5"

creator = PyblastafterglowData(outfile=outfile,
                               n_training=0, 
                               n_val=0,
                               n_test=0,
                               rank=rank,
                               path_to_exec="/home/aya/work/hkoehn/fiesta/PyBlastAfterglowMag/src/pba.out", 
                               pbag_kwargs=pbag_kwargs)

X = np.loadtxt("special_parameters_01.txt")
X = np.array_split(X, size)
X = X[rank]

creator.create_special_data(X_raw=X, label="01", comment="thetaWing in (0.5, 1.3) * inclination_EM, log10_E0 (52, 57)")