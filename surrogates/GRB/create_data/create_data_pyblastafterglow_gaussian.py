import numpy as np 

from fiesta.train.AfterglowData import PyblastafterglowData
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#############
### SETUP ###
#############

tmin = 1e-4 # days
tmax = 2000
n_times = 250


numin = 1e9 # Hz 
numax = 2.5e19 # Hz (100 keV)
n_nu = 256

parameter_distributions = {
    'inclination_EM': (0, np.pi/2, "uniform"),
    'log10_E0': (47, 57, "uniform"), 
    'thetaCore': (0.01, np.pi/5, "loguniform"),
    'alphaWing': (0.2, 3.5, "uniform"),
    'log10_n0': (-6, 2, "uniform"),
    'p': (2.01, 3, "uniform"),
    'log10_epsilon_e': (-4, 0, "uniform"),
    'log10_epsilon_B': (-8,0, "uniform"),
    'Gamma0': (100, 1000, "uniform")
}
    
jet_type = 0 # -1 for tophat, 0 for gaussian jet

n_training = 20
n_val = 4
n_test = 4

outfile = f"../training_data/pyblastafterglow_gaussian_raw_data_{rank}.h5"

#######################
### CREATE RAW DATA ###
#######################

creator = PyblastafterglowData(outfile=outfile,
                               jet_type=jet_type,
                               n_training=n_training, 
                               n_val=n_val,
                               n_test=n_test,
                               tmin=tmin,
                               tmax=tmax,
                               n_times=n_times,
                               numin=numin,
                               numax=numax,
                               n_nu=n_nu,
                               parameter_distributions=parameter_distributions,
                               rank=rank,
                               path_to_exec="/hppfs/scratch/06/di35kuf/pyblastafterglow/PyBlastAfterglowMag/src/pba.out")
