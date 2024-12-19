import numpy as np 
import matplotlib.pyplot as plt

from fiesta.train.AfterglowData import AfterglowpyData

#############
### SETUP ###
#############

tmin = 0.1 # days
tmax = 2000 # days
n_times = 200


numin = 1e9 # Hz 
numax = 2.5e18 # Hz (10keV)
n_nu = 256


parameter_distributions = {
    'inclination_EM': (0, np.pi/2, "uniform"),
    'log10_E0': (47, 57, "uniform"),
    'thetaCore': (0.01, np.pi/5, "loguniform"),
    'log10_n0': (-6, 2, "uniform"),
    'p': (2.01, 3, "uniform"),
    'log10_epsilon_e': (-4, 0, "uniform"),
    'log10_epsilon_B': (-8, 0, "uniform")
}

    

jet_name = "tophat"
jet_conversion = {"tophat": -1,
                  "gaussian": 0}

n_training = 0
n_val = 0
n_test = 0

n_pool = 1



#######################
### CREATE RAW DATA ###
#######################
name = jet_name
outdir = f"./model/"

jet_type = jet_conversion[jet_name]

creator = AfterglowpyData(outdir = outdir,
                          jet_type = jet_type,
                          n_training = n_training, 
                          n_val = n_val,
                          n_test = n_test,
                          parameter_distributions = parameter_distributions,
                          n_pool = n_pool,
                          tmin = tmin,
                          tmax = tmax,
                          n_times = n_times,
                          numin = numin,
                          numax = numax,
                          n_nu = n_nu)
