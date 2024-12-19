import numpy as np 
from fiesta.train.AfterglowData import AfterglowpyData

#############
### SETUP ###
#############


parameter_distributions = {
    'inclination_EM': (0, np.pi/2, "uniform"),
    'log10_E0': (47, 57, "uniform"),
    'thetaCore': (0.01, np.pi/5, "loguniform"),
    'alphaWing': (0.2, 3.5, "uniform"),
    'log10_n0': (-6, 2, "uniform"),
    'p': (2.01, 3, "uniform"),
    'log10_epsilon_e': (-4, 0, "uniform"),
    'log10_epsilon_B': (-8, 0, "uniform")
}

    

name = "tophat"
outdir = f"./model/"

n_training = 0
n_val = 0
n_test = 0

n_pool = 24
size = 20_000


#######################
### CREATE RAW DATA ###
#######################
creator = AfterglowpyData(outdir = outdir,
                          n_training = 0, 
                          n_val = 0,
                          n_test = 0,
                          n_pool = n_pool)

#import h5py
#with h5py.File(creator.outfile, "r+") as f:
#    unproblematic = np.unique(np.where(~np.isinf(f["special_train"]["01"]["y"]))[0])
#
#    X = f["special_train"]["01"]["X"][unproblematic]
#    y = f["special_train"]["01"]["y"][unproblematic]
#    breakpoint()
#    creator._save_to_file(X, y, group = "special_train", label = "02", comment = "log10_E0 (54, 57)  log10_n0 (-6, -4) thetaCore (0.4, np.pi/5)")
    
    

inclination = np.random.uniform(0, np.pi/2, size = size)
log10_E0 = np.random.uniform(54, 57, size = size)
thetaCore = np.random.uniform(0.4, np.pi/5, size= size)
alphaWing = np.random.uniform(0.2, 3.5, size = size)
log10_n0 = np.random.uniform(-6, -4, size = size)
p = np.random.uniform(2, 3, size = size)
log10_epsilon_e = np.random.uniform(-4, 0, size = size)
log10_epsilon_B = np.random.uniform(-8, 0, size = size)

X = np.array([inclination, log10_E0, thetaCore, alphaWing, log10_n0, p, log10_epsilon_e, log10_epsilon_B]).T

creator.create_special_data(X, label = "01", comment = "log10_E0 (54, 57)  log10_n0 (-6, -4) thetaCore (0.4, np.pi/5)")