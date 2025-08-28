import numpy as np 
from fiesta.train.AfterglowData import AfterglowpyData

#############
### SETUP ###
#############


outfile = "../training_data/afterglowpy_tophat_raw_data.h5"

n_training = 0
n_val = 0
n_test = 0

n_pool = 24
size = 5000


#######################
### CREATE RAW DATA ###
#######################
creator = AfterglowpyData(outfile=outfile,
                          n_training=0, 
                          n_val=0,
                          n_test=0,
                          n_pool=n_pool)


inclination = np.random.uniform(0, np.pi/2, size = size)
log10_E0 = np.random.uniform(54, 57, size = size)
thetaCore = np.random.uniform(0.01, np.pi/5, size= size)
log10_n0 = np.random.uniform(-6, 2, size = size)
p = np.random.uniform(2,3,size = size)
log10_epsilon_e = np.random.uniform(-3, 0, size = size)
log10_epsilon_B = np.random.uniform(-3, 0, size = size)

X = np.array([inclination, log10_E0, thetaCore, log10_n0, p, log10_epsilon_e, log10_epsilon_B]).T

creator.create_special_data(X, label="01", comment="higher E0, epsilon_e, epsilon_B")