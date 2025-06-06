from jax.random import PRNGKey
import numpy as np 

from fiesta.train.AfterglowData import AfterglowpyData
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.prior import Uniform, Constraint

#############
### SETUP ###
#############


name = "gaussian"
outfile = "../training_data/afterglowpy_gaussian_raw_data.h5"

n_training = 0
n_val = 0
n_test = 0

n_pool = 24

size = 20_000


#######################
### CREATE RAW DATA ###
#######################
creator = AfterglowpyData(outfile=outfile,
                          n_training=0, 
                          n_val=0,
                          n_test=0,
                          n_pool=n_pool)

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

prior = ConstrainedPrior([
                    Uniform(xmin=0., xmax=np.pi/2, naming=["inclination_EM"]),
                    Uniform(xmin=54., xmax=57., naming=["log10_E0"]),
                    Uniform(xmin=0.35, xmax=np.pi/5, naming=["thetaCore"]),
                    Uniform(0.2, 3.5, naming=["alphaWing"]),
                    Uniform(xmin=-6.,xmax=-4.,naming=["log10_n0"]),
                    Uniform(xmin=2., xmax=3., naming=["p"]),
                    Uniform(xmin=-4., xmax=0., naming=["log10_epsilon_e"]),
                    Uniform(xmin=-8.,xmax=0., naming=["log10_epsilon_B"]),
                    Constraint(xmin=0., xmax=1., naming=["epsilon_tot"]),
                    Constraint(xmin=0., xmax=np.pi/2, naming=["thetaWing"])
                    ],
                    conversion_function)

X = prior.sample(PRNGKey(2728), n_samples=size)
X = [X[p] for p in creator.parameter_names]
X = np.transpose(X)

creator.create_special_data(X, label = "01", comment = "log10_E0 (54, 57)  log10_n0 (-6, -4) thetaCore (0.35, np.pi/5)")