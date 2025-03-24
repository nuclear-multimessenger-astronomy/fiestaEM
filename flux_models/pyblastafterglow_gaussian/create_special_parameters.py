from jax.random import PRNGKey
import numpy as np 
import h5py

from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.prior import Uniform, Constraint

from fiesta.train.AfterglowData import PyblastafterglowData

with h5py.File("./model/pyblastafterglow_raw_data.h5") as f:
    parameter_names = f["parameter_names"][:].astype(str).tolist()

size = 30_000

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

prior = ConstrainedPrior([
                    Uniform(xmin=0., xmax=np.pi/2, naming=["inclination_EM"]),
                    Uniform(xmin=52., xmax=57., naming=["log10_E0"]),
                    Uniform(xmin=0.01, xmax=np.pi/5, naming=["thetaCore"]),
                    Uniform(0.2, 3.5, naming=["alphaWing"]),
                    Uniform(xmin=-6.,xmax=2.,naming=["log10_n0"]),
                    Uniform(xmin=2., xmax=3., naming=["p"]),
                    Uniform(xmin=-4., xmax=0., naming=["log10_epsilon_e"]),
                    Uniform(xmin=-8.,xmax=0., naming=["log10_epsilon_B"]),
                    Uniform(xmin=100., xmax=1000., naming=["Gamma0"]),
                    Constraint(xmin=0., xmax=1., naming=["epsilon_tot"]),
                    Constraint(xmin=0., xmax=np.pi/2, naming=["thetaWing"])
                    ],
                    conversion_function)


X = prior.sample(PRNGKey(272814), n_samples=size)
X = [X[p] for p in parameter_names]
X = np.transpose(X)
Xwing = np.random.uniform(0.5, 1.3, size=size) * X[:,0]
Xalpha = Xwing / X[:,2]

mask = Xalpha > 3.5
Xalpha[mask] = np.random.uniform(0.2, 3.5, size=np.sum(mask))

X[:,3] = Xalpha
X[mask,0] = np.minimum(Xalpha[mask] * X[mask,2] * np.random.uniform(1/1.3, 2, size=np.sum(mask)), np.pi/2)

np.savetxt("special_parameters_01.txt", X)