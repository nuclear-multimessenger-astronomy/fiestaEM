import numpy as np
import jax

from fiesta.inference.prior import Uniform, Constraint, ConstrainedPrior, Sine
from fiesta.inference.fiesta import Fiesta
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.utils import load_event_data


########
# DATA #
########

data = load_event_data("./data/GRB170817A.dat")
trigger_time = 57982.52851852
FILTERS = data.keys()

#########
# MODEL #
#########

model = AfterglowFlux(name="afgpy_gaussian_CVAE",
                      filters = FILTERS)


#########
# PRIOR #
#########

def conversion_function(sample):
    converted_sample = sample
    converted_sample["thetaWing"] = converted_sample["thetaCore"] * converted_sample["alphaWing"]
    converted_sample["epsilon_tot"] = 10**(converted_sample["log10_epsilon_B"]) + 10**(converted_sample["log10_epsilon_e"]) 
    return converted_sample

GRB_prior = [Sine(xmin=0.0, xmax=np.pi/4, naming=['inclination_EM']), 
             Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0']),
             Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore']),
             Uniform(xmin = 0.2, xmax = 3.5, naming= ["alphaWing"]),
             Constraint(xmin = 0, xmax = np.pi/2, naming = ["thetaWing"]),
             Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0']),
             Uniform(xmin=2.01, xmax=3.0, naming=['p']),
             Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e']),
             Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B']),
             Constraint(xmin = 0., xmax = 1., naming=["epsilon_tot"])]

prior = ConstrainedPrior(GRB_prior, conversion_function)

################
# LIKELIHOOD & #
# SAMPLING     #
################
  
  
detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          data_tmin=1.,
                          data_tmax=2000.0,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 43.58, "redshift": 0.009727}
                          )




# Save for postprocessing
outdir = f"./outdir_GRB/"

fiesta = Fiesta(likelihood,
                prior,
                systematics_file="./systematics_file_GRB.yaml",
                n_chains=200,
                outdir = outdir)

if __name__ == "__main__":
    fiesta.sample(jax.random.PRNGKey(42))
    fiesta.print_summary()
    fiesta.save_results()
    fiesta.plot_lightcurves()
    fiesta.plot_corner()