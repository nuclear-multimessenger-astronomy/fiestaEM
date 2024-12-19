"""Test run on GRB170817A data."""

import os
import jax 
print(f"GPU found? {jax.devices()}")
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import corner

from fiesta.inference.lightcurve_model import AfterglowpyLightcurvemodel
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.prior import Uniform, CompositePrior
from fiesta.inference.fiesta import Fiesta
from fiesta.utils import load_event_data

import time
start_time = time.time()

################
### Preamble ###
################

jax.config.update("jax_enable_x64", True)

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

#############
### SETUP ###
#############

trigger_time = 57982.5285236896

##############
### MODEL  ###
##############

name = "tophat"
model_dir = f"../../lightcurve_models/afterglowpy/{name}/"
FILTERS = ["radio-3GHz", "radio-6GHz", "X-ray-1keV", "bessellv"]

model = AfterglowpyLightcurvemodel(name,
                                   model_dir, 
                                   filters = FILTERS)


############
### DATA ###
############

data = load_event_data("../data/GRB170817A.dat") # only one filter of the GRB170817A data

#############################
### PRIORS AND LIKELIHOOD ###
#############################

inclination_EM = Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM'])
log10_E0 = Uniform(xmin=46.0, xmax=55.0, naming=['log10_E0'])
thetaCore = Uniform(xmin=0.01, xmax=np.pi/10, naming=['thetaCore'])
log10_n0 = Uniform(xmin=-7.0, xmax=1.0, naming=['log10_n0'])
p = Uniform(xmin=2.01, xmax=3.0, naming=['p'])
log10_epsilon_e = Uniform(xmin=-5.0, xmax=0.0, naming=['log10_epsilon_e'])
log10_epsilon_B = Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B'])

# luminosity_distance = Uniform(xmin=30.0, xmax=50.0, naming=['luminosity_distance'])

prior_list = [inclination_EM, 
              log10_E0, 
              thetaCore, 
              log10_n0, 
              p, 
              log10_epsilon_e, 
              log10_epsilon_B
            #   luminosity_distance
]

prior = CompositePrior(prior_list)

detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          FILTERS,
                          tmax = 500.0,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 40.0}
) 

##############
### FIESTA ###
##############

mass_matrix = jnp.eye(prior.n_dim)
eps = 5e-3
local_sampler_arg = {"step_size": mass_matrix * eps}

# Save for postprocessing
outdir = f"./outdir_GRB170817_tophat/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

fiesta = Fiesta(likelihood,
                prior,
                n_chains = 1_000,
                n_loop_training = 7,
                n_loop_production = 3,
                num_layers = 4,
                hidden_size = [64, 64],
                n_epochs = 20,
                n_local_steps = 50,
                n_global_steps = 200,
                local_sampler_arg=local_sampler_arg,
                outdir = outdir)

fiesta.sample(jax.random.PRNGKey(42))

fiesta.print_summary()

name = outdir + f'results_training.npz'
print(f"Saving samples to {name}")
state = fiesta.Sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state[
"log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, log_prob=log_prob, local_accs=local_accs,
        global_accs=global_accs, loss_vals=loss_vals)

#  - production phase
name = outdir + f'results_production.npz'
print(f"Saving samples to {name}")
state = fiesta.Sampler.get_sampler_state(training=False)
chains, log_prob, local_accs, global_accs = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, chains=chains, log_prob=log_prob,
            local_accs=local_accs, global_accs=global_accs)
    
################
### PLOTTING ###
################

# Fixed names: do not include them in the plotting, as will break corner
parameter_names = prior.naming

n_chains, n_steps, n_dim = np.shape(chains)
samples = np.reshape(chains, (n_chains * n_steps, n_dim))
samples = np.asarray(samples) # convert from jax.numpy array to numpy array for corner consumption

corner.corner(samples, labels = parameter_names, hist_kwargs={'density': True}, **default_corner_kwargs)
plt.savefig(os.path.join(outdir, "corner.png"), bbox_inches = 'tight')
plt.close()

end_time = time.time()
runtime_seconds = end_time - start_time
number_of_minutes = runtime_seconds // 60
number_of_seconds = np.round(runtime_seconds % 60, 2)
print(f"Total runtime: {number_of_minutes} m {number_of_seconds} s")

print("Plotting lightcurves")
fiesta.plot_lightcurves()
print("Plotting lightcurves . . . done")

print("DONE")