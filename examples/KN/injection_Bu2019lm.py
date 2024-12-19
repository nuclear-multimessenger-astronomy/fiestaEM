"""
Perform an injection-recovery on the Bu2019lm model.

NOTE: Ideally, we should fetch the true lightcurve from POSSIS (in the training dataset). Here, we inject with the surrogate model for simplicity and to test the performance of the surrogate and the likelihood functions. The injection-recovery setup is still in experimental stage and might be updated in the near future.
"""

import os
import json
import copy
import jax 
print(f"GPU found? {jax.devices()}")
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner

from fiesta.inference.lightcurve_model import BullaLightcurveModel
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.prior import Uniform, CompositePrior
from fiesta.inference.injection import InjectionRecovery
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
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3)

#############
### SETUP ###
#############

# Choose specific filters
filters = ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y", "2massj", "2massh", "2massks", "sdssu"] 
name = "Bu2019lm"
trigger_time = 0.0 

###########################
### MODEL AND INJECTION ###
###########################

model = BullaLightcurveModel(name,
                             f"../../lightcurve_models/KN/{name}/",
                             filters)

injection_dict = {"KNtheta": 30.0,
                  "KNphi": 30.0,
                  "log10_mej_dyn": -2.5,
                  "log10_mej_wind": -1.5,
                  "luminosity_distance": 40.0,}

injection = InjectionRecovery(model, 
                              injection_dict,
                              filters,
                              randomize_nondetections=True)

injection.create_injection()

#############################
### PRIORS AND LIKELIHOOD ###
#############################

KNtheta        = Uniform(xmin=0.0, xmax=90.0, naming=['KNtheta'])
KNphi          = Uniform(xmin=0.0, xmax=90.0, naming=['KNphi'])
log10_mej_dyn  = Uniform(xmin=-3.0, xmax=-1.7, naming=['log10_mej_dyn'])
log10_mej_wind = Uniform(xmin=-2.0, xmax=-0.89, naming=['log10_mej_wind'])
luminosity_distance = Uniform(xmin=30.0, xmax=50.0, naming=['luminosity_distance'])

prior_list = [KNtheta,
              KNphi,
              log10_mej_dyn, 
              log10_mej_wind,
              luminosity_distance
              ]

prior = CompositePrior(prior_list)

detection_limit = None
likelihood = EMLikelihood(model,
                          injection.data,
                          filters,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                        #   fixed_params={"luminosity_distance": 44.0},
                          )

##############
### FIESTA ###
##############

mass_matrix = jnp.eye(prior.n_dim)
eps = 1e-4
local_sampler_arg = {"step_size": mass_matrix * eps}

fiesta = Fiesta(likelihood,
                prior,
                n_chains = 1_000,
                n_loop_training = 3,
                n_loop_production = 3,
                num_layers = 4,
                hidden_size = [64, 64],
                local_sampler_arg=local_sampler_arg)

fiesta.sample(jax.random.PRNGKey(0))

# Save for postprocessing
outdir = "./outdir_injection_Bu2019lm/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
   
### New attempt

print("Saving injection dict")
save_name = os.path.join(outdir, "injection_dict.npz")
np.savez(save_name, **injection_dict)

save_name = outdir + f'results_training.npz'
print(f"Saving samples to {save_name}")
state = fiesta.Sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state[
"log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(save_name, log_prob=log_prob, local_accs=local_accs,
        global_accs=global_accs, loss_vals=loss_vals)

#  - production phase
save_name = outdir + f'results_production.npz'
print(f"Saving samples to {save_name}")
state = fiesta.Sampler.get_sampler_state(training=False)
chains, log_prob, local_accs, global_accs = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(save_name, chains=chains, log_prob=log_prob,
            local_accs=local_accs, global_accs=global_accs)
    
################
### PLOTTING ###
################

filters = likelihood.filters
fig, ax = plt.subplots(nrows = len(filters), ncols = 1, figsize = (10, 20))

for i, filter_name in enumerate(filters):
    ax = plt.subplot(len(filters), 1, i + 1)
    
    # Load the data
    t, mag, err = injection.data[filter_name].T
    idx_det = np.where(err != np.inf)[0]
    idx_nondet = np.where(err == np.inf)[0]
    
    # Detections
    ax.errorbar(t[idx_det] - trigger_time, mag[idx_det], yerr = err[idx_det], fmt = "o", color = "red", label = "Data (det.)")
    
    # Non-detections
    ax.scatter(t[idx_nondet] - trigger_time, mag[idx_nondet], marker = "v", color = "red", label = "Data (nondet.)")
    
    # Make pretty
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(filter_name)
    ax.invert_yaxis()
    
plt.savefig(os.path.join(outdir, f"test_injection_{name}_data.png"), bbox_inches = 'tight')
plt.close()

# Fixed names: do not include them in the plotting, as will break corner
parameter_names = prior.naming
truths = np.array([injection_dict[name] for name in parameter_names])

n_chains, n_steps, n_dim = np.shape(chains)
samples = np.reshape(chains, (n_chains * n_steps, n_dim))
samples = np.asarray(samples) # convert from jax.numpy array to numpy array for corner consumption

corner.corner(samples, labels = parameter_names, truths = truths, truth_color="red", hist_kwargs={'density': True}, **default_corner_kwargs)
plt.savefig(f"{outdir}/injection_{name}_corner.png", bbox_inches = 'tight')
plt.close()

print("DONE")

end_time = time.time()
runtime_seconds = end_time - start_time
number_of_minutes = runtime_seconds // 60
number_of_seconds = np.round(runtime_seconds % 60, 2)
print(f"Total runtime: {number_of_minutes} m {number_of_seconds} s")