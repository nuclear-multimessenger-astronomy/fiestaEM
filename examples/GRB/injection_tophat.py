"""Injection runs with afterglowpy"""

import os
import jax

print(f"GPU found? {jax.devices()}")
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import corner

from fiesta.inference.lightcurve_model import AfterglowpyPCA
from fiesta.inference.injection import InjectionRecoveryAfterglowpy
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.prior import Uniform, Composite
from fiesta.inference.fiesta import Fiesta
from fiesta.utils import load_event_data, write_event_data

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

##############
### MODEL  ###
##############

name = "tophat"
model_dir = f"../../trained_fluxes/afterglowpy/{name}/"
FILTERS = ["radio-3GHz", "radio-6GHz", "X-ray-1keV", "bessellv"]

model = AfterglowpyPCA(name,
                        model_dir, 
                        filters = FILTERS)


###################
###    INJECT   ###
### AFTERGLOWPY ###
###################

trigger_time = 58849 # 01-01-2020 in mjd
remake_injection = False
injection_dict = {"inclination_EM": 0.32, "log10_E0": 53.79, "thetaCore": 0.1, "p": 2.47, "log10_n0": -2.1, "log10_epsilon_e": -1.326, "log10_epsilon_B": -3.89, "luminosity_distance": 40.0}

if remake_injection:
    injection = InjectionRecoveryAfterglowpy(injection_dict, jet_type = -1, filters = FILTERS, N_datapoints = 70, error_budget = 0.5, tmin = 1, tmax = 2000, trigger_time = trigger_time)
    injection.create_injection()
    data = injection.data
    write_event_data("./injection_tophat/injection_tophat.dat", data)

data = load_event_data("./injection_tophat/injection_tophat.dat")


#############################
### PRIORS AND LIKELIHOOD ###
#############################

inclination_EM = Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM'])
log10_E0 = Uniform(xmin=47.0, xmax=57.0, naming=['log10_E0'])
thetaCore = Uniform(xmin=0.01, xmax=np.pi/5, naming=['thetaCore'])
log10_n0 = Uniform(xmin=-6.0, xmax=2.0, naming=['log10_n0'])
p = Uniform(xmin=2.01, xmax=3.0, naming=['p'])
log10_epsilon_e = Uniform(xmin=-4.0, xmax=0.0, naming=['log10_epsilon_e'])
log10_epsilon_B = Uniform(xmin=-8.0, xmax=0.0, naming=['log10_epsilon_B'])

# luminosity_distance = Uniform(xmin=30.0, xmax=50.0, naming=['luminosity_distance'])

prior_list = [inclination_EM, 
              log10_E0, 
              thetaCore, 
              log10_n0, 
              p, 
              log10_epsilon_e, 
              log10_epsilon_B
            # luminosity_distance
]

prior = Composite(prior_list)

detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          FILTERS,
                          tmax = 2000.0,
                          trigger_time=trigger_time,
                          detection_limit = detection_limit,
                          fixed_params={"luminosity_distance": 40.0},
                          error_budget = 1e-5
) 

##############
### FIESTA ###
##############

mass_matrix = jnp.eye(prior.n_dim)
eps = 5e-3
local_sampler_arg = {"step_size": mass_matrix * eps}

# Save for postprocessing
outdir = f"./injection_{name}/"
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
truths = [injection_dict[key] for key in parameter_names]

n_chains, n_steps, n_dim = np.shape(chains)
samples = np.reshape(chains, (n_chains * n_steps, n_dim))
samples = np.asarray(samples) # convert from jax.numpy array to numpy array for corner consumption

corner.corner(samples, labels = parameter_names, hist_kwargs={'density': True}, truths = truths, **default_corner_kwargs)
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