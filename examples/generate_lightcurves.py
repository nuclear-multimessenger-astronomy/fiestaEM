"""
This is an example showing how to generate lightcurves from a set of parameters.
Also showing how to sample from a prior distribution on those parameters and pass them to the model
This example script is based on examples/inference/inference_KN.py, but the same principle applies
for any fiesta model.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fiesta.inference.prior import Uniform, ConstrainedPrior
from fiesta.inference.lightcurve_model import BullaFlux

# Define filters to generate lightcurves for
# NOTE: the training script for this model is found at fiestaEM/surrogates/KN/Bu2025_MLP/train_Bu2025.py
FILTERS = ["ps1::y", "besselli", "bessellv", "bessellux"]
print(f"Using filters: {FILTERS}")

# Define the model we want to use
model = BullaFlux(name="Bu2025_MLP",
                  filters=FILTERS)

print(f"Using model: {model.name}")

# Define the prior distribution on the model parameters
KN_prior = [
    Uniform(xmin=0., xmax=np.pi/2, naming=["inclination_EM"]),
    Uniform(xmin=-3.0, xmax=-1.3, naming=["log10_mej_dyn"]),
    Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"]),
    Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"]),
    Uniform(xmin=-2., xmax=-0.886, naming=["log10_mej_wind"]),
    Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"]),
    Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
]
prior = ConstrainedPrior(KN_prior)

print(f"Sampling from prior with parameter names: {prior.naming}")

# Number of lightcurves to generate
n_samples = 10

# Fixed parameters (distance and redshift) -- these can optionally also be added into the prior above and varied as well
fixed_params = {
    "luminosity_distance": 43.583656,  # Mpc
    "redshift": 0.009727
}
print(f"Using fixed parameters: {fixed_params}")

# Time array for evaluation (in days since trigger)
time_array = jnp.logspace(np.log10(0.3), np.log10(28.), 100)


# Plot the lightcurves
fig, axes = plt.subplots(len(FILTERS), 1, figsize=(10, 3*len(FILTERS)), sharex=True)
if len(FILTERS) == 1:
    axes = [axes]

colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

print(f"Generating {n_samples} lightcurves from prior samples...")
key = jax.random.PRNGKey(42)
for i in range(n_samples):
    # Sample one parameter set from the prior
    key, subkey = jax.random.split(key)
    sample = prior.sample(subkey, 1)

    # Get parameters for this sample (sample is a dict with arrays)
    params = {name: float(sample[name][0]) for name in prior.naming}
    params.update(fixed_params)

    # Generate lightcurves for all filters at once
    # model.predict() returns times array and mag dict {filter_name: magnitudes_array}
    times, mag = model.predict(params)

    # Plot each filter
    for filter_idx, filter_name in enumerate(FILTERS):
        # Extract magnitudes for this filter from the dict
        magnitudes = mag[filter_name]

        # Plot
        axes[filter_idx].plot(times, magnitudes,
                            color=colors[i], alpha=0.7, linewidth=2.5)
        axes[filter_idx].set_ylabel(f'{filter_name} (AB mag)', fontsize=12)
        axes[filter_idx].set_xscale('log')

for ax in axes:
    ax.invert_yaxis()  # Invert y-axis for magnitudes

axes[-1].set_xlabel('Time since trigger (days)', fontsize=12)
axes[0].set_title(f'Kilonova Lightcurves from Prior Samples (n={n_samples})',
                  fontsize=14, fontweight='bold')

plt.tight_layout()

# Save the plot
plt.savefig("lightcurves.png", bbox_inches='tight')
print(f"\nPlot saved to: lightcurves.png")
plt.close()