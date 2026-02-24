"""
Post-processing script for inference_KN_SMC.py results.

Run from anywhere – it locates itself via __file__ and resolves paths relative
to the parent examples/inference/ directory so data and model are found correctly.

Usage:
    python outdir_KN_SMC/plot_results.py
    # or from within outdir_KN_SMC/:
    python plot_results.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── make sure we can import fiesta even if the venv is not activated ──────────
# (the script resolves the inference dir as its working root)
HERE    = os.path.dirname(os.path.abspath(__file__))          # .../outdir_KN_SMC
INFDIR  = os.path.dirname(HERE)                               # .../inference
os.chdir(INFDIR)                                              # relative paths now work

# ── fiesta imports ────────────────────────────────────────────────────────────
from fiesta.inference.prior import Uniform
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import BullaFlux
from fiesta.utils import load_event_data
from fiesta.plot import corner_plot, LightcurvePlotter

########################################################################
# RELOAD DATA / MODEL / LIKELIHOOD  (same as inference script)         #
########################################################################

data = load_event_data("./data/AT2017gfo.dat")
trigger_time = 57982.52851852
FILTERS = list(data.keys())

model = BullaFlux(name="Bu2025_MLP", filters=FILTERS)

fiesta_likelihood = EMLikelihood(
    model,
    data,
    tmin=0.3,
    tmax=28.0,
    trigger_time=trigger_time,
    detection_limit=None,
    fixed_params={"luminosity_distance": 43.583656, "redshift": 0.009727},
)

KN_prior = [
    Uniform(xmin=0.0,   xmax=np.pi / 2, naming=["inclination_EM"]),
    Uniform(xmin=-3.0,  xmax=-1.3,      naming=["log10_mej_dyn"]),
    Uniform(xmin=0.12,  xmax=0.28,      naming=["v_ej_dyn"]),
    Uniform(xmin=0.15,  xmax=0.35,      naming=["Ye_dyn"]),
    Uniform(xmin=-2.0,  xmax=-0.886,    naming=["log10_mej_wind"]),
    Uniform(xmin=0.05,  xmax=0.15,      naming=["v_ej_wind"]),
    Uniform(xmin=0.2,   xmax=0.4,       naming=["Ye_wind"]),
]
fiesta_prior = ConstrainedPrior(KN_prior)
parameter_names = list(fiesta_prior.naming)

########################################################################
# LOAD SAMPLES                                                          #
########################################################################

npz_path = os.path.join(HERE, "samples.npz")
print(f"Loading samples from {npz_path}")
data_npz = np.load(npz_path)

samples  = {name: data_npz[name] for name in parameter_names}
log_prob = data_npz["log_prob"]
weights  = data_npz["weights"]

n_particles = len(log_prob)
print(f"  {n_particles} particles loaded")

########################################################################
# WEIGHTED RESAMPLING → equal-weight posterior for plotting            #
########################################################################

rng = np.random.default_rng(0)
w   = weights / weights.sum()
n_resample = min(2000, n_particles)
idx = rng.choice(n_particles, size=n_resample, p=w, replace=True)

posterior = {name: samples[name][idx] for name in parameter_names}
posterior["log_prob"] = log_prob[idx]

########################################################################
# CORNER PLOT                                                           #
########################################################################

print("Generating corner plot …")
fig, ax = corner_plot(posterior, parameter_names)
out_corner = os.path.join(HERE, "corner.pdf")
fig.savefig(out_corner, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_corner}")

########################################################################
# LIGHTCURVE PLOT                                                       #
########################################################################

print("Generating lightcurve plot …")
filters   = fiesta_likelihood.filters
lc_plotter = LightcurvePlotter(posterior, fiesta_likelihood)

height = len(filters) * 2.5
fig, axes = plt.subplots(nrows=len(filters), ncols=1, figsize=(8, height))
if len(filters) == 1:
    axes = [axes]

for cax, filt in zip(axes, filters):
    lc_plotter.plot_data(cax, filt, color="red")
    lc_plotter.plot_best_fit_lc(cax, filt, color="blue")
    lc_plotter.plot_sample_lc(cax, filt)
    cax.set_ylabel(filt)
    cax.set_xlim(left=max(fiesta_likelihood.tmin, 1e-4), right=fiesta_likelihood.tmax)
    cax.set_xscale("log")
    ymin = np.min(np.concatenate([lc_plotter.mag_det[filt], lc_plotter.mag_nondet[filt]])) - 2
    ymax = np.max(np.concatenate([lc_plotter.mag_det[filt], lc_plotter.mag_nondet[filt]])) + 2
    cax.set_ylim(ymax, ymin)

axes[-1].set_xlabel("$t$ in days")
fig.tight_layout()
out_lc = os.path.join(HERE, "lightcurves.pdf")
fig.savefig(out_lc, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved → {out_lc}")

print("\nDone.")
