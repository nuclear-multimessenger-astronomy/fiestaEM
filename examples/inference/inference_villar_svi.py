"""Villar model SVI fitting with superphot+ priors (de Soto et al. 2024).

Demonstrates how to use fiesta's numpyro-svi sampler with informed
TruncatedNormal priors from the superphot+ population study for fast
Villar/Bazin light curve fitting of ZTF photometry.

Usage
-----
    python examples/inference/inference_villar_svi.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from fiesta.inference.analytical_models.phenomenological_models import VillarModel
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.prior import TruncatedNormal, Uniform, CompositePrior
from fiesta.inference.fiesta import Fiesta

# ── Superphot+ priors (de Soto et al. 2024, Table 2) ────────────────────────
# These are the population-level posteriors from fitting ~6000 ZTF SNe.

VILLAR_PRIORS = {
    # shape params: (mu, sigma, xmin, xmax)
    "t0":              (-17.878, 30.0,   -100.0, 30.0),
    "log10_tau_rise":  (  0.666,  1.2,    -2.0,   4.0),
    "log10_tau_fall":  (  1.526,  0.9,     0.0,   4.0),
    "beta_slope":      (  0.008,  0.012,  -0.01,  0.03),
    "log10_gamma":     (  1.426,  0.9,     0.0,   3.5),
}


def build_villar_prior(filters, amp_range=(-5.0, 5.0)):
    """Build a CompositePrior for the Villar model with superphot+ priors.

    Shape parameters get TruncatedNormal priors from the population study.
    Per-filter amplitudes get Uniform priors.
    """
    priors = []

    # Shape parameters with informed priors
    for name, (mu, sigma, xmin, xmax) in VILLAR_PRIORS.items():
        priors.append(TruncatedNormal(
            mu=mu, sigma=sigma, xmin=xmin, xmax=xmax, naming=[name],
        ))

    # Per-filter amplitude (magnitude offset, weakly constrained)
    for filt in filters:
        priors.append(Uniform(
            xmin=amp_range[0], xmax=amp_range[1],
            naming=[f"amp_mag_{filt}"],
        ))

    return CompositePrior(priors)


# ── Example: Fit synthetic ZTF data ──────────────────────────────────────────

if __name__ == "__main__":

    # Generate synthetic 2-band lightcurve
    np.random.seed(42)
    t = np.linspace(1, 80, 30)
    phase = (t - 25) / 12
    flux_shape = np.exp(-0.5 * phase ** 2)

    data = {}
    for filt, offset in [("ztfr", 0), ("ztfg", 0.3)]:
        mag = 19.0 + offset - 2.5 * np.log10(np.maximum(flux_shape, 1e-10))
        mag += np.random.normal(0, 0.1, len(t))
        mag_err = np.full(len(t), 0.1)
        data[filt] = np.column_stack([t + 59000, mag, mag_err])

    filters = ["ztfr", "ztfg"]

    # Model
    t_grid = jnp.linspace(0.5, 90.0, 200)
    model = VillarModel(filters=filters, times=t_grid)

    # Likelihood
    likelihood = EMLikelihood(
        model, data, trigger_time=59000.0,
        data_tmin=0.5, data_tmax=90.0, error_budget=0.3,
    )

    # Prior with superphot+ informed shape parameters
    prior = build_villar_prior(filters)

    # Fit with numpyro SVI
    fiesta = Fiesta(
        likelihood, prior,
        outdir="./villar_svi_results/",
        sampler="numpyro-svi",
        num_iter=5_000,
        step_size=0.001,
        num_samples=1000,
    )

    key = jax.random.PRNGKey(42)
    fiesta.sample(key)
    fiesta.print_summary()
    fiesta.save_results()
    fiesta.plot_lightcurves()
    fiesta.plot_corner()

    print("\nDone! Results saved to ./villar_svi_results/")
