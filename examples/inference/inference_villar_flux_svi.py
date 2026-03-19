"""Villar model flux-space SVI fitting with superphot+ priors.

Uses the FluxLikelihood class which operates in normalized flux space
(matching the superphot+ pipeline) instead of magnitude space.
This produces more stable SVI gradients and better convergence.

Usage
-----
    python examples/inference/inference_villar_flux_svi.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from fiesta.inference.analytical_models.phenomenological_models import VillarModel
from fiesta.inference.likelihood import FluxLikelihood
from fiesta.inference.prior import TruncatedNormal, Uniform, CompositePrior
from fiesta.inference.fiesta import Fiesta

# ── Superphot+ priors (de Soto et al. 2024, Table 2) ────────────────────────

VILLAR_SHAPE_PRIORS = {
    "t0":              (-17.878, 30.0,   -100.0, 30.0),
    "log10_tau_rise":  (  0.666,  1.2,    -2.0,   4.0),
    "log10_tau_fall":  (  1.526,  0.9,     0.0,   4.0),
    "beta_slope":      (  0.008,  0.012,  -0.01,  0.03),
    "log10_gamma":     (  1.426,  0.9,     0.0,   3.5),
}


def build_villar_flux_prior(filters):
    """Build a CompositePrior for flux-space Villar fitting.

    Shape parameters use TruncatedNormal priors from superphot+.
    Per-filter parameters include linear amplitude and extra_sigma.
    """
    priors = []

    for name, (mu, sigma, xmin, xmax) in VILLAR_SHAPE_PRIORS.items():
        priors.append(TruncatedNormal(
            mu=mu, sigma=sigma, xmin=xmin, xmax=xmax, naming=[name],
        ))

    for filt in filters:
        # Amplitude in normalized flux (peak ~ 1.0)
        priors.append(Uniform(xmin=0.01, xmax=5.0, naming=[f"amp_{filt}"]))
        # Extra sigma (intrinsic scatter in normalized flux)
        priors.append(Uniform(xmin=0.001, xmax=0.5, naming=[f"extra_sigma_{filt}"]))

    return CompositePrior(priors)


if __name__ == "__main__":

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

    # Flux-space likelihood
    likelihood = FluxLikelihood(
        model, data, trigger_time=59000.0,
        tmin=0.5, tmax=90.0,
    )

    # Prior with superphot+ shape priors + per-filter flux params
    prior = build_villar_flux_prior(filters)

    # Fit with numpyro SVI
    fiesta = Fiesta(
        likelihood, prior,
        outdir="./villar_flux_svi_results/",
        sampler="numpyro-svi",
        num_iter=5_000,
        step_size=0.001,
        num_samples=1000,
    )

    key = jax.random.PRNGKey(42)
    fiesta.sample(key)
    fiesta.print_summary()
    fiesta.save_results(bestfit_params=False)

    print("\nDone! Results saved to ./villar_flux_svi_results/")
