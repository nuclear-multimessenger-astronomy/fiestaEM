"""
Kilonova inference using the jesterTOV SMC Random Walk sampler.

This script mirrors inference_KN.py but uses the BlackJAX SMC (Sequential Monte
Carlo) sampler from jesterTOV instead of flowMC.  Thin wrapper classes bridge
the two APIs:

  FiestaJesterPrior     – exposes `parameter_names` (jester) as an alias for
                           fiesta's `naming` attribute.
  FiestaJesterLikelihood – exposes `evaluate(params_dict)` (jester) and
                           internally applies `prior.transform()` before
                           calling the fiesta EMLikelihood.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

# ── fiesta imports ────────────────────────────────────────────────────────────
from fiesta.inference.prior import Uniform
from fiesta.inference.prior_dict import ConstrainedPrior
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.lightcurve_model import BullaFlux
from fiesta.utils import load_event_data

# ── jesterTOV imports ─────────────────────────────────────────────────────────
from jesterTOV.inference.samplers.blackjax.smc.random_walk import (
    BlackJAXSMCRandomWalkSampler,
)
from jesterTOV.inference.config.schemas.samplers import SMCRandomWalkSamplerConfig

########################################################################
# ADAPTER CLASSES                                                       #
########################################################################

class FiestaJesterPrior:
    """Wrap a fiesta ConstrainedPrior to satisfy the jesterTOV Prior interface.

    jesterTOV samplers look for ``parameter_names`` (not ``naming``).
    We expose it as a property and forward ``sample`` / ``log_prob``.
    """

    def __init__(self, fiesta_prior: ConstrainedPrior) -> None:
        self._prior = fiesta_prior
        # jesterTOV uses .parameter_names; fiesta uses .naming
        self.parameter_names = list(fiesta_prior.naming)

    @property
    def n_dim(self) -> int:
        return len(self.parameter_names)

    def sample(self, rng_key, n_samples: int) -> dict:
        return self._prior.sample(rng_key, n_samples)

    def log_prob(self, x: dict) -> float:
        return self._prior.log_prob(x)


class FiestaJesterLikelihood:
    """Wrap a fiesta EMLikelihood to satisfy the jesterTOV LikelihoodBase interface.

    jesterTOV calls ``likelihood.evaluate(params_dict)``.
    Fiesta's Fiesta class calls ``prior.transform(params)`` before the
    likelihood; we replicate that here so the adapter is fully general.
    """

    def __init__(
        self,
        fiesta_likelihood: EMLikelihood,
        fiesta_prior: ConstrainedPrior,
    ) -> None:
        self._likelihood = fiesta_likelihood
        self._prior = fiesta_prior

    def evaluate(self, params: dict) -> float:
        # apply the prior's transform (identity for KN, but kept for generality)
        transformed = self._prior.transform(params)
        return self._likelihood.evaluate(transformed)


########################################################################
# DATA                                                                  #
########################################################################

data = load_event_data("./data/AT2017gfo.dat")
trigger_time = 57982.52851852
FILTERS = list(data.keys())

########################################################################
# MODEL                                                                 #
########################################################################

model = BullaFlux(name="Bu2025_MLP", filters=FILTERS)

########################################################################
# PRIOR                                                                 #
########################################################################

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

########################################################################
# LIKELIHOOD                                                            #
########################################################################

fiesta_likelihood = EMLikelihood(
    model,
    data,
    tmin=0.3,
    tmax=28.0,
    trigger_time=trigger_time,
    detection_limit=None,
    fixed_params={"luminosity_distance": 43.583656, "redshift": 0.009727},
)

########################################################################
# ADAPT TO JESTER INTERFACES                                            #
########################################################################

prior      = FiestaJesterPrior(fiesta_prior)
likelihood = FiestaJesterLikelihood(fiesta_likelihood, fiesta_prior)

########################################################################
# SMC SAMPLER CONFIG  (moderate settings for a first run)              #
########################################################################

outdir = "./outdir_KN_SMC/"
os.makedirs(outdir, exist_ok=True)

sampler_config = SMCRandomWalkSamplerConfig(
    n_particles=500,
    n_mcmc_steps=5,
    target_ess=0.9,
    random_walk_sigma=0.3,
    output_dir=outdir,
    log_prob_batch_size=100,
)

########################################################################
# BUILD & RUN SAMPLER                                                   #
########################################################################

sampler = BlackJAXSMCRandomWalkSampler(
    likelihood=likelihood,
    prior=prior,
    sample_transforms=[],
    likelihood_transforms=[],
    config=sampler_config,
    seed=42,
)

if __name__ == "__main__":
    print("=" * 60)
    print("KN inference with jesterTOV SMC Random Walk sampler")
    print("=" * 60)
    print(f"Parameters : {prior.parameter_names}")
    print(f"n_particles: {sampler_config.n_particles}")
    print(f"n_mcmc_steps: {sampler_config.n_mcmc_steps}")
    print()

    sampler.sample(jax.random.PRNGKey(42))

    # ── diagnostics ──────────────────────────────────────────────────
    sampler.plot_diagnostics(outdir=outdir, filename="smc_diagnostics.png")

    # ── retrieve posterior samples ───────────────────────────────────
    output = sampler.get_sampler_output()
    samples  = output.samples          # dict[str, Array]  (n_particles,)
    log_prob = output.log_prob         # Array  (n_particles,)
    weights  = output.metadata["weights"]
    ess      = output.metadata["ess"]

    print("\n── Sampling complete ──────────────────────────────────────")
    print(f"  Final ESS : {ess:.1f} / {sampler_config.n_particles}")
    print(f"  log Z     : {sampler.metadata['logZ']:.2f}")
    print(f"  Annealing steps: {sampler.metadata['annealing_steps']}")

    # ── save results ─────────────────────────────────────────────────
    save_dict = {k: np.array(v) for k, v in samples.items()}
    save_dict["log_prob"] = np.array(log_prob)
    save_dict["weights"]  = np.array(weights)
    np.savez(os.path.join(outdir, "samples.npz"), **save_dict)
    print(f"\nResults saved to {outdir}samples.npz")

    # ── quick posterior summary ───────────────────────────────────────
    print("\n── Posterior summary (weighted mean ± std) ────────────────")
    w = np.array(weights)
    w = w / w.sum()
    for name in prior.parameter_names:
        vals = np.array(samples[name])
        mean = np.sum(w * vals)
        std  = np.sqrt(np.sum(w * (vals - mean) ** 2))
        print(f"  {name:25s}: {mean:.4f} ± {std:.4f}")

    # ── plots ─────────────────────────────────────────────────────────
    # SMC produces weighted samples; resample to equal-weight before plotting.
    rng = np.random.default_rng(0)
    n_resample = min(2000, sampler_config.n_particles)
    idx = rng.choice(sampler_config.n_particles, size=n_resample, p=w, replace=True)
    posterior_plot = {name: np.array(samples[name])[idx] for name in prior.parameter_names}
    posterior_plot["log_prob"] = np.array(log_prob)[idx]

    try:
        import matplotlib.pyplot as plt
        from fiesta.plot import corner_plot, LightcurvePlotter

        # corner plot
        fig, ax = corner_plot(posterior_plot, prior.parameter_names)
        fig.savefig(os.path.join(outdir, "corner.pdf"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Corner plot  → {outdir}corner.pdf")
    except Exception as e:
        print(f"[WARNING] Corner plot failed: {e}")

    try:
        import matplotlib.pyplot as plt
        from fiesta.plot import LightcurvePlotter

        filters = fiesta_likelihood.filters
        lc_plotter = LightcurvePlotter(posterior_plot, fiesta_likelihood)

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
        fig.savefig(os.path.join(outdir, "lightcurves.pdf"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Lightcurves  → {outdir}lightcurves.pdf")
    except Exception as e:
        print(f"[WARNING] Lightcurve plot failed: {e}")
