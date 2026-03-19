import copy
import json
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray

from fiesta.conversions import mag_app_from_mag_abs
from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.inference.prior import Prior
from fiesta.inference.likelihood import EMLikelihood
from fiesta.logging import logger
from fiesta.plot import corner_plot, LightcurvePlotter
from fiesta.inference.systematic import setup_systematics_basic, setup_systematic_from_file


class Fiesta(object):
    """
    Master inference class for interfacing with the sampler.

    Args:
        "likelihood": "(EMLikelihood) likelihood object used for the inference",
        "prior": "(Prior) prior object used for the inference. It has to contain the parameters needed to evaluate likelihood.evaluate().",
        "outdir": (str) directory to which the output should be saved.
        "sampler": "(str) The sampler to use. Can be 'flowmc', 'blackjax-smc', or 'numpyro-svi'. Defaults to 'flowmc'.
        "error_budget": "(float) fixed systematic error to use in the inference in mag. Defaults to 0.3 but is ignored when systematics file is provided.",
        "systematics_file": "(str) path to the .yaml file that provides the setup for the systematic uncertainty parameters. Will overwrite error_budget.",
        "seed": "(int) Value of the random seed used.",
        **kwargs: Additional sampling parameters that are passed to the sampler.
    """
    
    likelihood: EMLikelihood
    prior: Prior

    def __init__(self, 
                 likelihood: EMLikelihood, 
                 prior: Prior,
                 outdir: str = "./outdir/",
                 sampler: str = 'flowmc',
                 error_budget: float = 0.3,
                 systematics_file: str | None = None,
                 seed: int = 42,
                 **kwargs):
               
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
      
        rng_key = jax.random.PRNGKey(seed)

        logger.info("Initializing Fast Inference of Electromagnetic Transients with JAX...")

        # setup the systematic uncertainty
        if systematics_file is not None:
            self.likelihood, self.prior = setup_systematic_from_file(likelihood, prior, systematics_file)
        else:
            self.likelihood, self.prior = setup_systematics_basic(likelihood, prior, error_budget)

        # setup sampler
        match sampler:
            case "flowmc":
                from fiesta.inference.samplers.flowmc import FlowMCSampler
                sampler_cls = FlowMCSampler
            case "blackjax-smc":
                from fiesta.inference.samplers.blackjax_smc import BlackJaxSMC
                sampler_cls = BlackJaxSMC
            case "numpyro-svi":
                from fiesta.inference.samplers.numpyro_svi import SVISampler
                sampler_cls = SVISampler
            case _:
                raise ValueError(
                    f"Unknown sampler '{sampler}'. "
                    "Supported samplers are 'flowmc', 'blackjax-smc', 'numpyro-svi'."
                )

        
        self.sampler = sampler_cls(self.likelihood,
                                   self.prior,
                                   rng_key,
                                   **kwargs)

        logger.info("Initializing Fast Inference of Electromagnetic Transients with JAX... DONE")

    def sample(self, key: PRNGKeyArray, **kwargs):
        """
        Starts the sampling algorithm from ``.sampler`` to obtain the posterior. 
        After running, the posterior samples are stored as a ``.posterior_samples`` attribute.

        Args:
            key (PRNGKeyArray): Random seed to start sampling.
            **kwargs: Additional arguments that are passed to the sample method of the ``.sampler``.
        """
        
        logger.info("Starting sampling.")
        start_time = time.perf_counter()
        self.posterior_samples = self.sampler.sample(key, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Sampling finished. Sampling took {end_time-start_time:.2f} seconds.")

    def _check_sampled(self):
        """Raise if sample() has not been called yet."""
        if not hasattr(self, "posterior_samples"):
            raise RuntimeError(
                "No posterior samples available. Call .sample() before "
                "print_summary(), save_results(), or plot methods."
            )

    def print_summary(self):
        self._check_sampled()
        self.sampler.print_summary()
        for key, value in self.posterior_samples.items():
            if key in ["log_prob", "log_likelihood"]:
                continue
            lower_lim, median, upper_lim = jnp.quantile(value, q=jnp.array([0.16, 0.5, 0.84]))
            print(f"{key}: {median:.3f} + {upper_lim-median:.3f} - {median-lower_lim:.3f}")

    def save_results(self, bestfit_params: bool = True, sampler_extra_output: bool = False):
        """
        Saves the posterior samples to .npz files in ``outdir``.

        Args:
            bestfit_params (bool): Whether to print an extra file with the best fit parameters and light curves. Defaults to True.
            sampler_extra_output (bool): Whether to save additional sampler output to the outdir. Defaults to False.
        """
        self._check_sampled()
        self.sampler.save(sampler_extra_output, self.outdir)

        if bestfit_params:
            # - best fit params
            name = os.path.join(self.outdir, f'bestfit_params.pkl')
            logger.info(f"Saving best fit params to {name}.")

            lc_plotter = LightcurvePlotter(self.posterior_samples,
                                           self.likelihood)
            lc_plotter._get_best_fit_lc()
            chisq_dict = lc_plotter.get_chisquared(per_dof=True)

            with open(name, "wb") as f:
                pickle.dump({"bestfit_params": lc_plotter.best_fit_params,
                             "light_curves": {"times": lc_plotter.t_best_fit, **lc_plotter.best_fit_lc},
                             "chisq": chisq_dict}, f)


        name = os.path.join(self.outdir, f"posterior.npz")
        logger.info(f"Saving posterior samples to {name}.")
        jnp.savez(name, **self.posterior_samples)
            

    def plot_lightcurves(self):
        """Plot the data and the posterior lightcurves and the best fit lightcurve more visible on top."""
        self._check_sampled()
        lc_plotter = LightcurvePlotter(self.posterior_samples,
                                       self.likelihood)

        filters = self.likelihood.filters

        ### Plot the data
        height = len(filters) * 2.5
        fig, ax = plt.subplots(nrows = len(filters), ncols = 1, figsize = (8, height))
        
        for cax, filt in zip(ax, filters):

            lc_plotter.plot_data(cax, filt, color="red")
            lc_plotter.plot_best_fit_lc(cax, filt, color="blue")
            lc_plotter.plot_sample_lc(cax, filt)
            
            # Make pretty
            cax.set_ylabel(filt)
            cax.set_xlim(left=np.maximum(self.likelihood.tmin, 1e-4), right=self.likelihood.tmax)
            cax.set_xscale("log")
            ymin = np.min(np.concatenate([lc_plotter.mag_det[filt], lc_plotter.mag_nondet[filt]])) - 2
            ymax = np.max(np.concatenate([lc_plotter.mag_det[filt], lc_plotter.mag_nondet[filt]])) + 2
            cax.set_ylim(ymax, ymin)
        
        ax[-1].set_xlabel("$t$ in days")
        
        # Save
        fig.savefig(os.path.join(self.outdir, "lightcurves.pdf"), bbox_inches = 'tight', dpi=250)
    
    def plot_corner(self, truths: dict | None = None):
        self._check_sampled()
        fig, ax = corner_plot(self.posterior_samples,
                              self.prior.naming,
                              truths=truths)

        if fig is None:
            return

        fig.savefig(os.path.join(self.outdir, "corner.pdf"), dpi=250, bbox_inches='tight')


