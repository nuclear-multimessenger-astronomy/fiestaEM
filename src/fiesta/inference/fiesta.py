import copy
import json
import os
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

from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

# see https://github.com/kazewong/flowMC/blob/main/src/flowMC/resource_strategy_bundle/RQSpline_MALA.py#L22
# for all the other arguments that can be set to the strategy-resource bundle
default_bundle_hyperparameters = {
        "n_local_steps": 50,
        "n_global_steps": 200,
        "n_training_loops": 20,
        "n_production_loops": 15,
        "n_epochs": 100,
        "rq_spline_n_layers": 4,
        "rq_spline_hidden_units": [64, 64],
        "rq_spline_n_bins": 8,
        "mala_step_size": 2e-3,
        "learning_rate": 4e-4,
        "n_max_examples": 10_000,
        "n_NFproposal_batch_size": 10_000,
        "chain_batch_size": 100,
        "batch_size": 10_000,
        "verbose": True,
        }


class Fiesta(object):
    """
    Master inference class for interfacing with flowMC.

    Args:
        "likelihood": "(EMLikelihood) likelihood object used for the inference",
        "prior": "(Prior) prior object used for the inference. It has to contain the parameters needed to evaluate likelihood.evaluate().",
        "error_budget": "(float) fixed systematic error to use in the inference in mag. Defaults to 0.3 but is ignored when systematics file is provided.",
        "systematics_file": "(str) path to the .yaml file that provides the setup for the systematic uncertainty parameters. Will overwrite error_budget.",
        "seed": "(int) Value of the random seed used.",
        "n_chains": "(int) Number of chains to be run in parallel by the flowMC sampler.",
        "num_layers": "(int) Number of hidden layers of the NF",
        "hidden_size": "List[int, int] Sizes of the hidden layers of the NF",
        "num_bins": "(int) Number of bins used in MaskedCouplingRQSpline",
        "local_sampler_arg": "(dict) Additional arguments to be used in the local sampler",
        "n_walkers_maximize_likelihood": "(int) Number of walkers used in the maximization of the likelihood with the evolutionary optimizer",
        "n_loops_maximize_likelihood": "(int) Number of loops to run the evolutionary optimizer in the maximization of the likelihood",
        "which_local_sampler": "(str) Name of the local sampler to use",
    """
    
    likelihood: EMLikelihood
    prior: Prior

    def __init__(self, 
                 likelihood: EMLikelihood, 
                 prior: Prior,
                 outdir: str = "./outdir/",
                 error_budget: float = 0.3,
                 systematics_file: str = None,
                 seed: int = 42,
                 n_chains: int = 200,
                 **kwargs):
        
        self.likelihood = likelihood
        self.prior = prior
        
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
      
        rng_key = jax.random.PRNGKey(seed)

        logger.info(f"Initializing Fast Inference of Electromagnetic Transients with JAX...")

        # setup the systematic uncertainty
        if systematics_file is not None:
            self.likelihood, self.prior = setup_systematic_from_file(self.likelihood, self.prior, systematics_file)
        else:
            self.likelihood, self.prior = setup_systematics_basic(self.likelihood, self.prior, error_budget)

        # Set and override any given hyperparameters, and save as attribute
        self.bundle_hyperparameters = default_bundle_hyperparameters

        for key, value in kwargs.items():
            if key in self.bundle_hyperparameters:
                self.bundle_hyperparameters[key] = value


        # TODO: what if we don't want to use MALA as local sampler?
        rng_key, subkey = jax.random.split(rng_key)
        bundle = RQSpline_MALA_Bundle(
            rng_key=subkey,
            n_chains=n_chains,
            n_dims=self.prior.n_dim,
            logpdf=self.log_posterior,
            **self.bundle_hyperparameters)
        
        rng_key, subkey = jax.random.split(rng_key)
        self.Sampler = Sampler(
            self.prior.n_dim,
            n_chains,
            subkey,
            resource_strategy_bundles=bundle,
        )
        logger.info(f"Initializing Fast Inference of Electromagnetic Transients with JAX... DONE")

    def log_posterior(self, params: Float[Array, "n_dims"], data: dict[str, any]) -> Float:
        prior_params = self.prior.add_name(params.T)
        log_prior = self.prior.log_prob(prior_params)
        log_posterior = self.likelihood.evaluate(self.prior.transform(prior_params), data) + log_prior
        return log_posterior

    def sample(self, key: PRNGKeyArray, initial_guess: Array = jnp.array([])):
        if initial_guess.size == 0:
            initial_guess_named = self.prior.sample(key, self.Sampler.n_chains)
            initial_guess = jnp.stack([initial_guess_named[key] for key in self.prior.naming]).T
        
        logger.info(f"Starting sampling.")
        start_time = time.perf_counter()
        self.Sampler.sample(initial_guess, data={"data": jnp.zeros(self.prior.n_dim)}) # the data argument is ignored because data is setup in the likelihood
        end_time = time.perf_counter()
        logger.info(f"Sampling finished. Sampling took {end_time-start_time:.2f} seconds.")

        # setup the production samples
        samples = self.Sampler.resources["positions_production"].data
        log_prob = self.Sampler.resources["log_prob_production"].data
        
        samples = samples.reshape(-1, self.prior.n_dim).T
        self.posterior_samples = self.prior.add_name(samples)
        self.posterior_samples["log_prob"] = log_prob.reshape(-1,)
        
        # TODO: memory issues cause crash here
        #self.posterior["log_likelihood"] = self.likelihood.v_evaluate(self.posterior)

    
    def _get_summary_statistics(self,):

        resources = self.Sampler.resources

        self.training_chain = resources["positions_training"].data.reshape(-1, self.prior.n_dim).T

        self.training_log_prob = resources["log_prob_training"].data
        training_local_acceptance = resources["local_accs_training"].data
        self.training_local_acceptance = training_local_acceptance[~jnp.isneginf(training_local_acceptance)]
        training_global_acceptance = resources["global_accs_training"].data
        self.training_global_acceptance = training_global_acceptance[~jnp.isneginf(training_global_acceptance)]
        self.training_loss = resources["loss_buffer"].data

        self.production_chain = resources["positions_production"].data.reshape(-1, self.prior.n_dim).T
        self.production_log_prob = resources["log_prob_production"].data
        production_local_acceptance = resources["local_accs_production"].data
        self.production_local_acceptance = production_local_acceptance[~jnp.isneginf(production_local_acceptance)]
        production_global_acceptance = resources["global_accs_production"].data
        self.production_global_acceptance = production_global_acceptance[~jnp.isneginf(production_global_acceptance)]



    def print_summary(self, transform: bool = True):
        """
        Generate summary of the run

        """
        self._get_summary_statistics()

        print("Training summary")
        print("=" * 10)
        training_chain = self.prior.add_name(self.training_chain)
        for key, value in training_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")

        print(
            f"Log probability: {self.training_log_prob.mean():.3f} +/- {self.training_log_prob.std():.3f}"
        )

        training_local_acceptance = jnp.mean(self.training_local_acceptance, axis=0)
        print(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        
        training_global_acceptance = jnp.mean(self.training_global_acceptance, axis=0)
        print(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )

        print(
            f"Max loss: {self.training_loss.max():.3f}, Min loss: {self.training_loss.min():.3f}"
        )
        
        print("\n \n")

        print("Production summary")
        print("=" * 10)
        production_chain = self.prior.add_name(self.production_chain)
        for key, value in production_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")

        print(
            f"Log probability: {self.production_log_prob.mean():.3f} +/- {self.production_log_prob.std():.3f}"
        )

        production_local_acceptance = jnp.mean(self.production_local_acceptance, axis=0)
        print(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )

        production_global_acceptance = jnp.mean(self.production_global_acceptance, axis=0)
        print(
            f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
        )
        print("=" * 10)
    
    def save_results(self):

        self._get_summary_statistics()
        
        # - training phase
        name = os.path.join(self.outdir, f'results_training.npz')
        logger.info(f"Saving training samples to {name}.")

        jnp.savez(name, log_prob=self.training_log_prob,
                        chains = self.training_chain,
                        local_accs=jnp.mean(self.training_local_acceptance, axis=0),
                        global_accs=jnp.mean(self.training_global_acceptance, axis=0), 
                        loss_vals=self.training_loss)
        
        #  - production phase
        name = os.path.join(self.outdir, f'results_production.npz')
        logger.info(f"Saving production samples to {name}")
        
        jnp.savez(name, chains=self.production_chain, 
                        log_prob=self.production_log_prob,
                        local_accs=jnp.mean(self.production_local_acceptance, axis=0),
                        global_accs=jnp.mean(self.production_global_acceptance, axis=0)
        )
        
        jnp.savez(os.path.join(self.outdir, f"posterior.npz"), **self.posterior_samples)

    
    def save_hyperparameters(self):
        
        hyperparameters_dict = {"flowmc": self.Sampler.hyperparameters}
        
        try:
            name = os.path.join(self.outdir, "hyperparams.json")
            with open(name, 'w') as file:
                json.dump(hyperparameters_dict, file)
        except Exception as e:
            logger.error(f"Error occurred saving jim hyperparameters, are all hyperparams JSON compatible?: {e}")
            

    def plot_lightcurves(self,):
        
        """
        Plot the data and the posterior lightcurves and the best fit lightcurve more visible on top
        """      

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
    
    def plot_corner(self,):

        fig, ax = corner_plot(self.posterior_samples,
                              self.prior.naming)
        
        if fig==1:
            return
        
        fig.savefig(os.path.join(self.outdir, "corner.pdf"), dpi=250, bbox_inches='tight')


