import copy
import json
import os

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


default_hyperparameters = {
        "n_chains": 20,
        "num_layers": 10,
        "hidden_size": [128,128],
        "num_bins": 8,
        "local_sampler_arg": {'eps': 5e-3},
        "which_local_sampler": "MALA"
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
                 error_budget: float = 0.3,
                 systematics_file: str = None,
                 seed: int = 42,
                 n_chains: int = 20,
                 **kwargs):
        self.likelihood = likelihood
        self.prior = prior
        
        self.outdir = kwargs.get("outdir", "./outdir/")
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
        self.hyperparameters = default_hyperparameters
        hyperparameter_names = list(self.hyperparameters.keys())
        
        for key, value in kwargs.items():
            if key in hyperparameter_names:
                self.hyperparameters[key] = value
        
        self.hyperparameters["local_sampler_arg"]["step_size"] = self.hyperparameters["local_sampler_arg"]["eps"]*jnp.eye(self.prior.n_dim)

        for key, value in self.hyperparameters.items():
            setattr(self, key, value)

        # TODO: what if we don't want to use MALA as local sampler?
        rng_key, subkey = jax.random.split(rng_key)
        bundle = RQSpline_MALA_Bundle(
            rng_key=subkey,
            n_chains=n_chains,
            n_dims=self.prior.n_dim,
            logpdf=self.posterior,
            # 5 more arguments
            )
        
        rng_key, subkey = jax.random.split(rng_key)
        self.Sampler = Sampler(
            self.prior.n_dim,
            n_chains,
            subkey,
            resource_strategy_bundle=bundle,
            **kwargs,
        )
        logger.info(f"Initializing Fast Inference of Electromagnetic Transients with JAX... DONE")

    def posterior(self, params: Float[Array, " n_dim"], data: dict):
        prior_params = self.prior.add_name(params.T)
        prior = self.prior.log_prob(prior_params)
        return (
            self.likelihood.evaluate(self.prior.transform(prior_params), data) + prior
        )

    def sample(self, key: PRNGKeyArray, initial_guess: Array = jnp.array([])):
        if initial_guess.size == 0:
            initial_guess_named = self.prior.sample(key, self.Sampler.n_chains)
            initial_guess = jnp.stack([initial_guess_named[key] for key in self.prior.naming]).T
        
        logger.info(f"Starting sampling.")
        self.Sampler.sample(initial_guess, None)  # type: ignore
        logger.info(f"Sampling finished.")

        # setup the production samples
        production_state = self.Sampler.get_sampler_state(training=False)
        samples, log_prob = production_state["chains"], production_state["log_prob"]
        
        samples = samples.reshape(-1, self.prior.n_dim).T
        self.posterior_samples = self.prior.add_name(samples)
        self.posterior_samples["log_prob"] = log_prob.reshape(-1,)
        
        # TODO: memory issues cause crash here
        #self.posterior["log_likelihood"] = self.likelihood.v_evaluate(self.posterior)


    def print_summary(self, transform: bool = True):
        """
        Generate summary of the run

        """

        train_summary = self.Sampler.get_sampler_state(training=True)
        production_summary = self.Sampler.get_sampler_state(training=False)

        training_chain = train_summary["chains"].reshape(-1, self.prior.n_dim).T
        training_chain = self.prior.add_name(training_chain)
        if transform:
            training_chain = self.prior.transform(training_chain)
        training_log_prob = train_summary["log_prob"]
        training_local_acceptance = train_summary["local_accs"]
        training_global_acceptance = train_summary["global_accs"]
        training_loss = train_summary["loss_vals"]

        production_chain = production_summary["chains"].reshape(-1, self.prior.n_dim).T
        production_chain = self.prior.add_name(production_chain)
        if transform:
            production_chain = self.prior.transform(production_chain)
        production_log_prob = production_summary["log_prob"]
        production_local_acceptance = production_summary["local_accs"]
        production_global_acceptance = production_summary["global_accs"]

        print("Training summary")
        print("=" * 10)
        for key, value in training_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        print(
            f"Log probability: {training_log_prob.mean():.3f} +/- {training_log_prob.std():.3f}"
        )
        print(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        print(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )
        print(
            f"Max loss: {training_loss.max():.3f}, Min loss: {training_loss.min():.3f}"
        )

        print("Production summary")
        print("=" * 10)
        for key, value in production_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")
        print(
            f"Log probability: {production_log_prob.mean():.3f} +/- {production_log_prob.std():.3f}"
        )
        print(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )
        print(
            f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
        )
        print("=" * 10)

    def get_samples(self, training: bool = False) -> dict:
        """
        Get the samples from the sampler

        Parameters
        ----------
        training : bool, optional
            Whether to get the training samples or the production samples, by default False

        Returns
        -------
        dict
            Dictionary of samples

        """
        if training:
            chains = self.Sampler.get_sampler_state(training=True)["chains"]
        else:
            chains = self.Sampler.get_sampler_state(training=False)["chains"]

        chains = self.prior.transform(self.prior.add_name(chains.transpose(2, 0, 1)))
        return chains
    
    def save_results(self):
        # - training phase
        name = os.path.join(self.outdir, f'results_training.npz')
        logger.info(f"Saving training samples to {name}")
        state = self.Sampler.get_sampler_state(training=True)
        chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
        local_accs = jnp.mean(local_accs, axis=0)
        global_accs = jnp.mean(global_accs, axis=0)
        jnp.savez(name, log_prob=log_prob, local_accs=local_accs,
                global_accs=global_accs, loss_vals=loss_vals)
        
        #  - production phase
        name = os.path.join(self.outdir, f'results_production.npz')
        logger.info(f"Saving production samples to {name}")
        state = self.Sampler.get_sampler_state(training=False)
        chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
        local_accs = jnp.mean(local_accs, axis=0)
        global_accs = jnp.mean(global_accs, axis=0)
        jnp.savez(name, chains=chains, log_prob=log_prob,
                    local_accs=local_accs, global_accs=global_accs)
        
        jnp.savez(os.path.join(self.outdir, f"posterior.npz"), **self.posterior_samples)

    
    def save_hyperparameters(self):
        
        # Convert step_size to list for JSON formatting
        if "step_size" in self.hyperparameters["local_sampler_arg"].keys():
            self.hyperparameters["local_sampler_arg"]["step_size"] = np.asarray(self.hyperparameters["local_sampler_arg"]["step_size"]).tolist()
        
        hyperparameters_dict = {"flowmc": self.Sampler.hyperparameters,
                                "jim": self.hyperparameters}
        
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


