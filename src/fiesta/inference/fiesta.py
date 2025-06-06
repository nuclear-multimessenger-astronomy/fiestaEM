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
from fiesta.plot import corner_plot
from fiesta.inference.systematic import setup_systematic

from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.utils.PRNG_keys import initialize_rng_keys

default_hyperparameters = {
        "seed": 1,
        "n_chains": 20,
        "num_layers": 10,
        "hidden_size": [128,128],
        "num_bins": 8,
        "local_sampler_arg": {'eps': 5e-3},
        "which_local_sampler": "MALA"
}

class Fiesta(object):
    """
    Master class for interfacing with flowMC

    Args:
        "seed": "(int) Value of the random seed used",
        "n_chains": "(int) Number of chains to be used",
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
                 systematics_file: str = None,
                 **kwargs):
        self.likelihood = likelihood
        self.prior = prior
        
        self.outdir = kwargs.get("outdir", "./outdir/")
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
      
      
        logger.info(f"Initializing Fast Inference of Electromagnetic Transients with JAX...")

        # setup the systematic uncertainty
        self.likelihood, self.prior = setup_systematic(self.likelihood, self.prior, systematics_file)

        # Set and override any given hyperparameters, and save as attribute
        self.hyperparameters = default_hyperparameters
        hyperparameter_names = list(self.hyperparameters.keys())
        
        for key, value in kwargs.items():
            if key in hyperparameter_names:
                self.hyperparameters[key] = value
        
        self.hyperparameters["local_sampler_arg"]["step_size"] = self.hyperparameters["local_sampler_arg"]["eps"]*jnp.eye(self.prior.n_dim)

        for key, value in self.hyperparameters.items():
            setattr(self, key, value)

        rng_key_set = initialize_rng_keys(self.hyperparameters["n_chains"], seed=self.hyperparameters["seed"])
        
        # set local sampling method
        if self.hyperparameters["which_local_sampler"] == "MALA":
            logger.info("Using MALA as local sampler.")
            local_sampler = MALA(
                self.posterior, True, self.local_sampler_arg
            )  # Remember to add routine to find automated mass matrix
        elif self.hyperparameters["which_local_sampler"] == "GaussianRandomWalk":
            logger.info("Using gaussian random walk as local sampler")
            local_sampler = GaussianRandomWalk(
                self.posterior, True, self.local_sampler_arg
            )  # Remember to add routine to find automated mass matrix
        else:   
            sampler = self.hyperparameters["which_local_sampler"]
            raise ValueError(f"Local sampler {sampler} not recognized")

        model = MaskedCouplingRQSpline(
            self.prior.n_dim, self.num_layers, self.hidden_size, self.num_bins, rng_key_set[-1]
        )

        self.Sampler = Sampler(
            self.prior.n_dim,
            rng_key_set,
            None,  # type: ignore
            local_sampler,
            model,
            global_sampler=None,
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
        
        chains = chains.reshape(-1, self.prior.n_dim).T
        posterior = dict(zip(self.prior.naming, np.array(chains)))
        posterior["log_prob"] = log_prob.reshape(-1,)
        jnp.savez(os.path.join(self.outdir, f"posterior.npz"), **posterior)

    
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
            

    def plot_lightcurves(self,
                         N_curves: int = 200):
        
        """
        Plot the data and the posterior lightcurves and the best fit lightcurve more visible on top
        """

        production_state = self.Sampler.get_sampler_state(training=False)
        samples, log_prob = production_state["chains"], production_state["log_prob"]
        
        # Reshape both
        samples = samples.reshape(-1, self.prior.n_dim).T
        log_prob = log_prob.reshape(-1)
        
        # Get the best fit lightcurve
        best_fit_idx = np.argmax(log_prob)
        best_fit_params = samples[:, best_fit_idx]
        best_fit_params_named = self.prior.add_name(best_fit_params)
        
        # Limit N_curves if necessary
        total_nb_samples = samples.shape[1]
        N_curves = min(N_curves, total_nb_samples)
        
        # Randomly sample N_curves samples
        idx = np.random.choice(total_nb_samples, N_curves, replace=False)
        samples = samples[:, idx]
        log_prob = log_prob[idx]
        
        filters = self.likelihood.filters
        tmin, tmax = self.likelihood.tmin, self.likelihood.tmax
        
        ### Plot the data
        height = len(filters) * 2.5
        plt.subplots(nrows = len(filters), ncols = 1, figsize = (8, height))
        zorder = 3
        for i, filt in enumerate(filters):
            ax = plt.subplot(len(filters), 1, i + 1)
            
            # Detections
            t, mag, err = self.likelihood.times_det[filt], self.likelihood.mag_det[filt], self.likelihood.mag_err[filt]
            ax.errorbar(t, mag, yerr=err, fmt = "o", color = "red", label = "Data")
            
            # Non-detections
            t, mag = self.likelihood.times_nondet[filt], self.likelihood.mag_nondet[filt]
            ax.scatter(t, mag, marker = "v", color = "red", zorder = zorder)
            
        ### Plot bestfit LC
        best_fit_params_named.update(self.likelihood.fixed_params)
        
        zorder = 2
        # Predict and convert to apparent magnitudes
        best_fit_params_named = self.likelihood.conversion(best_fit_params_named)
        time_obs, mag_bestfit = self.likelihood.model.predict(best_fit_params_named)

        for i, filter_name in enumerate(filters):
            ax = plt.subplot(len(filters), 1, i + 1)
            mag = mag_bestfit[filter_name]
            mask = (time_obs >= tmin) & (time_obs <= tmax)
            ax.plot(time_obs[mask], mag[mask], color = "blue", label = "Best fit", zorder = zorder)
            
        # Other samples
        zorder = 1
        for sample in samples.T:
            sample_named = self.prior.add_name(sample)
            sample_named.update(self.likelihood.fixed_params)
            sample_named = self.likelihood.conversion(sample_named)
            time_obs, mag = self.likelihood.model.predict(sample_named)
            mask = (time_obs >= tmin) & (time_obs <= tmax)

            for i, filter_name in enumerate(filters):
                ax = plt.subplot(len(filters), 1, i + 1)
                ax.plot(time_obs[mask], mag[filter_name][mask], color = "gray", alpha = 0.05, zorder = zorder)
        
        ### Make pretty
        for i, filter_name in enumerate(filters):
            ax = plt.subplot(len(filters), 1, i + 1)
            ax.set_xlabel("Time [days]")
            ax.set_ylabel(filter_name)
            ax.set_xlim(left=np.maximum(self.likelihood.tmin, 1e-4), right=self.likelihood.tmax)
            ax.invert_yaxis() 
            ax.set_xscale("log")
        
        # Save
        plt.savefig(os.path.join(self.outdir, "lightcurves.png"), bbox_inches = 'tight', dpi=250)
        plt.close()
    
    def plot_corner(self,):
        state = self.Sampler.get_sampler_state(training=False)
        samples = state["chains"]
        
        # Reshape both
        samples = samples.reshape(-1, self.prior.n_dim)
        fig, ax = corner_plot(np.array(samples),
                              self.prior.naming)
        
        fig.savefig(os.path.join(self.outdir, "corner.pdf"), dpi=250)


