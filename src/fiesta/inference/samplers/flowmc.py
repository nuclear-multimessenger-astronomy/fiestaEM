import os

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Float, Array, PRNGKeyArray

from fiesta.logging import logger

from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle


class FlowMCSampler:
    """
    Sampler from the flowmc package.
    Uses Metropolis-adjusted Langevin algorithm and trains normalizing flows during sampling to have efficient proposals.
    See https://arxiv.org/abs/2211.06397 for details.
    Original code from https://github.com/kazewong/flowMC/.
    """
        
    def __init__(self,
                 likelihood,
                 prior,
                 rng_key: PRNGKey,
                 n_chains: int = 500,
                 n_local_steps = 50,
                 n_global_steps = 200,
                 n_training_loops = 20,
                 n_production_loops = 15,
                 n_epochs = 100,
                 rq_spline_n_layers =4,
                 rq_spline_hidden_units=None,
                 rq_spline_n_bins = 8,
                 mala_step_size = 2e-3,
                 learning_rate = 4e-4,
                 n_max_examples=10_000,
                 n_NFproposal_batch_size = 10_000,
                 chain_batch_size = 100,
                 batch_size = 10_000,
                 verbose = True,):
        
        self.prior = prior
        self.likelihood = likelihood

        if rq_spline_hidden_units is None:
            rq_spline_hidden_units = [64, 64]

        def log_posterior_fn(params: Float[Array, "n_dims"], data: dict[str, any]) -> Float:
            params_named = self.prior.add_name(params.T)
            log_prior = self.prior.log_prob(params_named)
            params_named = self.prior.transform(params_named)
            log_posterior = self.likelihood.evaluate(params_named) + log_prior
            return log_posterior
        
        # TODO: what if we don't want to use MALA as local sampler?
        rng_key, subkey = jax.random.split(rng_key)
        bundle = RQSpline_MALA_Bundle(
                                      rng_key=subkey,
                                      n_chains=n_chains,
                                      n_dims=self.prior.n_dim,
                                      logpdf=log_posterior_fn,
                                      n_local_steps=n_local_steps,
                                      n_global_steps=n_global_steps,
                                      n_training_loops=n_training_loops,
                                      n_production_loops=n_production_loops,
                                      n_epochs=n_epochs,
                                      rq_spline_n_layers=rq_spline_n_layers,
                                      rq_spline_hidden_units=rq_spline_hidden_units,
                                      rq_spline_n_bins=rq_spline_n_bins,
                                      mala_step_size=mala_step_size,
                                      learning_rate=learning_rate,
                                      n_max_examples=n_max_examples,
                                      n_NFproposal_batch_size=n_NFproposal_batch_size,
                                      chain_batch_size=chain_batch_size,
                                      batch_size=batch_size,
                                      verbose=verbose,
                                    )
        
        rng_key, subkey = jax.random.split(rng_key)
        self.Sampler = Sampler(
            self.prior.n_dim,
            n_chains,
            subkey,
            resource_strategy_bundles=bundle,
        )

        logger.info(f"Set up flowmc sampler with {n_chains} chains, "
                    f"{n_training_loops} training loops "
                    f"and {n_production_loops} production loops.")

    def sample(self, key: PRNGKey, initial_position: Array | None = None):
        """
        Starts the flowmc sampling algorithm.
        After running, the posterior samples are stored as ``.posterior_samples`` attribute.

        Args:
            key (PRNGKey): Random seed to start sampling.
            initial_position (Array, optional): Initial positions of the chains. If None, samples from the prior.
        """
        if initial_position is None:
            initial_position_named = self.prior.sample(key, self.Sampler.n_chains)
            initial_position = jnp.stack([initial_position_named[p] for p in self.prior.naming]).T
        
        self.Sampler.sample(initial_position, data={"data": jnp.zeros(self.prior.n_dim)}) # the data argument is ignored because data is setup in the likelihood

        # setup the production samples
        samples = self.Sampler.resources["positions_production"].data
        log_prob = self.Sampler.resources["log_prob_production"].data
        
        samples = samples.reshape(-1, self.prior.n_dim).T
        posterior_samples = self.prior.add_name(samples)
        posterior_samples["log_prob"] = log_prob.reshape(-1,)
        posterior_samples["log_likelihood"] = posterior_samples["log_prob"] - self.prior.log_prob(posterior_samples)

        return posterior_samples

    def get_summary_statistics(self):

        if not hasattr(self, "production_global_acceptance"):

            resources = self.Sampler.resources

            self.training_chain = resources["positions_training"].data.reshape(-1, self.prior.n_dim).T

            self.training_log_prob = resources["log_prob_training"].data
            training_local_acceptance = resources["local_accs_training"].data
            self.training_local_acceptance = jnp.where(jnp.isfinite(training_local_acceptance), training_local_acceptance, jnp.nan)
            training_global_acceptance = resources["global_accs_training"].data
            self.training_global_acceptance = jnp.where(jnp.isfinite(training_global_acceptance), training_global_acceptance, jnp.nan)
            self.training_loss = resources["loss_buffer"].data

            self.production_chain = resources["positions_production"].data.reshape(-1, self.prior.n_dim).T
            self.production_log_prob = resources["log_prob_production"].data
            production_local_acceptance = resources["local_accs_production"].data
            self.production_local_acceptance = jnp.where(jnp.isfinite(production_local_acceptance), production_local_acceptance, jnp.nan)
            production_global_acceptance = resources["global_accs_production"].data
            self.production_global_acceptance = jnp.where(jnp.isfinite(production_global_acceptance), production_global_acceptance, jnp.nan)
    
    def save(self, sampler_extra_output: bool, outdir: str) -> None:

        if sampler_extra_output:

            self.get_summary_statistics()

            # - training phase
            name = os.path.join(outdir, 'results_training.npz')
            logger.info(f"FlowMC sampler saving training samples to {name}.")
    
            jnp.savez(name, log_prob=self.training_log_prob,
                            chains = self.training_chain,
                            local_accs=jnp.nanmean(self.training_local_acceptance, axis=0),
                            global_accs=jnp.nanmean(self.training_global_acceptance, axis=0), 
                            loss_vals=self.training_loss
            )
            
            #  - production phase
            name = os.path.join(outdir, 'results_production.npz')
            logger.info(f"FlowMC sampler saving production samples to {name}.")
            
            jnp.savez(name, chains=self.production_chain, 
                            log_prob=self.production_log_prob,
                            local_accs=jnp.nanmean(self.production_local_acceptance, axis=0),
                            global_accs=jnp.nanmean(self.production_global_acceptance, axis=0)
            )            
      
    def print_summary(self):
        """
        Print summary statement of the run
        """
        
        self.get_summary_statistics()
        print("\n")
        print("=" * 20)
        print("Training summary")
        print("=" * 20)

        print(
            f"Log probability: {self.training_log_prob.mean():.3f} +/- {self.training_log_prob.std():.3f}"
        )

        training_local_acceptance = jnp.nanmean(self.training_local_acceptance, axis=0)
        print(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        
        training_global_acceptance = jnp.nanmean(self.training_global_acceptance, axis=0)
        print(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )

        print(
            f"Max loss: {self.training_loss.max():.3f}, Min loss: {self.training_loss.min():.3f}"
        )
        print("\n")
        print("=" * 20)
        print("Production summary")
        print("=" * 20)

        print(
            f"Log probability: {self.production_log_prob.mean():.3f} +/- {self.production_log_prob.std():.3f}"
        )

        production_local_acceptance = jnp.nanmean(self.production_local_acceptance, axis=0)
        print(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )

        production_global_acceptance = jnp.nanmean(self.production_global_acceptance, axis=0)
        print(
            f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
        )
        print("\n")


