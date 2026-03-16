import os

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Float, Array, PRNGKeyArray

from fiesta.logging import logger

# see https://github.com/kazewong/flowMC/blob/main/src/flowMC/resource_strategy_bundle/RQSpline_MALA.py#L22
# for all the other arguments that can be set to the strategy-resource bundle

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
                 rq_spline_hidden_units =[64, 64],
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

        def log_posterior_fn(self, params: Float[Array, "n_dims"], data: dict[str, any]) -> Float:
            params_named = self.prior.add_name(params.T)
            log_prior = self.prior.log_prob(params_named)
            params_named = self.prior.tranform(params_named)
            log_posterior = self.likelihood.evaluate(params_named) + log_prior
            return log_posterior
        
        from flowMC.Sampler import Sampler
        from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

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

    def sample(self, key: PRNGKey, initial_position: Array = jnp.array([])):
        """
        Starts the flowmc sampling algorithm.
         After running, the posterior samples are stored as ``.posterior_samples`` attribute.

        Args:
            key (PRNGKey): Random seed to start sampling.
            initial_guess (Array, optional): Initial posisions of the chains. If empty, will get initial position as random samples from the prior.
        """
        if initial_position.size == 0:
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
    
    def save(self, sampler_extra_output: bool, outdir: str):

        if sampler_extra_output:

            self.get_summary_statistics()

            # - training phase
            name = os.path.join(outdir, f'results_training.npz')
            logger.info(f"FlowMC sampler saving training samples to {name}.")
    
            jnp.savez(name, log_prob=self.training_log_prob,
                            chains = self.training_chain,
                            local_accs=jnp.mean(self.training_local_acceptance, axis=0),
                            global_accs=jnp.mean(self.training_global_acceptance, axis=0), 
                            loss_vals=self.training_loss
            )
            
            #  - production phase
            name = os.path.join(outdir, f'results_production.npz')
            logger.info(f"FlowMC sampler saving production samples to {name}.")
            
            jnp.savez(name, chains=self.production_chain, 
                            log_prob=self.production_log_prob,
                            local_accs=jnp.mean(self.production_local_acceptance, axis=0),
                            global_accs=jnp.mean(self.production_global_acceptance, axis=0)
            )            
      
    def print_summary(self):
        """
        Generate summary of the run

        """
        
        self.get_summary_statistics()

        print("Training summary")
        print("=" * 10)

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


class BlackJaxSMC:
    """
    Sampler from the blackjax package.
    Uses sequential Monte Carlo algorithm, where initially only the prior is sampled and then gradually the inverse temperature is decreased to eventually sample the posterior distribution.
    Allows for calculation of the evidence. See e.g. https://arxiv.org/pdf/2506.18977 for details.
    Original code from https://github.com/blackjax-devs/blackjax.
    This API is inspired by jester (https://github.com/nuclear-multimessenger-astronomy/jester/).
    """

    def __init__(self,
                 likelihood,
                 prior,
                 rng_key: PRNGKey,
                 n_particles: int = 5000,
                 target_ess: float = 0.9,
                 num_mcmc_steps: int = 10):
        
        self.prior = prior
        self.likelihood = likelihood

        self.n_particles = n_particles


        def logprior_fn(x: Float[Array, "n_particles, ndims"]) -> Float:
            x = jnp.atleast_1d(x)
            x_dict = self.prior.add_name(x.T)
            return self.prior.log_prob(x_dict)
        
        def loglikelihood_fn(x: Float[Array, "n_particles, ndims"]) -> Float:
            x = jnp.atleast_1d(x)
            x_dict = self.prior.add_name(x.T)
            x_dict = self.prior.transform(x_dict)
            return self.likelihood.evaluate(x_dict)
        
        def logposterior_fn(x: Float[Array, "n_particles, ndims"]) -> Float:
            return logprior_fn(x) + loglikelihood_fn(x)

        from blackjax import inner_kernel_tuning, adaptive_tempered_smc
        from blackjax.smc import extend_params
        from blackjax.smc.resampling import systematic

        initial_position_named = self.prior.sample(rng_key, self.n_particles)
        initial_position = jnp.stack([initial_position_named[p] for p in self.prior.naming]).T

        mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn = self._setup_mcmc_kernel(logprior_fn, 
                                                                                                    loglikelihood_fn,
                                                                                                    logposterior_fn,
                                                                                                    initial_position
                                                                                                   )
        
        self.smc_alg = inner_kernel_tuning(
            smc_algorithm=adaptive_tempered_smc,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            resampling_fn=systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params(init_params),  # type: ignore[arg-type]
            target_ess=target_ess,
            num_mcmc_steps=num_mcmc_steps,
        )            
    

    def sample(self, key: PRNGKey):
        

        # Initialize SMC state
        key, subkey = jax.random.split(key)
        state = self.smc_alg.init(initial_position, subkey)

        # Progress callback for live updates during sampling
        def progress_callback(
            step: int, tempering_param: float, ess: float, acceptance: float
        ) -> None:
            """Print progress update during sampling (called via io_callback)."""
            # Create progress bar
            bar_length = 30
            filled = int(tempering_param * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Print update
            logger.info(
                f"Step {step:4d} | λ={tempering_param:.6f} | ESS={ess*100:5.1f}% | "
                f"Accept={acceptance*100:5.1f}% | {bar}"
            )
    
    def _setup_mcmc_kernel(self,
                           logprior_fn,
                           loglikelihood_fn,
                           **kwargs):
        pass
        

    


class BlackJaxNestedSampling:
    raise NotImplementedError(f"blackjax nested sampling still needs to be implemented.")