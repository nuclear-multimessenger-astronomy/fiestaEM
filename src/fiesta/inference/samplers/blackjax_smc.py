import os
from typing import Any, Callable, cast
import json

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Float, Array, PRNGKeyArray
from jax.experimental import io_callback

from fiesta.logging import logger

try: 
    from blackjax import inner_kernel_tuning, adaptive_tempered_smc
    from blackjax.mcmc import random_walk
    from blackjax.smc import extend_params
    from blackjax.smc.base import SMCInfo
    from blackjax.smc.resampling import systematic
    from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
    from blackjax.smc.tempered import TemperedSMCState
    from blackjax.smc.tuning.from_particles import particles_covariance_matrix
    from blackjax import nuts

except ImportError as err:
    raise ImportError("Please install blackjax if you want to use the blackjax-smc sampler.") from err

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
                 n_particles: int = 8000,
                 target_ess: float = 0.9,
                 num_mcmc_steps: int = 10,
                 random_walk_sigma = 0.05,
                 kernel: str = "random_walk"):
        
        self.prior = prior
        self.likelihood = likelihood

        self.n_particles = n_particles
        self.random_walk_sigma = random_walk_sigma # is ignored if kernel is 'nuts'
        self.num_mcmc_steps = num_mcmc_steps
        self.target_ess = target_ess
        self.kernel = kernel


        def logprior_fn(x: Float[Array, "n_particles ndims"]) -> Float:
            x = jnp.atleast_1d(x)
            x_dict = self.prior.add_name(x.T)
            return self.prior.log_prob(x_dict)
        
        def loglikelihood_fn(x: Float[Array, "n_particles ndims"]) -> Float:
            x = jnp.atleast_1d(x)
            x_dict = self.prior.add_name(x.T)
            x_dict = self.prior.transform(x_dict)
            return self.likelihood.evaluate(x_dict)
        
        def logposterior_fn(x: Float[Array, "n_particles ndims"]) -> Float:
            return logprior_fn(x) + loglikelihood_fn(x)


        initial_position_named = self.prior.sample(rng_key, self.n_particles)
        initial_position = jnp.stack([initial_position_named[p] for p in self.prior.naming]).T
        self.initial_position = initial_position

        mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn = self.setup_mcmc_kernel(logprior_fn, 
                                                                                                   loglikelihood_fn,
                                                                                                   logposterior_fn,
                                                                                                   self.initial_position,
                                                                                                   kernel=kernel)
        
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

        logger.info(f"Set up blackjax-smc sampler with {kernel} kernel, "
                    f"{self.n_particles} particles, "
                    f"{target_ess} target ESS, "
                    f"and {num_mcmc_steps} MCMC steps per tempering stage.")

    def sample(self, key: PRNGKey):
        
        key, subkey = jax.random.split(key)
        state = self.smc_alg.init(self.initial_position, subkey)
        
        ##########################
        # Prepare jax-while loop #
        ##########################

        # Carry value will be: (StateWithParameterOverride, 
        #                       key, 
        #                       step_count, 
        #                       tempering_param_history, 
        #                       ess_history, 
        #                       acceptance_history, 
        #                       log_evidence)

        
        # Define loop conditions with proper type hints
        def cond_fn(carry: tuple[StateWithParameterOverride, PRNGKeyArray, int, Array, Array, Array, float]) -> bool:
            state, _, _, _, _, _, _ = carry
            # Cast to proper type for type checker (runtime type is correct)
            sampler_state = cast(TemperedSMCState, state.sampler_state)
            return sampler_state.tempering_param < 1  # type: ignore[return-value]

        def body_fn(
                    carry: tuple[StateWithParameterOverride, PRNGKeyArray, int, Array, Array, Array, float]
                    )-> tuple[StateWithParameterOverride, PRNGKeyArray, int, Array, Array, Array, float]:

            (state,
             key,
             step_count,
             tempering_param_history,
             ess_history,
             acceptance_history,
             log_evidence) = carry
            
            key, subkey = jax.random.split(key)
            state, info = self.smc_alg.step(subkey, state)

            # Cast to proper types for type checker (runtime types are correct)
            state = cast(StateWithParameterOverride, state)
            info = cast(SMCInfo, info)
            sampler_state = cast(TemperedSMCState, state.sampler_state)

            # Accumulate log evidence from log_likelihood_increment
            log_evidence = log_evidence + info.log_likelihood_increment

            # Compute ESS
            weights = sampler_state.weights
            ess_value = (jnp.sum(weights) ** 2 / jnp.sum(weights**2) / self.n_particles)

            # Extract acceptance rate
            # Note: update_info is kernel-specific NamedTuple, not fully typed in blackjax
            acceptance_rate = info.update_info.acceptance_rate.mean()  # type: ignore[attr-defined]

            # Update histories
            tempering_param_history = tempering_param_history.at[step_count].set(
                sampler_state.tempering_param
            )
            ess_history = ess_history.at[step_count].set(ess_value)
            acceptance_history = acceptance_history.at[step_count].set(acceptance_rate)

            # Print progress update using io_callback
            io_callback(
                progress_callback,
                None,  # No return value
                step_count,
                sampler_state.tempering_param,
                ess_value,
                acceptance_rate,
            )

            return (
                state,
                key,
                step_count + 1,
                tempering_param_history,
                ess_history,
                acceptance_history,
                log_evidence,
            )
        
        # Progress callback for live updates during sampling
        def progress_callback(step: int, tempering_param: float, ess: float, acceptance: float) -> None:
            """
            Print progress update during sampling (called via io_callback).
            """
            # Create progress bar
            bar_length = 30
            filled = int(tempering_param * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Print update
            logger.info(
                f"Step {step:4d} | λ={tempering_param:.6f} | ESS={ess*100:5.1f}% | "
                f"Accept={acceptance*100:5.1f}% | {bar}"
            )
        
        #####################
        # Do jax-while loop #
        #####################

        max_steps = 1000

        init_carry = (
            state, # state
            key, # key
            0, # step_count
            jnp.zeros(max_steps), # tempering_param_history
            jnp.zeros(max_steps), # ess_history
            jnp.zeros(max_steps), # acceptance_history
            0.0, # initial log evidence
        )

        (state,
         key,
         steps,
         tempering_param_history,
         ess_history,
         acceptance_history,
         log_evidence) = jax.lax.while_loop(
                                            cond_fn, 
                                            body_fn, 
                                            init_carry  # type: ignore[arg-type]
        )

        final_sampler_state = cast(TemperedSMCState, state.sampler_state)

        ###################
        # post processing #
        ###################

        weights = final_sampler_state.weights
        ess = jnp.sum(weights) **2 / jnp.sum(weights**2)

        mean_ess = float(jnp.mean(ess_history[:steps]))
        min_ess = float(jnp.min(ess_history[:steps]))
        mean_acceptance = float(jnp.mean(acceptance_history[:steps]))

        log_evidence_err = 0.0 # TODO   

        self.metadata = {
            "kernel_type": self.kernel,
            "n_particles": self.n_particles,
            "num_mcmc_steps": self.num_mcmc_steps,
            "target_ess": self.target_ess,
            "annealing_steps": int(steps),
            "final_ess": float(ess),
            "final_ess_percent": float(ess / self.n_particles * 100),
            "mean_ess": float(mean_ess),
            "min_ess": float(min_ess),
            "mean_acceptance": mean_acceptance,
            "logZ": float(log_evidence),
            "logZ_err": float(log_evidence_err),
            "tempering_param_history": tempering_param_history[:steps].tolist(),
            "ess_history": ess_history[:steps].tolist(),
            "acceptance_history": acceptance_history[:steps].tolist(),
        }


        ###############
        # get samples #
        ###############

        samples = cast(Array, final_sampler_state.particles)
        samples = samples.reshape(-1, self.prior.n_dim).T
        posterior_samples = self.prior.add_name(samples)
        log_prior = self.prior.log_prob(posterior_samples)
        posterior_samples["log_likelihood"] = self.likelihood.vectorized_evaluate(posterior_samples)
        posterior_samples["log_prob"] = posterior_samples["log_likelihood"] + log_prior

        return posterior_samples
    
    def save(self, sampler_extra_output: bool, outdir: str) -> None:
        
        logger.info(f"Estimated log-evidence from sampling: {self.metadata['logZ']:.3f}")
        
        if sampler_extra_output:
            name = os.path.join(outdir, 'smc_metadata.json')
            logger.info(f"BlackJaxSMC sampler saving metadata to {name}.")

            with open(name, "w") as f:
                json.dump(self.metadata, f)
    
    def print_summary(self,):
        """
        Print summary statement of the run
        """
        print("\n \n")
        print("=" * 20)
        print(f"kernel_type: {self.metadata['kernel_type']}")
        print(f"n_particles: {self.metadata['n_particles']}")
        print(f"n_steps: {self.metadata['annealing_steps']}")
        for key in ["final_ess", 
                    "mean_ess", 
                    "min_ess", 
                    "mean_acceptance", 
                    "logZ"]:
            print(f"{key}: {self.metadata[key]:.3f}")

        print("=" * 20)
        print("\n")
       
    def setup_mcmc_kernel(self,                       
                          logprior_fn: Callable, 
                          loglikelihood_fn: Callable, 
                          logposterior_fn: Callable, 
                          initial_particles: Array,
                          kernel: str = "random_walk") -> tuple[Callable, Callable, dict, Callable]:
            
        match kernel: 
            case "random_walk":
                return self._setup_random_walk_kernel(initial_particles)
            case "nuts":
                return self._setup_nuts_kernel(logprior_fn,
                                               loglikelihood_fn,
                                               logposterior_fn,
                                               initial_particles)
            case _:
                raise ValueError("Kernel for blackjax-smc must either be 'random_walk' or 'nuts'.")
                                            
    
    def _setup_random_walk_kernel(self, initial_particles: Array) -> tuple[Callable, Callable, dict, Callable]:
            """
            Setup Random Walk kernel with covariance adaptation.

            The proposal covariance is computed from current particles and scaled by a
            fixed sigma^2 factor. Only the covariance shape is adapted, not the overall scale.

            Parameters
            ----------
            initial_particles : Array
                Initial particle positions for computing initial covariance

            Returns
            -------
            tuple[Callable, Callable, dict, Callable]
                (mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn)
            """

            # Setup random walk kernel with additive step
            kernel = random_walk.build_additive_step()

            # Compute initial covariance from initial particles
            init_cov = particles_covariance_matrix(initial_particles)
            # Ensure 2D array (n_dim, n_dim) even for 1D problems
            init_cov = jnp.atleast_2d(init_cov)
            # Scale by fixed sigma^2
            init_cov = init_cov * (self.random_walk_sigma**2)

            init_params = {"cov": init_cov}

            # Define parameter update function with covariance adaptation only
            def mcmc_parameter_update_fn(key, state, info):
                """Adapt proposal covariance based on current particle distribution.

                The covariance matrix is computed from current particles and scaled by
                the fixed sigma^2 parameter. No scale adaptation is performed.
                """
                # Note: state here is TemperedSMCState, particles are at state.particles

                # Compute covariance matrix from current particles
                cov = particles_covariance_matrix(state.particles)
                # Ensure 2D array (n_dim, n_dim) even for 1D problems
                cov = jnp.atleast_2d(cov)

                # Scale covariance by fixed sigma^2
                scaled_cov = cov * (self.random_walk_sigma**2)

                return extend_params({"cov": scaled_cov})  # type: ignore[arg-type]

            # Wrap kernel to match expected signature
            def mcmc_step_fn(rng_key, state, logdensity_fn, **params):
                """Random walk step function with multivariate normal proposal."""
                cov = params.get("cov", init_cov)

                def proposal_distribution(key, position):
                    """Multivariate normal proposal using covariance matrix."""
                    x, ravel_fn = jax.flatten_util.ravel_pytree(position)
                    return ravel_fn(
                        jax.random.multivariate_normal(key, jnp.zeros_like(x), cov)
                    )

                return kernel(rng_key, state, logdensity_fn, proposal_distribution)

            # Init function for random walk
            mcmc_init_fn = random_walk.init

            return mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn
    
    def _setup_nuts_kernel(self, logprior_fn: Callable, loglikelihood_fn: Callable,
                               logposterior_fn: Callable, initial_particles: Array) -> tuple[Callable, Callable, dict, Callable]:
        """Setup NUTS kernel with Hessian adaptation.

        Parameters
        ----------
        logposterior_fn : Callable
            Log posterior function for computing Hessian

        Returns
        -------
        tuple[Callable, Callable, dict, Callable]
            (mcmc_step_fn, mcmc_init_fn, init_params, mcmc_parameter_update_fn)
        """
        raise NotImplementedError
        
        