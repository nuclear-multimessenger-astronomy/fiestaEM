import os

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Float, Array, PRNGKeyArray

from fiesta.logging import logger

class BlackJaxNestedSampling:

    def __init__(self,
                 likelihood,
                 prior,
                 rng_key: PRNGKey,
                 n_live: int,
                 delete_frac: float):

        raise NotImplementedError
            
        self.prior = prior
        self.likelihood = likelihood
        
        self.n_live = n_live
        self.delete_frac = delete_frac

        n_delete = int(self.n_live * self.delete_frac)

        logger.info(f"Set up blackjax-nested-sampling with "
                    f"{self.n_live} live points and "
                    f"{self.delete_frac} deletion fraction.")        
    
    def sample(self, key: PRNGKey, initial_points: dict[str, Array] = {}):
        
        if not initial_points:
            key, subkey = jax.random.split(key)
            initial_points = self.prior.sample(subkey)
        
        key, init_key = jax.random.split(key)
        self.nested_sampler.init(initial_points, rng_key=init_key)

        # JIT compile step function for performance
        step_fn = jax.jit(self.nested_sampler.step)

        def terminate(state):
            """
            Termination condition: stop when remaining evidence is small.
            """
            dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
            return jnp.isfinite(dlogz) and dlogz < self.config.termination_dlogz

        def progress_callback(iteration: int, logZ: float, dlogZ: float) -> None:
            """Print progress update during nested sampling (called via io_callback)."""
            # Format logZ and dlogZ with appropriate precision
            logZ_str = f"{logZ:+10.2f}" if jnp.isfinite(logZ) else "      -inf"
            dlogZ_str = f"{dlogZ:8.4f}" if jnp.isfinite(dlogZ) else "     inf"

            # Print update
            logger.info(
                f"Iteration {iteration:4d} | logZ={logZ_str} | dlogZ={dlogZ_str}"
            ) 


    def save(self, sampler_extra_output: bool, outdir: str) -> None:
        raise NotImplementedError
    
    def print_summary(self,):
        raise NotImplementedError