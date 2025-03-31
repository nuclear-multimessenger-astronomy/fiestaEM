import jax
import jax.numpy as jnp

from typing import Callable
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from .prior import Prior, Constraint, CompositePrior


class ConstrainedPrior(CompositePrior):
    priors: CompositePrior
    constraints: list[Constraint]
    conversion: Callable
    factor: Float
    def __init__(self, priors: list, conversion_function: Callable=lambda x: x, transforms: dict[str, tuple[str, Callable]] = {}):

        super().__init__([prior for prior in priors if not isinstance(prior, Constraint)])

        self.constraints = [constraint for constraint in priors if isinstance(constraint, Constraint)]
        self.conversion = conversion_function
        
        self._estimate_normalization()
    
    def _estimate_normalization(self, nrepeats: int = 10, sampling_chunk: int = 50_000):
        rng_key = jax.random.key(314159265)
        factor_estimates = []
        for _ in range(nrepeats):
            rng_key, subkey = jax.random.split(rng_key)
            samples = super().sample(subkey, n_samples = sampling_chunk)
            constr = ~jnp.isneginf(self.evaluate_constraints(samples))
            factor_estimates.append(sampling_chunk/jnp.sum(constr))
        factor_estimates = jnp.array(factor_estimates)
        decimals = int( -jnp.floor(jnp.log10(3*jnp.std(factor_estimates))) )
        self.factor = jnp.round(jnp.mean(factor_estimates), decimals)
       
    def evaluate_constraints(self, samples):
        converted_sample = self.conversion(samples)
        log_prob = jnp.zeros_like(samples[self.naming[0]])
        for constraint in self.constraints:
            log_prob+=constraint.log_prob(converted_sample)
        return log_prob
    
    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, "n_samples"]]:
    
        rng_key, subkey = jax.random.split(rng_key)
        samples = super().sample(subkey, n_samples)
        constr = ~jnp.isneginf(self.evaluate_constraints(samples))

        while jnp.any(~constr): # not really jax-y but no idea atm how to do implement this logic better
            idx = jnp.where(~constr, jnp.arange(n_samples), 0)
            idx = jnp.unique(idx)# problems with jit here
            rng_key, subkey = jax.random.split(rng_key)
            new_samples = super().sample(subkey, idx.shape[0])
            new_constr = ~jnp.isneginf(self.evaluate_constraints(new_samples))
            def update_arrays(old_arr, new_arr):
                return old_arr.at[idx].set(new_arr)
            samples = jax.tree_util.tree_map(update_arrays, samples, new_samples) # update the samples dic by mapping update_arrays function over it
            constr = constr.at[idx].set(new_constr)
        
        for constraint in self.constraints:
            if constraint.naming[0] in samples.keys():
                del samples[constraint.naming[0]]

        return samples
            

        """
        def check_constraint(state):
            _, constr, _ , _ = state
            return jnp.all(constr)
        
        def update_samples(state):
            samples, constr, rng_key, super = state
            idx = jnp.where(~constr, jnp.arange(constr.shape[0]), 0)
            rng_key, subkey = jax.random.split(rng_key)
            new_samples = super.sample(subkey, jnp.sum(idx!=0))
            new_constr = self.evaluate_constraints(new_samples)

            samples = jax.tree_util.tree_map(update_arrays, samples, new_samples) 
            constr = constr.at[idx].set(new_constr)
            return samples, constr, rng_key, super
        
        rng_key, subkey = jax.random.split(rng_key)
        init_sample = super().sample(subkey, n_samples)
        init_constr = ~jnp.isneginf(self.evaluate_constraints(init_sample))
        init_state = (init_sample, init_constr, rng_key, super())

        final_state = jax.lax.while_loop(check_constraint, update_samples, init_state)
        return final_state[0]
        """        

    def log_prob(self, x: dict[str, Float]) -> Float:
        output = self.evaluate_constraints(x)
        for prior in self.priors:
            output += prior.log_prob(x)
        output += jnp.log(self.factor)
        return output

