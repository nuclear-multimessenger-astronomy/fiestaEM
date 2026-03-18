import os
import json

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Float, Array, PRNGKeyArray
from jax.experimental import io_callback

from fiesta.logging import logger

try:
    import numpyro
    import numpyro.distributions as npdist
    from numpyro.infer import SVI as NumpyroSVI, Trace_ELBO
except ImportError:
    raise ImportError(f"Please install numpyro if you want to use the numpyro-svi sampler.")

class SVISampler:
    """
    Run numpyro SVI to approximate the posterior.

    Constructs a numpyro model that wraps self.log_posterior (reusing the
    existing likelihood + prior) and a diagonal-normal guide with
    per-parameter learned location and scale.  After optimisation,
    ``self.svi_num_samples`` draws are produced.
    """
    def __init__(self,
                 likelihood,
                 prior,
                 rng_key: PRNGKey,
                 num_iter: int = 10_000,
                 step_size: int = 0.001,
                 num_samples: int = 1000,
                 ):

        self.prior = prior
        self.likelihood = likelihood

        self.num_iter = num_iter
        self.step_size = step_size
        self.num_samples = num_samples

        self.n_dim = self.prior.n_dim


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

        # Extract parameter names and bounds from the prior
        # CompositePrior has .priors list; bare Prior wraps itself
        prior_list = getattr(self.prior, 'priors', [self.prior])
        param_names = []; prior_mins = []; prior_maxs = []; prior_means = []; prior_stds = []
        for sub_prior in prior_list:
            for name in sub_prior.naming:
                param_names.append(name)
                xmin = float(sub_prior.xmin) if hasattr(sub_prior, 'xmin') else -10.0
                xmax = float(sub_prior.xmax) if hasattr(sub_prior, 'xmax') else 10.0
                prior_mins.append(xmin)
                prior_maxs.append(xmax)
                prior_means.append(0.5 * (xmin + xmax))
                prior_stds.append(0.25 * (xmax - xmin))

        prior_mins_j = jnp.array(prior_mins)
        prior_maxs_j = jnp.array(prior_maxs)
        init_loc = jnp.array(prior_means)
        init_scale = jnp.array(prior_stds) / 5.0
        

        # SVI model distribution
        def model():
            with numpyro.plate("params", self.n_dim):
                params = numpyro.sample("theta",
                                        npdist.TruncatedNormal(loc=init_loc, 
                                                               scale=jnp.array(prior_stds),
                                                               low=prior_mins_j, 
                                                               high=prior_maxs_j),
                                        )
            numpyro.factor("logposterior", logposterior_fn(params))
    
        # constraints
        def guide():
            loc = numpyro.param(
                "loc", init_loc,
                constraint=npdist.constraints.interval(prior_mins_j, prior_maxs_j),
            )
            scale = numpyro.param(
                "scale", init_scale,
                constraint=npdist.constraints.positive,
            )
            with numpyro.plate("params", self.n_dim):
                numpyro.sample("theta", npdist.Normal(loc, scale))

        # initialize svi_sampler
        optimizer = numpyro.optim.Adam(self.step_size)
        self.svi_sampler = NumpyroSVI(model, guide, optimizer, loss=Trace_ELBO())

        logger.info(f"Set up numpyro-svi sampler with {self.num_iter} iterations,"
                    f" {self.step_size} step_size, and "
                    f" {self.num_samples} number of samples.")

    def sample(self, key: PRNGKeyArray):

        key, subkey = jax.random.split(key)
        init_state = self.svi_sampler.init(subkey)

        # JIT-compiled scan loop with NaN safety
        @jax.jit
        def body_fn(num_state, _):
            step, state = num_state
            new_state, loss = self.svi_sampler.update(state)
            safe_state = jax.lax.cond(
                jnp.isfinite(loss),
                lambda: new_state,
                lambda: state,
            )

            def log_fn(_):
                bar_length = 30
                filled =  (step + 1) / self.num_iter * bar_length
                jax.debug.print("SVI sampling step {}/{}. Loss = {}.", step + 1, self.num_iter, loss)

            jax.lax.cond(
                (step+1) % (self.num_iter // 10) ==0,
                log_fn,
                lambda _: None,
                operand=None
            )

            return (step+1, safe_state), loss
        
        # main sampling loop
        (step, svi_state), losses = jax.lax.scan(body_fn, (0, init_state), None, length=self.num_iter)

        # Extract learned parameters and draw posterior samples
        params = self.svi_sampler.get_params(svi_state)
        loc = params["loc"]
        scale = params["scale"]

        key, subkey = jax.random.split(key)
        z_norm = jax.random.normal(subkey, shape=(self.num_samples, self.n_dim))
        draws = jnp.array(loc) + z_norm * jnp.array(scale)

        samples = draws.T  # (n_dim, n_samples)
        posterior_samples = self.prior.add_name(samples)

        # Compute log_prob for each sample (vectorized)
        
        log_prior = self.prior.log_prob(posterior_samples)
        posterior_samples["log_likelihood"] = self.likelihood.vectorized_evaluate(posterior_samples)
        posterior_samples["log_prob"] = posterior_samples["log_likelihood"] + log_prior
        

        # Store SVI-specific attributes for diagnostics
        self.metadata = {
            "elbo": float(losses[-1]),
            "svi_losses": jnp.array(losses).tolist(),
            "svi_loc": jnp.array(loc).tolist(),
            "svi_scale": jnp.array(scale).tolist()
        }


        return posterior_samples
    
    def save(self, sampler_extra_output: bool, outdir: str) -> None:

        logger.info(f"Evidence lower-bound: {self.metadata['elbo']:.3f}")

        if sampler_extra_output:
            name = os.path.join(outdir, "svi_metadata.json")
            logger.info(f"SVISampler saving metadata to {name}.")

            with open(name, "w") as f:
                json.dump(self.metadata, f)

    def print_summary(self,):
        """
        Print summary statement of the run
        """

        print("\n \n")
        print("=" * 20)
        print(f"num_iter: {self.num_iter}")
        print(f"ELBO: {self.metadata['elbo']:.3f}")
        for j, key in enumerate(self.prior.naming):
            print(f"loc {key}: {self.metadata["svi_loc"][j]}")
            print(f"scale {key}: {self.metadata["svi_scale"][j]}")
        print("=" * 20)
        print("\n")


