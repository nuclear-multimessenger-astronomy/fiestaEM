""" TODO: This is copied over from jim, later on, we can make jim a dependence and then remove this?"""
from dataclasses import field
from typing import Callable

import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker

class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    naming: list[str]
    transforms: dict[str, tuple[str, Callable]] = field(default_factory=dict)

    @property
    def n_dim(self):
        return len(self.naming)

    def __init__(
        self, naming: list[str], transforms: dict[str, tuple[str, Callable]] = {}
    ):
        """
        Parameters
        ----------
        naming : list[str]
            A list of names for the parameters of the prior.
        transforms : dict[tuple[str,Callable]]
            A dictionary of transforms to apply to the parameters. The keys are
            the names of the parameters and the values are a tuple of the name
            of the transform and the transform itself.
        """
        self.naming = naming
        self.transforms = {}

        def make_lambda(name):
            return lambda x: x[name]

        for name in naming:
            if name in transforms:
                self.transforms[name] = transforms[name]
            else:
                # Without the function, the lambda will refer to the variable name instead of its value,
                # which will make lambda reference the last value of the variable name
                self.transforms[name] = (name, make_lambda(name))

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Apply the transforms to the parameters.

        Parameters
        ----------
        x : dict
            A dictionary of parameters. Names should match the ones in the prior.

        Returns
        -------
        x : dict
            A dictionary of parameters with the transforms applied.
        """
        output = {}
        for value in self.transforms.values():
            output[value[0]] = value[1](x)
        return output

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).
        """

        return dict(zip(self.naming, x))

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        raise NotImplementedError

    def log_prob(self, x: dict[str, Array]) -> Float:
        raise NotImplementedError


@jaxtyped(typechecker=typechecker)
class Uniform(Prior):
    xmin: float = 0.0
    xmax: float = 1.0

    def __repr__(self):
        return f"Uniform(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Uniform needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.uniform(
            rng_key, (n_samples,), minval=self.xmin, maxval=self.xmax
        )
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        output = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        return output + jnp.log(1.0 / (self.xmax - self.xmin))


@jaxtyped(typechecker=typechecker)
class Normal(Prior):
    mu: float = 0.0
    sigma: float = 1.0

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    def __init__(
        self,
        mu: Float,
        sigma: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Normal needs to be 1D distributions"
        self.mu = mu
        self.sigma = sigma

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a normal distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.normal(rng_key, (n_samples,),)
        samples = self.mu + self.sigma * samples
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        return -1/(2*self.sigma**2) * (variable-self.mu)**2 - jnp.sqrt(2*jnp.pi*self.sigma**2)
    
# class DiracDelta(Prior):
    
#     value: float
    
#     def __init__(self, 
#                  value: float,
#                  naming: list[str],
#                  transforms: dict[str, tuple[str, Callable]] = {},
#                  **kwargs):
#         super().__init__(naming, transforms)
#         self.value = value
        
#     def sample(self,
#                 rng_key: PRNGKeyArray,
#                 n_samples: int) -> dict[str, Float[Array, " n_samples"]]:
#           return self.add_name(jnp.ones(n_samples) * self.value)
      
#     def log_prob(self, x: dict[str, Array]) -> Float:
#         variable = x[self.naming[0]]
#         output = jnp.where(variable == self.value, jnp.zeros_like(variable), jnp.zeros_like(variable) - jnp.inf)
#         return output
    
class Composite(Prior):
    priors: list[Prior] = field(default_factory=list)

    def __repr__(self):
        return f"Composite(priors={self.priors}, naming={self.naming})"

    def __init__(
        self,
        priors: list[Prior],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        naming = []
        self.transforms = {}
        for prior in priors:
            naming += prior.naming
            self.transforms.update(prior.transforms)
        self.priors = priors
        self.naming = naming
        self.transforms.update(transforms)

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = {}
        for prior in self.priors:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, x: dict[str, Float]) -> Float:
        output = 0.0
        for prior in self.priors:
            output += prior.log_prob(x)
        return output