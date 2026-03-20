from dataclasses import field
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker

from astropy.cosmology import Planck18, z_at_value
import astropy.units as u

class Prior(object):
    """
    A thin base clase to do book keeping.

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

class InterpedPrior(Prior):

    xx: Array
    yy: Array

    def __repr__(self,):
        return f"InterpedPrior(x={self.xx}, y={self.yy})"
    
    def __init__(self,
                 xx: Array,
                 yy: Array,
                 naming: list[str]):
        

        normalization_factor = jnp.trapezoid(y=yy, x=xx)
        
        self.xx = jnp.array(xx)
        self.yy = jnp.array(yy) / normalization_factor       
        
        dx = jnp.diff(self.xx)
        increments = 0.5 * (self.yy[:-1] + self.yy[1:]) * dx
        cdf = jnp.concatenate(
            [jnp.array([0.0]), jnp.cumsum(increments)]
        )
        self.cdf = cdf

        super().__init__(naming, {})
        assert self.n_dim == 1, "UniformSourceFrame needs to be a 1D distribution"


    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> dict[str, Float[Array, "n_samples"]]:
        alpha = jax.random.uniform(rng_key, shape=(n_samples))
        samples = jnp.interp(alpha, self.cdf, self.xx)
        return self.add_name(samples[None])
    
    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        return jnp.log(jnp.interp(variable, self.xx, self.yy, left=0, right=0))

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
        return -1/(2*self.sigma**2) * (variable-self.mu)**2 - 0.5 * jnp.log(2*jnp.pi*self.sigma**2)


class TruncatedNormal(Prior):
    """Truncated normal distribution with explicit bounds.

    Useful for informed priors from population studies (e.g., superphot+).
    The SVISampler uses xmin/xmax for its guide constraints and mu/sigma
    for the model's TruncatedNormal distribution.
    """
    mu: float = 0.0
    sigma: float = 1.0
    xmin: float = -10.0
    xmax: float = 10.0

    def __repr__(self):
        return f"TruncatedNormal(mu={self.mu}, sigma={self.sigma}, xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        mu: Float,
        sigma: Float,
        xmin: Float,
        xmax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "TruncatedNormal needs to be 1D distributions"
        self.mu = mu
        self.sigma = sigma
        self.xmin = xmin
        self.xmax = xmax

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        # Standardize bounds for truncated standard normal, then shift/scale
        lower = (self.xmin - self.mu) / self.sigma
        upper = (self.xmax - self.mu) / self.sigma
        samples = jax.random.truncated_normal(
            rng_key, lower, upper, (n_samples,),
        )
        samples = self.mu + self.sigma * samples
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        in_bounds = (variable >= self.xmin) & (variable <= self.xmax)
        # Gaussian kernel
        log_p = -0.5 * ((variable - self.mu) / self.sigma) ** 2 - 0.5 * jnp.log(2 * jnp.pi * self.sigma ** 2)
        # Truncation normalization: divide by (Phi(upper) - Phi(lower))
        lower = (self.xmin - self.mu) / self.sigma
        upper = (self.xmax - self.mu) / self.sigma
        log_Z = jnp.log(jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower))
        return jnp.where(in_bounds, log_p - log_Z, -jnp.inf)


@jaxtyped(typechecker=typechecker)
class UniformVolume(Prior):
    xmin: float = 10.
    xmax: float = 1e5

    def __repr__(self):
        return f"UniformVolume(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "UniformComovingVolume needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin     

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample luminosity distance from a distribution uniform in volume.

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
        vol_max = 4/3 * jnp.pi * self.xmax**3
        vol_min = 4/3 * jnp.pi * self.xmin**3  
        samples = jax.random.uniform(
            rng_key, (n_samples,), minval= vol_min, maxval=vol_max
        )
        samples = (3 / (4*jnp.pi) * samples)**(1/3)
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]

        vol_max = 4/3 * jnp.pi * self.xmax**3
        vol_min = 4/3 * jnp.pi * self.xmin**3  

        output = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.log(
                4*jnp.pi*variable**2 / (vol_max-vol_min)
            ),
        )
        return output


@jaxtyped(typechecker=typechecker)
class UniformSourceFrame(InterpedPrior):
    xmin: float = 10.
    xmax: float = 1e5

    """
    Prior that is uniform in comoving volume and source frame time, analogue to the corresponding bilby prior.
    Uses the default cosmology in fiesta which is Planck18.
    """

    def __repr__(self):
        return f"UniformSourceFrame(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        dmin: Float,
        dmax: Float,
        naming: list[str],
        cosmology = Planck18,
        **kwargs,
    ):
        """
        Args:
           dmin (Float): Minimum luminosity distance in Mpc.
           dmax (Float): Maximum luminosity distance in Mpc.
           naming (list[str]): Parameter name. Must be ['luminosity_distance'].
           cosmology (astropy.cosmology): Astropy cosmology. Defaults to Planck18.
        """

        self.dmax = dmax
        self.dmin = dmin

        xx = jnp.linspace(self.dmin, self.dmax, 500)
        redshift_arr = jnp.array(z_at_value(cosmology.luminosity_distance, xx * u.Mpc))
        ddl_dz = jnp.gradient(xx, redshift_arr)
        yy = cosmology.differential_comoving_volume(redshift_arr).value / (1 + redshift_arr) * 1/ ddl_dz
        
        if naming != ["luminosity_distance"]:
            raise NotImplementedError(f"For now, parameter must be 'luminosity_distance'. {naming[0]} not yet supported.")
        super().__init__(xx, yy, naming)

@jaxtyped(typechecker=typechecker)
class Sine(Prior):
    xmin: float = 0.0
    xmax: float = 1.0

    def __repr__(self):
        return f"Sine(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        naming: list[str],
        xmin: Float = 0,
        xmax: Float = jnp.pi,
        transforms: dict[str, tuple[str, Callable]] = {},
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "Sine needs to be 1D distributions"
        assert xmax > xmin, f"Provided xmax {xmax} is smaller than xmin, needs to be larger than {xmin}."

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
            rng_key, (n_samples,), minval=jnp.cos(self.xmax), maxval=jnp.cos(self.xmin)
        )
        samples = jnp.arccos(samples)
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        output = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.log(jnp.sin(variable) / (jnp.cos(self.xmin) - jnp.cos(self.xmax))),
        )
        return output


@jaxtyped(typechecker=typechecker)
class LogUniform(Prior):
    xmin: float = 0.0
    xmax: float = 1.0

    def __repr__(self):
        return f"LogUniform(xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        xmin: Float,
        xmax: Float,
        naming: list[str],
        transforms: dict[str, tuple[str, Callable]] = {},
        **kwargs,
    ):
        super().__init__(naming, transforms)
        assert self.n_dim == 1, "LogUniform needs to be 1D distributions"
        assert xmin > 0, f"Provided xmin {xmin} is negative, needs to be larger than 0."
        assert xmax > xmin, f"Provided xmax {xmax} is smaller than xmin, needs to be larger than {xmin}."

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
            rng_key, (n_samples,), minval=jnp.log(self.xmin), maxval=jnp.log(self.xmax)
        )
        samples = jnp.exp(samples)
        return self.add_name(samples[None])

    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        output = jnp.where(
            (variable >= self.xmax) | (variable <= self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        return output + jnp.log(1.0 / (jnp.log(self.xmax) - jnp.log(self.xmin)) ) - jnp.log(variable)


class CompositePrior(Prior):
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

class Constraint(Prior):
    xmin: float
    xmax: float
    def __init__(self,
                 naming: list[str], 
                 xmin: Float,
                 xmax: Float,
                 transforms: dict[str, tuple[str, Callable]] = {})->None:
        super().__init__(naming = naming, transforms=transforms)
        self.xmin = xmin
        self.xmax = xmax
    
    def log_prob(self, x: dict[str, Array]) -> Float:
        variable = x[self.naming[0]]
        output = jnp.where(
            (variable > self.xmax) | (variable < self.xmin),
            jnp.zeros_like(variable) - jnp.inf,
            jnp.zeros_like(variable),
        )
        return output