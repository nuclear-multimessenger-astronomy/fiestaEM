from functools import partial

import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from flax.core import FrozenDict
import jax

###########################
## BASIC TRANSFORMATIONS ##
###########################

class Scaler(object):
    """
    Base class for all the scalers that depend on some analytic algorithm to transform data.
    """

    def __init__(self,):
        pass

    def fit(self, x: Array) -> None:
        raise NotImplementedError
    
    def transform(self, x: Array) -> Array:
        return x
    
    def inverse_transform(self, x: Array) -> Array:
        return x
    
    def fit_transform(self, x: Array) -> Array:
        self.fit(x)
        return self.transform(x)
    
    def __call__(self, x: Array) -> Array:
        return self.transform(x)

class MinMaxScalerJax(Scaler):
    """
    JAX compatible MinMaxScaler. 
    API inspired by sklearn.
    """
    
    def __init__(self,
                 min_val: Array = 1.,
                 max_val: Array = 0.):
        
        self.min_val = min_val
        self.max_val = max_val
    
    def fit(self, x: Array) -> None:
        self.min_val = jnp.min(x, axis=0)
        self.max_val = jnp.max(x, axis=0)
        assert jnp.all(self.max_val > self.min_val), f"Some components of the input data have no range."
        
    def transform(self, x: Array) -> Array:
        return (x - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, x: Array) -> Array:
        return x * (self.max_val - self.min_val) + self.min_val

    
class StandardScalerJax(Scaler):
    """
    JAX compatible StandardScaler. 
    API inspired by sklearn.
    """
    
    def __init__(self,
                 mu: Array = 0.,
                 sigma: Array = 1.):
        self.mu = mu
        self.sigma = sigma
    
    def fit(self, x: Array) -> None:
        self.mu = jnp.average(x, axis=0)
        self.sigma = jnp.std(x, axis=0)
        assert jnp.all(self.sigma>0), f"Some components of the input have vanishing standard deviation."
        
    def transform(self, x: Array) -> Array:
        return (x - self.mu) / self.sigma
    
    def inverse_transform(self, x: Array) -> Array:
        return x * self.sigma + self.mu


class PCADecomposer(Scaler):
    """
    JAX compatible PCA decomposition.
    API inspired by sklearn.   
    Based on https://github.com/alonfnt/pcax/tree/main.
    """

    def __init__(self, 
                 n_components: int, 
                 solver: str = "randomized"):
        self.n_components = n_components
        self.solver = solver
    
    def fit(self, x: Array)-> None:
        if self.solver == "full":
            self._fit_full(x)
        elif self.solver == "randomized":
            rng = jax.random.PRNGKey(self.n_components)
            self._fit_randomized(x, rng)
        else:
            raise ValueError("solver parameter is not correct")
    
    def _fit_full(self, x: Array):
        n_samples, n_features = x.shape
        self.means = jnp.mean(x, axis=0, keepdims=True)
        x = x - self.means

        _, S, Vt = jax.scipy.linalg.svd(x, full_matrices= False)

        self.explained_variance_  = (S[:self.n_components] ** 2) / (n_samples - 1)
        total_var = jnp.sum(S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        self.Vt = Vt[:self.n_components]

    def _fit_randomized(self, x: Array, rng, n_iter = 5):
        """Randomized PCA based on Halko et al [https://doi.org/10.48550/arXiv.1007.5510]."""
        n_samples, n_features = x.shape
        self.means = jnp.mean(x, axis=0, keepdims=True)
        x = x - self.means
    
        # Generate n_features normal vectors of the given size
        size = jnp.minimum(2 * self.n_components, n_features)
        Q = jax.random.normal(rng, shape=(n_features, size))
    
        def step_fn(q, _):
            q, _ = jax.scipy.linalg.lu(x @ q, permute_l=True)
            q, _ = jax.scipy.linalg.lu(x.T @ q, permute_l=True)
            return q, None
    
        Q, _ = jax.lax.scan(step_fn, init=Q, xs=None, length=n_iter)
        Q, _ = jax.scipy.linalg.qr(x @ Q, mode="economic")
        B = Q.T @ x
    
        _, S, Vt = jax.scipy.linalg.svd(B, full_matrices=False)
        
        self.explained_variance_  = (S[:self.n_components] ** 2) / (n_samples - 1)
        total_var = jnp.sum(S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        self.Vt = Vt[:self.n_components]
    
    def transform(self, x: Array)->Array:
        return jnp.dot(x - self.means, self.Vt.T)
    
    def inverse_transform(self, x: Array)->Array:
        return jnp.dot(x, self.Vt) + self.means

    
class SVDDecomposer(Scaler):
    """
    JAX compatible SVD Decomposition.
    Based on the old NMMA approach to decompose lightcurves into SVD coefficients.
    """
    def __init__(self,
                 svd_ncoeff: Int):
        self.svd_ncoeff = svd_ncoeff
        self.scaler = MinMaxScalerJax()
    
    def fit(self, x: Array):
        xcopy = x.copy()
        xcopy = self.scaler.fit_transform(xcopy)
           
        # Do SVD decomposition on the training data
        UA, _, VA = jnp.linalg.svd(xcopy, full_matrices=True)
        self.VA = VA[:self.svd_ncoeff]
    
    def transform(self, x: Array) -> Array:
        x = self.scaler.transform(x)
        x = jnp.dot(x, self.VA.T)
        return x
    
    def inverse_transform(self, x: Array) -> Array:
        x = jnp.dot(x, self.VA)
        x = self.scaler.inverse_transform(x)
        return x


class ImageScaler(Scaler):
    """
    Scaler that down samples 2D arrays of shape upscale to downscale and the inverse.
    Note that the methods always assume that the input array x is flattened along the last axis, i.e. it will reshape the input x.reshape(-1, *upscale). 
    The down sampled image is scaled once more with a scaler object.
    Attention, this object has no proper fit method, because of its application in FluxTrainerCVAE and the way the data is loaded there to avoid memory issues.
    """
    def __init__(self, 
                 downscale: Int[Array, "shape=(2,)"],
                 upscale: Int[Array, "shape=(2,)"]):
        
        #these are defined here so that upscale and downscale become static
        def transform(x: Array) -> Array:
            x = x.reshape(-1, upscale[0], upscale[1])
            x = jax.image.resize(x, shape=(x.shape[0], downscale[0], downscale[1]), method="cubic")
            x = x.reshape(-1, downscale[0]*downscale[1])
            return x
        
        def inverse_transform(x: Array) -> Array:
            x = x.reshape(-1, downscale[0], downscale[1])
            x = jax.image.resize(x, shape = (x.shape[0], upscale[0], upscale[1]), method = "cubic")
            out = jax.vmap(self.fix_edges)(x[:, :, 4:-4]) # this is necessary because jax.image.resize produces artefacts at the edges when upsampling
            return out

        self.transform = transform
        self.inverse_transform = inverse_transform

    
    def fit(self, x: Array):
        pass    
    
    @staticmethod
    @jax.vmap
    def fix_edges(yp: Array)-> Array:
        """Extrapolate at early and late times from the reconstructed array to avoid artefacts at the edges from jax.image.resize.""" # TODO: a bit hacky
        xp = jnp.arange(4, yp.shape[0]+4)
        xl = jnp.arange(0,4)
        xr = jnp.arange(yp.shape[0]+4, yp.shape[0]+8)
        yl = jnp.interp(xl, xp, yp, left = "extrapolate", right = "extrapolate")
        yr = jnp.interp(xr, xp, yp, left = "extrapolate", right = "extrapolate")
        out = jnp.concatenate([yl, yp, yr])
        return out



#########################
### PARAMETER SCALERS ###
#########################

class ParameterScaler(Scaler):

    def __init__(self,
                 scaler: Scaler,
                 parameter_names: list[str],
                 conversion: str):
        
        self.parameter_names = parameter_names

        if conversion == "thetaWing_inclination":
            self.conversion = thetaWing_inclination
        else:
            self.conversion = identity
            
        self.scaler = scaler
    
    def fit(self, x: Array) -> None:
        x = self.conversion(x)
        self.scaler.fit(x)

    def transform(self, x: Array) -> Array:
        x = self.conversion(x)
        return self.scaler.transform(x)
    
def thetaWing_inclination(x):
    return jnp.hstack((x, (x[:,3]*x[:,2]-x[:,0]).reshape(-1,1) ))

def identity(x):
    return x



####################
### DATA SCALERS ###
####################

class DataScaler(Scaler):

    def __init__(self,
                 scalers: list[Scaler]):
        
        self.scalers = scalers
        self.scalers_transform = [scaler.transform for scaler in self.scalers]

    def fit(self, x: Array) -> None:
        for scaler in self.scalers:
            x = scaler.fit_transform(x)
        self.scalers_transform = [scaler.transform for scaler in self.scalers]
    
    @partial(jax.jit, static_argnames=("self",))
    def transform(self, x: Array) -> Array:
        # here we can use a for loop, 
        # as we typically only chain two or three scalers,
        # so the compile time will not increase to drastically
        for scaler in self.scalers:
            x = scaler.transform(x)
        return x
    
    @partial(jax.jit, static_argnames=("self",))
    def inverse_transform(self, x: Array) -> Array:

        for scaler in reversed(self.scalers):
            x = scaler.inverse_transform(x)
        return x

        
