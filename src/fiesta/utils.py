import jax.numpy as jnp
from jaxtyping import Array, Float

class MinMaxScalerJax(object):
    """
    MinMaxScaler like sklearn does it, but for JAX arrays since sklearn might not be JAX-compatible?
    
    Note: assumes that input has dynamical range: it will not catch errors due to constant input (leading to zero division)
    """
    
    def __init__(self,
                 min_val: Array = None,
                 max_val: Array = None):
        
        self.min_val = min_val
        self.max_val = max_val
    
    def fit(self, x: Array) -> None:
        self.min_val = jnp.min(x, axis=0)
        self.max_val = jnp.max(x, axis=0)
        
    def transform(self, x: Array) -> Array:
        return (x - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, x: Array) -> Array:
        return x * (self.max_val - self.min_val) + self.min_val
    
    def fit_transform(self, x: Array) -> Array:
        self.fit(x)
        return self.transform(x)
    
def inverse_svd_transform(x: Array, 
                          VA: Array, 
                          nsvd_coeff: int = 10) -> Array:

    # TODO: check the shapes etc, transforms and those things
    return jnp.dot(VA[:, :nsvd_coeff], x)