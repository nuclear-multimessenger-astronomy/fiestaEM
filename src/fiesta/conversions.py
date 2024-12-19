from fiesta.constants import pc_to_cm
import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np

def Mpc_to_cm(d: float):
    return d * 1e6 * pc_to_cm

# TODO: need a np and jnp version?
# TODO: account for extinction
def mJys_to_mag_np(mJys: np.array):
    Jys = 1e-3 * mJys
    mag = -48.6 + -1 * np.log10(Jys / 1e23) * 2.5
    return mag

@jax.jit
def mJys_to_mag_jnp(mJys: Array):
    Jys = 1e-3 * mJys
    mag = -48.6 + -1 * jnp.log10(Jys / 1e23) * 2.5
    return mag

def mag_app_from_mag_abs(mag_abs: Array,
                         luminosity_distance: Float) -> Array:
    return mag_abs + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)