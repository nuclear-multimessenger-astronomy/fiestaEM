import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from fiesta.constants import pc_to_cm, h_erg_s, c


def Mpc_to_cm(d: float) -> float:
    """
    Convert distance in Mpc to centimeters.

    Args:
        d (float): Distance in Mpc.

    Returns:
        float: Distance in centimeters.
    """
    return d * 1e6 * pc_to_cm


def Flambda_to_Fnu(F_lambda: Float[Array, "n_lambdas n_times"], 
                   lambdas: Float[Array, "n_lambdas"]) -> tuple[Float[Array, "n_lambdas n_times"], Float[Array, "n_lambdas"]]:
    """
    JAX-compatible conversion of wavelength flux in erg cm^{-2} s^{-1} Angström^{-1} to spectral flux density in mJys.
    We take the log of the flux to avoid numerical issues with large factors.
    
    For the conversion used, see https://en.wikipedia.org/wiki/AB_magnitude

    Args: 
        flux_lambda (Float[Array]): 2D flux density array in erg cm^{-2} s^{-1} Angström^{-1}. The rows correspond to the wavelengths provided in second argument lambdas.
        lambdas (Float[Array]): 1D wavelength array in Angström.
    Returns:
        mJys (Float[Array]): 2D spectral flux density array in mJys, shape (n_lambdas, n_times)
        nus (Float[Array]): 1D frequency array in Hz, length n_lambdas
    """
    F_lambda = F_lambda.reshape(lambdas.shape[0], -1)
    log_F_lambda = jnp.log10(F_lambda)
    log_F_nu = log_F_lambda + 2 * jnp.log10(lambdas[:, None]) + jnp.log10(3.3356) + 4
    F_nu = 10 ** (log_F_nu)
    F_nu = F_nu[::-1, :] # reverse the order to get lowest frequencies in first row
    mJys = 1e3 * F_nu # convert Jys to mJys
    
    nus = c / (lambdas * 1e-10)
    nus = nus[::-1]

    return mJys, nus

def Fnu_to_Flambda(F_nu: Float[Array, "n_nus n_times"], 
                   nus: Float[Array, "n_nus"]) -> tuple[Float[Array, "n_nus n_times"], Float[Array, "n_nus"]]:
    """
    JAX-compatible conversion of spectral flux density in mJys to wavelength flux in erg cm^{-2} s^{-1}.
    We take the log of the flux to avoid numerical issues with large factors.
    
    For the conversion used, see https://en.wikipedia.org/wiki/AB_magnitude

    Args: 
        flux_nu (Float[Array]): 2D flux density array in mJys. The rows correspond to the frequencies provided in second argument nus.
        nus (Float[Array]): 1D frequency array in Hz.
    Returns:
        flux_lambda (Float[Array]): 2D wavelength flux density array in erg cm^{-2} s^{-1} Angström^{-1}, shape is (n_nus, n_times)
        lambdas (Float[Array]): 1D wavelength array in Angström, length is n_nus
    """
    F_nu = F_nu.reshape(nus.shape[0], -1)
    log_F_nu = jnp.log10(F_nu)
    log_F_nu  = log_F_nu - 3 # convert mJys to Jys
    log_F_lambda = log_F_nu + 2 * jnp.log10(nus[:, None]) + jnp.log10(3.3356) - 42
    F_lambda = 10 ** (log_F_lambda)  
    F_lambda = F_lambda[::-1, :] # reverse the order to get the lowest wavelegnths in first row
    
    lambdas = c / nus
    lambdas = lambdas[::-1] * 1e10

    return F_lambda, lambdas

def apply_redshift(F_nu: Float[Array, "n_nus n_times"], 
                   times: Float[Array, "n_times"], 
                   nus: Float[Array, "n_nus"], z: Float) -> tuple[Float[Array, "n_nus n_times"], Float[Array, "n_times"], Float[Array, "n_nus"]]:
    """
    Rescale by redshift. This is just the frequency redshift, cosmological energy loss and time elongation are taken into account by luminosity_distance
    TODO: make sure a reference is provided here since there was some discussion about this

    Args:
        F_nu (Float[Array,]): Flux density in mJys
        times (Float[Array]): Time grid in seconds
        nus (Float[Array]): Frequency grid in Hz
        z (Float): Redshift value

    Returns:
        _type_: _description_
    """
    F_nu = F_nu * (1 + z)
    times = times * (1 + z)
    nus = nus / (1 + z)

    return F_nu, times, nus

def monochromatic_AB_mag(flux: Float[Array, "n_nus n_times"],
                         nus: Float[Array, "n_nus"],
                         nus_filt: Float[Array, "n_nus_filt"],
                         trans_filt: Float[Array, "n_nus_filt"],
                         ref_flux: Float) -> Float[Array, "n_times"]:
    """
    TODO: documentation

    Returns:
        Float[Array]: Magnitudes
    """
    # TODO: ref flux is not used?
    
    interp_col = lambda col: jnp.interp(nus_filt, nus, col)
    # Apply vectorized interpolation to interpolate columns of 2D array
    mJys = jax.vmap(interp_col, in_axes = 1, out_axes = 1)(flux) 

    mJys = mJys * trans_filt[:, None]
    mag = mJys_to_mag_jnp(mJys)
    return mag[0]

def bandpass_AB_mag(flux: Float[Array, "n_nus n_times"],
                    nus: Float[Array, "n_nus"],
                    nus_filt: Float[Array, "n_nus_filt"],
                    trans_filt: Float[Array, "n_nus_filt"],
                    ref_flux: Float) -> Float[Array, "n_times"]:
    """
    This is a JAX-compatile equivalent of sncosmo.TimeSeriesSource.bandmag(). 
    Unlike sncosmo, we use the frequency flux and not wavelength flux, but this function is tested to yield the same results as the sncosmo version.
    
    For more information, see https://en.wikipedia.org/wiki/AB_magnitude

    Args:
        flux (Float[Array, "n_nus n_times"]): Spectral flux density as a 2D array in mJys.
        nus (Float[Array, "n_nus"]): Associated frequencies in Hz
        nus_filt (Float[Array, "n_nus_filt"]): frequency array of the filter in Hz
        trans_filt (Float[Array, "n_nus_filt"]): transmissivity array of the filter in transmitted photons / incoming photons
        ref_flux (Float): flux in mJy for which the filter is 0 mag
    """
    
    # Apply vectorized interpolation to interpolate columns of 2D array
    interp_col = lambda col: jnp.interp(nus_filt, nus, col)
    mJys = jax.vmap(interp_col, in_axes = 1, out_axes = 1)(flux)

    # Work in log space for numerical stability
    log_mJys = jnp.log10(mJys) 
    log_mJys = log_mJys + jnp.log10(trans_filt[:, None])
    log_mJys = log_mJys - jnp.log10(h_erg_s) - jnp.log10(nus_filt[:, None])

    max_log_mJys = jnp.max(log_mJys)
    integrand = 10**(log_mJys - max_log_mJys) # make the integrand between 0 and 1, otherwise infs could appear
    integrate_col = lambda col: jnp.trapezoid(y = col, x = nus_filt)
    norm_band_flux = jax.vmap(integrate_col, in_axes = 1)(integrand) # normalized band flux

    log_integrated_flux = jnp.log10(norm_band_flux) + max_log_mJys # reintroduce scale here
    mag = -2.5 * log_integrated_flux + 2.5 * jnp.log10(ref_flux) 
    return mag

def integrated_AB_mag(flux: Float[Array, "n_nus n_times"],
                      nus: Float[Array, "n_nus"],
                      nus_filt: Float[Array, "n_nus_filt"],
                      trans_filt: Float[Array, "n_nus_filt"]) -> Float[Array, "n_times"]:
    """
    Compute integrated AB magnitude. 
    
    Args:
        flux (Float[Array, "n_nus n_times"]): Spectral flux density as a 2D array in mJys.
        nus (Float[Array, "n_nus"]): Associated frequencies in Hz
        nus_filt (Float[Array, "n_nus_filt"]): frequency array of the filter in Hz

    Returns:
        Float[Array]: Integrated AB magnitudes
    """
    
    # Apply vectorized interpolation to interpolate columns of 2D array
    interp_col = lambda col: jnp.interp(nus_filt, nus, col)
    mJys = jax.vmap(interp_col, in_axes = 1, out_axes = 1)(flux)

    # Work in log space for numerical stability
    log_mJys = jnp.log10(mJys)
    log_mJys = log_mJys + jnp.log10(trans_filt[:, None])

    # Make the integrand between 0 and 1, otherwise infs could appear
    max_log_mJys = jnp.max(log_mJys)
    integrand = 10 ** (log_mJys - max_log_mJys) 
    integrate_col = lambda col: jnp.trapezoid(y = col, x = nus_filt)
    # Normalize band flux
    norm_band_flux = jax.vmap(integrate_col, in_axes = 1)(integrand)

    # Rescale then divide by integration range
    log_integrated_flux = jnp.log10(norm_band_flux) + max_log_mJys
    log_integrated_flux = log_integrated_flux - jnp.log10(nus_filt[-1] - nus_filt[0])
    mJys = 10 ** log_integrated_flux
    mag = mJys_to_mag_jnp(mJys) 
    return mag

# TODO: can we remove the np vesion and only use the jnp version? Perhaps need to remove jit decorator in that case
@jax.jit
def mJys_to_mag_jnp(mJys: float) -> float:
    """
    Simple conversion function from mJys to AB magnitude, see https://en.wikipedia.org/wiki/AB_magnitude

    Args:
        mJys (float): The flux in mJys

    Returns:
        float: The AB magnitude
    """
    mag = -48.6 + -1 * jnp.log10(mJys) * 2.5 + 26 * 2.5
    return mag

# TODO: account for extinction
def mJys_to_mag_np(mJys: float) -> float:
    """
    Numpy-compatible conversion function from mJys to AB magnitude, see https://en.wikipedia.org/wiki/AB_magnitude

    Args:
        mJys (np.array): The flux in mJys

    Returns:
        float: The AB magnitude
    """
    Jys = 1e-3 * mJys
    mag = -48.6 + -1 * np.log10(Jys / 1e23) * 2.5
    return mag

def mag_app_from_mag_abs(mag_abs: Array, luminosity_distance: Float) -> Array:
    """
    Simple conversion function to go from absolute to apparent magnitude.

    Args:
        mag_abs (Array): Abolsute magnitudes
        luminosity_distance (Float): The luminosity distance in Mpc

    Returns:
        Array: Apparent magnitudes
    """
    return mag_abs + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)