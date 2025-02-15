import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from fiesta.constants import pc_to_cm, h_erg_s, c, H0


#######################
# DISTANCE CONVERSION #
#######################

def Mpc_to_cm(d: float):
    return d * 1e6 * pc_to_cm

def redshift_to_luminosity_distance(z: Array, Omega_m=0.321):
    
    def correction_factor(z: Float):
        z_arr = jnp.linspace(0, z, 100)
        integrand = ( Omega_m* (1+z_arr)**3 + (1-Omega_m) )**(-0.5)
        return jnp.trapezoid(x=z_arr, y=integrand)
    
    correction = jax.vmap(correction_factor)(z)
    luminosity_distance = c / H0 * (1+z) * correction
    return luminosity_distance

z_arr = jnp.logspace(-6, jnp.log10(15), 200)
dL_arr = redshift_to_luminosity_distance(z_arr)

def luminosity_distance_to_redshift(dL: Array):
    return jnp.interp(dL, dL_arr, z_arr)
    

###################
# FLUX CONVERSION #
###################

def Flambda_to_Fnu(F_lambda: Float[Array, "n_lambdas n_times"], lambdas: Float[Array, "n_lambdas"]) -> Float[Array, "n_lambdas n_times"]:
    """
    JAX-compatible conversion of wavelength flux in erg cm^{-2} s^{-1} Angström^{-1} to spectral flux density in mJys.

    Args: 
        flux_lambda (Float[Array]): 2D flux density array in erg cm^{-2} s^{-1} Angström^{-1}. The rows correspond to the wavelengths provided in lambdas.
        lambdas (Float[Array]): 1D wavelength array in Angström.
    Returns:
        mJys (Float[Array]): 2D spectral flux density array in mJys
        nus (Float[Array]): 1D frequency array in Hz
    """
    F_lambda = F_lambda.reshape(lambdas.shape[0], -1)
    log_F_lambda = jnp.log10(F_lambda) # got to log because of large factors
    log_F_nu = log_F_lambda + 2* jnp.log10(lambdas[:, None]) + jnp.log10(3.3356) + 4 # https://en.wikipedia.org/wiki/AB_magnitude
    F_nu = 10**(log_F_nu)
    F_nu = F_nu[::-1, :] # reverse the order to get lowest frequencies in first row
    mJys = 1e3 * F_nu # convert Jys to mJys
    
    nus = c / (lambdas*1e-10)
    nus = nus[::-1]

    return mJys, nus

def Fnu_to_Flambda(F_nu: Float[Array, "n_nus n_times"], nus: Float[Array, "n_nus"]) -> Float[Array, "n_nus n_times"]:
    """
    JAX-compatible conversion of spectral flux density in mJys to wavelength flux in erg cm^{-2} s^{-1}.

    Args: 
        flux_nu (Float[Array]): 2D flux density array in mJys. The rows correspond to the frequencies provided in nus.
        nus (Float[Array]): 1D frequency array in Hz.
    Returns:
        flux_lambda (Float[Array]): 2D wavelength flux density array in erg cm^{-2} s^{-1} Angström^{-1}.
        lambdas (Float[Array]): 1D wavelength array in Angström.
    """
    F_nu = F_nu.reshape(nus.shape[0], -1)
    log_F_nu = jnp.log10(F_nu) # go to log because of large factors
    log_F_nu  = log_F_nu - 3 # convert mJys to Jys
    log_F_lambda = log_F_nu + 2 * jnp.log10(nus[:, None]) + jnp.log10(3.3356) - 42
    F_lambda = 10**(log_F_lambda)  
    F_lambda = F_lambda[::-1, :] # reverse the order to get the lowest wavelegnths in first row
    
    lambdas = c / nus
    lambdas = lambdas[::-1] * 1e10

    return F_lambda, lambdas

def apply_redshift(F_nu: Float[Array, "n_nus n_times"], times: Float[Array, "n_times"], nus: Float[Array, "n_nus"], z: Float):
    
    F_nu = F_nu * (1 + z) # this is just the frequency redshift, cosmological energy loss and time elongation are taken into account by luminosity_distance
    times = times * (1 + z)
    nus = nus / (1 + z)

    return F_nu, times, nus

########################
# MAGNITUDE CONVERSION #
########################

def monochromatic_AB_mag(flux: Float[Array, "n_nus n_times"],
                         nus: Float[Array, "n_nus"],
                         nus_filt: Float[Array, "n_nus_filt"],
                         trans_filt: Float[Array, "n_nus_filt"],
                         ref_flux: Float) -> Float[Array, "n_times"]:
    
    interp_col = lambda col: jnp.interp(nus_filt, nus, col)
    mJys = jax.vmap(interp_col, in_axes = 1, out_axes = 1)(flux) # apply vectorized interpolation to interpolate columns of 2D array

    mJys = mJys * trans_filt[:, None]
    mag = mJys_to_mag_jnp(mJys)
    return mag[0]

def bandpass_AB_mag(flux: Float[Array, "n_nus n_times"],
                    nus: Float[Array, "n_nus"],
                    nus_filt: Float[Array, "n_nus_filt"],
                    trans_filt: Float[Array, "n_nus_filt"],
                    ref_flux: Float) -> Float[Array, "n_times"]:
    """
    This is a JAX-compatile equivalent of sncosmo.TimeSeriesSource.bandmag(). Unlike sncosmo, we use the frequency flux and not wavelength flux,
    but this function is tested to yield the same results as the sncosmo version.

    Args:
        flux (Float[Array, "n_nus n_times"]): Spectral flux density as a 2D array in mJys.
        nus (Float[Array, "n_nus"]): Associated frequencies in Hz
        nus_filt (Float[Array, "n_nus_filt"]): frequency array of the filter in Hz
        trans_filt (Float[Array, "n_nus_filt"]): transmissivity array of the filter in transmitted photons / incoming photons
        ref_flux (Float): flux in mJy for which the filter is 0 mag
    """
    
    interp_col = lambda col: jnp.interp(nus_filt, nus, col)
    mJys = jax.vmap(interp_col, in_axes = 1, out_axes = 1)(flux) # apply vectorized interpolation to interpolate columns of 2D array

    log_mJys = jnp.log10(mJys) # go to log because of large factors
    log_mJys = log_mJys + jnp.log10(trans_filt[:, None])
    log_mJys = log_mJys - jnp.log10(h_erg_s) - jnp.log10(nus_filt[:, None])  # https://en.wikipedia.org/wiki/AB_magnitude

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
    
    interp_col = lambda col: jnp.interp(nus_filt, nus, col)
    mJys = jax.vmap(interp_col, in_axes = 1, out_axes = 1)(flux) # apply vectorized interpolation to interpolate columns of 2D array

    log_mJys = jnp.log10(mJys) # go to log because of large factors
    log_mJys = log_mJys + jnp.log10(trans_filt[:, None])

    max_log_mJys = jnp.max(log_mJys)
    integrand = 10**(log_mJys - max_log_mJys) # make the integrand between 0 and 1, otherwise infs could appear
    integrate_col = lambda col: jnp.trapezoid(y = col, x = nus_filt)
    norm_band_flux = jax.vmap(integrate_col, in_axes = 1)(integrand) # normalized band flux

    log_integrated_flux = jnp.log10(norm_band_flux) + max_log_mJys # reintroduce scale here
    log_integrated_flux = log_integrated_flux - jnp.log10(nus_filt[-1] - nus_filt[0]) # divide by integration range
    mJys = 10**log_integrated_flux
    mag = mJys_to_mag_jnp(mJys) 
    return mag

@jax.jit
def mJys_to_mag_jnp(mJys: Array):
    mag = -48.6 + -1 * jnp.log10(mJys) * 2.5 + 26 * 2.5 # https://en.wikipedia.org/wiki/AB_magnitude
    return mag

# TODO: need a np and jnp version?
# TODO: account for extinction
def mJys_to_mag_np(mJys: np.array):
    Jys = 1e-3 * mJys
    mag = -48.6 + -1 * np.log10(Jys / 1e23) * 2.5
    return mag

def mag_app_from_mag_abs(mag_abs: Array,
                         luminosity_distance: Float) -> Array:
    return mag_abs + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)