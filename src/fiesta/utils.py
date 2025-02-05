import copy
import re

import numpy as np
import pandas as pd
from astropy.time import Time
from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS
from sncosmo import get_bandpass
import scipy.interpolate as interp

import jax.numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, Float, Int
import jax

from fiesta.conversions import monochromatic_AB_mag, bandpass_AB_mag
import fiesta.constants as constants



####################
### DATA SCALERS ###
####################

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
    
class StandardScalerJax(object):
    """
    StandardScaler like sklearn does it, but for JAX arrays since sklearn might not be JAX-compatible?
    
    Note: assumes that input has dynamical range: it will not catch errors due to constant input (leading to zero division)
    """
    
    def __init__(self,
                 mu: Array = None,
                 sigma: Array = None):
        
        self.mu = mu
        self.sigma = sigma
    
    def fit(self, x: Array) -> None:
        self.mu = jnp.average(x, axis=0)
        self.sigma = jnp.std(x, axis=0)
        
    def transform(self, x: Array) -> Array:
        return (x - self.mu) / self.sigma
    
    def inverse_transform(self, x: Array) -> Array:
        return x * self.sigma + self.mu
    
    def fit_transform(self, x: Array) -> Array:
        self.fit(x)
        return self.transform(x)

class PCADecomposer(object):
    """
    PCA decomposer like sklearn does it. Based on https://github.com/alonfnt/pcax/tree/main.
    """
    def __init__(self, n_components: int, solver: str = "randomized"):
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
    
    def fit_transform(self, x: Array)-> Array:
        self.fit(x)
        return self.transform(x)
    
class SVDDecomposer(object):
    """
    SVDDecomposer that uses the old NMMA approach to decompose lightcurves into SVD coefficients.
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
    
    def fit_transform(self, x: Array)-> Array:
        self.fit(x)
        return self.transform(x)

class ImageScaler(object):
    """
    Scaler that down samples 2D arrays of shape upscale to downscale and the inverse.
    Note that the methods always assume that the input array x is flattened along the last axis, i.e. it will reshape the input x.reshape(-1, *upscale). 
    The down sampled image is scaled once more with a scaler object.
    Attention, this object has no proper fit method, because of its application in FluxTrainerCVAE and the way the data is loaded there to avoid memory issues.
    """
    def __init__(self, 
                 downscale: Int[Array, "shape=(2,)"],
                 upscale: Int[Array, "shape=(2,)"],
                 scaler: object):
        self.downscale = downscale
        self.upscale = upscale
        self.scaler = scaler
    
    def resize_image(self, x: Array) -> Array:
        x = x.reshape(-1, *self.upscale)
        return jax.image.resize(x, shape = (x.shape[0], *self.downscale), method = "cubic")
    
    def transform(self, x: Array)-> Array:
        x = x.reshape(-1, *self.upscale)
        x = jax.image.resize(x, shape = (x.shape[0], *self.downscale), method = "cubic")
        x = x.reshape(-1, jnp.prod(self.downscale))
        x = self.scaler.transform(x)
        return x

    def inverse_transform(self, x: Array)-> Array:
        x = self.scaler.inverse_transform(x)
        x = x.reshape(-1, *self.downscale)
        x = jax.image.resize(x, shape = (x.shape[0], *self.upscale), method = "cubic")
        out = jax.vmap(self.fix_edges)(x[:, :, 4:-4]) # this is necessary because jax.image.resize produces artefacts at the edges when upsampling
        return out
    
    def fit_transform_scaler(self, x: Array) -> Array:
        """Method that will fit the scaling object. Here, the array already has to be down sampled."""
        out = self.scaler.fit_transform(x)
        return out
        
    @staticmethod
    @jax.vmap
    def fix_edges(yp: Array):
        """Extrapolate at early and late times from the reconstructed array to avoid artefacts at the edges from jax.image.resize."""
        xp = jnp.arange(4, yp.shape[0]+4)
        xl = jnp.arange(0,4)
        xr = jnp.arange(yp.shape[0]+4, yp.shape[0]+8)
        yl = jnp.interp(xl, xp, yp, left = "extrapolate", right = "extrapolate")
        yr = jnp.interp(xr, xp, yp, left = "extrapolate", right = "extrapolate")
        out = jnp.concatenate([yl, yp, yr])
        return out
    

# TODO: Remove this   
def inverse_svd_transform(x: Array, 
                          VA: Array, 
                          nsvd_coeff: int = 10) -> Array:

    # TODO: check the shapes etc, transforms and those things
    return jnp.dot(VA[:, :nsvd_coeff], x)


#######################
### BULLA UTILITIES ###
#######################

# TODO: place that somewhere else?

def get_filters_bulla_file(filename: str,
                           drop_times: bool = False) -> list[str]:
    
    assert filename.endswith(".dat"), "File should be of type .dat"
    
    # Open up the file and read the first line to get the header
    with open(filename, "r") as f:
        names = list(filter(None, f.readline().rstrip().strip("#").split(" ")))
    # Drop the times column if required, to get only the filters
    if drop_times:
        names = [name for name in names if name != "t[days]"]
    # Replace  colons with underscores
    names = [name.replace(":", "_") for name in names]
    
    return names

def get_times_bulla_file(filename: str) -> list[str]:
    
    assert filename.endswith(".dat"), "File should be of type .dat"
    
    names = get_filters_bulla_file(filename, drop_times=False)
    
    data = pd.read_csv(filename, 
                       delimiter=" ", 
                       comment="#", 
                       header=None, 
                       names=names, 
                       index_col=False)
    
    times = data["t[days]"].to_numpy()

    return times

def read_single_bulla_file(filename: str) -> dict:
    """
    Load lightcurves from Bulla type .dat files

    Args:
        filename (str): Name of the file

    Returns:
        dict: Dictionary containing the light curve data
    """
    
    # Extract the name of the file, without extensions or directories
    name = filename.split("/")[-1].replace(".dat", "")
    with open(filename, "r") as f:
        names = get_filters_bulla_file(filename)
    
    df = pd.read_csv(
        filename,
        delimiter=" ",
        comment="#",
        header=None,
        names=names,
        index_col=False,
    )
    df.rename(columns={"t[days]": "t"}, inplace=True)

    lc_data = df.to_dict(orient="series")
    lc_data = {
        k.replace(":", "_"): v.to_numpy() for k, v in lc_data.items()
    }
    
    return lc_data

#########################
### GENERAL UTILITIES ###
#########################

def interpolate_nans(data: dict[str, Float[Array, " n_files n_times"]],
                     times: Array, 
                     output_times: Array = None) -> dict[str, Float[Array, " n_files n_times"]]:
    """
    Interpolate NaNs and infs in the raw light curve data. 

    Args:
        data (dict[str, Float[Array, 'n_files n_times']]): The raw light curve data
        diagnose (bool): If True, print out the number of NaNs and infs in the data etc to inform about quality of the grid.

    Returns:
        dict[str, Float[Array, 'n_files n_times']]: Raw light curve data but with NaNs and infs interpolated
    """
    
    if output_times is None:
        output_times = times
    
    # TODO: improve this function overall!
    copy_data = copy.deepcopy(data)
    output = {}
    
    for filt, lc_array in copy_data.items():
        
        n_files = np.shape(lc_array)[0]
        
        if filt == "t":
            continue
        
        for i in range(n_files):
            lc = lc_array[i]
            # Get NaN or inf indices
            nan_idx = np.isnan(lc)
            inf_idx = np.isinf(lc)
            bad_idx = nan_idx | inf_idx
            good_idx = ~bad_idx
            
            # Interpolate through good values on given time grid
            if len(good_idx) > 1:
                # Make interpolation routine at the good idx
                good_times = times[good_idx]
                good_mags = lc[good_idx]
                interpolator = interp.interp1d(good_times, good_mags, fill_value="extrapolate")
                # Apply it to all times to interpolate
                mag_interp = interpolator(output_times)
                
            else:
                raise ValueError("No good values to interpolate from")
            
            if filt in output:
                output[filt] = np.vstack((output[filt], mag_interp))
            else:
                output[filt] = np.array(mag_interp)

    return output

def truncated_gaussian(mag_det: Array, 
                       mag_err: Array, 
                       mag_est: Array, 
                       lim: Float = jnp.inf):
    
    """
    Evaluate log PDF of a truncated Gaussian with loc at mag_est and scale mag_err, truncated at lim above.

    Returns:
        _type_: _description_
    """
    
    loc, scale = mag_est, mag_err
    a_trunc = -999 # TODO: OK if we just fix this to a large number, to avoid infs?
    a, b = (a_trunc - loc) / scale, (lim - loc) / scale
    logpdf = truncnorm.logpdf(mag_det, a, b, loc=loc, scale=scale)
    return logpdf

def load_event_data(filename):
    """
    Takes a file and outputs a magnitude dict with filters as keys.
    
    Args:
        filename (str): path to file to be read in
    
    Returns:
        data (dict[str, Array]): Data dictionary with filters as keys. The array has the structure [[mjd, mag, err]].

    """
    mjd, filters, mags, mag_errors = [], [], [], []

    with open(filename, "r") as input:

        for line in input:
            line = line.rstrip("\n")
            t, filter, mag, mag_err = line.split(" ")

            mjd.append(Time(t, format="isot").mjd) # convert to mjd
            filters.append(filter)
            mags.append(float(mag))
            mag_errors.append(float(mag_err))
    
    mjd = np.array(mjd)
    filters = np.array(filters)
    mags = np.array(mags)
    mag_errors = np.array(mag_errors)
    data = {}

    unique_filters = np.unique(filters)
    for filt in unique_filters:
        filt_inds = np.where(filters==filt)[0]
        data[filt] = np.array([ mjd[filt_inds], mags[filt_inds], mag_errors[filt_inds] ]).T

    return data

def write_event_data(filename: str, data: dict):
    """
    Takes a magnitude dict and writes it to filename. 
    The magnitude dict should have filters as keys, the arrays should have the structure [[mjd, mag, err]].
    """
    with open(filename, "w") as out:
        for filt in data.keys():
            for data_point in data[filt]:
                time = Time(data_point[0], format = "mjd")
                filt_name = filt.replace("_", ":")
                line = f"{time.isot} {filt_name} {data_point[1]:f} {data_point[2]:f}"
                out.write(line +"\n")

#########################
### Filters           ###
#########################


class Filter:

    def __init__(self,
                 name: str,):
        """
        Filter class that uses the bandpass properties from sncosmo or just a simple monochromatic filter based on the name.
        The necessary attributes are stored as jnp arrays.

        Args: 
            name (str): Name of the filter. Will be either passed to sncosmo to get the optical bandpass, or the unit at the end will be used to create a monochromatic filter. Supported units are keV and GHz.
        """
        self.name = name
        if (self.name, None) in _BANDPASSES._primary_loaders:
            bandpass = get_bandpass(self.name) # sncosmo bandpass
            self.nu = constants.c / (bandpass.wave_eff*1e-10)
            self.nus = constants.c / (bandpass.wave[::-1]*1e-10)
            self.trans = bandpass.trans[::-1] # reverse the array to get the transmission as function of frequency (not wavelength)
            
        elif (self.name, None) in _BANDPASS_INTERPOLATORS._primary_loaders:
            bandpass = get_bandpass(self.name, 0) # these bandpass interpolators require a radius (here by default 0 cm)
            self.nu = constants.c/(bandpass.wave_eff*1e-10)
            self.nus = constants.c / (bandpass.wave[::-1]*1e-10)
            self.trans = bandpass.trans[::-1] # reverse the array to get the transmission as function of frequency (not wavelength)

        elif self.name.endswith("GHz"):
            freq = re.findall(r"[-+]?(?:\d*\.*\d+)", self.name.replace("-",""))
            freq = float(freq[-1])
            self.nu = freq*1e9
            self.nus = jnp.array([self.nu])
            self.trans = jnp.ones(1)

        elif self.name.endswith("keV"):
            energy = re.findall(r"[-+]?(?:\d*\.*\d+)", self.name.replace("-",""))
            energy = float(energy[-1])
            self.nu = energy*1000*constants.eV / constants.h
            self.nus = jnp.array([self.nu])
            self.trans = jnp.ones(1)
        else:
            print(f"Warning: Filter {self.name} not recognized")
            self.nu = jnp.nan
            
        self.wavelength = constants.c/self.nu
        self._calculate_ref_flux()

        if len(self.nus)>1:
            self.get_mag = lambda Fnu, nus: bandpass_AB_mag(Fnu, nus, self.nus, self.trans, self.ref_flux)
        else:
            self.get_mag = lambda Fnu, nus: monochromatic_AB_mag(Fnu, nus, self.nus, self.trans, self.ref_flux)

    
    def _calculate_ref_flux(self,):
        """method to determine the reference flux for the magnitude conversion."""
        if self.trans.shape[0] == 1:
            self.ref_flux = 3631000. # mJy
        else:
            integrand = self.trans / (constants.h_erg_s * self.nus) # https://en.wikipedia.org/wiki/AB_magnitude
            integral = jnp.trapezoid(y = integrand, x = self.nus)
            self.ref_flux = 3631000. * integral.item() # mJy
    
    def get_mags(self, fluxes: Float[Array, "n_samples n_nus n_times"], nus: Float[Array, "n_nus"]) -> Float[Array, "n_samples n_times"]:

        def get_single(flux):
            return self.get_mag(flux, nus)
        
        mags = jax.vmap(get_single)(fluxes)
        return mags