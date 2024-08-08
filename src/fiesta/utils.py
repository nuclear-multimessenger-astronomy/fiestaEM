import jax
import jax.numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, Float
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import copy
import re
from sncosmo.bandpasses import _BANDPASSES
from astropy.time import Time

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

# @jax.jit
def mag_app_from_mag_abs(mag_abs: Array,
                         luminosity_distance: Float) -> Array:
    return mag_abs + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)


#######################
### BULLA UTILITIES ###
#######################

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
                     output_times: Array) -> dict[str, Float[Array, " n_files n_times"]]:
    """
    Interpolate NaNs and infs in the raw light curve data. 

    Args:
        data (dict[str, Float[Array, 'n_files n_times']]): The raw light curve data
        diagnose (bool): If True, print out the number of NaNs and infs in the data etc to inform about quality of the grid.

    Returns:
        dict[str, Float[Array, 'n_files n_times']]: Raw light curve data but with NaNs and infs interpolated
    """
    
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
    # TODO: polish?
    lines = [line.rstrip("\n") for line in open(filename)]
    lines = filter(None, lines)

    sncosmo_filts = [val["name"] for val in _BANDPASSES.get_loaders_metadata()]
    sncosmo_maps = {name: name.replace(":", "_") for name in sncosmo_filts}

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        lineSplit = list(filter(None, lineSplit))
        mjd = Time(lineSplit[0], format="isot").mjd
        filt = lineSplit[1]

        if filt in sncosmo_maps:
            filt = sncosmo_maps[filt]

        mag = float(lineSplit[2])
        dmag = float(lineSplit[3])

        if filt not in data:
            data[filt] = np.empty((0, 3), float)
        data[filt] = np.append(data[filt], np.array([[mjd, mag, dmag]]), axis=0)

    return data

