import copy
from multiprocessing import Value
import os
import re
import tqdm

import numpy as np
import pandas as pd
import h5py
from astropy.time import Time
import astropy.units as u
import scipy.interpolate as interp

import jax
import jax.numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, Float, Int

from fiesta.conversions import Flambda_to_Fnu
from fiesta.constants import c, days_to_seconds


#######################
### BULLA UTILITIES ###
#######################

def read_parameters_POSSIS(filename):
    num_str = re.findall(r'\d+\.\d+', filename) 
    parlist = list(map(float, num_str)) # the first entry here is the number of photon packets
    return parlist[1:]

def read_POSSIS_file(filename):
    parameters = read_parameters_POSSIS(filename)
    with h5py.File(filename) as f:

        waves = f["observables"]["wave"][:]
        
        n_inclinations, _, _, _ = f["observables"]["stokes"].shape
        inclinations = np.arccos(np.linspace(0, 1, n_inclinations))

        intensity = f["observables"]["stokes"][:,:,:,0] 
        intensity = intensity / ((10*u.pc).to(u.Mpc).value)**2
        intensity = np.maximum(intensity, 1e-15)
        flux = intensity
        flux = np.transpose(flux, axes = [0,2,1])
    
    mJys, _ = jax.vmap(Flambda_to_Fnu, in_axes = (0, None), out_axes = (0, None))(flux, waves)
    y_file = np.log10(mJys).reshape(-1, 1000, 100)
    
    X_file = np.array([[*parameters, obs_angle] for obs_angle in inclinations])
    
    return X_file, y_file

def convert_POSSIS_outputs_to_h5(possis_dirs: list[str] | str,
                                 outfile: str,
                                 parameter_names: list[str] = ["log10_mej_dyn", "v_ej_dyn", "Ye_dyn", "log10_mej_wind", "v_ej_wind", "Ye_wind", "inclination_EM"],
                                 clip: float = 6.5144,
                                 log_arguments = [0, 3]):
    
    if isinstance(possis_dirs, str):
        possis_dirs = [possis_dirs]
    
    files = []
    for dir in possis_dirs:
        files.extend([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".hdf5")])
    
    with h5py.File(files[0]) as f:
        waves = f["observables"]["wave"][:]
        times = f["observables"]["time"][:] / days_to_seconds
        nus = c / (waves[::-1] * 1e-10)
        
        X_file, y_file = read_POSSIS_file(files[0])
        if X_file.shape[1] != len(parameter_names):
            raise ValueError(f"parameter_names do not match parameters stored in POSSIS file ({X.shape[1]} parameters in POSSIS files).")
        
        X_file[:,log_arguments] = np.log10(X_file[:,log_arguments]) # make mej_dyn and mej_wind to log10
        y_file = np.maximum(y_file, clip)

        # initialize training data file
        write_training_data(outfile, X_file[:4], y_file[:4], X_file[4:7], y_file[4:7], X_file[7:], y_file[7:], times, nus, parameter_names, {})
    

    for file in tqdm.tqdm(files[1:]):
        
        X_file, y_file = read_POSSIS_file(file)
        
        if X_file.shape[1] != len(parameter_names):
            raise ValueError(f"parameter_names do not match parameters stored in POSSIS file ({X.shape[1]} parameters in POSSIS files).")
        
        X_file[:,log_arguments] = np.log10(X_file[:,log_arguments]) # make mej_dyn and mej_wind to log10
        y_file = np.maximum(y_file, clip)

        train_X, val_X, train_y, val_y = train_test_split(X_file, y_file, train_size=0.8)
        val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, train_size=0.5)

        append_training_data_file(outfile, train_X, train_y, val_X, val_y, test_X, test_y)

    with h5py.File(outfile, "a") as f:
        train_X = f["train"]["X"][:]
        parameter_distributions = {p: (np.min(train_X[:,j]).item(), np.max(train_X[:,j]).item(), "uniform") for j, p in enumerate(parameter_names)}
        del f["parameter_distributions"]
        f["parameter_distributions"] = str(parameter_distributions)

    
#####################
# GWEMOPT UTILITIES #
#####################

def read_gwemopt_parameters(filename: str):
    num_str = re.findall(r'\d+\.?\d*E?-?\d*', filename)
    parlist = list(map(float, num_str)) # the last entry here is 1D and h5
    return parlist[:-2]

def read_gwemopt_file(filename: str):

    parameters = read_gwemopt_parameters(filename)
    with h5py.File(filename) as f:
        Lnu = f["Lnu"][::, ::2]
        Lnu = np.maximum(Lnu, 1e-15)
        Lnu /= 4*np.pi* ((10*u.pc).to(u.cm).value)**2 # to erg / (s Hz cm^2)
        Lnu *= 1e26 # to mJy
        
        y_file = np.log10(Lnu.T)
    
    X_file = np.array(parameters)
    
    return X_file, y_file


def convert_gwemopt_to_h5(dirs: list[str],
                          outfile: str,
                          parameter_names: list[str] = ["dens_slope", "log10_X_lan", "vkin", "log10_mej"],
                          clip: float = 6.5144,
                          log_arguments = [1, 3]):
    if isinstance(dirs, str):
        dirs = [dirs]
    
    files = []
    for dir in dirs:
        files.extend([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".h5")])
    
    with h5py.File(files[0]) as f:
        nus = f["nu"][::2]
        times = f["time"][:] / days_to_seconds

    X, y = [], []
    for file in files:
        
        X_file, y_file = read_gwemopt_file(file)
        X.append(X_file)
        y.append(y_file)
    
    X, y = np.array(X), np.array(y)

    if X.shape[1] != len(parameter_names):
        raise ValueError(f"parameter_names do not match parameters stored in POSSIS file ({X.shape[1]} parameters in POSSIS files).")
    
    y = np.maximum(y, clip)
    X[:,log_arguments] = np.log10(X[:,log_arguments]) # make mej_dyn and mej_wind to log10

    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8)
    val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, train_size=0.5)
    
    parameter_distributions = {p: (np.min(train_X[:,j]).item(), np.max(train_X[:,j]).item(), "uniform") for j, p in enumerate(parameter_names)}

    write_training_data(outfile, 
                        train_X,
                        train_y,
                        val_X,
                        val_y,
                        test_X,
                        test_y,
                        times,
                        nus,
                        parameter_names,
                        parameter_distributions)


###############################
### TRAINING DATA UTILITIES ###
###############################

def train_test_split(X, y, train_size: float | int):
    
    if isinstance(train_size, int):
        assert train_size > 0 and train_size <= X.shape[0], f"train_size needs to be smaller than X shape, it was {train_size:.2f}."
        train_size /= X.shape[0]
        
    elif isinstance(train_size, float):
        assert train_size > 0 and train_size < 1, f"train_size needs to be between 0 and 1, it was {train_size:.2f}."
    
    else:
        raise ValueError(f"train_size needs to be float or int")

    mask = np.random.choice(a=[True, False], size=X.shape[0], replace=True, p=[train_size, 1-train_size])
    return X[mask], X[~mask], y[mask], y[~mask]


def write_training_data(outfile: str,
                        train_X: Array,
                        train_y: Array,
                        val_X: Array,
                        val_y: Array,
                        test_X: Array,
                        test_y: Array,
                        times: Array,
                        nus: Array,
                        parameter_names: list[str],
                        parameter_distributions: str):
    
    with h5py.File(outfile, "w") as f:
        f.create_dataset("times", data = times)
        f.create_dataset("nus", data = nus)
        f.create_dataset("parameter_names", data = parameter_names)
        f.create_dataset("parameter_distributions", data = str(parameter_distributions))
        f.create_group("train"); f.create_group("val"); f.create_group("test"); f.create_group("special_train")
        f["train"].create_dataset("X", data = train_X, maxshape=(None, len(parameter_names)), chunks = (1, len(parameter_names)))
        f["train"].create_dataset("y", data = train_y, maxshape=(None, len(nus), len(times)), chunks = (1, len(nus), len(times)))
        f["val"].create_dataset("X", data = val_X, maxshape=(None, len(parameter_names)), chunks=(1, len(parameter_names)))
        f["val"].create_dataset("y", data = val_y, maxshape=(None, len(nus), len(times)), chunks = (1, len(nus), len(times)))
        f["test"].create_dataset("X", data= test_X, maxshape=(None, len(parameter_names)), chunks=(1, len(parameter_names)))
        f["test"].create_dataset("y", data = test_y, maxshape=(None, len(nus), len(times)), chunks = (1, len(nus), len(times)))
    

def append_training_data_file(outfile: str,
                              train_X: Array,
                              train_y: Array,
                              val_X: Array,
                              val_y: Array,
                              test_X: Array,
                              test_y: Array):

    with h5py.File(outfile, "a") as f:

        for Xnew, ynew, group in zip([train_X, val_X, test_X],[train_y, val_y, test_y], ["train", "val", "test"]):

            Xset = f[group]["X"]
            yset = f[group]["y"]
            
            if Xnew.shape[0] > 0:
                Xset.resize(Xset.shape[0]+Xnew.shape[0], axis = 0)
                Xset[-Xnew.shape[0]:] = Xnew
                
                yset.resize(yset.shape[0]+ynew.shape[0], axis=0)
                yset[-ynew.shape[0]:] = ynew




##########################
### I/O DATA UTILITIES ###
##########################

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

##############
### LEGACY ###
##############

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