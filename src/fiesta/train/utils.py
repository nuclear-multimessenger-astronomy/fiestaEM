import os
import re

import h5py
import numpy as np
import jax
from jaxtyping import Array, Float, Int
import tqdm
import astropy.units as u

from fiesta.conversions import Flambda_to_Fnu
from fiesta.constants import days_to_seconds, c

###############################
### TRAINING DATA UTILITIES ###
###############################

def train_test_split(X: Array, y: Array, train_size: float | int) -> tuple[Array, Array, Array, Array]:
    """
    Split arrays into training and test sets.

    Args:
        X (Array): Input features, with samples along the first axis.
        y (Array): Target values corresponding to the samples in ``X``.
        train_size (float | int): Number or fraction of samples to include
            in the training set. If a float, must be between 0 and 1.
            If an int, specifies the exact number of training samples.

    Returns:
        tuple[Array, Array, Array, Array]:
            A tuple containing ``X_train, X_test, y_train, y_test``.
            The training arrays contain ``train_size`` samples, while the
            test arrays contain the remaining samples.

    Raises:
        ValueError: If ``X`` and ``y`` have different numbers of samples,
            or if ``train_size`` is invalid.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    n_samples = X.shape[0]

    if isinstance(train_size, int):
        if not 0 < train_size < n_samples:
            raise ValueError(
                f"train_size must be between 1 and {n_samples - 1}, "
                f"got {train_size}."
            )
        n_train = train_size

    elif isinstance(train_size, float):
        if not 0 < train_size < 1:
            raise ValueError(
                f"train_size must be between 0 and 1, got {train_size:.2f}."
            )
        n_train = int(train_size * n_samples)

    else:
        raise TypeError("train_size must be a float or int.")

    indices = np.random.permutation(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def write_training_data(
        outfile: str,
        train_X: Array,
        train_y: Array,
        val_X: Array,
        val_y: Array,
        test_X: Array,
        test_y: Array,
        times: Array,
        nus: Array,
        parameter_names: list[str],
        parameter_distributions: str
    ):
    
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

#######################
### BULLA UTILITIES ###
#######################

def convert_POSSIS_outputs_to_h5(
        dirs: str | list[str],
        outfile: str,
        parameter_names: list[str],
        log_arguments: list[int],
        train_size: float = 0.8,
        clip: float = 6.5144,
    ) -> None:
    """
    Merges a directory (or several directories) full of POSSIS ``.h5`` outputs to a single training data file in the fiesta format.

    Args:
        dirs (str | list[str]): directory or list of directories with the outputs that should be merged into a single training data file.
        outfile (str): Name of the ``.h5`` training data file to create.
        parameter_names (list[str]): Parameter names in the order they appear in the file names. 
                                     These will be the parameters of the trained surrogate in the end.
                                     Note that function expects the possis file to contain different inclinations.
        log_arguments (list[int]): Parameters, which are not log10 when read from the filenames, but should be converted to log10 for the training.
        train_size (float): Relative proportion of the training data. Defaults to 0.8.
        clip (float): Lower floor value for the minimum log10(mJy) at 10 pc. Every flux density below that will be set to that value. Defaults to 6.5144 (appr. 0 abs. mag).
    """
    
    if isinstance(dirs, str):
        possis_dirs = [dirs]
    
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

        train_X, val_X, train_y, val_y = train_test_split(X_file, y_file, train_size=train_size)
        val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, train_size=0.5)

        append_training_data_file(outfile, train_X, train_y, val_X, val_y, test_X, test_y)

    with h5py.File(outfile, "a") as f:
        train_X = f["train"]["X"][:]
        parameter_distributions = {p: (np.min(train_X[:,j]).item(), np.max(train_X[:,j]).item(), "uniform") for j, p in enumerate(parameter_names)}
        del f["parameter_distributions"]
        f["parameter_distributions"] = str(parameter_distributions)

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


####################
# SEDONA UTILITIES #
####################

def read_SEDONA_parameters(filename: str):
    num_str = re.findall(r'\d+\.?\d*E?-?\d*', filename)
    parlist = list(map(float, num_str))
    parlist = parlist[2:-2] # The first two numbers are the year and a density slope we ignore
                            # The last two numbers are 1D and h5
    return parlist

def read_SEDONA_file(filename: str):

    parameters = read_SEDONA_parameters(filename)
    with h5py.File(filename) as f:
        Lnu = f["Lnu"][::, ::2]
        Lnu = np.maximum(Lnu, 1e-15)
        Lnu /= 4*np.pi* ((10*u.pc).to(u.cm).value)**2 # to erg / (s Hz cm^2)
        Lnu *= 1e26 # to mJy
        
        y_file = np.log10(Lnu.T)
    
    X_file = np.array(parameters)
    
    return X_file, y_file


def convert_SEDONA_outputs_to_h5(
        dirs: list[str],
        outfile: str,
        parameter_names: list[str],
        log_arguments: list[int],
        train_size: float = 0.8,
        clip: float = 6.5144,
    ) -> None:

    """
    Merges a directory (or several directories) full of SEDONA ``.h5`` outputs to a single training data file in the fiesta format.

    Args:
        dirs (str | list[str]): directory or list of directories with the outputs that should be merged into a single training data file.
        outfile (str): Name of the ``.h5`` training data file to create.
        parameter_names (list[str]): Parameter names in the order they appear in the file names. 
                                     These will be the parameters of the trained surrogate in the end.
        log_arguments (list[int]): Parameters, which are not log10 when read from the filenames, but should be converted to log10 for the training.
        train_size (float): Relative proportion of the training data. Defaults to 0.8.
        clip (float): Lower floor value for the minimum log10(mJy) at 10 pc. Every flux density below that will be set to that value. Defaults to 6.5144 (appr. 0 abs. mag).
    """
    
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
        
        X_file, y_file = read_SEDONA_file(file)
        X.append(X_file)
        y.append(y_file)
    
    X, y = np.array(X), np.array(y)

    if X.shape[1] != len(parameter_names):
        raise ValueError(f"parameter_names do not match parameters stored in POSSIS file ({X.shape[1]} parameters in POSSIS files).")
    
    y = np.maximum(y, clip)
    X[:,log_arguments] = np.log10(X[:,log_arguments]) # make mej_dyn and mej_wind to log10

    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=train_size)
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