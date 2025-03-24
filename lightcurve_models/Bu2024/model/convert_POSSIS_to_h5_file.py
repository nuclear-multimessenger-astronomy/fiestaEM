import os
import re
from sklearn.model_selection import train_test_split
import tqdm

import astropy.units as u
import h5py
from jax import vmap
import numpy as np
from sklearn.model_selection import train_test_split

from fiesta.conversions import Flambda_to_Fnu
from fiesta.constants import c


#################################################################################

dir = "/home/kingu/work/markin/possis-chemical/newheat/possis_all_newheat_unpacked"

parameter_names = ["log10_mej_dyn", "v_ej_dyn", "Ye_dyn", "log10_mej_wind", "v_ej_wind", "inclination_EM"]

#################################################################################

files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("30.hdf5")]

# define some fixed quantities
with h5py.File(files[0]) as f:
    waves = f["observables"]["wave"][:]
    times = f["observables"]["time"][:]
    nus = c / (waves[::-1] * 1e-10)
    inclinations = np.arccos(np.linspace(0,1,11)) * 180 / np.pi


def read_parameters(filename):
    num_str = re.findall(r'\d+\.\d+', filename)
    _, mej_dyn, v_ej_dyn, Ye_dyn, mej_wind, v_ej_wind, _ = list(map(float, num_str))
    return [mej_dyn, v_ej_dyn, Ye_dyn, mej_wind, v_ej_wind]


def read_file(filename):   
    parameters = read_parameters(filename)
    with h5py.File(filename) as f:
        # see Ivan's script
        intensity = f["observables"]["stokes"][:,:,:,0] 
        intensity = intensity / ((10*u.pc).to(u.Mpc).value)**2
        intensity = np.maximum(intensity, 1e-15)
        flux = intensity
        flux = np.transpose(flux, axes = [0,2,1])
    
    mJys, _ = vmap(Flambda_to_Fnu, in_axes = (0, None), out_axes = (0, None))(flux, waves)
    y_file = np.log(mJys).reshape(-1, 1000 *100)
    
    X_file = np.array([[*parameters, obs_angle] for obs_angle in inclinations])
    
    return X_file, y_file


def get_data_from_files(files_list):
     
    X, y = [], []
    for file in tqdm.tqdm(files_list):
        
        X_file, y_file = read_file(file)
        
        X.extend(X_file)
        y.extend(y_file)
    
    X, y = np.array(X), np.array(y)

    X[:,0] = np.log10(X[:,0]) # make mej_dyn to log
    X[:,3] = np.log10(X[:, 3]) # make mej_wind to log
    return X, y

X, y = get_data_from_files(files)

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.8)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, train_size = 0.5)

parameter_distributions = {p: (min(train_X[:,j]), max(train_X[:,j]), "uniform") for j, p in enumerate(parameter_names)}


with h5py.File("Bu2024_raw_data.h5", "w") as f:
    f.create_dataset("times", data = times)
    f.create_dataset("nus", data = nus)
    f.create_dataset("parameter_names", data = parameter_names)
    f.create_dataset("parameter_distributions", data = str(parameter_distributions))
    f.create_group("train"); f.create_group("val"); f.create_group("test"); f.create_group("special_train")
    f["train"].create_dataset("X", data = train_X, maxshape=(None, len(parameter_names)), chunks = (1000, len(parameter_names)))
    f["train"].create_dataset("y", data = train_y, maxshape=(None, len(times)*len(nus)), chunks = (1000, len(times)*len(nus)))
    f["val"].create_dataset("X", data = val_X)
    f["val"].create_dataset("y", data = val_y)
    f["test"].create_dataset("X", data= test_X)
    f["test"].create_dataset("y", data = test_y)