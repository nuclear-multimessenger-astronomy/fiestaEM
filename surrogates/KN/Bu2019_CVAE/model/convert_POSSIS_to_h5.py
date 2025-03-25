import numpy as np
import h5py
from sklearn.model_selection import train_test_split

import re
import os
import tqdm


c = 299792458


#################################################################################
dir = "./2020_dietrich_Bu2019lm"

parameter_names = ["log10_mej_dyn", "log10_mej_wind", "KNphi", "inclination_EM"]

#################################################################################


####################
### PREPARATION ####
####################
# get the files
files = [f for f in os.listdir("./2020_dietrich_Bu2019lm") if ".txt" in f]

# set up times and nus
n_nus = int(np.loadtxt(os.path.join(dir, files[0]), skiprows = 1, max_rows = 1))
n_times, ti, tf = np.loadtxt(os.path.join(dir, files[0]), skiprows = 2, max_rows = 1).astype(int)

dt = (tf - ti)/n_times
times = np.arange(ti + 0.5 * dt, ti + (n_times + 0.5) * dt, dt)
a = np.genfromtxt(os.path.join(dir, files[0]), skip_header=3)
nus = c / (1e-10* a[:n_nus, 0]) # in Hz


# set up the parameter arrays
n_params = len(parameter_names)

X = []; y = []

###########################
### loop over the files ###
###########################

for f in tqdm.tqdm(files):

    with open(os.path.join(dir, f)) as handle:
        Nobs = int(np.loadtxt(os.path.join(dir, f),  max_rows = 1))
    a = np.genfromtxt(os.path.join(dir, f), skip_header = 3)

    obs_angles = np.arccos(np.linspace(0, 1, Nobs)) * 180 / np.pi
    for j, obs in enumerate(obs_angles):
        params = [float(p) for p in re.findall(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", f)[-3:]]
        params.append(obs)
        X.append(params)
        y.append(a[j * n_nus : (j+1) * n_nus, 1:])
    


####################################
### convert to the correct units ###
####################################

y = np.array(y)
X = np.array(X)

y = np.divide(y*c/1e-10, (nus**2)[:, None]) * 1e26  # convert to mJy (milli-Jansky)

nus = nus[::-1]
y = y[:, ::-1, :]

# summarize frequency bins
bin_edges = np.logspace(np.log10(nus[0]), np.log10(nus[-1]), 101)
bins = [np.where( (nus>=bin_edges[j]) & (nus <=bin_edges[j+1]) )[0] for j in range(len(bin_edges)-1)]
bins = [bin for bin in bins if len(bin)>0]
#import matplotlib.pyplot as plt
#for bin in bins:
#    plt.plot(times, np.log10(np.mean(y[0, bin, :], axis = 0)), label = f"$\\nu = {np.mean(nus[bin])}$ Hz")
#plt.xlabel("$t$ in days")
#plt.ylabel("flux in mJys")
#plt.legend()
#plt.show()

nus = np.array([np.mean(nus[bin]) for bin in bins])
y = np.array([[np.mean(y[j, bin, :], axis = 0) for bin in bins] for j in range(len(y))])
y = np.log10(y)

def remove_infs(yy):
    out = yy.copy()
    #min_val = out[~np.isinf(out)].min()
    out[out<=-15] = -15
    return out

y = np.array([remove_infs(yy) for yy in y])

import matplotlib.pyplot as plt
for yy in y[0]:
    plt.plot(times, yy)
plt.xlabel("$t$ in days")
plt.ylabel("log(flux/ mJys)")
#plt.legend()
plt.show()
breakpoint()

y = (y.reshape(y.shape[0], -1))
X[:,0] = np.log10(X[:, 0]) # to log10
X[:,1] = np.log10(X[:, 1]) # to log10

parameter_distributions = {p: (min(X[:,j]), max(X[:,j]), "uniform") for j, p in enumerate(parameter_names)}


train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.8)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, train_size = 0.5)


######################
### create h5 file ###
######################

with h5py.File("Bu2019_raw_data.h5", "w") as f:
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


        



