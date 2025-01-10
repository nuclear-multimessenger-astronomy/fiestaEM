from re import L
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import h5py

from fiesta.train.FluxTrainer import PCATrainer, DataManager
import fiesta.train.neuralnets as fiesta_nn
from fiesta.inference.lightcurve_model import AfterglowpyPCA
from fiesta.train.neuralnets import NeuralnetConfig, CNN
from fiesta.utils import Filter


#############
### SETUP ###
#############

tmin = 1 # days
tmax = 2000

numin = 1e9 # Hz 
numax = 2.5e18

n_training = 30_000
n_val = 5000
n_pca = 100

name = "gaussian"
outdir = f"./model/"
file = outdir + "afterglowpy_raw_data.h5"


config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=1000,
                         hidden_layer_sizes = [256, 512, 256],
                         learning_rate =8e-3)


############

import jax
import jax.numpy as jnp
import h5py
from fiesta.utils import StandardScalerJax
import os
import pickle

import tqdm
import scipy.interpolate as interpolate

key = jax.random.key(24)
model = CNN(dense_layer_sizes=[64, 1], conv_layer_sizes= [ 1, 1, 1], spatial = 64, kernel_sizes= [3,3,1], output_shape=(64, 50))

key, subkey = jax.random.split(key)
state = fiesta_nn.create_train_state(model, jnp.ones(8), subkey, config)

with h5py.File("../afterglowpy_gaussian/model/afterglowpy_raw_data.h5") as f:
    
    nus = f["nus"][:]
    times = f["times"][:]

    train_X_raw  = f["train"]["X"][:]
    train_y_raw = []
    for chunk in (f['train']['y'].iter_chunks()):
        data = f['train']['y'][chunk].reshape(-1, 256, 200)
        train_y_raw.append(jax.image.resize(data, shape = (data.shape[0], 64, 50), method = "bilinear"))
    train_y_raw = jnp.array(train_y_raw, dtype = jnp.float16).reshape(-1, 64, 50)
    
    val_X_raw = f["val"]["X"][:5000]
    val_y_raw = jax.image.resize(f["val"]["y"][:5000].reshape(-1, 256, 200), shape = (5000, 64, 50), method = "bilinear")


Xscaler = StandardScalerJax()
yscaler = StandardScalerJax()

train_X = Xscaler.fit_transform(train_X_raw)
val_X = Xscaler.transform(val_X_raw)

train_y = yscaler.fit_transform(train_y_raw)
val_y = yscaler.transform(val_y_raw)


state, train_losses, val_losses = fiesta_nn.train_loop(state, config, train_X, train_y, val_X, val_y, verbose=True)
plt.figure(figsize=(10, 5))
ls = "-o"
ms = 3
plt.plot([i+1 for i in range(len(train_losses))], train_losses, ls, markersize=ms, label="Train", color="red")
plt.plot([i+1 for i in range(len(val_losses))], val_losses, ls, markersize=ms, label="Validation", color="blue")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.yscale('log')
plt.title("Learning curves")
plt.savefig(os.path.join("benchmarks", f"learning_curves_{name}.png"))
plt.close()   


breakpoint()
#### TEST #### 
from fiesta.utils import Filter
filt = Filter("bessellv")

predict = state.apply_fn({'params': state.params}, val_X[3]).T
predict = jax.image.resize(yscaler.inverse_transform(predict[0]), shape = (256, 200), method = "bilinear")

logflux_predict = interpolate.interp1d(nus, predict, axis = 0)(filt.nu)
logflux_val = interpolate.interp1d(nus, jax.image.resize(val_y_raw[3], shape = (256, 200), method = "bilinear") , axis = 0)(filt.nu)

plt.plot(times, logflux_predict, color = "blue")
plt.plot(times, logflux_val, color = 'red')
plt.xlabel("$t$ in days")
plt.ylabel("log(flux/mJys)")
plt.xscale("log")
plt.savefig(f"benchmarks/example_{filt.name}.png")





meta_filename = os.path.join(outdir, f"{name}_metadata.pkl")
        
save = {}
save["times"] = times
save["nus"] = nus
save["parameter_names"] = ["inclination_EM", 'log10_E0', 'thetaCore', 'alphaWing', 'log10_n0', 'p', 'log10_epsilon_e', 'log10_epsilon_B']
save["Xscaler"] = Xscaler
save["yscaler"] = yscaler
with open(meta_filename, "wb") as meta_file:
    pickle.dump(save, meta_file)

# Save the NN
model = state
fiesta_nn.save_model(model, config, out_name=outdir + f"/{name}.pkl")
exit()



###############
### TRAINER ###
###############


data_manager = DataManager(file = file,
                           n_training= n_training,
                           n_val= n_val, 
                           tmin= tmin,
                           tmax= tmax,
                           numin = numin,
                           numax = numax,
                           special_training=["02"])

trainer = PCATrainer(name,
                     outdir,
                     data_manager = data_manager,
                     plots_dir=f"./benchmarks/",
                     n_pca = n_pca,
                     save_preprocessed_data=False
                     )

###############
### FITTING ###
###############

trainer.fit(config=config)
trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")
FILTERS = ["radio-3GHz", "X-ray-1keV", "radio-6GHz", "bessellv"]

lc_model = AfterglowpyPCA(name,
                          outdir, 
                          filters = FILTERS)

for filt in lc_model.Filters:
    with h5py.File(file, "r") as f:
        X_example = f["val"]["X"][-1]
        y_raw = f["val"]["y"][-1, data_manager.mask]

    y_raw = y_raw.reshape(len(lc_model.nus), len(lc_model.times))
    y_raw = np.exp(y_raw)
    y_raw = np.array([np.interp(filt.nu, lc_model.metadata["nus"], column) for column in y_raw.T]) 
    y_raw = -48.6 + -1 * np.log10(y_raw*1e-3 / 1e23) * 2.5
    
    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt.name]
    
    plt.plot(lc_model.times, y_raw, color = "red", label="afterglowpy")
    plt.plot(lc_model.times, y_predict, color = "blue", label="Surrogate prediction")
    upper_bound = y_predict + 1
    lower_bound = y_predict - 1
    plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)

    plt.ylabel(f"mag for {filt.name}")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.xlim(lc_model.times[0], lc_model.times[-1])

    plt.savefig(f"./benchmarks/afterglowpy_{name}_{filt.name}_example.png")
    plt.close()