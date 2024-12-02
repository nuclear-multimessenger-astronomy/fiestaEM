import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import h5py

from fiesta.train.FluxTrainer import PCATrainer, DataManager
from fiesta.inference.lightcurve_model import AfterglowpyPCA
from fiesta.train.neuralnets import NeuralnetConfig
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

name = "tophat"
outdir = f"./model/"
file = outdir + "afterglowpy_raw_data.h5"


config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=100_000,
                         hidden_layer_sizes = [256, 512, 256],
                         learning_rate =8e-3)


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

    y_raw = y_raw.reshape(256, len(lc_model.times))
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