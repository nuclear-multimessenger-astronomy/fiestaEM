import numpy as np 
import matplotlib.pyplot as plt
import h5py

from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS

from fiesta.train.LightcurveTrainer import SVDTrainer
from fiesta.inference.lightcurve_model import BullaLightcurveModel
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.filters import Filter


#############
### SETUP ###
#############

tmin = 0.2 # days
tmax = 26


numin = 1e13 # Hz 
numax = 1e16

n_training = 17_899 
n_val = 2237

svd_ncoeff = 50
FILTERS = list(map(lambda x: x[0], _BANDPASSES._primary_loaders))
FILTERS = list(set(FILTERS) - set(("f1000w", "f1130w", "f2550w", "f2300c", "f1140c", "f1280w", "f1550c", "f2100w", "f1800w", "f1500w", "f1065c")) )

name = "Bu2025"
outdir = f"./model/"
file = "../training_data/Bu2025_raw_data.h5"

config = NeuralnetConfig(output_size= svd_ncoeff,
                         nb_epochs=100_000,
                         hidden_layer_sizes = [64, 128, 64],
                         learning_rate =2e-3)


###############
### TRAINER ###
###############


data_manager_args = dict(file = file,
                           n_training= n_training, 
                           n_val= n_val, 
                           tmin= tmin,
                           tmax= tmax,
                           numin = numin,
                           numax = numax, 
                           )

trainer = SVDTrainer(name,
                     outdir,
                     data_manager_args = data_manager_args,
                     plots_dir=f"./benchmarks/",
                     svd_ncoeff= svd_ncoeff,
                     filters= FILTERS,
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

lc_model = BullaLightcurveModel(name,
                                filters=FILTERS, 
                                directory=outdir)

with h5py.File(file, "r") as f:
    X_example = f["val"]["X"][-1]
    y_raw = f["val"]["y"][-1, trainer.data_manager.mask]
    y_raw = y_raw.reshape(-1, len(lc_model.times))
    mJys = np.exp(y_raw)

    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    _, y_predict = lc_model.predict_abs_mag(X_example)

    
    for filt in lc_model.Filters:

        y_val = filt.get_mag(mJys, trainer.data_manager.nus)

        plt.plot(lc_model.times, y_val, color = "red", label="POSSIS")
        plt.plot(lc_model.times, y_predict[filt.name], color = "blue", label="Surrogate prediction")
        upper_bound = y_predict[filt.name] + 1
        lower_bound = y_predict[filt.name] - 1
        plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)
    
        plt.ylabel(f"mag for {filt.name}")
        plt.xlabel("$t$ in days")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.xscale('log')
        plt.xlim(lc_model.times[0], lc_model.times[-1])
    
        plt.savefig(f"./benchmarks/{name}_{filt.name}_example.png")
        plt.close()
