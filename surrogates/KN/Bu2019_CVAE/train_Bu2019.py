import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import CVAETrainer
from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.utils import Filter

#############
### SETUP ###
#############

tmin = 1 # days
tmax = 20


numin = 1e13 # Hz 
numax = 6e15

n_training = 1276
n_val = 160
image_size = np.array([64, 99])

name = "Bu2019"
outdir = f"./model/"
file = "./model/Bu2019_raw_data.h5"

config = NeuralnetConfig(output_size= int(np.prod(image_size)),
                         nb_epochs=100_000,
                         hidden_layer_sizes = [1000, 500, 250],
                         learning_rate =1e-4)


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

trainer = CVAETrainer(name,
                     outdir,
                     data_manager_args = data_manager_args,
                     plots_dir=f"./benchmarks/",
                     image_size=image_size,
                     save_preprocessed_data=False
                     )

###############
### FITTING ###
###############


#trainer.fit(config=config)
#trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")
FILTERS = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y", "2massj", "2massh", "2massks", "sdssu"]

lc_model = AfterglowFlux(name,
                         outdir, 
                         filters = FILTERS,
                         model_type = "CVAE")

for filt in lc_model.Filters:
    with h5py.File(file, "r") as f:
        X_example = f["val"]["X"][-2]
        y_raw = f["val"]["y"][-2, trainer.data_manager.mask]

    y_raw = y_raw.reshape(len(lc_model.nus), len(lc_model.times))
    y_raw = np.exp(y_raw)
    y_raw = np.array([np.interp(filt.nu, lc_model.metadata["nus"], column) for column in y_raw.T]) 
    y_raw = -48.6 + -1 * np.log10(y_raw*1e-3 / 1e23) * 2.5
    
    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt.name]
    
    plt.plot(lc_model.times, y_raw, color = "red", label="POSSIS")
    plt.plot(lc_model.times, y_predict, color = "blue", label="Surrogate prediction")
    upper_bound = y_predict + 1
    lower_bound = y_predict - 1
    plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)

    plt.xlabel(f"$t$ in days")
    plt.ylabel(f"mag for {filt.name}")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.xlim(lc_model.times[0], lc_model.times[-1])

    plt.savefig(f"./benchmarks/afterglowpy_{name}_{filt.name}_example.png")
    plt.close()