import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import PCATrainer
from fiesta.inference.lightcurve_model import FluxModel
from fiesta.train.neuralnets import NeuralnetConfig

#############
### SETUP ###
#############

tmin = 0.3 # days
tmax = 16


numin = 1e14 # Hz 
numax = 2e15

n_training = 17_899 
n_val = 2237

n_pca = 100

name = "Bu2025_MLP"
outdir = f"./model/"
file = "../training_data/Bu2025_raw_data.h5"

config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=240_000,
                         hidden_layer_sizes = [256, 512, 256],
                         learning_rate =1e-3)


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

trainer = PCATrainer(name,
                     outdir,
                     data_manager_args = data_manager_args,
                     plots_dir=f"./benchmarks/",
                     n_pca=n_pca,
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

FILTERS = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y", "2massj", "2massh", "2massks", "sdssu"]
lc_model = FluxModel(name,
                     directory=outdir, 
                     filters=FILTERS)

trainer.plot_example_lc(lc_model)

