import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import PCATrainer
from fiesta.inference.lightcurve_model import AfterglowFlux
from fiesta.train.neuralnets import NeuralnetConfig

#############
### SETUP ###
#############

tmin = 1e-4 # days
tmax = 2000


numin = 1e9 # Hz 
numax = 5e18

n_training = 91670
n_val = 7676
n_pca = 50

name = "pbag_gaussian"
outdir = f"./model/"
file = "../data/pyblastafterglow_gaussian_raw_data.h5"

config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=200_000,
                         hidden_layer_sizes = [256, 512, 256],
                         learning_rate =5e-3)

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
                           special_training=["01"])

trainer = PCATrainer(name,
                     outdir,
                     data_manager_args = data_manager_args,
                     plots_dir=f"./benchmarks/",
                     n_pca = n_pca,
                     conversion="thetaWing_inclination",
                     save_preprocessed_data=True
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

lc_model = AfterglowFlux(name,
                          outdir, 
                          filters = FILTERS)

trainer.plot_example_lc(lc_model)
