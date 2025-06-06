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

n_training = 56_930
n_val = 8750
n_pca = 50

name = "pbag_tophat"
outdir = f"../../../src/fiesta/surrogates/GRB/pbag_tophat_MLP/model/"
file = "../training_data/pyblastafterglow_tophat_raw_data.h5"

config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=300_000,
                         hidden_layer_sizes = [300, 600, 300],
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
                           special_training=[])

trainer = PCATrainer(name,
                     outdir,
                     data_manager_args = data_manager_args,
                     plots_dir=f"./benchmarks/",
                     n_pca = n_pca,
                     conversion="thetaCore_inclination",
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

lc_model = AfterglowFlux(name,
                         directory="./model",
                         filters = FILTERS)

trainer.plot_example_lc(lc_model)
