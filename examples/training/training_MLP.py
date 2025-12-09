import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import PCATrainer
from fiesta.inference.lightcurve_model import FluxModel
from fiesta.train.neuralnets import NeuralnetConfig

#############
### SETUP ###
#############

tmin = 1e-4 # days
tmax = 1e3


numin = 1e9 # Hz 
numax = 1e18

n_training = 200
n_val = 10

n_pca = 20

name = "test_MLP"
outdir = f"./model/"
file = "./data/afterglowpy_tophat_reduced_set.h5"

config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=100_000,
                         hidden_layer_sizes = [32, 32],
                         learning_rate =2e-4)


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
                         special_training=["special_1"],
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

FILTERS = ["ps1::y", "besselli", "bessellv", "bessellux"]
lc_model = FluxModel(name,
                     directory=outdir, 
                     filters=FILTERS)

trainer.plot_example_lc(lc_model)