import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import PCATrainer
from fiesta.inference.lightcurve_model import FluxModel
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.train.Benchmarker import Benchmarker

#############
### SETUP ###
#############

tmin = 0.2 # days
tmax = 26


numin = 1e14 # Hz 
numax = 2e15

n_training = 13_692 
n_val = 1712

n_pca = 100

name = "Bu2026_MLP"
outdir = f"./model/"
file = "../_training_data/Bu2026_raw_data.h5"

config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=240_000,
                         hidden_layer_sizes = [256, 512, 256],
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

for metric_name in ["L2", "Linf"]:
    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "../_training_data/Bu2026_raw_data.h5",
                    metric_name = metric_name
                    )
    benchmarker.benchmark()
    benchmarker.plot_lightcurves_mismatch()
