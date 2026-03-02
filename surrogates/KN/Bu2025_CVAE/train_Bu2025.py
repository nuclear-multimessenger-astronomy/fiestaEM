import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import CVAETrainer
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

image_size= np.array([64, 40])

name = "Bu2025_CVAE"
outdir = f"./model/"
file = "../_training_data/Bu2025_raw_data.h5"

config = NeuralnetConfig(output_size= int(np.prod(image_size)),
                         nb_epochs=250_000,
                         hidden_layer_sizes = [600, 400, 200],
                         learning_rate =5e-5)


###############
### TRAINER ###
###############


data_manager_args = dict(file = file,
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


trainer.fit(config=config)
trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")

FILTERS = ["2massj", "ps1::y", "besselli", "bessellv", "bessellux"]
lc_model = FluxModel(name,
                     directory=outdir, 
                     filters = FILTERS)

trainer.plot_example_lc(lc_model)


for metric_name in ["L2", "Linf"]:
    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "../_training_data/Bu2025_raw_data.h5",
                    metric_name = metric_name
                    )
    benchmarker.benchmark()
    benchmarker.plot_lightcurves_mismatch()
