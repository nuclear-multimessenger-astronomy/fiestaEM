import numpy as np 
import matplotlib.pyplot as plt
import h5py

from fiesta.train import FluxSurrogateTrainer, DataLoader, MLP
from fiesta.models.surrogate_models import FluxSurrogate
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.train.Benchmarker import Benchmarker

#############
### SETUP ###
#############

tmin = 0.2 # days
tmax = 26


numin = 1e14 # Hz 
numax = 2e15

n_pca = 100

surrogate_name = "Bu2026_MLP"
outdir = f"./model/"
file = "../_training_data/Bu2026_raw_data.h5"

###############
### TRAINER ###
###############


data = DataLoader(
    file = file,
    tmin= tmin,
    tmax= tmax,
    numin = numin,
    numax = numax,
)

config = NeuralnetConfig(output_size=n_pca,
                         input_size=len(data.parameter_names),
                         nb_epochs=300_000,
                         hidden_layer_sizes = [256, 512, 256],
                         learning_rate =2e-3)

network = MLP(config=config)

trainer = FluxSurrogateTrainer(
    surrogate_name,
    data,
    outdir,
    network,
    plots_dir=f"./benchmarks/",
    save_preprocessed_data=False
)

###############
### FITTING ###
###############

trainer.fit()
trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")

FILTERS = ["ps1::y", "besselli", "bessellv", "bessellux"]
trainer.plot_example_lc(FILTERS)


####################
### BENCHMARKING ###
####################


lc_model = FluxSurrogate(surrogate_name, directory=outdir, filters=FILTERS)

benchmarker = Benchmarker(
    model=lc_model,
    data_file=file
)
benchmarker.benchmark()
benchmarker.plot_lightcurves_mismatch()
