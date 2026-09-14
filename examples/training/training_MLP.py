import numpy as np
import matplotlib.pyplot as plt
import h5py

from fiesta.train import FluxSurrogateTrainer, DataLoader, NeuralnetConfig, MLP

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


###############
### TRAINER ###
###############


data = DataLoader(
    file=file,
    n_training=n_training,
    n_val=n_val,
    tmin=tmin,
    tmax=tmax,
    numin=numin,
    numax=numax,
    special_training=["special_1"],
)

config = NeuralnetConfig(
    output_size=n_pca,
    input_size=len(data.parameter_names),
    nb_epochs=100_000,
    hidden_layer_sizes=[32, 32],
    learning_rate =2e-4
)

network = MLP(config=config)

trainer = FluxSurrogateTrainer(
    name,
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

trainer.plot_example_lc(["ps1::y", "besselli", "bessellv", "bessellux"])