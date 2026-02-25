import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import PCATrainer
from fiesta.train.neuralnets import NeuralnetConfig

#############
### SETUP ###
#############

tmin = 1e-4 # days
tmax = 2000


numin = 1e9 # Hz
numax = 5e18

n_training = 20_000
n_val = 2_000
n_pca = 100

name = "jetsimpy_gaussian"
outdir = "/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/jetsimpy_gaussian_MLP/model/"
file = "/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/jetsimpy_gaussian_raw_data.h5"

config = NeuralnetConfig(output_size=n_pca,
                         nb_epochs=200_000,
                         hidden_layer_sizes=[128, 256, 128],
                         learning_rate=5e-4)

###############
### TRAINER ###
###############


data_manager_args = dict(file=file,
                         n_training=n_training,
                         n_val=n_val,
                         tmin=tmin,
                         tmax=tmax,
                         numin=numin,
                         numax=numax)

trainer = PCATrainer(name,
                     outdir,
                     data_manager_args=data_manager_args,
                     plots_dir="/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/jetsimpy_gaussian_MLP/benchmarks/",
                     n_pca=n_pca,
                     conversion="thetaCore_inclination",
                     save_preprocessed_data=False
                     )

###############
### FITTING ###
###############

trainer.fit(config=config)
trainer.save()

print("Training complete. Model saved to", outdir)
