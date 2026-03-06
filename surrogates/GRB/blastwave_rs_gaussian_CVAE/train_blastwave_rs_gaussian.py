import numpy as np
import matplotlib.pyplot as plt
import h5py

from fiesta.train.FluxTrainer import CVAETrainer, DataManager
from fiesta.train.neuralnets import NeuralnetConfig

#############
### SETUP ###
#############

tmin = 1e-4 # days
tmax = 2000

numin = 1e9 # Hz
numax = 5e18

n_training = 18_000
n_val = 2_000
image_size = np.array([32, 42])

name = "blastwave_rs_gaussian"
outdir = "/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_rs_gaussian_CVAE/model/"
file = "/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/_training_data/blastwave_rs_gaussian_raw_data.h5"

config = NeuralnetConfig(output_size=int(np.prod(image_size)),
                         nb_epochs=100_000,
                         hidden_layer_sizes=[128, 256, 128],
                         latent_dim=32,
                         learning_rate=5e-5,
                         weight_decay=1e-4)


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

trainer = CVAETrainer(name,
                     outdir,
                     data_manager_args=data_manager_args,
                     plots_dir="/fred/oz480/mcoughli/fiestaEM_build/surrogates/GRB/blastwave_rs_gaussian_CVAE/benchmarks/",
                     image_size=image_size,
                     conversion="thetaCore_inclination",
                     save_preprocessed_data=False
                     )

###############
### FITTING ###
###############

trainer.fit(config=config)
trainer.save()

print("Training complete. Model saved to", outdir)
