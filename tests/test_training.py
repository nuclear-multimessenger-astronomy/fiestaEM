import os
from pathlib import Path
import shutil

import numpy as np

from fiesta.train.FluxTrainer import PCATrainer, CVAETrainer
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

fiesta_dir = Path(__file__).parent.parent.absolute()
file = os.path.join(fiesta_dir, "examples/training/data/afterglowpy_tophat_reduced_set.h5")


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

def test_train_MLP():
    name = "test_MLP"
    outdir = f"./model/"

    config = NeuralnetConfig(output_size=20,
                             nb_epochs=100,
                             hidden_layer_sizes = [32, 32],
                             learning_rate =2e-4)
    

    trainer = PCATrainer(name,
                         outdir,
                         data_manager_args = data_manager_args,
                         n_pca=20,
                         save_preprocessed_data=False
                         )
    
    trainer.fit(config=config)
    trainer.save()

    shutil.rmtree("./model")




def test_train_CVAE():
    name = "test_CVAE"
    outdir = f"./model/"
    image_size = np.array([42, 57])

    config = NeuralnetConfig(output_size= int(np.prod(image_size)),
                             nb_epochs=100,
                             hidden_layer_sizes = [200, 100],
                             learning_rate =2e-4)

    trainer = CVAETrainer(name,
                          outdir,
                          data_manager_args = data_manager_args,
                          image_size=image_size,
                          save_preprocessed_data=False
                          )

    trainer.fit(config=config)
    trainer.save()

    shutil.rmtree("./model")