import os
import numpy as np
import matplotlib.pyplot as plt

from fiesta.train.SurrogateTrainer import BullaSurrogateTrainer
from fiesta.inference.lightcurve_model import BullaLightcurveModel

from fiesta.train.neuralnets import NeuralnetConfig

print("Checking whether we found a GPU:")
import jax
print(jax.devices())

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# All filters that are in the files for this model:
FILTERS = ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y", "2massj", "2massh", "2massks", "sdssu"]

# TODO: need to find a way to locate the files/help users
lc_dir = "/home/urash/twouters/projects/fiesta_dev/fiesta_test/lightcurves/Bu2019lm/lcs/"
name = "Bu2019lm"
outdir = f"./{name}/"
plots_dir = "./figures/" # to make plots

if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

lc_files = os.listdir(lc_dir)
example_file = os.path.join(lc_dir, lc_files[0])    

print("example_filename")
print(example_file)

###############
### TRAINER ###
###############

print("Defining trainer object, will take around 1 minute for loading and preprocessing")

bulla_trainer = BullaSurrogateTrainer(name,
                                      outdir, 
                                      filters = FILTERS,
                                      data_dir=lc_dir, 
                                      tmin = 0.1,
                                      tmax = 14.0,
                                      dt = 0.1,
                                      plots_dir = plots_dir,
                                      save_raw_data = True,
                                      save_preprocessed_data = True)

print("Filters to train on:")
print(bulla_trainer.filters)

full_lc_files = [os.path.join(lc_dir, f) for f in lc_files]
print("Example fetching parameters from filename:")
for filename in full_lc_files[:4]:
    print("filename")
    print(filename)
    p = bulla_trainer.extract_parameters_function(filename)
    print("Parameters extracted")
    print(p)
 
# Define the config if you want to change a default parameter
# Here we change the number of epochs to 10_000
config = NeuralnetConfig(nb_epochs = 10_000,
                         output_size=bulla_trainer.svd_ncoeff)

bulla_trainer.fit(config=config, verbose=True)
bulla_trainer.save()

bulla_trainer._save_raw_data()
bulla_trainer._save_preprocessed_data()

########################
### LIGHTCURVE MODEL ###
########################

print("Producing example lightcurve . . .")

lc_model = BullaLightcurveModel(name,
                                outdir, 
                                FILTERS)

times = bulla_trainer.times

print("Training done!")