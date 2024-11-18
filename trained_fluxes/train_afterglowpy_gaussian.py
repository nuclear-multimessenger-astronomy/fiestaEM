import numpy as np 
import matplotlib.pyplot as plt

from fiesta.train.FluxTrainer import PCATrainer, AfterglowpyData, DataManager
from fiesta.train.BenchmarkerFluxes import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowpyPCA
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.utils import Filter

#############
### SETUP ###
#############

tmin = 0.1 # days
tmax = 2000
n_times = 200


numin = 1e9 # Hz 
numax = 2.5e18
n_nu = 256


parameter_distributions = {
    'inclination_EM': (0, np.pi/2, "uniform"),
    'log10_E0': (47, 57, "uniform"),
    'alphaCore': (0.01, 4, "loguniform"),
    'thetaWing': (0.01, np.pi/5, "uniform"),
    'log10_n0': (-6, 2, "uniform"),
    'p': (2.01, 3, "uniform"),
    'log10_epsilon_e': (-4, 0, "uniform"),
    'log10_epsilon_B': (-8,0, "uniform")
}

    

jet_name = "gaussian"
jet_conversion = {"tophat": -1,
                  "gaussian": 0,
                  "powerlaw": 4}

n_training = 60_000
n_val = 5000
n_test = 2000

n_pool = 24

retrain_weights = None


#######################
### CREATE RAW DATA ###
#######################
name = jet_name
outdir = f"./afterglowpy/{name}/"

jet_type = jet_conversion[jet_name]



creator = AfterglowpyData(outdir = outdir,
                          jet_type = jet_type,
                          n_training = n_training, 
                          n_val = n_val,
                          n_test = n_test,
                          n_pool = n_pool,
                          tmin = tmin,
                          tmax = tmax,
                          n_times = n_times,
                          numin = numin,
                          numax = numax,
                          n_nu = n_nu,
                          parameter_distributions = parameter_distributions)


###############
### TRAINER ###
###############


data_manager = DataManager(outdir = outdir,
                           n_training= 60_000, 
                           n_val= 5000, 
                           tmin= 1,
                           tmax= 2000,
                           numin = 1e9,
                           numax = 2.5e18,
                           retrain_weights = retrain_weights)

trainer = PCATrainer(name,
                     outdir,
                     data_manager = data_manager,
                     plots_dir=f"./benchmarks/{name}",
                     n_pca = 50,
                     save_preprocessed_data=False
                     )

###############
### FITTING ###
###############

config = NeuralnetConfig(output_size=trainer.n_pca,
                         nb_epochs=100_000,
                         hidden_layer_sizes = [128, 256, 128],
                         learning_rate =8e-3)

trainer.fit(config=config)
trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")
FILTERS = ["radio-3GHz", "X-ray-1keV", "radio-6GHz", "bessellv"]

lc_model = AfterglowpyPCA(name,
                          outdir, 
                          filters = FILTERS)

for filt in lc_model.Filters:
    X_example = trainer.val_X_raw[-1]

    y_raw = trainer.val_y_raw[-1]
    y_raw = y_raw.reshape(256, len(lc_model.times))
    y_raw = np.exp(y_raw)
    y_raw = np.array([np.interp(filt.nu, lc_model.metadata["nus"], column) for column in y_raw.T]) 
    y_raw = -48.6 + -1 * np.log10(y_raw*1e-3 / 1e23) * 2.5
    
    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt.name]
    
    plt.plot(lc_model.times, y_raw, color = "red", label="afterglowpy")
    plt.plot(lc_model.times, y_predict, color = "blue", label="Surrogate prediction")
    upper_bound = y_predict + 1
    lower_bound = y_predict - 1
    plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)

    plt.ylabel(f"mag for {filt.name}")
    plt.legend()
    plt.gca().invert_yaxis()

    plt.savefig(f"./benchmarks/{name}/afterglowpy_{name}_{filt.name}_example.png")
    plt.close()