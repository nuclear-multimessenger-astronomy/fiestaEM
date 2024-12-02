import numpy as np 
import matplotlib.pyplot as plt

from fiesta.train.SurrogateTrainer import AfterglowpyTrainer
from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowpyLightcurvemodel
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.utils import Filter

#############
### SETUP ###
#############

FILTERS = ["X-ray-1keV", "radio-6GHz", "radio-3GHz", "bessellv"]
for filter in FILTERS:
    filter = Filter(filter)
    print(filter.name, filter.nu)

tmin = 1
tmax = 1000

"""
#grid for radio-6GHz and radio-3GHz
parameter_grid = {
    'inclination_EM': [0.0, np.pi/24, np.pi/12, np.pi/8, np.pi/6, np.pi*5/24, np.pi/4, np.pi/3, 5*np.pi/12, 1.4, np.pi/2],
    'log10_E0': [46.0, 46.5, 48, 50, 51, 52., 53, 53.5, 54., 54.5, 55.],
    'thetaCore': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.3, np.pi/10],
    'log10_n0': [-7.0, -6.5, -6.0, -5.0, -4.0, -3.0, -1.0, 1.0],
    'p': [2.01, 2.1, 2.2, 2.4, 2.6, 2.8, 2.9, 3.0],
    'log10_epsilon_e': [-4, -3.5,  -3,  -2, -1, -0.66, -0.33, 0],
    'log10_epsilon_B': [-8, -6, -4, -2., -1., 0]
}

#grid for X-ray-1keV and bessellv
parameter_grid = {
    'inclination_EM': [0.0, np.pi/24, np.pi/12, np.pi/8, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, 1.4, np.pi/2],
    'log10_E0': [46.0, 46.5, 48, 50, 51, 52., 53, 53.5, 54., 54.5, 55.],
    'thetaCore': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.3, np.pi/10],
    'log10_n0': [-7.0, -6.5, -6.0, -5.0, -4.0, -3.0, -1.0, 1.0],
    'p': [2.01, 2.1, 2.2, 2.4, 2.6, 2.8, 2.9, 3.0],
    'log10_epsilon_e': [-4, -3.5,  -3,  -2, -1, -0.66, -0.33, 0],
    'log10_epsilon_B': [-8, -6, -4, -2., -1., 0]
}


"""

FILTERS = ["X-ray-1keV", "radio-6GHz", "radio-3GHz", "bessellv"]
FILTERS = ["radio-6GHz"]
parameter_grid = {
    'inclination_EM': np.linspace(0, np.pi/4, 12),
    'log10_E0': np.linspace(47, 56, 19),
    'thetaCore': np.logspace(-2, np.log10(np.pi/5), 12),
    'log10_n0': np.linspace(-6, 2, 17),
    'p': np.linspace(2.01, 3.0, 10),
    'log10_epsilon_e': np.linspace(-4, 0, 9),
    'log10_epsilon_B': np.linspace(-8, 0, 9)
}


jet_name = "tophat"
jet_conversion = {"tophat": -1,
                  "gaussian": 0,
                  "powerlaw": 4}

name = "tophat"
outdir = f"./afterglowpy/{name}/"

###############
### TRAINER ###
###############

B = Benchmarker(name = name,
                parameter_grid = parameter_grid,
                model_dir = outdir,
                filters = FILTERS,
                n_test_data = 2000,
                metric_name = "$\\mathcal{L}_\infty$",
                remake_test_data = False,
                jet_type = jet_conversion[jet_name],
                )

ww = B.error_distribution

weight_grids = ww["radio-6GHz"]

# TODO: perhaps also want to train on the full LC, without the SVD?
# TODO: train to output flux, not the mag?
trainer = AfterglowpyTrainer(name,
                             outdir,
                             FILTERS,
                             parameter_grid,
                             weight_grids = weight_grids,
                             jet_type = jet_conversion[jet_name],
                             tmin = tmin,
                             tmax = tmax,
                             use_log_spacing = True,
                             plots_dir=f"./benchmarks/{name}",
                             svd_ncoeff=30,
                             save_raw_data=True,
                             save_preprocessed_data=True,
                             remake_training_data = True,
                             n_training_data = 7000
                             )

###############
### FITTING ###
###############

config = NeuralnetConfig(output_size=trainer.svd_ncoeff,
                         nb_epochs=50_000,
                         hidden_layer_sizes = [64, 128, 64],
                         learning_rate = 8e-3)

trainer.fit(config=config)
trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")

lc_model = AfterglowpyLightcurvemodel(name,
                                      outdir, 
                                      filters = FILTERS)

for filt in lc_model.filters:
    X_example = trainer.val_X_raw[0]
    y_raw = trainer.val_y_raw[filt][0]
    
    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt]
    
    plt.plot(lc_model.times, y_raw, color = "red", label="afterglowpy")
    plt.plot(lc_model.times, y_predict, color = "blue", label="Surrogate prediction")
    upper_bound = y_predict + 1
    lower_bound = y_predict - 1
    plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)

    plt.ylabel(f"mag for {filt}")
    plt.legend()
    plt.gca().invert_yaxis()

    plt.savefig(f"./benchmarks/{name}/afterglowpy_{name}_{filt}_example.png")
    plt.close()