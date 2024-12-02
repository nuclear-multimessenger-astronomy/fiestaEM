import numpy as np 
import matplotlib.pyplot as plt

from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowpyLightcurvemodel
from fiesta.utils import Filter


name = "gaussian"
model_dir = f"./afterglowpy/{name}/"
FILTERS = ["radio-3GHz", "radio-6GHz", "bessellv", "X-ray-1keV"]

parameter_grid = {
    'inclination_EM': np.linspace(0, np.pi/4, 12),
    'log10_E0': np.linspace(47, 56, 19),
    'thetaWing': np.linspace(0.01, np.pi/5, 12),
    'xCore': np.linspace(0.05, 1, 20),
    'log10_n0': np.linspace(-6, 2, 17),
    'p': np.linspace(2.01, 3.0, 10),
    'log10_epsilon_e': np.linspace(-4, 0, 9),
    'log10_epsilon_B': np.linspace(-8, 0, 9)
}

for metric_name in ["$\\mathcal{L}_2$", "$\\mathcal{L}_\infty$"]:    
    if metric_name == "$\\mathcal{L}_2$":
        file_ending = "L2"
    else:
        file_ending = "Linf"
    
    B = Benchmarker(name = name,
                parameter_grid = parameter_grid,
                model_dir = model_dir,
                MODEL = AfterglowpyLightcurvemodel,
                filters = FILTERS,
                n_test_data = 2000,
                metric_name = metric_name,
                remake_test_data = False,
                jet_type = 0,
                )

    fig, ax = B.plot_error_distribution("radio-6GHz")

    
    for filt in FILTERS:
        
        fig, ax = B.plot_lightcurves_mismatch(filter =filt, parameter_labels = ["$\\iota$", "$\log_{10}(E_0)$", "$\\theta_{\\mathrm{w}}$", "$x_c$", "$\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\epsilon_E$", "$\\epsilon_B$"])
        fig.savefig(f"./benchmarks/{name}/benchmark_{filt}_{file_ending}.pdf", dpi = 200)
    
        B.print_correlations(filter = filt)


        if metric_name == "$\\mathcal{L}_\infty$":
            fig, ax = B.plot_error_distribution(filt)
            fig.savefig(f"./benchmarks/{name}/error_distribution_{filt}.pdf", dpi = 200)
    
    
    fig, ax = B.plot_worst_lightcurves()
    fig.savefig(f"./benchmarks/{name}/worst_lightcurves_{file_ending}.pdf", dpi = 200)






