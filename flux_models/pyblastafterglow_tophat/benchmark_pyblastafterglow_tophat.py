import numpy as np 
import matplotlib.pyplot as plt

from fiesta.train.BenchmarkerFluxes import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowpyPCA
from fiesta.utils import Filter


name = "tophat"
model_dir = f"./model/"
FILTERS = ["radio-3GHz", "radio-6GHz", "bessellv"]

for metric_name in ["$\\mathcal{L}_2$", "$\\mathcal{L}_\infty$"]:   
    if metric_name == "$\\mathcal{L}_2$":
        file_ending = "L2"
    else:
        file_ending = "Linf"
    
    B = Benchmarker(name = name,
                model_dir = model_dir,
                MODEL = AfterglowpyPCA,
                filters = FILTERS,
                metric_name = metric_name,
                )
    
  
    for filt in FILTERS:
        
        fig, ax = B.plot_lightcurves_mismatch(filter =filt, parameter_labels = ["$\\iota$", "$\log_{10}(E_0)$", "$\\theta_{\\mathrm{c}}$", "$\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\epsilon_E$", "$\\epsilon_B$", "$\\Gamma_0$"])
        fig.savefig(f"./benchmarks/benchmark_{filt}_{file_ending}.pdf", dpi = 200)
    
        B.print_correlations(filter = filt)    
    
    fig, ax = B.plot_worst_lightcurves()
    fig.savefig(f"./benchmarks/worst_lightcurves_{file_ending}.pdf", dpi = 200)


fig, ax = B.plot_error_distribution()
fig.savefig(f"./benchmarks/error_distribution.pdf", dpi = 200)

fig, ax = B.plot_error_over_time()
fig.savefig(f"./benchmarks/error_over_time.pdf", dpi = 200)



