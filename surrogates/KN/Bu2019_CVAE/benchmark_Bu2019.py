from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowFlux



name = "Bu2019"
model_dir = f"./model/"
FILTERS = ["2massj", "2massks", "sdssu", "ps1::r"]

lc_model = AfterglowFlux(name,
                         directory = model_dir, 
                         filters = FILTERS,
                         model_type= "CVAE")
 
for metric_name in ["L2", "Linf"]:   

    
    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "./model/Bu2019_raw_data.h5",
                    metric_name = metric_name
                    )
    
    benchmarker.benchmark()

    benchmarker.plot_lightcurves_mismatch(parameter_labels = ["$\\log_{10}(m_{\\mathrm{ej, dyn}})$", "$\\log_{10}(m_{\\mathrm{ej, wind}})$", "$\\Phi_{\\mathrm{KN}}$", "$\\iota$"])



"""
name = "Bu2019"
model_dir = f"./model/"
FILTERS = ["2massj", "2massks", "sdssu", "ps1::r"]

for metric_name in ["$\\mathcal{L}_2$", "$\\mathcal{L}_\infty$"]:   
    if metric_name == "$\\mathcal{L}_2$":
        file_ending = "L2"
    else:
        file_ending = "Linf"
    
    B = Benchmarker(name = name,
                    model_dir = model_dir,
                    MODEL = AfterglowFlux,
                    filters = FILTERS,
                    metric_name = metric_name,
                    model_type = "CVAE",
                    file = "./model/Bu2019_raw_data.h5"
                    )
        
  
    for filt in FILTERS:
        
        fig, ax = B.plot_lightcurves_mismatch(filter =filt, parameter_labels = ["$\\log_{10}(m_{\\mathrm{ej, dyn}})$", "$\\log_{10}(m_{\\mathrm{ej, wind}})$", "$\\Phi_{\\mathrm{KN}}$", "$\\iota$"])
        fig.savefig(f"./benchmarks/benchmark_{filt}_{file_ending}.pdf", dpi = 200)
    
        B.print_correlations(filter = filt)    
    
    fig, ax = B.plot_worst_lightcurves()
    fig.savefig(f"./benchmarks/worst_lightcurves_{file_ending}.pdf", dpi = 200)


fig, ax = B.plot_error_distribution()
fig.savefig(f"./benchmarks/error_distribution.pdf", dpi = 200)

fig, ax = B.plot_error_over_time()
fig.savefig(f"./benchmarks/error_over_time.pdf", dpi = 200)
"""