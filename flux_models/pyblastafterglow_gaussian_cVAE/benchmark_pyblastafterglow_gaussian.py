from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowFlux


name = "gaussian"
model_dir = f"./model/"
FILTERS = ["radio-3GHz", "radio-6GHz", "bessellv", "X-ray-1keV"]


lc_model = AfterglowFlux(name,
                         directory = model_dir, 
                         filters = FILTERS,
                         model_type= "CVAE")
 
for metric_name in ["L2", "Linf"]:   

    
    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "../pyblastafterglow_gaussian/model/pyblastafterglow_raw_data.h5",
                    metric_name = metric_name
                    )
    
    benchmarker.benchmark()

    benchmarker.plot_lightcurves_mismatch(parameter_labels = ["$\\iota$", 
                                                              "$\log_{10}(E_0)$", 
                                                              "$\\theta_{\\mathrm{c}}$",
                                                              "$\\alpha_{\\mathrm{w}}$",
                                                              "$\log_{10}(n_{\mathrm{ism}})$", 
                                                              "$p$", 
                                                              "$\log_{10}(\\epsilon_e)$", 
                                                              "$\log_{10}(\\epsilon_B)$",
                                                              "$\\Gamma_0$"])




