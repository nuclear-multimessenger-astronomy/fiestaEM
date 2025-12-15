from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowFlux


name = "pbag_gaussian"
model_dir = f"./model/"
FILTERS = ["radio-3GHz", "radio-6GHz", "bessellv", "X-ray-1keV"]


lc_model = AfterglowFlux(name,
                         directory = model_dir, 
                         filters = FILTERS)
 
for metric_name in ["L2", "Linf"]:   

    
    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "../_training_data/pyblastafterglow_gaussian_raw_data.h5",
                    metric_name = metric_name
                    )
    
    benchmarker.benchmark()
    benchmarker.plot_lightcurves_mismatch() 
