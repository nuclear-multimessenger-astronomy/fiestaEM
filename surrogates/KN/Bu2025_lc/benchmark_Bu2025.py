from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import BullaLightcurveModel



name = "Bu2025"
model_dir = f"./model/"
FILTERS = ["ps1::y", "besselli", "bessellv", "bessellux"]

lc_model = BullaLightcurveModel(name,
                                directory=model_dir, 
                                filters = FILTERS)

for metric_name in ["L2", "Linf"]:

    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "../training_data/Bu2025_raw_data.h5",
                    metric_name = metric_name
                    )
    
    benchmarker.benchmark()

    benchmarker.plot_lightcurves_mismatch(parameter_labels = ["$\\log_{10}(m_{\\mathrm{ej, dyn}})$", "$v_{\\mathrm{ej, dyn}}$", "$Y_{e, \\mathrm{dyn}}$", "$\\log_{10}(m_{\\mathrm{ej, wind}})$", "$v_{\\mathrm{ej, wind}}$" , "$Y_{e, \\mathrm{wind}}$", "$\\iota$", "$z$"])






