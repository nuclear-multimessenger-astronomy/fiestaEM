from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowFlux



name = "Bu2024"
model_dir = f"./model/"
FILTERS = ["2massj", "2massks", "sdssu", "ps1::r"]

lc_model = AfterglowFlux(name,
                                model_dir, 
                                filters = FILTERS)

for metric_name in ["L2", "Linf"]:

    benchmarker = Benchmarker(
                    model = lc_model,
                    data_file = "../../lightcurve_models/Bu2024/model/Bu2024_raw_data.h5",
                    metric_name = metric_name
                    )
    
    benchmarker.benchmark()

    benchmarker.plot_lightcurves_mismatch(parameter_labels = ["$\\log_{10}(m_{\\mathrm{ej, dyn}})$", "$v_{\\mathrm{ej, dyn}}$", "$Y_{e, \\mathrm{dyn}}$", "$\\log_{10}(m_{\\mathrm{ej, wind}})$", "$v_{\\mathrm{ej, wind}}$", "$\\iota$"])






