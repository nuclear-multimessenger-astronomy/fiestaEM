from os.path import dirname, join
from pathlib import Path
from fiesta.inference.lightcurve_model import built_in_surrogates, FluxModel, BullaLightcurveModel

working_dir = Path(dirname(__file__))


def test_builtins():

    for model_name, model_dir, model_type in built_in_surrogates():
        if model_type=="KN":
            model = FluxModel(model_name, filters=["besselli", "bessellv"])
        elif model_type=="GRB":
            model = FluxModel(model_name, filters=["radio-3GHz", "bessellv", "X-ray-1keV"])
        
        params = {p: 0.5*(val[0] + val[1]) for p, val in model.parameter_distributions.items()}
        params["luminosity_distance"] = 40.0
        params["redshift"] = 0.0

        times, mag = model.predict(params)

def test_lc_surrogates():

    model = BullaLightcurveModel(name="Bu2025",
                                 filters=["besselli", "bessellv"],
                                 directory=working_dir.parent / "surrogates" / "KN" / "Bu2025_lc" / "model")
    
    
    params = {p: 0.5*(val[0] + val[1]) for p, val in model.parameter_distributions.items()}
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0

    times, mag = model.predict(params)


def test_CVAE_surrogates():

    model = FluxModel(name="afgpy_gaussian_CVAE",
                      filters=["radio-3GHz", "bessellv", "X-ray-1keV"],
                      directory=working_dir.parent / "surrogates" / "GRB" / "afgpy_gaussian_CVAE" / "model")
    
    
    params = {p: 0.5*(val[0] + val[1]) for p, val in model.parameter_distributions.items()}
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0

    times, mag = model.predict(params)


def test_MLP_surrogates():

    model = FluxModel(name="Bu2026_MLP",
                      filters=["besselli", "bessellv"],
                      directory=working_dir.parent / "surrogates" / "KN" / "Bu2026_MLP" / "model")
    
    
    params = {p: 0.5*(val[0] + val[1]) for p, val in model.parameter_distributions.items()}
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0

    times, mag = model.predict(params)