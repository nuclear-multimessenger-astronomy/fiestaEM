from os.path import dirname, join
from pathlib import Path

from fiesta.surrogates import built_in_surrogates
from fiesta.inference.lightcurve_model import FluxModel, BullaLightcurveModel

working_dir = Path(dirname(__file__))



def test_CVAE_surrogates():

    model = FluxModel(name="afgpy_gaussian_CVAE",
                      filters=["radio-3GHz", "bessellv", "X-ray-1keV"])
    
    
    params = {p: 0.5*(val[0] + val[1]) for p, val in model.parameter_distributions.items()}
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0

    times, mag = model.predict(params)


def test_MLP_surrogates():

    model = FluxModel(name="Bu2026_MLP",
                      filters=["besselli", "bessellv"])
    
    
    params = {p: 0.5*(val[0] + val[1]) for p, val in model.parameter_distributions.items()}
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0

    times, mag = model.predict(params)