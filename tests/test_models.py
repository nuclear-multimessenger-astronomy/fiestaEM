from os.path import dirname, join

from fiesta.inference.lightcurve_model import AfterglowFlux, BullaLightcurveModel


##############
# Flux model #
##############

working_dir = dirname(__file__)
surrogates_dir = join(dirname(working_dir), "surrogates")
FILTERS = ["radio-6GHz", "bessellv", "X-ray-1keV"]

def test_MLP():

    model = AfterglowFlux(name="pbag_gaussian_MLP", 
                          filters=FILTERS,
                          directory=join(surrogates_dir, "GRB/pbag_gaussian_MLP/model"))
    
    X = [3.141/30, 54., 0.05, 2., -1., 2.5, -2., -4., 500]
    params = dict(zip(model.parameter_names, X))
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0
    mag = model.predict(params)

def test_CVAE():

    model = AfterglowFlux(name="pbag_gaussian_CVAE", 
                          filters=FILTERS,
                          directory=join(surrogates_dir, "GRB/pbag_gaussian_CVAE/model"))
    
    X = [3.141/30, 54., 0.05, 2., -1., 2.5, -2., -4., 500]
    params = dict(zip(model.parameter_names, X))
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0
    mag = model.predict(params)

def test_LC():

    model = BullaLightcurveModel(name="Bu2025",
                                 filters=["besselli", "bessellg", "bessellr"],
                                 directory=join(surrogates_dir, "KN/Bu2025_lc/model"))
    
    X = [120, -2, 0.2, 0.3, -1.5, 0.4, 0.3, 0.1]
    params = dict(zip(model.parameter_names, X))
    params["luminosity_distance"] = 40.0
    params["redshift"] = 0.0
    mag = model.predict(params)