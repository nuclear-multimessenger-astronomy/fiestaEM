import os

from fiesta.inference.lightcurve_model import AfterglowFlux


##############
# Flux model #
##############

working_dir = os.path.dirname(__file__)
model_dir = os.path.join(working_dir, "models")

def test_models():

    model = AfterglowFlux(name="flux",
                          directory=model_dir,
                          filters=["radio-6GHz", "bessellv", "X-ray-1keV"])

    X = [3.141/30, 54., 0.05, -1., 2.5, -2., -4.]
    mag = model.predict_abs_mag(dict(zip(model.parameter_names, X)))

# TODO: Add more model types here