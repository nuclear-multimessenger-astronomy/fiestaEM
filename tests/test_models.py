from fiesta.inference.lightcurve_model import AfterglowFlux


##############
# Flux model #
##############

model = AfterglowFlux(name="flux",
                      directory="./models",
                      filters=["radio-6GHz", "bessellv", "X-ray-1keV"],
                      model_type="MLP")

X = [3.141/30, 54., 0.05, -1., 2.5, -2., -4.]
mag = model.predict(dict(zip(model.parameter_names, X)))


# TODO: Add more model types here