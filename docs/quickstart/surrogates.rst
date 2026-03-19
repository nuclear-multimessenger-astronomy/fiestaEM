Surrogates
----------

Fiesta essentially has two types of surrogates. The one type is the ``FluxModel``, where the surrogate is trained directly on the flux density array, and the other is the ``LightcurveModel``, where for each filter a separate NN is trained directly on the photometric lightcurves.
Each type can have different underlying NN architectures, which are inferred automatically upon loading.

Loading surrogate models
^^^^^^^^^^^^^^^^^^^^^^^^
Surrogates can be loaded within a Python script as

.. code:: python

    from fiesta.inference.lightcurve_model import FluxModel

    model = FluxModel(name=name, filters=filters, directory=directory)

The ``name`` argument is the name of the surrogate.
The  `directory` argument points to the `.pkl` files storing the neural network and surrogate metadata.
However, if a surrogate is stored in the installation under `src/fiesta/surrogates/KN` or `src/fiesta/surrogates/GRB`, it can be loaded as a built-in surrogate without providing the directory argument explicitely. 
``fiesta`` will find these built-in surrogates automatically solely based on the ``name`` argument.
The ``filters`` argument takes a list of strings that can be used to initialize a ``Filter`` class. The program will check automatically whether these filters are compatible with the frequency ranges of the surrogates and if not, it will remove them from the filter list.


Built-in surrogates
^^^^^^^^^^^^^^^^^^^
To see which surrogates are built-in and can simply be loaded without providing ``directory`` explicitely, you can run in your shell

.. code:: bash

    python -c "from fiesta.surrogates import print_built_in_surrogates; print_built_in_surrogates()"

Initially, there are no built-in surrogates. 
However, built-in surorgates can be installed by downloading them from our `huggingface repository <https://huggingface.co/nuclear-multimessenger-astronomy/fiesta-surrogates>`_.
This is automized, meaning that you can run from shell

.. code:: bash

    python -c "from fiesta.surrogates import download_recommended_surrogates; download_recommended_surrogates()"

to fetch a set of recommended built-in surrogates to your local installation. To obtain a specific surrogate ``name``, use

.. code:: python

    from fiesta.surrogates import download_surrogate
    download_surrogate(name)


Alternatively, if you attempt to load a surrogate in Python without providing a ``directory`` argument and the surrogate has not been stored locally yet, the program will try to fetch the surrogate from the hugging face repo automatically (obviously this requires an internet connection).
This means 

.. code-block:: python

    from fiesta.inference.lightcurve_model import FluxModel

    model_flux = FluxModel(name="afgpy_gaussian_CVAE",
                           filters = ["radio-3GHz", "bessellv", "X-ray-1keV"])


should run even without any prior download attempts.

To print a list of surrogates available for download, run

.. code:: bash

    python -c "from fiesta.surrogates import print_downloadable_surrogates; print_downloadable_surrogates()"


Predicting light curves
^^^^^^^^^^^^^^^^^^^^^^^
The main purpose of the surrogates is to predict light curves for a given set of parameters. There is a simple method for this

.. code:: python

    from fiesta.inference.lightcurve_model import FluxModel

    model_flux = FluxModel(name="Bu2026_MLP",
                           filters = ["radio-3GHz", "bessellv", "X-ray-1keV"])

    params = dict(inclination_EM=0.2,
                  log10_mej_dyn=-3.,
                  v_ej_dyn=0.2,
                  Ye_dyn=0.2,
                  log10_mej_wind=-1.5,
                  v_ej_wind=0.1,
                  Ye_wind=0.3,
                  redshift=0.0079,
                  luminosity_distance=40.)

    times, mags = model.predict(params)


The ``params`` dictionary has to contain redshift, luminosity distance (in Mpc), and all the model parameters. The surrogate usually prints the parameters upon loading. 
Alternatively, the surrogate parameters can also be checked from the ``model.parameter_names``.

When loading a surrogate, it will also print the range of parameters in which the surrogate has been trained.
These parameter ranges are not enforced, but **WITH THE BIGGEST EMPHASIS, WE HEAVILY ADVISE AGAINST GOING OUTSIDE THESE RANGES**. It is your responsibility to ensure only valid parameters are fed into the ``predict`` method.

The return values are a ``jnp`` array for the time in observer frame and a dictionary with the apparent magnitudes in each filter (i.e. ``radio-3GHz``, ``bessellv``, ``X-ray-1keV`` in this case).
If you want to simply get the absolute magnitudes use 

.. code:: python

    times, mags = model.predict_abs_mag(params)

where ``luminosity_distance`` and ``redshift`` don't have to be passed in params (if they are still in params, they will be ignored).

From each ``FluxModel`` surrogate, you can also get the flux density array in :math:`\log_{10}` mJy by using

.. code-block:: python

    times, nus, log10_flux = model.predict_log_flux(params)


Combining surrogates
^^^^^^^^^^^^^^^^^^^^
To perform light curve analyses where the emission might arise from multiple processes, ``fiesta`` offers a convenient way to combine multiple models into one instance:

.. code:: python

    from fiesta.inference.lightcurve_model import LightcurveModel, FluxModel, CombinedSurrogate
    
    model1 = AfterglowFlux(name="afgpy_gaussian_CVAE",
                           filters = FILTERS)
    
    model2 = BullaFlux(name="Bu2026_MLP",
                                  filters = FILTERS)
    
    model = CombinedSurrogate(models=[model1, model2],
                              sample_times=jnp.geomspace(0.3, 1000, 200))

The fluxes of ``model1`` and ``model2`` are simply added in the respective photometric bands. If ``model1`` or ``model2`` fall outside the time range of the ``sample_times`` argument, they are simply set to ``jnp.inf`` mag there.


