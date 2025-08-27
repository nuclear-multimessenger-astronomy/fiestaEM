Surrogates
----------

Fiesta essentially has two types of surrogates. The one type is the ``FluxModel``, where the surrogate is trained directly on the flux density array, and the other is the ``LightcurveModel``, where for each filter a separate NN is trained directly on the photometric lightcurves.
Each type can have different underlying NN architectures, which are inferred automatically upon loading. 

Built-in surrogates
^^^^^^^^^^^^^^^^^^^
In fiesta, a built-in model can easily be loaded with 

.. code:: python

    from fiesta.inference.lightcurve_model import BullaLightcurveModel, AfterglowFlux
    
    model_flux = AfterglowFlux(name="afgpy_gaussian_CVAE",
                               filters = ["radio-3GHz", "bessellv", "X-ray-1keV"])

    model_lc = BulllaLightcurveModel(name="Bu2025_lc",
                                     filters = ["besselli", "bessellr", "bessellux"]) 

Based on the ``name`` fiesta will load the model automatically from the surrogate directory in the source directory.
A list of available built-in surrogates can be obtained through ``list_built_in_surrogates()`` from ``fiesta.inference.lightcurve_model``.
The ``filters`` argument takes a list of strings that can be used to initialize a ``Filter`` class. The program will check automatically whether these filters are compatible with the frequency ranges of the surrogates and if not, it will remove them from the filter list.

Custom surrogates
^^^^^^^^^^^^^^^^^
If a custom surrogate was trained and the ``.pkl`` files are saved in ``/path/to/my/dir``, the surrogate can be loaded as

.. code:: python

    from fiesta.inference.lightcurve_model import BullaLightcurveModel, AfterglowFlux
    
    model_flux = AfterglowFlux(name="my_surrogate",
                               filters = ["radio-3GHz", "bessellv", "X-ray-1keV"],
                               directory= "/path/to/my/dir")

This works similarly for a ``LightcurveModel`` type surrogate.

Combining surrogates
^^^^^^^^^^^^^^^^^^^^

To perform light curve analyses where the emission might arise from multiple processes, fiesta offers a convenient way to combine multiple models into one instance:

.. code:: python

    from fiesta.inference.lightcurve_model import BullaLightcurveModel, AfterglowFlux
    
    model_flux = AfterglowFlux(name="my_surrogate",
                               filters = ["radio-3GHz", "bessellv", "X-ray-1keV"],
                               directory= "/path/to/my/dir")







