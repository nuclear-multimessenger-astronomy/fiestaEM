Training data
-------------

In fiesta, training a surrogate starts from an ``.hdf5`` file that contains all the training, validation, and test data, plus the meta data of the model.
This ``.hdf5`` file needs to follow a certain format, although how exactly it is created does not matter. 
There are some helper functions in ``fiesta`` that can create training data from GRB afterglow models (``fiesta.train.AfterglowData``) or summarize the output from ``possis`` and ``gwemopt`` into a ``.hdf5`` file (``fiesta.utils.convert_POSSIS_outputs_to_h5``).

Data file format
^^^^^^^^^^^^^^^^
The ``.hdf5`` file needs to have the following data sets as "metadata":
    - ``times``: An array for the time domain of the data in days.
    - ``nus``: An array for the frequency domain of the data in Hz.
    - ``parameter_names``: A list of strings that contains the parameter names. This determines which parameter names need to be present in the param-dict that is the argument for the surrogate prediction.
    - ``parameter_distributions``: A string-converted dictionary that has ``parameter_names`` as keys and the values are tuples ``tuple[float, float, str]``. The first two numbers are the minimum and maximum range of this parameter in the training data, i.e., the range in which the trained surrogate will be valid. The string should indicate which distribution the training parameter samples follow, though there are no negative side-effects should the distribution provided here be inaccurate.

Further, the file needs the following groups that contain the actual training data:
    - ``train``: Training data used for training the surrogate. Used by ``fiesta.train.FluxTrainer`` and ``fiesta.train.LightcurveTrainer`` through ``fiesta.training.DataManager``.
    - ``val``: Validation data used for validating the surrogate immediately during training and hyper-parameters tuning. Used by ``fiesta.train.FluxTrainer`` and ``fiesta.train.LightcurveTrainer`` through ``fiesta.training.DataManager``. 
    - ``test``: Test data used for testing the model once hyper-parameters are tuned. Used by ``fiesta.train.Benchmarker``.
    -  ``special_train`` (optional): Data supplementing ``train`` for specific areas in the parameter space that need extra coverage to get better training results.

Each of these groups contains exactly two arrays.
One array has to be named ``X`` and contains the model parameter values in the shape ``(n_samples, n_params)``. They need to match to the parameters provided in the ``parameter_names`` set.
The other array has to be named ``y`` and is of shape ``(n_samples, n_nus, n_times)`` where the last two shapes are the length of the ``nu`` and ``time`` array.
The entries are the corresponding flux densities at 10 pc (but zero redshift, i.e. source frame) in units of $log_{10}(\\mathrm{mJys}) = log_{10}(\\mathrm{1e-26 erg / (s Hz cm^2)})$.
The ``special_train`` group is further divided into subgroups (with arbitrary names) that then store the ``X`` and ``y`` data sets.

``DataLoader``
^^^^^^^^^^^^^^

The ``DataLoader`` class provides an interface to the ``.hdf5`` file.
It can load the raw data directly, but it can also take care of preprocessing the data before placing them into the training loop of the neural networks.
Which data points will be used is determined upon initialization through the ``n_training``, ``n_val``, and ``special_training`` arguments.
It can also cut the data to a custom time and frequency domain through the ``tmin``, ``tmax``, ``numin``, ``numax`` arguments.
The data is not actually loaded during initialization, but only when one of the following methods is called: 

    - ``DataLoader.load_from_file``: returns two arrays ``X`` and ``y`` depending on which data set is called. Possible data sets are ``'train'``, ``'val'``, and ``'test'``.
    - ``DataManager.preprocess_data``: returns ``train_X, train_y, val_X, val_y, X_scaler, y_scaler``, where the last to entries are ``fiesta.scaler`` objects that can be used to transform and back-transform data. The returned arrays here are already transformed.


