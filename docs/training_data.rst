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
Each of these groups contains exactly two sets. 
One set has to be named ``X`` and contains the parameters as a vector of shape ``(n_samples, n_params)``. They need to match to the parameters provided in the ``parameter_names`` set.
The other set has to be named ``y`` and contains the a vector of shape ``(n_samples, n_nus, n_times)`` where the last two shapes are the length of the ``nu`` and ``time`` array.
The entries are the corresponding flux densities at 10 pc (but zero redshift, i.e. source frame) in $log_{10}(\\mathrm{mJys}) = log_{10}(\\mathrm{1e-26 erg / (s Hz cm^2)})$.

Optionally, there can be a data group ``special_train``. This group can contain further groups, labeled arbitrarily, that contain ``X`` and ``y`` data sets, that can supplement the larger training data set in ``train``, should certain areas of the parameter space need extra coverage.
They can be used during training through the ``DataManager`` interface.

Data manager
^^^^^^^^^^^^

The ``DataManager`` class provides an interface to the .hdf5 file.
It can load the raw data directly, but it can also take care of preprocessing the data before placing them into the training loop of the NN.
Which data points will be used is determined upon initialization through the ``n_training``, ``n_val``, and ``special_training`` arguments.
It can also cut the data to a custom time and frequency domain through the ``tmin``, ``tmax``, ``numin``, ``numax`` arguments.
The data is not actually loaded during initialization, but only when one of the following methods is called: 

    - ``DataManager.load_raw_data_from_file``: returns four arrays ``train_X, train_y, val_X, val_y`` in raw format.
    - ``DataManager.preprocess_pca``: returns ``train_X, train_y, val_X, val_y, Xscaler, yscaler`` in rescaled format, where the last to entries are the scalers to rescale the values back. The ``yscaler`` is a ``PCAScaler``.
    - ``DataManager.preprocess_cVAE``: returns ``train_X, train_y, val_X, val_y, Xscaler, yscaler`` in rescaled format, where the last to entries are the scalers to rescale the values back. The ``yscaler`` is a ``ImageScaler`` concatenated with a ``StandardScaler``.



