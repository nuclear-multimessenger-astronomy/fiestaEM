"""DataLoader class to interact with the training data files"""
from typing import Callable
import tqdm

import numpy as np
import jax.numpy as jnp
import jax
import h5py
import gc
from jaxtyping import Array, Float, Int

import fiesta.scalers as scalers
from fiesta.scalers import ParameterScaler, DataScaler
from fiesta.conversions import apply_redshift
from fiesta.logging import logger


##############
# DATA UTILS #
##############

def array_mask_from_interval(sorted_array, amin, amax):
    """
    Return a mask selecting the grid points spanning [amin, amax].
    
    If a boundary exists exactly in the array, that exact value is used.
    Otherwise, the interval is expanded outward to the nearest grid point.

    Args:
        sorted_array (array): A sorted array
        amin (float): Lower interval bound
        amax (float): Upper interval bound

    Returns:
        A boolean array mask

    """
    indmin = np.searchsorted(sorted_array, amin, side="left")
    if indmin > 0 and sorted_array[indmin] != amin:
        indmin -= 1

    indmax = np.searchsorted(sorted_array, amax, side="right") - 1
    if indmax < len(sorted_array) - 1 and sorted_array[indmax] != amax:
        indmax += 1

    mask = np.zeros(len(sorted_array), dtype=bool)
    mask[indmin:indmax + 1] = True

    return mask

def _check_index_in_range(index: int | slice, n_entries: int, group: str) -> None:
    """Raises an IndexError if ``index`` (or, for a slice, either of its bounds) falls outside [-n_entries, n_entries) for ``group``."""
    if isinstance(index, slice):
        for bound in (index.start, index.stop):
            if bound is not None and not (-n_entries <= bound <= n_entries):
                raise IndexError(f"Slice {index} is out of range for group '{group}' with {n_entries} entries.")
    else:
        if not (-n_entries <= index < n_entries):
            raise IndexError(f"Index {index} is out of range for group '{group}' with {n_entries} entries.")

def concatenate_redshift(X_raw, max_z=0.5):
    redshifts = np.random.uniform(0, max_z, size= 3*X_raw.shape[0])
    X_raw = np.tile(X_raw, (3,1))
    X_raw = np.append(X_raw, redshifts.reshape(-1,1), axis=1)
    return X_raw

def redshifted_magnitude(filt, mJys, nus, redshifts):
    """
    This is a slow and inefficient implementation to get the redshifted magnitudes as training data.
    """
    nnus = nus / (1+redshifts[:, None])
    
    sample_factor_redshift = int(len(redshifts)/len(mJys))
    mJys = np.tile(mJys, (sample_factor_redshift, 1, 1))

    mJys = mJys * (1+redshifts[:, None, None])
    
    def get_mag(mJy_, nu_):
        return filt.get_mag(mJy_, nu_)
    mag = jax.vmap(get_mag, in_axes=0)(mJys, nnus)
    return np.array(mag)



###################
# DATA MANAGEMENT #       
###################

class DataLoader:
    
    def __init__(
        self,
        file: str,
        tmin: Float,
        tmax: Float,
        numin: Float = 1e9,
        numax: Float = 2.5e18,
        n_training: Int = None,
        n_val: Int = None,
        special_training: list = [],
    ) -> None:
        """
        DataLoader class used to handle and preprocess the training, validation, and test data from the base model.
        Initializing an instance of this class will only read in the meta data, 
        the actual data samples will only be loaded once the preprocessing methods is called.

        The training data file must be in ``.h5`` format and contain the following data sets:
            - "times": times in days associated to the spectral flux densities
            - "nus": frequencies in Hz associated to the spectral flux densities
            - "parameter_names": list of the parameter names that are present in the training data.
            - "parameter_distributions": utf-8-string of a dict containing the boundaries and distribution of the parameters.
        Additionally, it must contain three data groups "train", "val", "test". Each of these groups contains two data sets, namely "X" and "y". 
        The X arrays contain the model parameters with columns in the order of "parameter_names" and thus have shape (-1, #parameters). 
        The y array contains the associated log10 of the spectral flux densities in mJys and have shape (-1, #nus, #times).
        
        Args:
            file (str): Path to the .h5 file that contains the raw data.
            tmin (float): Minimum time for which the data will be read in. Fluxes earlier than this time will not be loaded. Defaults to the minimum time of the stored data, if smaller than that value.
            max (float): Maximum time for which the data will be read in. Fluxes later than this time will not be loaded. Defaults to the maximum time of the stored data, if larger than that value.
            numin (float): Minimum frequency for which the data will be read in. Fluxes with frequencies lower than this frequency will not be loaded. Will be set to the minimum frequency of the stored data, if smaller than that value. Defaults to 1e9 Hz (1 GHz).
            numax (float): Maximum frequency for which the data will be read in. Fluxes with frequencies higher than this frequency will not be loaded. Will be set to the maximum frequency of the stored data, if larger than that value. Defaults to 2.5e18 Hz.
            n_training (int): Number of training data points that will be read in and preprocessed. If used with a FluxTrainer, this is also the number of training data points used to train the model. 
                              Will raise a ValueError, if ``n_training`` is larger than the number of training data points stored in the file.
                              Defaults to ``None``, in which case all training samples from the file are used.
            n_val (int): Number of validation data points that will be read in and preprocessed. If used with a FluxTrainer, this is also the number of validation data points used to monitor the training progress. 
                              Will raise a ValueError, if ``n_val`` is larger than the number of validation data points stored in the file.
                              Defaults to ``None``, in which case all validation samples from the file are used.
    
            special_training (list[str]): Batch of 'special' training data to be added. 
                                          This can be customly designed training data to cover a certain area of the parameter space more intensely
                                          and should be stored in the ``.h5`` file as ``f['special_train'][label]['X']`` and ``f['special_train'][label]['y']``, 
                                          where ``label`` is an entry for this special_training argument. Defaults to [].
        """
        
        self.file = file
        self.n_training = n_training
        self.n_val = n_val

        self.tmin = tmin
        self.tmax = tmax
        self.numin = numin
        self.numax = numax

        self.special_training = special_training
        
        self.read_metadata_from_file()
        self.set_up_domain_mask()

    def __repr__(self):
        return f"DataManager({self.file})"

    def read_metadata_from_file(self,)->None:
        """
        Reads in the metadata of the raw data, i.e., times, frequencies and parameter names. 
        Also determines how many training and validation data points are available.
        """
        with h5py.File(self.file, "r") as f:
            self.times_data = f["times"][:]
            self.nus_data = f["nus"][:]
            self.parameter_names =  f["parameter_names"][:].astype(str).tolist()
            self.n_training_exists = f["train"]["X"].shape[0]
            self.n_val_exists = f["val"]["X"].shape[0]
            self.parameter_distributions = f['parameter_distributions'][()].decode('utf-8')

            if self.n_training is None:
                self.n_training = self.n_training_exists
            if self.n_val is None:
                self.n_val = self.n_val_exists

        self.n_times_data = len(self.times_data)
        self.n_nus_data = len(self.nus_data)
        
        # check if there is enough data
        if self.n_training > self.n_training_exists: 
            raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
        if self.n_val > self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to train with {self.n_val} data points.")
    
    def set_up_domain_mask(self,)->None:
        """Trims the stored data down to the time and frequency range desired for training. It sets the mask attribute which is a boolean mask used when loading the data arrays."""
        
        if self.tmin<self.times_data.min() or self.tmax>self.times_data.max():
            logger.warning(f"Provided time range {self.tmin, self.tmax} is too wide for the data stored in file. Using range {max(self.times_data.min(), self.tmin), min(self.times_data.max(), self.tmax)} instead.\n")
        time_mask = array_mask_from_interval(self.times_data, self.tmin, self.tmax)
        self.times = self.times_data[time_mask]
        self.n_times = len(self.times)

        if self.numin<self.nus_data.min() or self.numax>self.nus_data.max():
            logger.warning(f"Provided frequency range {self.numin, self.numax} is too wide for the data stored in file. Using range {max(self.nus_data.min(), self.numin), min(self.nus_data.max(), self.numax)} instead.\n")
        nu_mask = array_mask_from_interval(self.nus_data, self.numin, self.numax)
        self.nus = self.nus_data[nu_mask]
        self.n_nus = len(self.nus)

        mask = nu_mask[:, None] & time_mask
        self.mask = mask
        self.n_mask = np.sum(self.mask)
    
    def print_file_info(self,) -> None:
        """
        Prints the meta data of the raw data, i.e., time, frequencies, and parameter names to terminal. 
        Also prints how many training, validation, and test data points are available.
        """
        logger.info(f"File info for {self.file}:")
        with h5py.File(self.file, "r") as f:
            logger.info(f"   Time range in file: {f['times'][0]:.2f} {f['times'][-1]:.2f} days")
            logger.info(f"   Frequency range in file: {f['nus'][0]:.2e} {f['nus'][-1]:.2e} Hz")
            logger.info(f"   Parameter distributions: {f['parameter_distributions'][()].decode('utf-8')}")
            logger.info("")
            logger.info(f"   Training data: {self.n_training_exists}")
            logger.info(f"   Validation data: {self.n_val_exists}")
            logger.info(f"   Test data: {f['test']['X'].shape[0]}")
            logger.info(f"   Special data:")
            for key in f['special_train'].keys():
                logger.info(f"   \t {key}: {f['special_train'][key]['X'].shape[0]}   description: {f['special_train'][key].attrs['comment']}")
            logger.info("")

    def print_loaded_data_info(self,) -> None:
        """
        Prints the meta data of the loaded data, i.e., the actually requested time and frequency range to terminal. 
        Also prints how many training, validation, and test data points will actually be used.
        """

        logger.info(f"Using the following data from {self.file} for training the surrogate:")
        with h5py.File(self.file, "r") as f:
            logger.info(f"   Time range loaded: {self.times[0]:.2f} {self.times[-1]:.2f} days")
            logger.info(f"   Number of points in the time array: {self.n_times}")
            logger.info(f"   Frequency range loaded: {self.nus[0]:.2e} {self.nus[-1]:.2e} Hz")
            logger.info(f"   Number of points in the frequency array: {self.n_nus}")
            logger.info(f"   Parameter names: {self.parameter_names}")

            logger.info("")
            logger.info(f"   Training data: {self.n_training}")
            if self.special_training:
                logger.info(f"   Special data:")
                for key in self.special_training:
                    logger.info(f"   \t {key}: {f['special_train'][key]['X'].shape[0]}   description: {f['special_train'][key].attrs['comment']}")
            logger.info(f"   Validation data: {self.n_val}")
            logger.info("")
    
    def load_from_file(
            self,
            group: str,
            index: int | slice,
            special_label: str | None = None
        ) -> tuple[Array, Array]:
        """
        Loads raw data from the file and returns them as arrays.

        Args:
            group (str): The data group from which the file to load from.
                         Can be ``train``, ``val``, ``test``, or ``special_train``.
                         If ``special_train``, the argument ``special_label`` must also be provided.
            index (int | slice): Index or slice of indices to load (e.g. ``5`` or ``slice(5, 8)``).
            special_label (str): Special data set to load from the ``special_train`` data group.
                                 Only relevant when ``group`` is ``"special_train"``. Defaults to ``None``.

        Raises:
            IndexError: If ``index`` (or, for a slice, either of its bounds) falls outside the range of entries stored in ``group``.
        """

        with h5py.File(self.file, "r") as f:

            if group != "special_train":
                dataset = f[group]
            else:
                if not special_label in f["special_train"].keys():
                    if special_label is None:
                        raise ValueError("When loading special training data, please provide ``special_label``"
                                         "so that a particular special data set can be loaded.")
                    else:
                        raise ValueError(f"Could not find data set {special_label} in ``special_train``.")
                dataset = f["special_train"][special_label]

            n_entries = dataset["X"].shape[0]
            _check_index_in_range(index, n_entries, group)

            X_raw = dataset["X"][index]
            y_raw = dataset["y"][index][:, self.mask]

        return X_raw, y_raw

    def preprocess_data(
            self,
            X_scaler: ParameterScaler,
            y_scaler: DataScaler,
    ) -> tuple[Array, Array, Array, Array, ParameterScaler, DataScaler]:

        train_X, val_X, X_scaler = self.preprocess_parameters(X_scaler)
        train_y, val_y, y_scaler = self.preprocess_fluxes(y_scaler)       

        return train_X, train_y, val_X, val_y, X_scaler, y_scaler

    def preprocess_parameters(
        self,
        X_scaler: ParameterScaler,
    ) -> tuple[Array, Array, ParameterScaler]:
        """
        Fits and transforms the parameters (``X`` data sets) from ``self.file``.
        """
        
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            val_X_raw = f["val"]["X"][:self.n_val]

            # fit and transform
            train_X = X_scaler.fit_transform(train_X_raw)
            val_X = X_scaler.transform(val_X_raw)

            # add special training data
            for label in self.special_training:
                special_X_raw = f["special_train"][label]["X"][:]
                special_X = X_scaler.transform(special_X_raw)
                train_X = np.concatenate((train_X, special_X))

        return train_X, val_X, X_scaler

    def preprocess_fluxes(
        self,
        y_scaler: DataScaler
    ) -> tuple[Array, Array, DataScaler]:

        with h5py.File(self.file, "r") as f:

            train_set = f["train"]["y"]

            # First fit the y_scaler 
            # with a fit batch of max. 20k samples
            # to save memory
            n_fits = min(20_000, self.n_training)
            fit_batch = train_set[:n_fits][:, self.mask].astype(np.float16)
            self._check_array_for_garbage(fit_batch, "fit training batch")
            fit_batch = y_scaler.fit_transform(fit_batch)
            transformed_shape = fit_batch[0].shape
            del fit_batch; gc.collect() # remove fit_batch from memory

            # loop over the entire training data
            train_y = np.empty((self.n_training, *transformed_shape))
            chunk_size = train_set.chunks[0]
            nchunks, rest = divmod(self.n_training, chunk_size)

            for chunk in tqdm.tqdm(range(nchunks)):
                sl = slice(chunk * chunk_size, (chunk+1) * chunk_size)

                # read into raw batch
                raw_batch = np.empty((chunk_size, self.n_nus_data, self.n_times_data))
                train_set.read_direct(raw_batch, source_sel=np.s_[sl, :, :])
                raw_batch = raw_batch[:, self.mask]
                self._check_array_for_garbage(raw_batch, f"training data chunk {chunk}")

                train_y[sl] = y_scaler.transform(raw_batch)

            if rest:
                sl = slice(self.n_training - rest, self.n_training)
                raw_batch = np.empty((rest, self.n_nus_data, self.n_times_data))
                train_set.read_direct(raw_batch, source_sel=np.s_[sl, :, :])
                raw_batch = raw_batch[:, self.mask]
                self._check_array_for_garbage(raw_batch, "training data remainder")
                train_y[sl] = y_scaler.transform(raw_batch)


            # add special training data
            for label in self.special_training:
                special_train_y = f["special_train"][label]["y"][:]
                special_train_y = special_train_y[:, self.mask].astype(np.float16)
                special_train_y = y_scaler.transform(special_train_y)
                train_y = np.concatenate((train_y, special_train_y))

            # add val data
            val_y_raw = f["val"]["y"][:self.n_val][:, self.mask]
            val_y = y_scaler.transform(val_y_raw)

        return train_y, val_y, y_scaler

    def _check_array_for_garbage(self, y: Array, label: str):

        if np.any(np.isnan(y)):
            raise ValueError(
                f"Found nans in data ({label})."
            )

        if np.any(np.isinf(y)):
            raise ValueError(
                f"Found infinites in data ({label})."
            )

                
    def preprocess_cVAE(self,
                        image_size: Int[Array, "shape=(2,)"],
                        conversion: str=None) -> tuple[Array, Array, Array, Array, object, object]:
        """
        Loads in the training and validation data and performs data preprocessing for the CVAE using fiesta.utils.ImageScaler. 
        Because of memory issues, the training data set is loaded in chunks.
        The X arrays (parameter values) are standardized with fiesta.utils.StandardScalerJax.

        Args:
            image_size (Array[Int]): Image size the 2D flux arrays are down sampled to with jax.image.resize
            conversion (str): references how to convert the parameters for the training. Defaults to None, in which case it's the identity.
        Returns:
            train_X (Array): Standardized training parameters.
            train_y (Array): PCA coefficients of the training data. 
            val_X (Array): Standardized validation parameters
            val_y (Array): PCA coefficients of the validation data.
            Xscaler (StandardScalerJax): Standardizer object fitted to the mean and sigma of the raw training data. Can be used to transform and inverse transform parameter points.
            yscaler (ImageScaler): ImageScaler object fitted to part of the raw training data. Can be used to transform and inverse transform log spectral flux densities.
        """
        Xscaler = ParameterScaler(scaler=scalers.StandardScalerJax(),
                                  parameter_names=self.parameter_names,
                                  conversion=conversion)
        yscaler = DataScaler(scalers=[scalers.ImageScaler(downscale=image_size, upscale=(self.n_nus, self.n_times)), scalers.StandardScalerJax()])
        
        # load potentially large training data set
        train_X, train_y, Xscaler, yscaler.scalers[0] = self._preprocess_training_batches(Xscaler, yscaler.scalers[0], image_size)
        train_y = train_y.reshape(-1, jnp.prod(image_size))
        # standardize the down sampled fluxes
        train_y = yscaler.scalers[1].fit_transform(train_y)

        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)
        return train_X, train_y, val_X, val_y, Xscaler, yscaler
        
    def preprocess_svd(self,
                       svd_ncoeff: Int,
                       filters: list,
                       conversion: str=None) -> tuple[Array, dict[str, Array], Array, dict[str, Array], object, dict[str, object]]:
        """
        Loads in the training and validation data and performs data preprocessing for the SVD decomposition using fiesta.utils.SVDDecomposer. 
        This is done *per filter* supplied in the filters argument which is equivalent to the old NMMA procedure.
        The X arrays (parameter values) are scaled to [0,1] with MinMaxScalerJax()

        Args:
            svd_ncoeff (Int): Number of SVD coefficients to keep
            filters (Filter[list]): List of fiesta.utils.filter instances that are used to convert the fluxes to magnitudes
            conversion (str): references how to convert the parameters for the training. Defaults to None, in which case it's the identity.

        Returns:
            train_X (Array): Scaled training parameters.
            train_y (dict[Array]): Dictionary of the SVD coefficients of the training magnitude lightcurves with the filter names as keys
            val_X (Array): Scaled validation parameters
            val_y (dict[Array]): Dictionary of the SVD coefficients of the validation magnitude lightcurves with the filter names as keys
            Xscaler (ParameterScaler): MinMaxScaler object fitted to the minimum and maximum of the training data parameters. Can be used to transform and inverse transform parameter points.
            yscaler (dict[str, SVDDecomposer]): Dictionary of SVDDecomposer objects with the filter names as keys. The SVDDecomposer objects are fitted to the magnitude training data. Can be used to transform and inverse transform magnitudes in this filter.
        """
        Xscaler = ParameterScaler(conversion=conversion,
                                  scaler=scalers.MinMaxScalerJax(),
                                  parameter_names=self.parameter_names)
        yscaler = {filt.name: DataScaler([scalers.SVDDecomposer(svd_ncoeff)]) for filt in filters}
        train_y = {}
        val_y = {}

        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X_raw = concatenate_redshift(train_X_raw)
            train_X = Xscaler.fit_transform(train_X_raw) # fit the Xscaler and transform the train_X_raw

            for label in self.special_training:
                    special_train_X_raw = f["special_train"][label]["X"][:]
                    special_train_X_raw = concatenate_redshift(special_train_X_raw)
                    special_train_X = Xscaler.transform(special_train_X_raw)

                    train_X = np.concatenate((train_X, special_train_X))
            
            val_X_raw = f["val"]["X"][:self.n_val]
            val_X_raw = concatenate_redshift(val_X_raw)
            val_X = Xscaler.transform(val_X_raw)

            train_y_raw = f["train"]["y"][:self.n_training, self.mask].reshape(-1, self.n_nus, self.n_times)
            mJys_train = np.exp(train_y_raw)
            val_y_raw =  f["val"]["y"][:self.n_val, self.mask].reshape(-1, self.n_nus, self.n_times)
            mJys_val = np.exp(val_y_raw)
            
            for filt in filters:
                mag = redshifted_magnitude(filt, mJys_train, self.nus, train_X_raw[:,-1]) # convert to magnitudes
                train_data = yscaler[filt.name].fit_transform(mag)

                # preprocess the special training data
                for label in self.special_training:
                    special_train_y = np.exp(f["special_train"][label]["y"][:, self.mask].reshape(-1, self.n_nus, self.n_times))
                    special_mag = redshifted_magnitude(filt, special_train_y, self.nus, special_train_X_raw[:,-1]) # convert to magnitudes
                    special_train_data = yscaler[filt.name].transform(special_mag)
                    train_data = np.concatenate((train_data, special_train_data))

                train_y[filt.name] = train_data
    
                # preprocess validation data
                mag = redshifted_magnitude(filt, mJys_val, self.nus, val_X_raw[:,-1]) # convert to magnitudes
                val_data = yscaler[filt.name].transform(mag)
                val_y[filt.name] = val_data

        return train_X, train_y, val_X, val_y, Xscaler, yscaler