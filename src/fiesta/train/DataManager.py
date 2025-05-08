from typing import Callable

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

def array_mask_from_interval(sorted_array, amin, amax):
    indmin = max(0, np.searchsorted(sorted_array, amin, side='right') -1)
    indmax = min(len(sorted_array)-1, np.searchsorted(sorted_array, amax))
    mask = np.logical_and(sorted_array>=sorted_array[indmin], sorted_array<=sorted_array[indmax])
    return mask

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

class DataManager:
    
    def __init__(self,
                 file: str,
                 n_training: Int,
                 n_val: Int,
                 tmin: Float,
                 tmax: Float,
                 numin: Float = 1e9,
                 numax: Float = 2.5e18,
                 special_training: list = [],
                 ) -> None:
        """
        DataManager class used to handle and preprocess the raw data from the physical model computations stored in an .h5 file.
        Initializing an instance of this class will only read in the meta data, the actual training data and validation data will only be loaded if one of the preprocessing methods is called.

        The .h5 file must contain the following data sets:
            - "times": times in days associated to the spectral flux densities
            - "nus": frequencies in Hz associated to the spectral flux densities
            - "parameter_names": list of the parameter names that are present in the training data.
            - "parameter_distributions": utf-8-string of a dict containing the boundaries and distribution of the parameters.
        Additionally, it must contain three data groups "train", "val", "test". Each of these groups contains two data sets, namely "X" and "y". 
        The X arrays contain the model parameters with columns in the order of "parameter_names" and thus have shape (-1, #parameters). The y array contains the associated log of the spectral flux densities in mJys and have shape (-1, #nus * #times).
        To get the full 2D log spectral flux density arrays, one needs to reshape 1D entries of y to (#nus, #times). 
        
        Args:
            file (str): Path to the .h5 file that contains the raw data.
            n_training (int): Number of training data points that will be read in and preprocessed. If used with a FluxTrainer, this is also the number of training data points used to train the model. 
                              Will raise a ValueError, if n_training is larger than the number of training data points stored in the file.
            n_val (int): Number of validation data points that will be read in and preprocessed. If used with a FluxTrainer, this is also the number of validation data points used to monitor the training progress. 
                              Will raise a ValueError, if n_val is larger than the number of validation data points stored in the file.
            tmin (float): Minimum time for which the data will be read in. Fluxes earlier than this time will not be loaded. Defaults to the minimum time of the stored data, if smaller than that value.
            max (float): Maximum time for which the data will be read in. Fluxes later than this time will not be loaded. Defaults to the maximum time of the stored data, if larger than that value.
            numin (float): Minimum frequency for which the data will be read in. Fluxes with frequencies lower than this frequency will not be loaded. Defaults to the minimum frequency of the stored data, if smaller than that value.
            numax (float): Maximum frequency for which the data will be read in. Fluxes with frequencies higher than this frequency will not be loaded. Defaults to the maximum frequency of the stored data, if larger than that value. Defaults to 1e9 Hz (1 GHz).
            special_training (list[str]): Batch of 'special' training data to be added. This can be customly designed training data to cover a certain area of the parameter space more intensily and should be stored in the .h5 file as f['special_train'][label]['X'] and f['special_train'][label]['y'], where label is an entry in this special_training. Defaults to [].
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
        self.mask = mask.flatten()
    
    def print_file_info(self,) -> None:
        """
        Prints the meta data of the raw data, i.e., time, frequencies, and parameter names to terminal. 
        Also prints how many training, validation, and test data points are available.
        """
        with h5py.File(self.file, "r") as f:
            logger.info(f"Times: {f['times'][0]} {f['times'][-1]}")
            logger.info(f"Nus: {f['nus'][0]} {f['nus'][-1]}")
            logger.info(f"Parameter distributions: {f['parameter_distributions'][()].decode('utf-8')}")
            logger.info("\n")
            logger.info(f"Training data: {self.n_training_exists}")
            logger.info(f"Validation data: {self.n_val_exists}")
            logger.info(f"Test data: {f['test']['X'].shape[0]}")
            logger.info("Special data:")
            for key in f['special_train'].keys():
                logger.info(f"\t {key}: {f['special_train'][key]['X'].shape[0]}   description: {f['special_train'][key].attrs['comment']}")
            logger.info("\n \n")
    
    def load_raw_data_from_file(self, n_training: int=1, n_val: int=0) -> tuple[Array, Array, Array, Array]:
        """Loads raw data for training and validation data and returns them as arrays"""
        with h5py.File(self.file, "r") as f:
            if n_training>self.n_training_exists:
                raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
            train_X_raw = f["train"]["X"][:n_training]
            train_y_raw = f["train"]["y"][:n_training, self.mask]

            if n_val>self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to validate with {self.n_val} data points.")
            
            val_X_raw = f["val"]["X"][:n_val]
            val_y_raw = f["val"]["y"][:n_val, self.mask]
        
        return train_X_raw, train_y_raw, val_X_raw, val_y_raw
    
    def preprocess_pca(self, 
                       n_components: int,
                       conversion: str=None) -> tuple[Array, Array, Array, Array, object, object]:
        """
        Loads in the training and validation data and performs PCA decomposition using fiesta.utils.PCADecomposer. 
        Because of memory issues, the training data set is loaded in chunks.
        The X arrays (parameter values) are standardized with fiesta.utils.StandardScalerJax.

        Args:
            n_components(int): Number of PCA components to keep.
            conversion (str): references how to convert the parameters for the training. Defaults to None, in which case it's the identity.

        Returns:
            train_X (Array): Standardized training parameters.
            train_y (Array): PCA coefficients of the training data. 
            val_X (Array): Standardized validation parameters
            val_y (Array): PCA coefficients of the validation data.
            Xscaler (StandardScalerJax): Standardizer object fitted to the mean and sigma of the raw training data. Can be used to transform and inverse transform parameter points.
            yscaler (PCAdecomposer): PCADecomposer object fitted to part of the raw training data. Can be used to transform and inverse transform log spectral flux densities.
        """
        Xscaler = ParameterScaler(scaler=scalers.StandardScalerJax(),
                                  parameter_names=self.parameter_names,
                                  conversion=conversion)
        yscaler = DataScaler([scalers.PCADecomposer(n_components=n_components)])
        
        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X = Xscaler.fit_transform(train_X_raw) # fit the Xscaler and transform the train_X_raw
            
            y_set = f["train"]["y"]
            loaded = y_set[: min(20_000, self.n_training), self.mask].astype(np.float16) # only load max. 20k cause otherwise we might run out of memory at this step
            assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
            yscaler.fit(loaded) # fit the yscaler and transform with the loaded data
            del loaded; gc.collect() # remove loaded from memory

            train_y = np.empty((self.n_training, n_components))

            chunk_size = y_set.chunks[0] # load raw data in chunks of chunk_size
            nchunks, rest = divmod(self.n_training, chunk_size) # load raw data in chunks of chunk_size
            for j, chunk in enumerate(y_set.iter_chunks()):
                if j >= nchunks:
                    break
                loaded = y_set[chunk][:, self.mask]
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[j*chunk_size:(j+1)*chunk_size] = yscaler.transform(loaded)
            if rest > 0:
                loaded = y_set[-rest:, self.mask]
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[-rest:] = yscaler.transform(loaded)
        
        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)

        return train_X, train_y, val_X, val_y, Xscaler, yscaler
    
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
        
        # preprocess the training data
        with h5py.File(self.file, "r") as f:
            train_X_raw = f["train"]["X"][:self.n_training]
            train_X = Xscaler.fit_transform(train_X_raw) # fit the Xscaler and transform the train_X_raw

            y_set = f["train"]["y"]

            train_y = np.empty((self.n_training, jnp.prod(image_size)), dtype=jnp.float16)
            
            chunk_size = y_set.chunks[0]
            nchunks, rest = divmod(self.n_training, chunk_size) # create raw data in chunks of chunk_size
            for j, chunk in enumerate(y_set.iter_chunks()):
                if j>= nchunks:
                    break
                loaded = y_set[chunk][:, self.mask].astype(jnp.float16)
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[j*chunk_size:(j+1)*chunk_size] = yscaler.scalers[0].transform(loaded).reshape(-1, jnp.prod(image_size))

            if rest > 0:
                loaded = y_set[-rest:, self.mask].astype(jnp.float16)
                assert not np.any(np.isinf(loaded)), f"Found inftys in training data."
                train_y[-rest:] = yscaler.scalers[0].transform(loaded).reshape(-1, jnp.prod(image_size))
            
            train_y = yscaler.scalers[1].fit_transform(train_y) # this standardizes now the down sampled fluxes

        # preprocess the special training data as well ass the validation data
        train_X, train_y, val_X, val_y = self.__preprocess__special_and_val_data(train_X, train_y, Xscaler, yscaler)
        return train_X, train_y, val_X, val_y, Xscaler, yscaler
    

    def __preprocess__special_and_val_data(self, train_X, train_y, Xscaler, yscaler) -> tuple[Array, Array, Array, Array]:
        """ sub method that just applies the scaling transforms to the validation and special training data """
        with h5py.File(self.file, "r") as f:
            # preprocess the special training data       
            for label in self.special_training:
                special_train_X = Xscaler.transform(f["special_train"][label]["X"][:])
                train_X = np.concatenate((train_X, special_train_X))

                special_train_y = yscaler.transform(f["special_train"][label]["y"][:, self.mask])
                train_y = np.concatenate(( train_y, special_train_y.astype(jnp.float16) ))

            # preprocess validation data
            val_X_raw = f["val"]["X"][:self.n_val]
            val_X = Xscaler.transform(val_X_raw)
            val_y_raw = f["val"]["y"][:self.n_val, self.mask]
            val_y = yscaler.transform(val_y_raw)
        
        return train_X, train_y, val_X, val_y
    
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
                
    def pass_meta_data(self, object) -> None:
        """Pass training data meta data to another object. Used for the FluxTrainers."""
        object.parameter_names = self.parameter_names
        object.times = self.times
        object.nus = self.nus
        object.parameter_distributions = self.parameter_distributions