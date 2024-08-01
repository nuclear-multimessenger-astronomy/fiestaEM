"""Method to train the surrogate models"""

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float
from beartype import beartype as typechecker
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

from fiesta.train import utils

class SurrogateTrainer:
    """Abstract class for training the surrogate models"""
    
    name: str
    parameter_names: list[str]
    
    def __init__(self, 
                 name: str) -> None:
        self.name = name
        self.parameter_names = []
        
    def __repr__(self) -> str:
        return f"SurrogateTrainer(name={self.name})"
    
    # TODO: type hints
    def fit(self, 
            X: Array, 
            y: Array):
        raise NotImplementedError
    
    def preprocess(self):
        raise NotImplementedError

class BullaSurrogateTrainer(SurrogateTrainer):
    
    def __init__(self,
                 name: str,
                 lc_dir: list[str],
                 outdir: str,
                 filters: list[str] = None,
                 svd_ncoeff: int = 10
                 ):
        
        """
        Initialize the surrogate model trainer
        
        Note: currently, only models of Bulla type .dat files are supported
        
        Args:
            name (str): Name of the surrogate model. Will be used 
            lc_dir (list[str]): Directory where all the raw light curve files, to be read and processed into a surrogate model.
            outdir (str): Directory where the trained surrogate model has to be saved.
            filters (list[str], optional): List of all the filters used in the light curve files and for which surrogate has to be trained. If None, all the filters will be used. Defaults to None.
        """
        
        # Check if supported
        supported_models = list(utils.BULLA_PARAMETER_NAMES.keys())
        if name not in supported_models:
            raise ValueError(f"Model {name} is not supported yet. Supported models are: {supported_models}")
        
        super().__init__(name)
        self.lc_dir = lc_dir
        self.outdir = outdir
        self.svd_ncoeff = svd_ncoeff
        
        self.lc_files = [os.path.join(lc_dir, f) for f in os.listdir(lc_dir) if f.endswith(".dat")]

        # If no filters are given, we will read the filters from the first file and assume all files have the same filters
        if filters is None:
            filters = utils.get_filters_bulla_file(self.lc_files[0], drop_times=True)
        self.filters = filters
        
        # Fetch times
        self.times = utils.get_times_bulla_file(self.lc_files[0])
        
        # Assert all times are same
        for filename in self.lc_files:
            assert np.allclose(self.times, utils.get_times_bulla_file(filename)), "All the times should be same"
            
        # Fetch parameter names
        self.parameter_names = utils.BULLA_PARAMETER_NAMES[name]
        
        # TODO: change this if doesn't turn out well later on?
        self.preprocessing_metadata = {filt: {} for filt in self.filters}
        
        print("Reading data files and interpolating NaNs")
        self.raw_data, self.parameter_values = self.read_files()
        self.raw_data = utils.interpolate_nans(self.raw_data, self.times)
        
        # TODO: put the preprocess here etc
        
        
    def __repr__(self) -> str:
        return f"BullaSurrogateTrainer(name={self.name}, lc_dir={self.lc_dir}, outdir={self.outdir}, filters={self.filters})"
    
    #####################
    ### PREPROCESSING ###
    #####################
    
    def read_files(self) -> tuple[dict[str, Float[Array, " n_files n_times"]], Float[Array, "n_files n_params"]]:
        """
        Read the photometry files and interpolate the NaNs. 
        Output will be an array of shape (n_filters, n_files, n_times)

        Args:
            lc_files (list[str]): List of all the raw light curve files, to be read and processed into a surrogate model.
            
        Returns:
            tuple[dict[str, Float[Array, " n_files n_times"]], Float[Array, "n_files n_params"]]: First return value is a dictionary containing the filters and corresponding light curve data which has shape (n_files, n_times). Second return value is an array of all the parameter values extracted from the files.
        """
        
        # TODO: figure out how to save?
        data = {filt: [] for filt in self.filters}

        # Fetch the result for each filter and add it to already existing dataset
        for i, filename in enumerate(tqdm.tqdm(self.lc_files)):
            # Get a dictionary with keys being the filters and values being the light curve data
            lc_data = utils.read_single_bulla_file(filename)
            for filt in self.filters:
                if i == 0:
                    data[filt] = lc_data[filt]
                else:
                    data[filt] = np.vstack((data[filt], lc_data[filt]))
                    
            # Fetch the parameter values of this file
            params = utils.extract_Bu2022Ye_parameters(filename)
            if i == 0:
                parameter_values = params
            else:
                parameter_values = np.vstack((parameter_values, params))
                
        return data, parameter_values
                
    def preprocess(self) -> tuple[Float[Array, "n_files n_params"], dict[str, Float[Array, "n_files nsvd_coeff"]]]:
        # Scale inputs
        print("Preprocessing data")
        X_scaler = MinMaxScaler()
        X = X_scaler.fit_transform(self.parameter_values)
        
        # Scale outputs and save into y
        y = {filt: [] for filt in self.filters}
        for filt in tqdm.tqdm(self.filters):
            # TODO: this is now duplicate in each filter. Is this helpful?
            self.preprocessing_metadata[filt]["X_scaler"] = X_scaler
            
            data_scaler = MinMaxScaler()
            data = data_scaler.fit_transform(self.raw_data[filt])
            self.preprocessing_metadata[filt]["y_scaler"] = data_scaler
            
            # Do SVD decomposition
            # TODO: generalize this so that people can also easily train on the full lightcurve if they want to
            UA, _, VA = np.linalg.svd(data, full_matrices=True)
            VA = VA.T

            n, n = UA.shape
            m, m = VA.shape

            # This is taken over from NMMA
            cAmat = np.zeros((self.svd_ncoeff, n))
            cAvar = np.zeros((self.svd_ncoeff, n))
            for i in range(n):
                ErrorLevel = 1.0
                cAmat[:, i] = np.dot(
                    data[i, :], VA[:, : self.svd_ncoeff]
                )
                errors = ErrorLevel * np.ones_like(data[i, :])
                cAvar[:, i] = np.diag(
                    np.dot(
                        VA[:, : self.svd_ncoeff].T,
                        np.dot(np.diag(np.power(errors, 2.0)), VA[:, : self.svd_ncoeff]),
                    )
                )
            cAstd = np.sqrt(cAvar)
            
            self.preprocessing_metadata[filt]["cAmat"] = cAmat
            self.preprocessing_metadata[filt]["cAstd"] = cAstd
            self.preprocessing_metadata[filt]["VA"] = VA
            
            y[filt] = cAmat.T # Transpose to get the shape (n_files, n_svd_coeff)
            
        self.X = X 
        self.y = y
        
        return X, y
    
    ###############
    ### FITTING ###
    ###############
    
    # TODO: implement fitting 
    def fit(self, X, y):
        pass