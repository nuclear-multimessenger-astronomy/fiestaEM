"""Method to train the surrogate models"""

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp 
from jaxtyping import Array
from beartype import beartype as typechecker

from fiesta.train import utils

class SurrogateTrainer():
    
    name: str
    
    def __init__(self, 
                 name: str) -> None:
        self.name = name
        
    def __repr__(self) -> str:
        return f"SurrogateTrainer(name={self.name})"
    
    # TODO: type hints
    def fit(self, 
            X: Array, 
            y: Array):
        raise NotImplementedError

class BullaSurrogateTrainer(SurrogateTrainer):
    
    def __init__(self,
                 name: str,
                 lc_dir: list[str],
                 outdir: str,
                 filters: list[str] = None,
                 ):
        
        """
        Initialize the surrogate model trainer
        
        Note: currently, only models of Bulla type .dat files are supported
        
        Args:
            name (str): Name of the surrogate model. Will be used 
            lc_dir (list[str]): Directory where all the raw light curve files, to be read and processed into a surrogate model.
            outdir (str): Directory where the trained surrogate model has to be saved.
            filters (list[str], optional): List of all the filters used in the light curve files and for which surrogate has to be trained. If None, all the filters will be used. Defaults to None.
        
        Returns:
        
        """
        
        super().__init__(name)
        self.lc_dir = lc_dir
        self.lc_files = [os.path.join(lc_dir, f) for f in os.listdir(lc_dir) if f.endswith(".dat")]

        # If no filters are given, we will read the filters from the first file and assume all files have the same filters
        if filters is None:
            filters = utils.get_filters_bulla_file(self.lc_files[0], drop_times=True)
        self.filters = filters
        self.outdir = outdir
        
    def __repr__(self) -> str:
        return f"BullaSurrogateTrainer(name={self.name}, lc_dir={self.lc_dir}, outdir={self.outdir}, filters={self.filters})"
    
    # TODO: implement fitting 
    def fit(self, X, y):
        pass
    
    
    #####################
    ### PREPROCESSING ###
    #####################
    
    def read_files(self) -> Array:
        """
        Read the photometry files and interpolate the NaNs

        Args:
            lc_files (list[str]): List of all the raw light curve files, to be read and processed into a surrogate model.
        """
        
        # TODO: figure out how to save?
        data = {filt: np.array([]) for filt in self.filters}

        # Fetch the result for each filter and add it to already existing data            
        for filename in self.lc_files:
            lc_data = utils.read_single_bulla_file(filename)
            for filt in self.filters:
                data[filt] = np.concatenate([data[filt], lc_data[filt]])
        
        return data
                
    def preprocess(self):
        # Read the data
        pass