"""Method to train the surrogate models"""

import os
import numpy as np

import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int

from fiesta.utils import MinMaxScalerJax, StandardScalerJax, PCAdecomposer
from fiesta import utils
from fiesta import conversions
from fiesta import models_utilities
import fiesta.train.neuralnets as fiesta_nn

import matplotlib.pyplot as plt
import pickle
import h5py
from typing import Callable


class FluxTrainer:
    """Abstract class for training a collection of surrogate"""
    
    name: str
    outdir: str
    parameter_names: list[str]
    
    preprocessing_metadata: dict[str, dict[str, float]]
    
    X_raw: Float[Array, "n_batch n_params"]
    y_raw: dict[str, Float[Array, "n_batch n_times"]]
    
    X: Float[Array, "n_batch n_input_surrogate"]
    y: dict[str, Float[Array, "n_batch n_output_surrogate"]]
    
    trained_states: dict[str, fiesta_nn.TrainState]
    
    def __init__(self, 
                 name: str,
                 outdir: str,
                 plots_dir: str = None, 
                 ) -> None:
        
        self.name = name
        self.outdir = outdir
        # Check if directories exists, otherwise, create:
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        self.plots_dir = plots_dir
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
       
        # To be loaded by child classes
        self.parameter_names = None
                
        self.preprocessing_metadata = {}
        
        self.train_X_raw = None
        self.train_y_raw = None

        self.val_X_raw = None
        self.val_y_raw = None

    def __repr__(self) -> str:
        return f"FluxTrainer(name={self.name})"
    
    def preprocess(self):
        
        print("Preprocessing data by scaling to mean 0 and std 1. . .")
        self.X_scaler = StandardScalerJax()
        self.X = self.X_scaler.fit_transform(self.train_X_raw)
        
        self.y_scaler = StandardScalerJax()
        self.y = self.y_scaler.fit_transform(self.train_y_raw)
            
        # Save the metadata
        self.preprocessing_metadata["X_scaler"] = self.X_scaler
        self.preprocessing_metadata["y_scaler"] = self.y_scaler
        print("Preprocessing data . . . done")
    
    def fit(self,
            config: fiesta_nn.NeuralnetConfig = None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            verbose: bool = True):
        """
        The config controls which architecture is built and therefore should not be specified here.
        
        Args:
            config (nn.NeuralnetConfig, optional): _description_. Defaults to None.
        """
        
        # Get default choices if no config is given
        if config is None:
            config = fiesta_nn.NeuralnetConfig()
        self.config = config
            
        input_ndim = len(self.parameter_names)

           
        # Create neural network and initialize the state
        net = fiesta_nn.MLP(layer_sizes=config.layer_sizes)
        key, subkey = jax.random.split(key)
        state = fiesta_nn.create_train_state(net, jnp.ones(input_ndim), subkey, config)
        
        # Perform training loop
        state, train_losses, val_losses = fiesta_nn.train_loop(state, config, self.train_X, self.train_y, self.val_X, self.val_y, verbose=verbose)
        # Plot and save the plot if so desired
        if self.plots_dir is not None:
            plt.figure(figsize=(10, 5))
            ls = "-o"
            ms = 3
            plt.plot([i+1 for i in range(len(train_losses))], train_losses, ls, markersize=ms, label="Train", color="red")
            plt.plot([i+1 for i in range(len(val_losses))], val_losses, ls, markersize=ms, label="Validation", color="blue")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.yscale('log')
            plt.title("Learning curves")
            plt.savefig(os.path.join(self.plots_dir, f"learning_curves_{self.name}.png"))
            plt.close()     
       
        self.trained_state = state
        
    def save(self):
        """
        Save the trained model and all the used metadata to the outdir.
        """
        # Save the metadata
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        meta_filename = os.path.join(self.outdir, f"{self.name}_metadata.pkl")
        
        save = {}
        save["times"] = self.times
        save["nus"] = self.nus
        save["parameter_names"] = self.parameter_names
        save.update(self.preprocessing_metadata) 

        with open(meta_filename, "wb") as meta_file:
            pickle.dump(save, meta_file)
        
        # Save the NN
        model = self.trained_state
        fiesta_nn.save_model(model, self.config, out_name=self.outdir + f"{self.name}.pkl")
    
    def _save_preprocessed_data(self):
        print("Saving preprocessed data . . .")
        np.savez(os.path.join(self.outdir, "afterglow_preprocessed_data.npz"), train_X=self.train_X, train_y= self.train_y, val_X = self.val_X, val_y = self.val_y)
        print("Saving preprocessed data . . . done")

class PCATrainer(FluxTrainer):

    def __init__(self,
                 name: str,
                 outdir: str,
                 data_manager,
                 n_pca: Int = 100,
                 plots_dir: str = None,
                 save_preprocessed_data: bool = False):

        super().__init__(name = name,
                       outdir = outdir,
                       plots_dir = plots_dir)

        self.n_pca = n_pca
        self.save_preprocessed_data = save_preprocessed_data
        self.data_manager = data_manager
        self.parameter_names = data_manager.parameter_names
        self.times = data_manager.times
        self.nus = data_manager.nus
        
        self.plots_dir = plots_dir
        if self.plots_dir is not None and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        self.preprocess()
    
        if save_preprocessed_data:
            self._save_preprocessed_data()
        
    def preprocess(self):
        print(f"Fitting PCA model with {self.n_pca} components to the provided data.")
        self.train_X, self.train_y, self.val_X, self.val_y, self.X_scaler, self.y_scaler = self.data_manager.preprocess_data_from_file(self.n_pca)
        print(f"PCA model accounts for a share {np.sum(self.y_scaler.explained_variance_ratio_)} of the total variance in the training data. This value is hopefully close to 1.")
        self.preprocessing_metadata["X_scaler"] = self.X_scaler 
        self.preprocessing_metadata["y_scaler"] = self.y_scaler
        print("Preprocessing data . . . done")

    def load_parameter_names(self):
        raise NotImplementedError
        
    def load_times(self):
        raise NotImplementedError
       
    def load_raw_data(self):
        raise NotImplementedError

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
                 ):
        
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
        with h5py.File(self.file, "r") as f:
            self.times_data = f["times"][:]
            self.nus_data = f["nus"][:]
            self.parameter_names =  f["parameter_names"][:].astype(str).tolist()
            self.n_training_exists = f["train"]["X"].shape[0]
            self.n_val_exists = f["val"]["X"].shape[0]
    
    def set_up_domain_mask(self,)->None:
        """Trims the stored data down to the time and frequency range desired for training."""
        
        if self.tmin<self.times_data.min() or self.tmax>self.times_data.max():
            print(f"\nWarning: provided time range {self.tmin, self.tmax} is too wide for the data stored in file. Using range {max(self.times_data.min(), self.tmin), min(self.times_data.max(), self.tmax)} instead.\n")
        time_mask = np.logical_and(self.times_data>=self.tmin, self.times_data<=self.tmax)
        self.times = self.times_data[time_mask]
        self.n_times = len(self.times)

        if self.numin<self.nus_data.min() or self.numax>self.nus_data.max():
            print(f"\nWarning: provided frequency range {self.numin, self.numax} is too wide for the data stored in file. Using range {max(self.nus_data.min(), self.numin), min(self.nus_data.max(), self.numax)} instead.\n")
        nu_mask = np.logical_and(self.nus_data>=self.numin, self.nus_data<=self.numax)
        self.nus = self.nus_data[nu_mask]
        self.n_nus = len(self.nus)

        mask = nu_mask[:, None] & time_mask
        self.mask = mask.flatten()
    
    def get_data_from_file(self,):
        with h5py.File(self.file, "r") as f:
            if self.n_training>self.n_training_exists:
                raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
            self.train_X_raw = f["train"]["X"][:self.n_training]
            self.train_y_raw = f["train"]["y"][:self.n_training, self.mask]

            for label in self.special_training:
                self.train_X_raw = np.concatenate((self.train_X_raw, f["special_train"][label]["X"][:]))
                self.train_y_raw = np.concatenate((self.train_y_raw, f["special_train"][label]["y"][:, self.mask]))

            if self.n_val>self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to validate with {self.n_val} data points.")
            self.val_X_raw = f["val"]["X"][:self.n_val]
            self.val_y_raw = f["val"]["y"][:self.n_val, self.mask]
    
    def preprocess_data_from_file(self, n_components: int)->None:
        Xscaler, yscaler = StandardScalerJax(), PCAdecomposer(n_components=n_components)
        with h5py.File(self.file, "r") as f:
            # preprocess the training data
            if self.n_training>self.n_training_exists:
                raise ValueError(f"Only {self.n_training_exists} entries in file, not enough to train with {self.n_training} data points.")
            
            train_X_raw = f["train"]["X"][:self.n_training]
            for label in self.special_training:
                train_X_raw = np.concatenate((train_X_raw, f["special_train"][label]["X"][:]))
            train_X = Xscaler.fit_transform(train_X_raw)
            

            yscaler.fit(f["train"]["y"][:15_000, self.mask]) # only load 15k cause otherwise the array might get too large
            train_y = np.empty((self.n_training, n_components))
            n_loaded = 0
            for chunk in f["train"]["y"].iter_chunks():
                loaded = f["train"]["y"][chunk][:, self.mask]
                train_y[n_loaded:n_loaded+len(loaded)] = yscaler.transform(loaded)
                n_loaded += len(loaded)
                if n_loaded >= self.n_training:
                    break       
            for label in self.special_training:
                special_train_y = yscaler.transform(f["special_train"][label]["y"][:, self.mask])
                train_y = np.concatenate((train_y, special_train_y))

            # preprocess validation data
            if self.n_val>self.n_val_exists:
                raise ValueError(f"Only {self.n_val_exists} entries in file, not enough to train with {self.n_val} data points.")
            val_X_raw = f["val"]["X"][:self.n_val]
            val_X = Xscaler.transform(val_X_raw)
            val_y_raw = f["val"]["y"][:self.n_val, self.mask]
            val_y = yscaler.transform(val_y_raw)

            return train_X, train_y, val_X, val_y, Xscaler, yscaler
 
    
    def pass_data(self, object):
        object.parameter_names = self.parameter_names
        object.train_X_raw = self.train_X_raw
        object.train_y_raw = self.train_y_raw
        object.val_X_raw = self.val_X_raw
        object.val_y_raw = self.val_y_raw
        object.times = self.times
        object.nus = self.nus
    

    def print_file_info(self,):
        with h5py.File(self.file, "r") as f:
            print(f"Times: {f['times'][0]} {f['times'][-1]}")
            print(f"Nus: {f['nus'][0]} {f['nus'][-1]}")
            print(f"Parameter distributions: {f['parameter_distributions'][()].decode('utf-8')}")
            print("\n")
            print(f"Training data: {self.n_training_exists}")
            print(f"Validation data: {self.n_val_exists}")
            print(f"Test data: {f['test']['X'].shape[0]}")
            print("Special data:")
            for key in f['special_train'].keys():
                print(f"\t {key}: {f['special_train'][key]['X'].shape[0]}   description: {f['special_train'][key].attrs['comment']}")
            print("\n \n")